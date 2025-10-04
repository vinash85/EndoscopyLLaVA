#!/usr/bin/env python3
"""
Frame & Image Pipeline: Blur Filtering, Optional Preprocessing, and Feature Clustering

Combined from two user-provided snippets and cleaned for CLI use on a GPU server (e.g., H100).

- Utilities (OpenCV/Numpy): histogram equalization, FFT-based blur scoring, reflection mask
- Vision backbone (PyTorch): default ResNet-50 global features; optional CLIP ViT-B/32 if available
- K-Means clustering of labeled features to map clusters->classes, then predict labels for unlabeled
- Optional: filter unlabeled frames by blur score before clustering (keeps both sets for review)

Dependencies (tested with recent PyTorch/Torchvision + OpenCV):
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    python -m pip install opencv-python-headless scikit-learn pillow numpy
Optional (for --backbone clip_vit_b32):
    python -m pip install open_clip_torch  # preferred
    # or rely on torchvision>=0.16 which provides torchvision.models.clip

Example usage (two-stage):

# 1) (Optional) Pre-filter unlabeled frames by blur score and save into subfolders
python frame_cluster_pipeline.py \
  --mode preprocess \
  --unlabeled_dir /data/unlabeled_frames \
  --output_dir /data/outputs_pre \
  --blur_scale 1.0

# 2) Cluster and label using a small annotated set + the (possibly prefiltered) unlabeled dir
python frame_cluster_pipeline.py \
  --mode cluster \
  --annotated_dir /data/annotated \
  --unlabeled_dir /data/unlabeled_frames \
  --output_dir /data/outputs_cluster \
  --batch_size 64 \
  --backbone resnet50

One-shot pipeline:
python frame_cluster_pipeline.py \
  --mode all \
  --annotated_dir /data/annotated \
  --unlabeled_dir /data/unlabeled_frames \
  --output_dir /data/outputs_all \
  --blur_scale 1.0 \
  --batch_size 64

Notes:
- This script auto-detects CUDA; it will use the H100 if a recent CUDA-enabled PyTorch is installed.
- If you see CUDA/driver mismatches on H100, update PyTorch to a CUDA 12.x build (see pip command above).
"""

import os
import argparse
import shutil
import csv
import random
import pickle
import warnings
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# ---- OpenCV / Matplotlib utilities (no plotting required for headless use) ----
import cv2

# ---- Torch / Torchvision for feature extraction ----
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from PIL import Image

# ---- Clustering ----
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Image utility functions
# ----------------------------

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization.
    - For grayscale: standard equalization.
    - For color: equalize the luminance (Y) in YCrCb.

    Args:
        image: np.ndarray (BGR or grayscale)
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    elif image.ndim == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        raise ValueError("Unsupported image format for histogram_equalization.")


def get_all_images(folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
    folder = Path(folder_path)
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]


def energy_fft(image_bgr: np.ndarray, mask_size: int = 60) -> float:
    """Compute a high-frequency 'energy' score via FFT magnitude (for blur detection)."""
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    h, w = gray.shape[:2]
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    cx, cy = w // 2, h // 2
    ms = int(mask_size)
    y0, y1 = max(0, cy - ms), min(h, cy + ms)
    x0, x1 = max(0, cx - ms), min(w, cx + ms)
    magnitude[y0:y1, x0:x1] = 0  # suppress low-frequencies

    high_freq_energy = float(np.mean(magnitude))
    return high_freq_energy


def divide_dictionary_by_threshold(original_dict, threshold):
    """Split dict into <= threshold and > threshold."""
    below, above = {}, {}
    for k, v in original_dict.items():
        (below if v <= threshold else above)[k] = v
    return below, above


def get_mean_std(data):
    data = np.asarray(list(data), dtype=float)
    return float(np.mean(data)), float(np.std(data))


def eliminate_blur_frames(img_dir: str, scale: float = 1.0):
    """
    Score frames by FFT energy; compute threshold = mean - scale*std; split to blurry/non-blurry.
    Returns (blur_dict, non_blur_dict), each mapping path->score.
    """
    img_paths = get_all_images(img_dir)
    energy = {}

    print(f"[Blur] Processing {len(img_paths)} images in {img_dir} ...")
    for p in img_paths:
        # Use cv2 to ensure BGR np.uint8
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            warnings.warn(f"Could not read image: {p}")
            continue
        energy[p] = energy_fft(img)

    if not energy:
        print("[Blur] No images scored. Returning empty dicts.")
        return {}, {}

    scores = list(energy.values())
    mean, std = get_mean_std(scores)
    threshold = mean - scale * std
    print(f"[Blur] mean={mean:.4f}, std={std:.4f}, threshold={threshold:.4f} (<= = blurry)")
    blur, non_blur = divide_dictionary_by_threshold(energy, threshold)
    print(f"[Blur] Blurry: {len(blur)} | Non-blurry: {len(non_blur)}")
    return blur, non_blur


def save_frames_split(save_root: str, non_blur: dict, blur: dict):
    nb_dir = os.path.join(save_root, "non_blur_frames")
    b_dir = os.path.join(save_root, "blur_frames")
    os.makedirs(nb_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    for k in non_blur.keys():
        shutil.copy(str(k), os.path.join(nb_dir, os.path.basename(str(k))))
    for k in blur.keys():
        shutil.copy(str(k), os.path.join(b_dir, os.path.basename(str(k))))

    print(f"[Blur] Saved copies to:\n - {nb_dir}\n - {b_dir}")


def reflection_mask(frame_bgr: np.ndarray, thresh_val: int = 220, morph_k: int = 3):
    """Binary mask of bright/specular areas. Returns (mask_uint8, area_ratio_float)."""
    v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    _, mask = cv2.threshold(v, int(thresh_val), 255, cv2.THRESH_BINARY)

    if morph_k and morph_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    area_ratio = float((mask > 0).sum()) / float(mask.size)
    return mask.astype(np.uint8), area_ratio


# ----------------------------
# Datasets for torch
# ----------------------------

class LabeledImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, path


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None,
                 extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, path


# ----------------------------
# Backbones
# ----------------------------

def get_resnet50_extractor(device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(model.children())[:-1]  # drop fc
    model = torch.nn.Sequential(*modules)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    feat_dim = 2048
    return model, feat_dim


def try_get_clip_vit_b32_extractor(device):
    """
    Attempts to create a CLIP ViT-B/32 image encoder.
    Prefers open_clip; falls back to torchvision.models.clip if available.
    """
    # Try open_clip
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        def encode(batch):
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats

        feat_dim = model.visual.output_dim if hasattr(model.visual, 'output_dim') else 512
        # Build a transform that matches open_clip preprocess
        tfm = preprocess
        return encode, tfm, feat_dim, 'open_clip'
    except Exception as e:
        warnings.warn(f"open_clip unavailable or failed: {e}")

    # Try torchvision
    try:
        from torchvision.models.clip import CLIPVisionModelWithProjection, CLIPVisionConfig  # noqa: F401
        warnings.warn("torchvision CLIP path not fully implemented here; falling back to ResNet-50.")
    except Exception:
        pass

    return None, None, None, None


def get_backbone(backbone_name: str, device):
    """
    Returns a tuple: (forward_fn, transform, feat_dim, name)
    - For resnet50: forward_fn(tensor[B,3,224,224]) -> [B,2048]
    - For clip_vit_b32: forward_fn returns projected features
    """
    backbone_name = (backbone_name or 'resnet50').lower()

    if backbone_name == 'clip_vit_b32':
        encode_fn, tfm, feat_dim, tag = try_get_clip_vit_b32_extractor(device)
        if encode_fn is not None:
            def forward(batch):
                feats = encode_fn(batch)
                return feats
            return forward, tfm, feat_dim, 'clip_vit_b32'
        warnings.warn("Falling back to ResNet-50 due to CLIP unavailability.")

    # Default: ResNet-50
    model, feat_dim = get_resnet50_extractor(device)
    # Standard ImageNet transform for ResNet-50
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def forward(batch):
        with torch.no_grad():
            out = model(batch)
            out = out.view(out.size(0), -1)
            return out

    return forward, tfm, feat_dim, 'resnet50'


# ----------------------------
# Feature extraction helpers
# ----------------------------

def extract_features_labeled(forward_fn, dataset_root, transform, device,
                             batch_size=32, num_workers=4):
    ds = LabeledImageDataset(dataset_root, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    feats, labels, paths = [], [], []
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        f = forward_fn(x).detach().cpu().numpy()
        feats.append(f)
        labels.extend(y.numpy().tolist())
        paths.extend(p)

    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1))
    labels = np.array(labels, dtype=int)
    return feats, labels, paths, ds.classes


def extract_features_unlabeled(forward_fn, dataset_root, transform, device,
                               batch_size=32, num_workers=4):
    ds = UnlabeledImageDataset(dataset_root, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    feats, paths = [], []
    for x, p in loader:
        x = x.to(device, non_blocking=True)
        f = forward_fn(x).detach().cpu().numpy()
        feats.append(f)
        paths.extend(p)

    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1))
    return feats, paths


# ----------------------------
# Label normalization helper (NEW)
# ----------------------------

def normalize_kudo_folder(label: str):
    """
    Map a predicted label to one of the five Kudo folders.
    Accepts variants like 'I', 'Kudo I', 'kudo_ii', 'Type III', etc.
    Returns folder name like 'kudo_I'...'kudo_V' or None if no match.
    """
    s = str(label).strip().lower().replace('_', ' ').replace('-', ' ')
    s = s.replace('kudo', '').replace('type', '').strip()
    tokens = s.split()
    candidate = tokens[-1] if tokens else s

    roman_map = {
        'i': 'kudo_I',
        'ii': 'kudo_II',
        'iii': 'kudo_III',
        'iv': 'kudo_IV',
        'v': 'kudo_V',
        '1': 'kudo_I',
        '2': 'kudo_II',
        '3': 'kudo_III',
        '4': 'kudo_IV',
        '5': 'kudo_V',
    }
    return roman_map.get(candidate, None)


# ----------------------------
# Clustering + labeling
# ----------------------------

def cluster_and_label(annotated_dir, unlabeled_dir, output_dir, device,
                      backbone='resnet50', batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    forward_fn, tfm, feat_dim, used_bb = get_backbone(backbone, device)
    print(f"[Backbone] Using {used_bb} feature dim={feat_dim}")

    # 1) Labeled features
    feats_lab, labs, paths_lab, class_names = extract_features_labeled(
        forward_fn, annotated_dir, tfm, device, batch_size=batch_size
    )
    if feats_lab.shape[0] == 0:
        raise RuntimeError("No labeled features extracted. Check annotated_dir.")

    # 2) Standardize + KMeans
    scaler = StandardScaler()
    feats_lab_std = scaler.fit_transform(feats_lab)
    num_clusters = len(class_names)
    km = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cl_ids = km.fit_predict(feats_lab_std)

    # Map clusters -> class via majority vote of labeled data
    cl_to_cls = {}
    for i in range(num_clusters):
        idxs = np.where(cl_ids == i)[0]
        if len(idxs) == 0:
            continue
        cls_ids = labs[idxs]
        mc = Counter(cls_ids).most_common(1)[0][0]
        cl_to_cls[i] = class_names[mc]

    # Persist scaler & kmeans
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, 'kmeans.pkl'), 'wb') as f:
        pickle.dump(km, f)
    with open(os.path.join(output_dir, 'cluster_to_class.pkl'), 'wb') as f:
        pickle.dump(cl_to_cls, f)

    # 3) Unlabeled features
    feats_u, paths_u = extract_features_unlabeled(
        forward_fn, unlabeled_dir, tfm, device, batch_size=batch_size
    )
    if feats_u.shape[0] == 0:
        warnings.warn("No unlabeled features extracted. Check unlabeled_dir. Skipping predictions.")
        return

    feats_u_std = scaler.transform(feats_u)
    cl_ids_u = km.predict(feats_u_std)
    pred_labels = [cl_to_cls.get(i, f"cluster_{i}") for i in cl_ids_u]

    # Save CSV
    results_csv = os.path.join(output_dir, 'unlabeled_predictions.csv')
    with open(results_csv, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['path', 'predicted_label', 'cluster_id'])
        for p, lbl, cid in zip(paths_u, pred_labels, cl_ids_u):
            w.writerow([p, lbl, int(cid)])

    # 4) Save images into five folders, one per Kudo classification (NEW)
    #    Folders: kudo_I, kudo_II, kudo_III, kudo_IV, kudo_V
    kudo_bins = {
        'kudo_I': [],
        'kudo_II': [],
        'kudo_III': [],
        'kudo_IV': [],
        'kudo_V': [],
    }
    unknown_bin = []  # in case some class names don't map cleanly

    for p, lbl in zip(paths_u, pred_labels):
        folder = normalize_kudo_folder(lbl)
        if folder is None:
            unknown_bin.append(p)
        else:
            kudo_bins[folder].append(p)

    # create destination root
    dest_root = os.path.join(output_dir, 'kudo_folders')
    os.makedirs(dest_root, exist_ok=True)

    # copy files into their Kudo folders
    for folder, flist in kudo_bins.items():
        fdir = os.path.join(dest_root, folder)
        os.makedirs(fdir, exist_ok=True)
        for src in flist:
            dst = os.path.join(fdir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    # optionally save any unmapped labels to an 'unknown' folder for review
    if unknown_bin:
        udir = os.path.join(dest_root, 'unknown')
        os.makedirs(udir, exist_ok=True)
        for src in unknown_bin:
            dst = os.path.join(udir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    print(f"[Cluster] Complete. Predictions CSV: {results_csv}")
    print(f"[Cluster] Saved Kudo-sorted images to: {dest_root}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Blur filtering + optional reflection stats + feature clustering for frames/images."
    )
    parser.add_argument('--mode', type=str, choices=['preprocess', 'cluster', 'all'], default='all',
                        help="preprocess = blur split only; cluster = feature clustering; all = both.")
    parser.add_argument('--annotated_dir', type=str, help='Path to annotated images (subfolders per class).')
    parser.add_argument('--unlabeled_dir', type=str, help='Path to unlabeled images (or frames).')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--blur_scale', type=float, default=1.0,
                        help='Threshold = mean - scale*std (<= is blurry).')
    parser.add_argument('--save_blur_split', action='store_true',
                        help='Copy blurry/non-blurry to output subfolders.')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'clip_vit_b32'])
    parser.add_argument('--device', type=str, default=None, help="Force 'cuda' or 'cpu'. Default: auto.")

    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using device: {device}")

    # Ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Preprocess (blur split on unlabeled_dir)
    if args.mode in ('preprocess', 'all'):
        if not args.unlabeled_dir:
            parser.error("--unlabeled_dir is required for mode 'preprocess' or 'all'.")
        blur, non_blur = eliminate_blur_frames(args.unlabeled_dir, args.blur_scale)

        # Save a CSV with scores
        scores_csv = os.path.join(args.output_dir, 'blur_scores.csv')
        with open(scores_csv, 'w', newline='') as cf:
            w = csv.writer(cf)
            w.writerow(['path', 'fft_energy', 'is_blurry'])
            for dct, flag in ((blur, 1), (non_blur, 0)):
                for p, sc in dct.items():
                    w.writerow([str(p), float(sc), int(flag)])
        print(f"[Preprocess] Saved blur scores to {scores_csv}")

        if args.save_blur_split:
            save_frames_split(args.output_dir, non_blur, blur)

    # 2) Cluster
    if args.mode in ('cluster', 'all'):
        if not args.annotated_dir or not args.unlabeled_dir:
            parser.error("--annotated_dir and --unlabeled_dir are required for mode 'cluster' or 'all'.")

        cluster_and_label(
            annotated_dir=args.annotated_dir,
            unlabeled_dir=args.unlabeled_dir,
            output_dir=os.path.join(args.output_dir, 'cluster_results'),
            device=device,
            backbone=args.backbone,
            batch_size=args.batch_size,
        )


if __name__ == '__main__':
    main()
