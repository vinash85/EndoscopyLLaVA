from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
import os
import secrets
import hashlib
import csv
import json

app = Flask(__name__)
with open("questions.json", "r") as f:
    questions = json.load(f)
app.secret_key = "super_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///polypLabeler.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

### MODELS ###
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(200), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    progress = db.Column(db.Integer, default=0)
    password_reset_token = db.Column(db.String(200), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

### ROUTES ###
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form["fullname"]
        username = request.form["username"]
        password = request.form["password"]

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists. Please choose a different one.", "error")
            return redirect(url_for("signup"))

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(fullname=fullname, username=username, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("User not found or incorrect password. Please sign up if you don't have an account.", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))




@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", fullname=current_user.fullname, progress=current_user.progress, total=len(questions))

@app.route("/api/get_item/<int:progress>")
@login_required
def get_item(progress):
    global questions
    if progress < 0 or progress >= len(questions):
        return jsonify({"error": "Invalid progress index"}), 404
    
    data = questions[progress]
    image = data["image"]
    question = data["question"]
    options = data["options"]
    
    return jsonify({
        "image": url_for('static', filename="images/" + image),
        "question": question,
        "progress": progress,
        "options": options
    })

# New API endpoint to check if a question has been answered
@app.route("/api/get_answered_questions")
@login_required
def get_answered_questions():
    username = current_user.username
    csv_file_path = f"answers/{username}_answers.csv"
    
    answered_questions = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header row
            for row in reader:
                answered_questions.append(row[0])  # Assuming the first column is image_name
    
    return jsonify({"answered": answered_questions})

@app.route("/api/submit_answer", methods=["POST"])
@login_required
def submit_answer():
    data = request.json
    answer = data.get("answer")
    image_name = data.get("image_name")
    
    if not answer or not image_name:
        return jsonify({"status": "error", "message": "Missing answer or image name"}), 400
        
    username = current_user.username
    csv_file_path = f"answers/{username}_answers.csv"
    
    # Create the answers directory if it doesn't exist
    if not os.path.exists("answers"):
        os.makedirs("answers")
        
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_name", "answer"])
        writer.writerow([image_name, answer])

    # Now, save the progress as before
    current_user.progress = current_user.progress + 1
    db.session.commit()
    
    return jsonify({"status": "success", "progress": current_user.progress})

@app.route("/api/get_user_answer/<image_name>")
@login_required
def get_user_answer(image_name):
    username = current_user.username
    csv_file_path = f"answers/{username}_answers.csv"
    
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header row
            for row in reader:
                if row and len(row) > 1 and row[0] == image_name:
                    return jsonify({"answer": row[1]})
    
    return jsonify({"answer": None})

@app.route("/api/change_answer", methods=["POST"])
@login_required
def change_answer():
    data = request.json
    image_name = data.get("image_name")
    new_answer = data.get("new_answer")
    
    if not image_name or not new_answer:
        return jsonify({"status": "error", "message": "Missing image name or new answer"}), 400
    
    username = current_user.username
    csv_file_path = f"answers/{username}_answers.csv"
    
    if not os.path.exists(csv_file_path):
        return jsonify({"status": "error", "message": "No answers file found"}), 404
        
    temp_file_path = f"answers/{username}_answers_temp.csv"
    
    found = False
    with open(csv_file_path, "r") as read_f, open(temp_file_path, "w", newline="") as write_f:
        reader = csv.reader(read_f)
        writer = csv.writer(write_f)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            if row and len(row) > 0 and row[0] == image_name:
                writer.writerow([image_name, new_answer])
                found = True
            else:
                writer.writerow(row)
                
    if found:
        os.replace(temp_file_path, csv_file_path)
        return jsonify({"status": "success"})
    else:
        os.remove(temp_file_path)
        return jsonify({"status": "error", "message": "Question not found in answers"}), 404

@app.route("/api/save_progress", methods=["POST"])
@login_required
def save_progress():
    data = request.json
    new_progress = data.get("progress", current_user.progress)
    current_user.progress = new_progress
    db.session.commit()
    return jsonify({"status": "success", "progress": current_user.progress})

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form["username"]
        user = User.query.filter_by(username=username).first()
        if user:
            token = secrets.token_hex(16)
            hashed_token = hashlib.sha256(token.encode()).hexdigest()
            user.password_reset_token = hashed_token
            db.session.commit()
            flash(f"A password reset token has been generated. Please copy this token: {token}", "success")
            return redirect(url_for("forgot_password"))
        else:
            flash("User not found. Please check your username.", "error")
            return redirect(url_for("forgot_password"))
    return render_template("forgot_password.html")

@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        username = request.form["username"]
        token = request.form["token"]
        new_password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        
        hashed_token = hashlib.sha256(token.encode()).hexdigest()

        if user and user.password_reset_token == hashed_token:
            hashed_pw = bcrypt.generate_password_hash(new_password).decode('utf-8')
            user.password = hashed_pw
            user.password_reset_token = None
            db.session.commit()
            flash("Your password has been reset successfully.", "success")
            return redirect(url_for("login"))
        else:
            flash("Invalid username or token.", "error")
            return redirect(url_for("reset_password"))
    return render_template("reset_password.html")

@app.route("/find_username", methods=["GET", "POST"])
def find_username():
    if request.method == "POST":
        fullname = request.form["fullname"]
        user = User.query.filter_by(fullname=fullname).first()
        if user:
            flash(f"Your username is: {user.username}", "success")
        else:
            flash("No account found with that full name.", "error")
        return redirect(url_for("find_username"))
    return render_template("find_username.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False,host='0.0.0.0', port=7700)
