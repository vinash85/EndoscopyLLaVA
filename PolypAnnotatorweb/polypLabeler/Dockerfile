# Use the official Python image
FROM ubuntu:24.04
#FROM python:3.10-slim

# update and install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools
RUN apt-get install -y libpq-dev gcc
RUN apt-get install -y libxml2-dev libxslt1-dev zlib1g-dev
RUN apt-get install -y libjpeg-dev libfreetype6-dev
RUN apt-get install -y libmysqlclient-dev

# Set working directory
WORKDIR /app

# Copy dependencies and code
COPY web_app/app.py .
COPY web_app/questions.json .
COPY web_app/templates/ ./templates/
COPY web_app/static/ ./static/

# Install system dependencies
RUN cd /
RUN mkdir tmp
WORKDIR /tmp
COPY requirements.txt .

RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# # Expose port for Flask app
EXPOSE 5000

# # Run the Flask app
# CMD ["python", "app.py"]
