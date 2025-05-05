# Create Dockerfile for streamlit ui server
# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run streamlit
CMD ["python", "collectdata.py"]