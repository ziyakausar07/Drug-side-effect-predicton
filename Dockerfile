FROM python:3.11-slim

# Install system dependencies including Tesseract for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Run Flask application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
