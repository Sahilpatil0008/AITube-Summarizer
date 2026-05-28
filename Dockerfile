FROM python:3.11-slim

# System deps: ffmpeg for video processing, libGL for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first so sentence-transformers doesn't pull CUDA (~2 GB)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/

# Output directory for generated videos
RUN mkdir -p output

# HF Spaces listens on port 7860
EXPOSE 7860

CMD ["python", "app.py"]
