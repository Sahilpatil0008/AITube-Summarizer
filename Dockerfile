FROM python:3.11-slim

# System deps: ffmpeg for video processing, libGL for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY templates/ templates/

# Output directory for generated videos
RUN mkdir -p output

# HF Spaces listens on 7860
EXPOSE 7860

CMD ["python", "app.py"]
