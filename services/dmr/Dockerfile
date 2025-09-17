FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nvidia-smi \
    lspci \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model detection script
COPY model_detector.py .
COPY config.yaml .

# Set up model detection
RUN python model_detector.py

# Default command - this will be overridden by docker-compose
CMD ["python", "model_detector.py"]
