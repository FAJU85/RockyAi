FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy executor
COPY executor.py .

# Create non-root user for security
RUN useradd -m -u 1001 rocky && \
    chown -R rocky:rocky /app

USER rocky

# Default command
CMD ["python", "executor.py"]
