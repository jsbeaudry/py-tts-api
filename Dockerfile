FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py ./

# Expose the API port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8005/health')" || exit 1

# Run the API (model auto-downloads from HuggingFace on first run)
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8005"]
