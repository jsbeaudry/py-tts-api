FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

# Download models from Hugging Face using git clone to ensure LFS files are fetched
RUN git clone https://huggingface.co/Supertone/supertonic-2 hf_tmp/supertonic2 \
    && git clone https://huggingface.co/Supertone/supertonic hf_tmp/supertonic \
    && mkdir -p assets \
    && mv hf_tmp/supertonic2/onnx assets/onnx \
    && mv hf_tmp/supertonic/voice_styles assets/voice_styles \
    && rm -rf hf_tmp \
    && echo "=== ONNX files ===" && ls -lh assets/onnx/*.onnx \
    && echo "=== Voice styles ===" && ls -la assets/voice_styles/

# Copy application code
COPY api.py helper.py ./

# Environment variables with defaults
ENV ONNX_DIR=assets/onnx
ENV VOICE_STYLES_DIR=assets/voice_styles
ENV USE_GPU=false
ENV WHISPER_MODEL_SIZE=tiny

# Expose the API port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8005/health')" || exit 1

# Run the API
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8005"]
