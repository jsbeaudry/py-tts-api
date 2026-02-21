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

# Download models from Hugging Face
# supertonic-2 has onnx/ subdirectory, supertonic has voice_styles/ subdirectory
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Supertone/supertonic-2', local_dir='hf_tmp/supertonic2'); \
    snapshot_download('Supertone/supertonic', local_dir='hf_tmp/supertonic')" \
    && mkdir -p assets \
    && mv hf_tmp/supertonic2/onnx assets/onnx \
    && mv hf_tmp/supertonic/voice_styles assets/voice_styles \
    && rm -rf hf_tmp \
    && ls -la assets/onnx && ls -la assets/voice_styles

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
