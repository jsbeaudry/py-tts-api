import io
import logging
import os
import tempfile
from enum import Enum

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from supertonic import TTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supertonic TTS API", description="OpenAI-compatible TTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f">>> {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"<<< {request.method} {request.url.path} -> {response.status_code}")
    return response


# Load TTS model at startup
tts: TTS = None


class Voice(str, Enum):
    """Available voice styles mapped to OpenAI-style names."""
    alloy = "M1"
    echo = "M2"
    fable = "M3"
    onyx = "M4"
    nova = "F1"
    shimmer = "F2"
    # Additional voices
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"


class ResponseFormat(str, Enum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"
    pcm = "pcm"


class TTSRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model ID (ignored, for compatibility)")
    input: str = Field(..., description="The text to generate audio for", max_length=4096)
    voice: str = Field(default="alloy", description="Voice to use for synthesis")
    response_format: ResponseFormat = Field(default=ResponseFormat.wav, description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of speech (0.25 to 4.0)")


@app.on_event("startup")
async def startup_event():
    """Load TTS model on startup."""
    global tts

    # Load TTS model (auto-downloads from HuggingFace on first run)
    print("Loading Supertonic TTS model...")
    tts = TTS(auto_download=True)
    print("TTS model loaded successfully")


def get_voice_id(voice: str) -> str:
    """Get the voice ID by name, supporting both OpenAI-style and direct names."""
    # Check if it's an OpenAI-style voice name (lookup by enum name, not value)
    if voice.lower() in Voice.__members__:
        return Voice[voice.lower()].value
    else:
        # Try direct voice name (M1, F1, etc.)
        return voice.upper()


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generate speech from text (OpenAI-compatible endpoint).

    This endpoint is compatible with OpenAI's TTS API.
    """
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    logger.info(f"TTS request: voice={request.voice}, format={request.response_format}, text='{request.input[:50]}...'")

    # Get voice style using supertonic API
    voice_id = get_voice_id(request.voice)
    try:
        voice_style = tts.get_voice_style(voice_name=voice_id)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not found: {e}"
        )

    try:
        # Generate speech using supertonic
        wav, duration = tts.synthesize(
            request.input,
            voice_style=voice_style,
            speed=request.speed * 1.05,  # Convert OpenAI speed to internal
        )

        # Convert to requested format
        buffer = io.BytesIO()
        sample_rate = tts.sample_rate

        if request.response_format == ResponseFormat.wav:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tts.save_audio(wav, tmp.name)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                buffer.write(f.read())
            os.unlink(tmp_path)
            media_type = "audio/wav"
        elif request.response_format == ResponseFormat.flac:
            sf.write(buffer, wav, sample_rate, format="FLAC")
            media_type = "audio/flac"
        elif request.response_format == ResponseFormat.pcm:
            # Raw PCM (24kHz, 16-bit signed little-endian) - OpenAI TTS format
            import numpy as np

            # Resample to 24kHz if needed (OpenAI PCM standard)
            target_rate = 24000
            if sample_rate != target_rate:
                num_samples = int(len(wav) * target_rate / sample_rate)
                x_old = np.linspace(0, 1, len(wav))
                x_new = np.linspace(0, 1, num_samples)
                wav_resampled = np.interp(x_new, x_old, wav).astype(np.float32)
                logger.info(f"Resampled from {sample_rate}Hz to {target_rate}Hz")
            else:
                wav_resampled = wav

            pcm_data = (wav_resampled * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
            media_type = "audio/pcm"
        else:
            # For mp3, opus, aac - fall back to wav (would need ffmpeg for conversion)
            sf.write(buffer, wav, sample_rate, format="WAV")
            media_type = "audio/wav"

        buffer.seek(0)
        logger.info(f"TTS complete: {len(buffer.getvalue())} bytes, format={media_type}")

        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format.value}"}
        )

    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices."""
    # OpenAI-compatible voice names mapped to supertonic voice IDs
    openai_mapping = {
        "alloy": "M1",
        "echo": "M2",
        "fable": "M3",
        "onyx": "M4",
        "nova": "F1",
        "shimmer": "F2",
    }

    voices = []
    for name, voice_id in openai_mapping.items():
        voices.append({
            "voice_id": name,
            "name": name.capitalize(),
            "internal_id": voice_id,
            "type": "male" if voice_id.startswith("M") else "female"
        })

    # Add additional supertonic voices
    additional_voices = ["M3", "M4", "M5", "F3", "F4", "F5"]
    for voice_id in additional_voices:
        if voice_id not in openai_mapping.values():
            voices.append({
                "voice_id": voice_id,
                "name": voice_id,
                "internal_id": voice_id,
                "type": "male" if voice_id.startswith("M") else "female"
            })

    return {"voices": voices}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tts_model_loaded": tts is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
