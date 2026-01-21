import io
import logging
import os
import tempfile
from enum import Enum
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from helper import load_text_to_speech, load_voice_style, AVAILABLE_LANGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supertonic Audio API", description="OpenAI-compatible TTS and STT API")

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

# Configuration
ONNX_DIR = os.environ.get("ONNX_DIR", "assets/onnx")
VOICE_STYLES_DIR = os.environ.get("VOICE_STYLES_DIR", "assets/voice_styles")
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

# Load models at startup
text_to_speech = None
voice_styles = {}
whisper_model = None

# STT Configuration
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
# deepdml/faster-whisper-large-v3-turbo-ct2
# samll

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
    language: str = Field(default="en", description="Language code (en, ko, es, pt, fr)")


@app.on_event("startup")
async def startup_event():
    """Load models and voice styles on startup."""
    global text_to_speech, voice_styles, whisper_model

    # Load TTS model
    print(f"Loading TTS model from {ONNX_DIR}...")
    text_to_speech = load_text_to_speech(ONNX_DIR, USE_GPU)
    print("TTS model loaded successfully")

    # Load all available voice styles
    print(f"Loading voice styles from {VOICE_STYLES_DIR}...")
    for voice_file in os.listdir(VOICE_STYLES_DIR):
        if voice_file.endswith(".json"):
            voice_name = voice_file.replace(".json", "")
            voice_path = os.path.join(VOICE_STYLES_DIR, voice_file)
            voice_styles[voice_name] = load_voice_style([voice_path])
            print(f"  Loaded voice: {voice_name}")
    print(f"Loaded {len(voice_styles)} voice styles")

    # Load Whisper model for STT
    print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE)
    print("Whisper model loaded successfully")


def get_voice_style(voice: str):
    """Get the voice style by name, supporting both OpenAI-style and direct names."""
    # Check if it's an OpenAI-style voice name
    try:
        voice_enum = Voice(voice)
        voice_id = voice_enum.value
    except ValueError:
        # Try direct voice name (M1, F1, etc.)
        voice_id = voice.upper()

    if voice_id not in voice_styles:
        available = list(voice_styles.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available voices: {available}"
        )
    return voice_styles[voice_id]


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generate speech from text (OpenAI-compatible endpoint).

    This endpoint is compatible with OpenAI's TTS API.
    """
    if text_to_speech is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    logger.info(f"TTS request: voice={request.voice}, format={request.response_format}, text='{request.input[:50]}...'")

    # Validate language
    if request.language not in AVAILABLE_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{request.language}' not supported. Available: {AVAILABLE_LANGS}"
        )

    # Get voice style
    style = get_voice_style(request.voice)

    # Convert OpenAI speed (0.25-4.0) to internal speed
    # OpenAI: 1.0 = normal, >1 = faster, <1 = slower
    # Internal: higher = faster (default 1.05)
    internal_speed = request.speed * 1.05

    try:
        # Generate speech
        wav, duration = text_to_speech(
            text=request.input,
            lang=request.language,
            style=style,
            total_step=5,
            speed=internal_speed,
        )

        # Trim to actual duration
        wav_trimmed = wav[0, : int(text_to_speech.sample_rate * duration[0].item())]

        # Convert to requested format
        buffer = io.BytesIO()

        if request.response_format == ResponseFormat.wav:
            sf.write(buffer, wav_trimmed, text_to_speech.sample_rate, format="WAV")
            media_type = "audio/wav"
        elif request.response_format == ResponseFormat.flac:
            sf.write(buffer, wav_trimmed, text_to_speech.sample_rate, format="FLAC")
            media_type = "audio/flac"
        elif request.response_format == ResponseFormat.pcm:
            # Raw PCM (24kHz, 16-bit signed little-endian) - OpenAI TTS format
            import numpy as np
            pcm_data = (wav_trimmed * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
            media_type = "audio/pcm"
        else:
            # For mp3, opus, aac - fall back to wav (would need ffmpeg for conversion)
            sf.write(buffer, wav_trimmed, text_to_speech.sample_rate, format="WAV")
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


class STTResponseFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="whisper-1", description="Model ID (ignored, for compatibility)"),
    language: Optional[str] = Form(default=None, description="Language code (ISO-639-1)"),
    prompt: Optional[str] = Form(default=None, description="Optional prompt to guide transcription"),
    response_format: str = Form(default="json", description="Response format"),
    temperature: float = Form(default=0.0, ge=0.0, le=1.0, description="Sampling temperature"),
    timestamp_granularities: Optional[str] = Form(default=None, description="Timestamp granularities (word, segment)"),
):
    """
    Transcribe audio to text (OpenAI-compatible endpoint).

    This endpoint is compatible with OpenAI's Whisper API.
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    logger.info(f"STT request received: filename={file.filename}, model={model}, language={language}, format={response_format}")

    # Save uploaded file to temp location - use .wav as default for raw audio
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        logger.info(f"Saved audio to temp file: {tmp_path}, size={len(content)} bytes")

    try:
        # Transcribe with faster-whisper
        logger.info("Starting transcription...")
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
        )

        # Collect all segments
        segments_list = []
        full_text = ""
        for segment in segments:
            segments_list.append({
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            })
            full_text += segment.text

        logger.info(f"Transcription complete: '{full_text.strip()}' (detected language: {info.language})")

        # Return based on response format
        if response_format == "text":
            return PlainTextResponse(content=full_text.strip())

        elif response_format == "srt":
            srt_output = ""
            for i, seg in enumerate(segments_list, 1):
                start = format_timestamp_srt(seg["start"])
                end = format_timestamp_srt(seg["end"])
                srt_output += f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n"
            return PlainTextResponse(content=srt_output)

        elif response_format == "vtt":
            vtt_output = "WEBVTT\n\n"
            for seg in segments_list:
                start = format_timestamp_vtt(seg["start"])
                end = format_timestamp_vtt(seg["end"])
                vtt_output += f"{start} --> {end}\n{seg['text'].strip()}\n\n"
            return PlainTextResponse(content=vtt_output)

        elif response_format == "verbose_json":
            return JSONResponse(content={
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "text": full_text.strip(),
                "segments": segments_list,
            })

        else:  # json (default)
            return JSONResponse(content={"text": full_text.strip()})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(default="whisper-1", description="Model ID (ignored, for compatibility)"),
    prompt: Optional[str] = Form(default=None, description="Optional prompt to guide translation"),
    response_format: str = Form(default="json", description="Response format"),
    temperature: float = Form(default=0.0, ge=0.0, le=1.0, description="Sampling temperature"),
):
    """
    Translate audio to English text (OpenAI-compatible endpoint).

    This endpoint is compatible with OpenAI's Whisper API.
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".wav")[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Translate with faster-whisper (task="translate" translates to English)
        segments, info = whisper_model.transcribe(
            tmp_path,
            task="translate",
            initial_prompt=prompt,
            temperature=temperature,
        )

        # Collect all segments
        full_text = ""
        for segment in segments:
            full_text += segment.text

        # Return based on response format
        if response_format == "text":
            return full_text.strip()
        else:  # json (default)
            return {"text": full_text.strip()}

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices."""
    voices = []

    # OpenAI-compatible voice names
    openai_mapping = {
        "alloy": "M1",
        "echo": "M2",
        "fable": "M3",
        "onyx": "M4",
        "nova": "F1",
        "shimmer": "F2",
    }

    for name, voice_id in openai_mapping.items():
        if voice_id in voice_styles:
            voices.append({
                "voice_id": name,
                "name": name.capitalize(),
                "internal_id": voice_id,
                "type": "male" if voice_id.startswith("M") else "female"
            })

    # Add additional voices
    for voice_id in voice_styles:
        if voice_id not in openai_mapping.values() or voice_id in ["M5", "F3", "F4", "F5"]:
            voices.append({
                "voice_id": voice_id,
                "name": voice_id,
                "internal_id": voice_id,
                "type": "male" if voice_id.startswith("M") else "female"
            })

    return {"voices": voices}


@app.get("/v1/audio/languages")
async def list_languages():
    """List supported languages."""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "ko", "name": "Korean"},
            {"code": "es", "name": "Spanish"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "fr", "name": "French"},
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tts_model_loaded": text_to_speech is not None,
        "stt_model_loaded": whisper_model is not None,
        "voices_loaded": len(voice_styles),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
