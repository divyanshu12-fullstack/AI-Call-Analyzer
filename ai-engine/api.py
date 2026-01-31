"""
FastAPI server for AI Voice Detection.
Exposes a POST /detect endpoint that accepts base64-encoded audio.
"""
from fastapi import FastAPI, HTTPException, Header, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import os
from inference import VoiceDetector

# Configuration
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY", "default-key-change-me")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials",
    )

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detector API",
    description="Detects whether a voice sample is AI-generated or human-spoken",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (loads model once at startup)
detector = VoiceDetector()


class AudioRequest(BaseModel):
    """Request body containing base64-encoded audio or URL."""
    audio: Optional[str] = None      # Standard naming
    audio_url: Optional[str] = None  # URL naming
    audio_base64: Optional[str] = None # Alternate naming
    audio_base64_format: Optional[str] = None # Possible exact-label match
    language: Optional[str] = None   # Optional field from tester
    audio_format: Optional[str] = None # Optional field from tester


class DetectionResponse(BaseModel):
    """Response containing classification results."""
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidence: float    # 0.0 to 1.0
    explanation: str     # Technical reasoning


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "AI Voice Detector",
        "version": "1.0.0"
    }


@app.post("/detect", response_model=DetectionResponse)
def detect_voice(request: AudioRequest, api_key: str = Depends(get_api_key)):
    """
    Detect if audio is AI-generated or human.
    
    - **audio**: Base64-encoded MP3 or WAV audio file (Optional)
    - **audio_url**: URL to audio file (Optional)
    
    Requires exactly one of audio or audio_url.
    Returns classification, confidence score, and technical explanation.
    """
    try:
        # Consolidate audio input from possible field names
        audio_data = request.audio or request.audio_base64 or request.audio_base64_format
        audio_url = request.audio_url

        # Validate input
        if not audio_data and not audio_url:
            raise HTTPException(status_code=400, detail="Must provide either 'audio' (base64) or 'audio_url'")
        
        if audio_data and audio_url:
            raise HTTPException(status_code=400, detail="Provide only one of 'audio' or 'audio_url', not both")
        
        # Process request
        if audio_url:
            try:
                result = detector.predict_from_url(audio_url)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
        else:
            result = detector.predict_from_base64(audio_data)
            
        return DetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting AI Voice Detector API...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
