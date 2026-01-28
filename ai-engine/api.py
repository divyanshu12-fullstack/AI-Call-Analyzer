"""
FastAPI server for AI Voice Detection.
Exposes a POST /detect endpoint that accepts base64-encoded audio.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import VoiceDetector

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
    """Request body containing base64-encoded audio."""
    audio: str  # Base64-encoded MP3 or WAV


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
def detect_voice(request: AudioRequest):
    """
    Detect if audio is AI-generated or human.
    
    - **audio**: Base64-encoded MP3 or WAV audio file
    
    Returns classification, confidence score, and technical explanation.
    """
    try:
        if not request.audio:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        result = detector.predict_from_base64(request.audio)
        return DetectionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting AI Voice Detector API...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
