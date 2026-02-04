"""
FastAPI server for AI Voice Detection.
Exposes a POST /api/voice-detection endpoint that accepts base64-encoded audio.
"""
from fastapi import FastAPI, HTTPException, Header, Security, Depends, Request
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional
import os
import sys

# Add current directory to sys.path so 'inference' and 'model' can be imported
# when running from the project root (as Render does)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import VoiceDetector

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")
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

# Custom error handler for standardized error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return errors in submission spec format: {status: 'error', message: '...'}"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

# Custom error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI request validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": f"Invalid request format or missing fields: {str(exc.errors())}"
        }
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": f"Validation error: {str(exc)}"
        }
    )


class AudioRequest(BaseModel):
    """Request body containing base64-encoded audio."""
    # Primary fields per submission spec (required)
    language: str  # Tamil, English, Hindi, Malayalam, Telugu
    audioFormat: str  # mp3
    audioBase64: str  # Base64-encoded audio


class DetectionResponse(BaseModel):
    """Response containing classification results."""
    status: str = "success"  # Always "success" for successful responses
    language: str  # Echo back the language from request
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidenceScore: float  # 0.0 to 1.0
    explanation: str  # Technical reasoning


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "AI Voice Detector",
        "version": "1.0.0"
    }


@app.post("/api/voice-detection", response_model=DetectionResponse)
def detect_voice(request: AudioRequest, api_key: str = Depends(get_api_key)):
    """
    Detect if audio is AI-generated or human.
    
    - **language**: Must be one of: Tamil, English, Hindi, Malayalam, Telugu
    - **audioFormat**: Audio format (mp3)
    - **audioBase64**: Base64-encoded audio file
    
    Returns classification, confidence score, and technical explanation.
    """
    # Log the request for debugging
    print(f"Received request: language={request.language}, audioFormat={request.audioFormat}, audioBase64 length={len(request.audioBase64) if request.audioBase64 else 0}")
    
    # Supported languages per submission spec
    SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    try:
        # Validate language
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        
        # Validate audio format
        if request.audioFormat.lower() != "mp3":
            raise HTTPException(
                status_code=400,
                detail="Only MP3 format is supported"
            )
        
        # Process request using base64 audio
        result = detector.predict_from_base64(request.audioBase64)
        
        # Add required fields to response
        result['status'] = 'success'
        result['language'] = request.language
        
        # Rename confidence to confidenceScore if needed
        if 'confidence' in result:
            result['confidenceScore'] = result.pop('confidence')
            
        return DetectionResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting AI Voice Detector API...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
