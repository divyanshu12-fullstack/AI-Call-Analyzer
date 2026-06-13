#!/usr/bin/env python3
"""
Test script for the AI Voice Detector API.
Sends a sample audio file as base64 to the API.
"""
import requests
import base64
import json
import os

# API endpoint
API_URL = "https://ai-call-analyzer.onrender.com/api/voice-detection"
API_KEY = os.getenv("API_KEY")

def encode_audio_to_base64(file_path):
    """Encode audio file to base64 string."""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def test_api(audio_file_path, language="English"):
    """Test the API with a given audio file."""
    # Encode audio
    audio_base64 = encode_audio_to_base64(audio_file_path)

    audio_ext = os.path.splitext(audio_file_path)[1].lower().lstrip(".")
    audio_format = audio_ext if audio_ext else "mp3"

    # Prepare request payload
    payload = {
        "language": language,
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    print(f"Sending request to {API_URL}")
    print(f"Audio file: {audio_file_path}")
    print(f"Language: {language}")
    print(f"Base64 length: {len(audio_base64)}")
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("API_KEY env var is not set. Set API_KEY before running.")

    # Test with a sample file (replace with actual path)
    test_audio_path = "ai-test.mp3"  # Use an .mp3 file for compliance
    if not test_audio_path.lower().endswith(".mp3"):
        raise SystemExit("Test audio must be .mp3 to match API compliance.")

    test_api(test_audio_path, "English")