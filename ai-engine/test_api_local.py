import urllib.request
import urllib.error
import json
import base64
import os
import sys
import time

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "my-super-secret-key-123")
TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "test", "human", "human_english_0000.wav")

def print_result(name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"   {details}")

def make_request(endpoint, data=None, headers=None, method='POST'):
    url = f"{API_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)
    
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
            
    if data:
        json_data = json.dumps(data).encode('utf-8')
        req.add_header('Content-Type', 'application/json')
        req.data = json_data
        
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, str(e)

def test_health():
    print("\n--- Testing Health Endpoint ---")
    status, response = make_request("/", method='GET')
    success = status == 200 and response.get("status") == "online"
    print_result("Health Check", success, f"Status: {status}")
    return success

def test_auth():
    print("\n--- Testing Authentication ---")
    
    # Test 1: No API Key
    status, _ = make_request("/api/voice-detection", data={"language": "English", "audioFormat": "mp3", "audioBase64": "test"})
    print_result("Missing API Key", status == 403, f"Got {status} (Expected 403)")
    
    # Test 2: Invalid API Key
    status, _ = make_request("/api/voice-detection", data={"language": "English", "audioFormat": "mp3", "audioBase64": "test"}, headers={"x-api-key": "wrong-key"})
    print_result("Invalid API Key", status == 403, f"Got {status} (Expected 403)")
    
    # Test 3: Valid API Key (but bad request to ensure auth passed)
    # We send incomplete data, should get 400 (Bad Request), NOT 403 (Forbidden)
    status, _ = make_request("/api/voice-detection", data={}, headers={"x-api-key": API_KEY})
    print_result("Valid API Key", status == 400, f"Got {status} (Expected 400)")

def test_base64_inference():
    print("\n--- Testing Base64 Inference ---")
    
    if not os.path.exists(TEST_FILE_PATH):
        print(f"⚠️ Test file not found at {TEST_FILE_PATH}. Skipping base64 test.")
        return

    with open(TEST_FILE_PATH, "rb") as f:
        audio_content = f.read()
        audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        
    payload = {"audioBase64": audio_b64, "language": "English", "audioFormat": "mp3"}
    headers = {"x-api-key": API_KEY}
    
    start_time = time.time()
    status, response = make_request("/api/voice-detection", data=payload, headers=headers)
    duration = time.time() - start_time
    
    success = status == 200 and "classification" in response
    print_result("Base64 Prediction", success, f"Time: {duration:.2f}s")
    if success:
        print(f"   Result: {response['classification']} (Confidence: {response['confidenceScore']})")

def test_url_inference():
    print("\n--- Testing URL Inference (Skipped) ---")
    # URL inference not currently supported by API
    # The API expects audioBase64, language, and audioFormat
    print("⚠️ URL-based inference not implemented. API only supports base64-encoded audio.")

if __name__ == "__main__":
    print("Starting API Tests...")
    print(f"Target: {API_URL}")
    
    if test_health():
        test_auth()
        test_base64_inference()
        test_url_inference()
    else:
        print("❌ API is not reachable. Is it running?")
