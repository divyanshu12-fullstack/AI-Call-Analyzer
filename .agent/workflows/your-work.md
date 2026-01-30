---
description: More Details
---

====================================
YOUR ROLE AS AI ASSISTANT
When user asks for help:

ALWAYS prioritize the CRITICAL REQUIREMENTS

Authentication first
URL support second
Error handling third


Provide COMPLETE, RUNNABLE code

No pseudo-code
Include imports
Add error handling
Include comments


Give SPECIFIC commands

Exact file paths
Complete bash commands
Actual curl examples


Warn about COMMON MISTAKES

Preprocessing mismatches
Missing model.eval()
Temp file cleanup
API key exposure


Keep solutions MINIMAL

Don't add unnecessary features
Don't overcomplicate
Focus on requirements
Hackathon-ready code


Explain TRADE-OFFS

Speed vs accuracy
Complexity vs maintainability
Features vs time



When debugging:

Ask for error messages
Check logs
Verify file paths
Test incrementally

When improving:

Measure current performance
Identify bottleneck
Propose specific fix
Test improvement

====================================
PRIORITY ORDER
If time is limited, implement in this order:
CRITICAL (Must have):

✅ Authentication with X-API-Key
✅ URL input support
✅ Correct response format
✅ Error handling
✅ Model loads and predicts

HIGH (Should have):
6. ✅ Comprehensive testing
7. ✅ Deployment guide
8. ✅ Configuration file
9. ✅ Documentation
MEDIUM (Nice to have):
10. ✅ Better explanations
11. ✅ Logging
12. ✅ Performance monitoring
LOW (Optional):
13. ⏸️ Frontend UI (NOT needed for hackathon)
14. ⏸️ Database (NOT needed)
15. ⏸️ User accounts (NOT needed)
====================================
SUCCESS CRITERIA (FINAL)
The project is successful if:
✅ API is deployed and accessible via public URL
✅ POST /detect endpoint works
✅ Requires X-API-Key header authentication
✅ Accepts both "audio" (base64) and "audio_url" inputs
✅ Returns correct JSON format:
{
"classification": "AI_GENERATED" | "HUMAN",
"confidence": 0.0-1.0,
"explanation": "technical explanation"
}
✅ Model accuracy ≥75% on test set
✅ Response time <2 seconds
✅ No crashes or uncaught exceptions
✅ Automated tests pass
✅ Documentation exists (README)
✅ Can pass hackathon endpoint tester
BONUS POINTS:

Model accuracy ≥85%
Response time <1 second
Detailed technical explanations
Comprehensive error handling
Docker containerization
CI/CD pipeline

====================================
ANTI-PATTERNS TO AVOID
❌ DON'T build a frontend UI (waste of time)
❌ DON'T add Node.js wrapper (unnecessary complexity)
❌ DON'T use multiple servers (keep it simple)
❌ DON'T add databases (not needed)
❌ DON'T add user authentication (only API key needed)
❌ DON'T hardcode API keys (use env vars)
❌ DON'T skip error handling (causes crashes)
❌ DON'T ignore preprocessing consistency (ruins model)
❌ DON'T train without validation set (leads to overfitting)
❌ DON'T deploy without testing (guaranteed bugs)
====================================
QUICK REFERENCE COMMANDS
Training:
bashpython ai-engine/download_hf_human.py
python ai-engine/generate_ai_voices.py
python ai-engine/train.py
Testing locally:
bashpython ai-engine/api.py
python test_api.py
Testing deployed API:
bashcurl -X POST "https://your-api.onrender.com/detect" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.mp3"}'
Docker:
bashdocker build -t ai-voice-detector .
docker run -p 8000:8000 -e API_KEY=your-key ai-voice-detector
====================================
REMEMBER
This is a HACKATHON project, not a PhD thesis.
Goals:

✅ Working API
✅ Good accuracy
✅ Fast inference
✅ Clean code
✅ Deployed online

Non-goals:

❌ Perfect accuracy
❌ Beautiful UI
❌ Complex architecture
❌ Novel research
❌ Production-grade scalability

SHIP IT, DON'T PERFECT IT.
====================================
END OF INSTRUCTIONS
When user asks "what should I do next?":

Check CHECKLIST.md status
Identify missing CRITICAL requirements
Provide step-by-step instructions
Give runnable code
Warn about time traps

Good luck! 🚀