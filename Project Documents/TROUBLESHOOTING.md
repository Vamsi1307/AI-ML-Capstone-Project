# Troubleshooting Guide

## Common Issues & Solutions

### 1. **Port Already in Use**
**Error**: `Address already in use` or `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (Windows)
taskkill /PID <PID> /F

# Or change the port in launch.json:
# In "Python: FastAPI" configuration, change --port to 8001
```

### 2. **Streamlit Connection Refused**
**Error**: `ConnectionError: Error connecting to http://localhost:8000/api/v1/health-check`

**Cause**: FastAPI hasn't started yet when Streamlit tries to connect

**Solutions**:
- **Wait for FastAPI to start**: The FastAPI server needs ~2-3 seconds to initialize
- **Check FastAPI logs**: Look for startup messages in the Terminal 1
- **Manually start FastAPI first**: 
  - Run "Python: FastAPI" separately first
  - Wait for "Uvicorn running on..." message
  - Then run "Python: Streamlit UI"

### 3. **Environment Variables Not Loading**
**Error**: Missing API keys, provider not found, etc.

**Solutions**:
```bash
# Verify .env file exists and has correct values
cat .env

# Check if LOCAL_LLM_ENABLED is set correctly
# For local Ollama:
LOCAL_LLM_ENABLED=true
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama2

# For OpenAI:
OPENAI_ENABLED=true
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4
```

### 4. **Ollama Not Responding**
**Error**: `Failed to connect to http://localhost:11434` or `Connection refused`

**Solutions**:
```bash
# Check if Ollama is running
ollama serve

# If not installed, install Ollama from: https://ollama.ai

# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Pull a model if needed
ollama pull llama2
```

### 5. **Provider Initialization Error**
**Error**: `ValueError: No LLM providers are enabled`

**Cause**: Neither OpenAI nor Local LLM is properly enabled

**Solution**:
```env
# At least ONE must be enabled:

# Option 1: Use Local LLM (Ollama)
OPENAI_ENABLED=false
LOCAL_LLM_ENABLED=true
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama2

# Option 2: Use OpenAI
OPENAI_ENABLED=true
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
LOCAL_LLM_ENABLED=false
```

### 6. **Streamlit Reruns Causing Multiple Requests**
**Behavior**: Your query gets asked multiple times, API calls happen multiple times

**Solution**: This is normal Streamlit behavior. Use session state to prevent it:
```python
# Already implemented in streamlit_app.py
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
```

---

## Step-by-Step Startup Guide

### Method 1: Recommended - Run Separately First
1. **Start FastAPI**:
   - Click Debug ? Select "Python: FastAPI"
   - Wait for: `Uvicorn running on http://0.0.0.0:8000`
   - Check: Open browser to `http://localhost:8000/docs`

2. **Start Streamlit** (in a new Terminal):
   - Click Debug ? Select "Python: Streamlit UI"
   - Wait for: `You can now view your Streamlit app in your browser`
   - Check: Open browser to `http://localhost:8501`

3. **Verify Health**:
   - In Streamlit, you should see a "? Healthy" status

### Method 2: Compound Launch (Full App)
1. Click Debug ? Select "FastAPI + Streamlit (Full App)"
2. This starts both in parallel
3. **Wait 3-5 seconds** for FastAPI to be ready
4. Streamlit UI should appear at `http://localhost:8501`
5. Check health status in Streamlit

---

## Debug Mode - Increase Logging

### In .env, add:
```env
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log
```

### Then watch logs:
```bash
# Windows PowerShell
Get-Content logs/app.log -Wait -Tail 50

# Or just check if file exists
ls logs/
```

---

## Browser Ports

| Service | Default Port | URL |
|---------|-------------|-----|
| FastAPI API | 8000 | `http://localhost:8000` |
| FastAPI Docs | 8000 | `http://localhost:8000/docs` |
| Streamlit UI | 8501 | `http://localhost:8501` |

---

## Checking System Status

```bash
# Check Python version
python --version

# Check if required packages are installed
pip list | grep -E "(fastapi|streamlit|uvicorn|requests)"

# Check if port is free
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Verify .env file
type .env
```

---

## If Problems Persist

1. **Collect Logs**:
   - Copy Terminal output from both "Python: FastAPI" and "Python: Streamlit UI"
   - Copy contents of `logs/app.log`

2. **Check Configuration**:
   - Verify `.env` file has correct settings
   - Run health check: `GET http://localhost:8000/api/v1/health-check`

3. **Test Each Component Separately**:
   - Test just FastAPI: `python -m uvicorn main:app --reload`
   - Test just Streamlit: `streamlit run streamlit_app.py`

4. **Restart Everything**:
   - Stop all debug sessions (Ctrl+Shift+F5)
   - Stop any running Python processes
   - Kill any processes on port 8000/8501
   - Start fresh
