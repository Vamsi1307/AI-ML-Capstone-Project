# Quick Start Guide

## ?? Quick Test (2 minutes)

### Step 1: Run Diagnostic
```bash
python diagnostic.py
```
This will check:
- ? Python environment
- ? Required packages
- ? Port availability
- ? .env configuration
- ? Ollama status (if using local LLM)

### Step 2: Fix Any Issues Found
The diagnostic will tell you exactly what needs to be fixed.

### Step 3: Start the Application

#### Option A: Recommended - Run Separately
1. **Terminal 1 - Start FastAPI**:
   ```bash
   python -m uvicorn main:app --reload
   ```
   Wait for: `Uvicorn running on http://0.0.0.0:8000`

2. **Terminal 2 - Start Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Wait for: `You can now view your Streamlit app`

3. **Open Browser**:
   - FastAPI Docs: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

#### Option B: Using VS Code Debug
1. Click Debug icon (or press Ctrl+Shift+D)
2. Select "FastAPI + Streamlit (Full App)" from dropdown
3. Click green play button

---

## ? Prerequisites Checklist

### Local LLM Setup (Ollama)
- [ ] Ollama installed from https://ollama.ai
- [ ] Ollama running: `ollama serve`
- [ ] Model pulled: `ollama pull llama2` (or your model name)
- [ ] .env configured:
  ```env
  LOCAL_LLM_ENABLED=true
  LOCAL_LLM_URL=http://localhost:11434
  LOCAL_LLM_MODEL=llama2
  OPENAI_ENABLED=false
  ```

### OR OpenAI Setup
- [ ] OpenAI API key obtained
- [ ] .env configured:
  ```env
  OPENAI_ENABLED=true
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4
  LOCAL_LLM_ENABLED=false
  ```

### General Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] .env file created and configured
- [ ] Ports 8000, 8501 are free

---

## ?? Testing Workflow

### 1. Upload a Document
1. Open Streamlit UI: http://localhost:8501
2. Click "Upload Document" section
3. Select a PDF, TXT, or other supported file
4. See chunks created and vectors added

### 2. Ask a Question
1. In "Ask Question" section, enter a question
2. Click "Search"
3. See answer with context

### 3. Check API Directly
```bash
# Health check
curl http://localhost:8000/api/v1/health-check

# Provider status
curl http://localhost:8000/api/v1/providers

# Interactive docs
# Open: http://localhost:8000/docs
```

---

## ?? Troubleshooting Quick Fixes

### Issue: "Connection refused" in Streamlit
```
FastAPI not started yet. Wait 3-5 seconds after starting FastAPI.
```

### Issue: "Port 8000 already in use"
```bash
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: "Ollama not responding"
```bash
# Start Ollama
ollama serve

# Or if using OpenAI, set OPENAI_ENABLED=true
```

### Issue: "Provider not enabled"
```
Check your .env file - at least ONE provider must be enabled:
- LOCAL_LLM_ENABLED=true (requires Ollama running)
- OPENAI_ENABLED=true (requires API key)
```

---

## ?? Full App Architecture

```
???????????????????????????????????????????????????????????
?                    Streamlit UI                         ?
?                  (Port 8501)                            ?
???????????????????????????????????????????????????????????
?                 API Calls (HTTP)                        ?
???????????????????????????????????????????????????????????
?              FastAPI Backend                            ?
?               (Port 8000)                               ?
??????????????????????????????????????????????????????????
? RAG Pipeline     ? LLM Service      ? Vector Store     ?
??????????????????????????????????????????????????????????
? Retrieval        ? OpenAI/Ollama    ? FAISS            ?
? Chunking         ? Provider Manager ? Embeddings       ?
? Embedding        ? Multi-Provider   ? Document Index   ?
??????????????????????????????????????????????????????????
         ?                    ?
         ?                    ?
    ???????????????    ????????????????
    ?   Ollama    ?    ?    OpenAI    ?
    ? Local LLM   ?    ?    API       ?
    ? :11434      ?    ?              ?
    ???????????????    ????????????????
```

---

## ?? .env Configuration Reference

```env
# ===== API Configuration =====
API_TITLE=GenAI Document Assistant
API_VERSION=1.0.0

# ===== Server =====
HOST=0.0.0.0
PORT=8000

# ===== OPENAI (Choose ONE provider) =====
OPENAI_ENABLED=false
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# ===== LOCAL LLM (Ollama) =====
LOCAL_LLM_ENABLED=true
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama2

# ===== Vector Storage =====
VECTOR_STORE_PATH=app/data/vectors
VECTOR_STORE_TYPE=faiss

# ===== Document Processing =====
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=200
CHUNK_OVERLAP=50
CHUNKING_STRATEGY=overlap

# ===== Retrieval =====
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3

# ===== Logging =====
DEBUG=false
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

---

## ?? Still Having Issues?

1. **Run diagnostic**: `python diagnostic.py`
2. **Check logs**: Look in `logs/app.log`
3. **Check .env**: Verify at least ONE provider is enabled
4. **Kill processes**: Stop all Python processes and restart
5. **Restart Ollama/Services**: If using Ollama, restart it

For detailed troubleshooting, see: `TROUBLESHOOTING.md`
