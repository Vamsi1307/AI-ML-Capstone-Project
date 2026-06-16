# GenAI Document Assistant - Production-Ready RAG System

A production-ready Generative AI system combining Retrieval-Augmented Generation (RAG) with Agentic AI capabilities for intelligent document processing and question answering.

## Features

- **Streamlit Web UI**: Interactive web interface for document upload and Q&A
- **Multi-format Document Support**: PDF, TXT, CSV, Excel, JSON, YAML
- **Intelligent Chunking**: Fixed-size, overlap, and semantic boundary strategies
- **Vector Storage**: FAISS-based local vector store with embeddings
- **RAG Pipeline**: Retrieval-Augmented Generation for grounded responses
- **Agentic AI**: Specialized agents (Planner, Retriever, Reasoning, Response)
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, Groq, or Local LLM (Ollama)
- **Production Features**:
  - Structured logging with JSON output
  - Input validation and safety guardrails
  - Error handling and monitoring
  - CORS configuration
  - Request/response logging

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              Streamlit Web UI (Port 8501)                        │
│  • Document Upload • Ask Questions • Query History • Health Check│
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Application (Port 8000)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           API Routes                                    │   │
│  │  • POST /upload-document                                │   │
│  │  • POST /ask-question                                   │   │
│  │  • GET  /health-check                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Services Layer                                │   │
│  │  • Document Ingestion                                   │   │
│  │  • Chunking (Fixed/Overlap/Semantic)                    │   │
│  │  • Embedding Generation                                 │   │
│  │  • Vector Storage (FAISS)                               │   │
│  │  • Document Retrieval                                   │   │
│  │  • RAG Pipeline                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Agents Layer (Agentic AI)                      │   │
│  │  • Planner Agent (Query Planning)                        │   │
│  │  • Retriever Agent (Doc Retrieval)                       │   │
│  │  • Reasoning Agent (Analysis)                            │   │
│  │  • Response Agent (Answer Generation)                    │   │
│  │  • Agent Orchestrator                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Core Components                               │   │
│  │  • Configuration Management                             │   │
│  │  • Logging & Monitoring                                 │   │
│  │  • Validation & Guardrails                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
              │                      │
              ▼                      ▼
        ┌─────────────────┐  ┌──────────────────────┐
        │   FAISS Index   │  │  LLM Providers       │
        │  Vector Store   │  │  • OpenAI (GPT-4)    │
        └─────────────────┘  │  • Anthropic Claude  │
                             │  • Google Gemini     │
                             │  • Groq              │
                             │  • Local Ollama      │
                             └──────────────────────┘
```

## Project Structure

```
genai-doc-Assistant/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # FastAPI endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_ingestion.py   # Multi-format document extraction
│   │   ├── chunking.py             # Text splitting strategies
│   │   ├── embedding.py            # Embedding generation
│   │   ├── vector_store.py         # FAISS vector storage
│   │   ├── retrieval.py            # Semantic search
│   │   └── rag_pipeline.py         # RAG implementation
│   ├── agents/
│   │   ├── __init__.py
│   │   └── agent.py                # Agentic AI agents
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   └── logging_config.py       # Structured logging
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validation.py           # Input validation & safety
│   │   └── metrics.py              # Metrics collection
│   └── data/                       # Vector store and data
├── main.py                         # FastAPI application entry point
├── streamlit_app.py                # Streamlit web UI
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Docker Compose setup
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda
- At least one LLM provider configured (OpenAI, Anthropic, Google, Groq, or Local Ollama)

### Local Setup

1. **Clone and navigate to project:**
```bash
cd genai-doc-Assistant
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. **Run the application:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## Docker Deployment

### Using Docker

```bash
# Build image
docker build -t genai-doc-assistant .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  -e MODEL_NAME=gpt-4 \
  genai-doc-assistant
```

### Using Docker Compose

```bash
# Create .env file with your OpenAI key
echo "OPENAI_API_KEY=your-api-key" > .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/v1/health-check
```

**Response:**
```json
{
  "status": "healthy",
  "message": "GenAI Document Assistant API is running",
  "vector_count": 0
}
```

### Upload Document
Upload documents for indexing:

```bash
curl -X POST http://localhost:8000/api/v1/upload-document \
  -F "file=@path/to/document.pdf"
```

**Supported Formats:**
- PDF (.pdf)
- Text (.txt)
- CSV (.csv)
- Excel (.xlsx)
- JSON (.json)
- YAML (.yaml, .yml)

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "chunks_created": 42,
  "total_vectors": 42,
  "message": "Document processed successfully: 42 chunks created"
}
```

### Ask Question
Query the indexed documents:

```bash
curl -X POST http://localhost:8000/api/v1/ask-question \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "use_agents": true,
    "include_context": true
  }'
```

**Response:**
```json
{
  "query": "What is the main topic of the document?",
  "answer": "Based on the retrieved documents...",
  "context_count": 5,
  "confidence": 0.87
}
```

## Configuration

### Environment Variables

```env
# API Configuration
DEBUG=False
HOST=0.0.0.0
PORT=8000

# LLM Configuration
OPENAI_API_KEY=your-key
MODEL_NAME=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002

# Vector Storage
VECTOR_STORE_PATH=app/data/vectors
VECTOR_STORE_TYPE=faiss

# Document Processing
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=200          # Tokens per chunk
CHUNK_OVERLAP=50        # Overlap tokens
CHUNKING_STRATEGY=overlap  # fixed|overlap|semantic

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## Chunking Strategies

### Fixed Size
- Splits text into fixed-size chunks (default: 200 tokens)
- Simple and predictable
- May split sentences

### Overlap
- Chunks with specified overlap (default: 50 tokens)
- Preserves context between chunks
- **Recommended for most use cases**

### Semantic
- Splits at sentence/paragraph boundaries
- Maintains semantic coherence
- Better for longer documents

## Features in Detail

### Document Ingestion
- **Validation**: File format, size, existence checks
- **Extraction**: Format-specific text extraction
- **Cleaning**: Removes extra whitespace, normalizes content

### Vector Storage
- **FAISS**: Fast Approximate Nearest Neighbor search
- **Local Storage**: No external dependencies
- **Metadata**: Stores document source and chunk info
- **Scalable**: Supports millions of vectors

### RAG Pipeline
- **Retrieval**: Semantic search for relevant chunks
- **Augmentation**: Combines documents with user query
- **Generation**: LLM generates grounded responses

### Agentic AI
- **Planner**: Analyzes queries and creates execution plans
- **Retriever**: Fetches relevant documents
- **Reasoning**: Analyzes coherence and themes
- **Response**: Generates final answer with confidence

### Monitoring
- **Structured Logging**: JSON output for easy parsing
- **Request Tracing**: All requests logged
- **Metrics Collection**: Query counts, errors, documents
- **Performance Tracking**: Response times and confidence scores

## Safety & Guardrails

- **Input Validation**: Query length, format, pattern detection
- **SQL Injection Prevention**: Pattern matching for dangerous SQL
- **Code Injection Prevention**: Blocks dangerous Python keywords
- **Output Sanitization**: Removes harmful content from responses
- **Response Truncation**: Limits response length
- **Rate Limiting Ready**: Framework for request throttling

## Performance Tuning

### For Speed
```env
CHUNKING_STRATEGY=fixed       # Faster chunking
CHUNK_SIZE=100               # Smaller chunks
TOP_K_RESULTS=3              # Fewer retrievals
```

### For Quality
```env
CHUNKING_STRATEGY=semantic    # Better coherence
CHUNK_SIZE=300               # Larger chunks
TOP_K_RESULTS=10             # More context
SIMILARITY_THRESHOLD=0.5     # Higher quality
```

## Testing

### Quick Test
```bash
# Create test document
echo "This is a test document about Python programming." > test.txt

# Upload document
curl -X POST http://localhost:8000/api/v1/upload-document \
  -F "file=@test.txt"

# Ask question
curl -X POST http://localhost:8000/api/v1/ask-question \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

### pytest Framework
```bash
# Install pytest
pip install pytest pytest-asyncio

# Run tests (placeholder)
pytest
```

## Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt

# Restart Python
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Verify key is valid and has credits
# Check https://platform.openai.com/account/usage
```

### Vector Store Issues
```bash
# Reset vector store
rm -rf app/data/vectors

# Restart application
python main.py
```

### Memory Issues with Large Files
```env
CHUNK_SIZE=100        # Reduce chunk size
MAX_FILE_SIZE_MB=50   # Reduce max file size
```

## Logging

View logs in real-time:
```bash
tail -f logs/app.log
```

Or access via HTTP:
- Request logs are stored with timestamps
- Structured JSON format for parsing
- Log level configurable via `LOG_LEVEL`

## Production Deployment

### Recommended Setup
1. Use Docker/Kubernetes for orchestration
2. Configure load balancing
3. Set up monitoring and alerting
4. Use separate vector store for scale
5. Cache frequent queries
6. Implement rate limiting
7. Add authentication/authorization

### Example Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-doc-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genai-doc-assistant
  template:
    metadata:
      labels:
        app: genai-doc-assistant
    spec:
      containers:
      - name: app
        image: genai-doc-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Future Enhancements

- [ ] Multi-LLM support (Claude, Llama, etc.)
- [ ] Advanced caching strategies
- [ ] Batch query processing
- [ ] Fine-tuning capabilities
- [ ] Custom embedding models
- [ ] GraphQL API
- [ ] Web UI dashboard
- [ ] Advanced filtering and metadata search
- [ ] Persistent chat history
- [ ] User authentication and RBAC

## Contributing

Follow coding guidelines:
- Use type hints for all functions
- Include docstrings
- Keep functions small and testable
- Add error handling and logging
- Write defensive code

## License

This project is part of a Capstone project for GenAI education.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/app.log`
3. Check environment variables in `.env`

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI API](https://platform.openai.com/docs)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [RAG Overview](https://blog.google/technology/ai/retrieval-augmented-generation-prompt-engineering/)
