# Development Guide

This guide covers development setup, testing, and best practices.

## Development Setup

### Install Development Dependencies

```bash
# Activate virtual environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy
```

### Running the Application During Development

#### Terminal 1: Start FastAPI Backend
```bash
# Run with auto-reload on code changes
python main.py

# Server URL: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Terminal 2: Start Streamlit Frontend (Optional)
```bash
# Run Streamlit app with hot reload
streamlit run streamlit_app.py

# Web UI URL: http://localhost:8501
```

### Code Quality Tools

#### Format Code with Black
```bash
# Format all Python files
black app/

# Format specific file
black app/api/routes.py
```

#### Lint with Flake8
```bash
# Check code style
flake8 app/

# Show statistics
flake8 app/ --statistics
```

#### Type Check with MyPy
```bash
# Check type hints
mypy app/

# Ignore specific errors
mypy app/ --ignore-missing-imports
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_vectorstore.py

# Run with coverage
pytest --cov=app tests/

# Run with verbose output
pytest -v
```

### Integration Tests

```bash
# Start the application in one terminal
python main.py

# Run integration tests in another
pytest tests/integration/

# Test specific endpoint
pytest tests/integration/test_endpoints.py::test_upload_document
```

### Manual Testing with cURL

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health-check

# Test document upload with verbose output
curl -v -X POST http://localhost:8000/api/v1/upload-document \
  -F "file=@test_document.pdf"

# Test question endpoint
curl -X POST http://localhost:8000/api/v1/ask-question \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the document about?",
    "use_agents": true
  }' | python -m json.tool
```

## Project Structure Guidelines

### Adding New Services

1. Create file in `app/services/`
2. Add class inheriting production patterns
3. Include logging
4. Add type hints
5. Write docstrings

Example:
```python
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class MyService:
    """Service description."""

    def __init__(self):
        """Initialize service."""
        logger.info("Service initialized")

    def process(self, data: str) -> dict:
        """
        Process data.

        Args:
            data: Input data

        Returns:
            Processed result
        """
        try:
            result = self._internal_process(data)
            logger.info("Processing successful")
            return result
        except Exception as e:
            logger.error("Processing failed", error=str(e))
            raise
```

### Adding New Endpoints

1. Create in `app/api/routes.py` or new file
2. Use request/response Pydantic models
3. Include input validation
4. Add error handling
5. Return appropriate status codes

Example:
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class MyRequest(BaseModel):
    """Request model."""
    data: str

class MyResponse(BaseModel):
    """Response model."""
    result: str

@router.post("/my-endpoint")
async def my_endpoint(request: MyRequest) -> MyResponse:
    """
    My endpoint description.

    Args:
        request: Request data

    Returns:
        Response data
    """
    try:
        # Validate input
        if not request.data:
            raise HTTPException(status_code=400, detail="Data required")

        # Process
        result = process_data(request.data)

        return MyResponse(result=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Endpoint error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal error")
```

## Debugging

### Enable Debug Mode

```bash
# Set debug environment variable
export DEBUG=True

# Or in .env
DEBUG=True

# Run application
python main.py
```

### View Logs

```bash
# Real-time logs
tail -f logs/app.log

# View recent logs (last 100 lines)
tail -100 logs/app.log

# Search logs
grep "error" logs/app.log

# View logs with context
grep -A 5 -B 5 "specific_string" logs/app.log
```

### Debug with Python

Add breakpoints in code:
```python
import pdb

def my_function():
    pdb.set_trace()  # Code will pause here
    # Debug in interactive mode
```

Or use a debugger-friendly IDE like VS Code.

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Memory Usage

```bash
# Monitor memory with tracemalloc
python -m tracemalloc -c "your_script.py"

# Use memory_profiler
pip install memory-profiler
python -m memory_profiler your_script.py
```

### Caching

Add result caching:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(query: str) -> str:
    """Cached function."""
    return process_query(query)
```

## Database Integration (Future)

When adding database support:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database settings in config
DATABASE_URL = "sqlite:///./app/data/app.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Use in dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# In route
@router.get("/items")
async def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

## Version Management

### Update Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade fastapi

# Update all packages
pip install --upgrade -r requirements.txt

# Create updated requirements.txt
pip freeze > requirements.txt
```

### Semantic Versioning

Follow [semver](https://semver.org/):
- MAJOR: Breaking changes (1.0.0 → 2.0.0)
- MINOR: New features (1.0.0 → 1.1.0)
- PATCH: Bug fixes (1.0.0 → 1.0.1)

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function(param1: str, param2: int) -> dict:
    """
    Short description of function.

    Longer description if needed. Can span multiple
    lines and explain complex behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Examples:
        >>> function("test", 5)
        {'result': 'test5'}
    """
    pass
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add app/services/new_service.py
git commit -m "feat: add new service with enhancement"

# Push to remote
git push origin feature/new-feature

# Create pull request on GitHub
# After review, merge to main

# Update local main
git checkout main
git pull origin main
```

### Commit Message Format

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style
- `refactor:` Code refactor
- `test:` Test addition/update
- `chore:` Maintenance

Example:
```
feat: implement document ingestion for PDF files

- Add PDF extraction using pypdf2
- Include error handling
- Add comprehensive logging
```

## CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/test.yml`:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - run: pip install -r requirements.txt
      - run: pytest
      - run: black --check app/
      - run: flake8 app/
```

## Production Deployment Checklist

- [ ] All tests passing
- [ ] Code linting clean
- [ ] Type hints complete
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Logging enabled
- [ ] Error handling complete
- [ ] Security validation added
- [ ] Performance optimized
- [ ] Monitoring configured

---

Happy developing! 🚀
