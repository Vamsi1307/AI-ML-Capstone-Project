# VS Code Debugging Guide

This guide explains how to use the VS Code debugging configurations for the GenAI Document Assistant project.

## Quick Start

1. **Open Debug View**: Press `Ctrl+Shift+D` (or click Debug icon in sidebar)
2. **Select Configuration**: Choose a configuration from the dropdown at the top
3. **Start Debugging**: Press `F5` or click the green play button
4. **Set Breakpoints**: Click line number to add/remove breakpoints
5. **Debug Controls**:
   - `F5` - Continue
   - `F10` - Step over
   - `F11` - Step into
   - `Shift+F11` - Step out
   - `Ctrl+K Ctrl+I` - Evaluate expression

## Available Launch Configurations

### 1. Python: FastAPI (Recommended)
- **Purpose**: Debug the FastAPI application with auto-reload
- **Command**: Press `F5` or select from dropdown
- **Features**:
  - Hot reload on file changes
  - Breakpoint support
  - Integrated terminal output
  - Loads `.env` file automatically
  - Debug logging enabled

**Usage**:
```
1. Select "Python: FastAPI"
2. Press F5
3. Server starts at http://localhost:8000
4. Set breakpoints in your code
5. Open API at http://localhost:8000/docs
```

### 2. Python: Current File
- **Purpose**: Debug the currently open Python file
- **Command**: Select from dropdown, press F5
- **Features**:
  - Run any Python file directly
  - Good for testing individual modules
  - Automatic environment variable loading

**Usage**:
```
1. Open a Python file (e.g., app/services/embedding.py)
2. Select "Python: Current File"
3. Press F5
```

### 3. Python: Main App
- **Purpose**: Run the main application directly
- **Command**: Select from dropdown, press F5
- **Features**:
  - Direct Python execution
  - Full debugging support
  - Environment variables loaded

### 4. Python: Uvicorn Debug
- **Purpose**: Advanced debugging with detailed logging
- **Command**: Select from dropdown, press F5
- **Features**:
  - Debug log level enabled
  - Detailed request/response logging
  - Useful for diagnosing issues

**Usage**:
```
Select "Python: Uvicorn Debug" and press F5
Check console output for detailed debug information
```

### 5. Python: Tests
- **Purpose**: Run tests with debugging
- **Command**: Select from dropdown, press F5
- **Features**:
  - Run pytest with verbose output
  - Show full test output
  - Debugging support for tests

**Usage**:
```
1. Select "Python: Tests"
2. Press F5
3. View test results in integrated terminal
```

### 6. Python: Attach
- **Purpose**: Attach debugger to running process
- **Command**: Select from dropdown, press F5
**Features**:
  - Attach to port 5678
  - Useful for remote debugging
  - Requires running process

## Common Tasks

### Run Server (Keyboard Shortcut)
- **Shortcut**: `Ctrl+Shift+B` (Run Build Task)
- **Task**: "Run FastAPI Server"
- **Result**: Starts server with auto-reload

### Run Tests
- **Command**: Open command palette (`Ctrl+Shift+P`)
- **Type**: "Tasks: Run Test Task"
- **Task**: "Run Tests"
- **Result**: Runs pytest with verbose output

### Format Code
- **Command**: Open command palette (`Ctrl+Shift+P`)
- **Type**: "Tasks: Run Build Task"
- **Task**: "Format Code with Black"
- **Result**: Auto-formats all Python files

### Lint Code
- **Command**: Open command palette (`Ctrl+Shift+P`)
- **Type**: "Tasks: Run Build Task"
- **Task**: "Lint with Flake8"
- **Result**: Shows linting issues

### Type Check
- **Command**: Open command palette (`Ctrl+Shift+P`)
- **Type**: "Tasks: Run Build Task"
- **Task**: "Type Check with MyPy"
- **Result**: Checks type annotations

## Breakpoint Setup

### Setting Breakpoints

1. **Click Line Number**: Click on the line number where you want to break
   - Red dot appears = breakpoint set
   - Click again to remove

2. **Conditional Breakpoint**: Right-click line number
   - Choose "Add Conditional Breakpoint"
   - Enter condition: `len(query) > 100`
   - Breaks only when condition is true

3. **Logpoint**: Right-click line number
   - Choose "Add Logpoint"
   - Enter message: `Query: {query}`
   - Logs without stopping execution

### Debug Console

While debugging:

1. **View Variables**:
   - Left panel shows local and global variables
   - Hover over variables in editor to see values

2. **Evaluate Expressions**:
   - Click "Debug Console" tab
   - Type expressions: `len(documents)`, `query.upper()`
   - Press Enter to evaluate

3. **Watch Expressions**:
   - In "Watch" section, click "+"
   - Enter expression: `len(retrieved_documents)`
   - Updates as you step through code

## Environment Variables

The `.env` file is automatically loaded. Add your configuration:

```env
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-4
DEBUG=True
LOG_LEVEL=DEBUG
```

## Debugging Tips

### 1. Debug the Embedding Service
```python
# Add breakpoint in embedding.py
def embed_text(self, text):
    # Line 53 - breakpoint here
    embeddings = self.client.encode(text)
    return embeddings
```

### 2. Debug Document Upload
```python
# Breakpoint in routes.py upload_document()
# Step through file processing, chunking, embedding
```

### 3. Debug Query Processing
```python
# Breakpoint in routes.py ask_question()
# Watch retrieved_documents, confidence scores
```

### 4. Debug Agent Execution
```python
# Breakpoint in agent.py orchestrate()
# Follow agent execution pipeline
```

## Troubleshooting

### "Python Interpreter not Found"
```
1. Open command palette (Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose ".venv"
```

### "Module not found" Errors
```
1. Ensure virtual environment is activated
2. Run: pip install -r requirements.txt
3. Reload VS Code window (Ctrl+R)
```

### Breakpoints Not Working
```
1. Ensure you're using the correct Python interpreter
2. Check "Justification Code" is enabled (it is)
3. Restart debugger (Shift+F5 then F5)
```

### Port 8000 Already in Use
```
1. Change port in launch.json: --port 8001
2. Or kill existing process on port 8000
```

## VS Code Settings

Key settings in `.vscode/settings.json`:

- `python.defaultInterpreterPath`: Points to `.venv/Scripts/python.exe`
- `python.formatting.provider`: Uses Black for formatting
- `python.linting.flake8Enabled`: Enables real-time linting
- `python.analysis.typeCheckingMode`: Basic type checking
- `editor.formatOnSave`: Auto-format on save

## Recommended Extensions

The `.vscode/extensions.json` file recommends:
- **Python**: Microsoft official Python support
- **Pylance**: Advanced Python type checking
- **Black Formatter**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **REST Client**: Test API endpoints
- **GitLens**: Git integration

Install recommended extensions:
1. Click Extensions icon (`Ctrl+Shift+X`)
2. Type "@recommended"
3. Install all

## Keyboard Shortcuts Cheat Sheet

| Action | Shortcut |
|--------|----------|
| Start/Continue | `F5` |
| Stop | `Shift+F5` |
| Step Over | `F10` |
| Step Into | `F11` |
| Step Out | `Shift+F11` |
| Toggle Breakpoint | `Ctrl+B` |
| Open Debug Console | `Ctrl+J` then Debug |
| Run Task | `Ctrl+Shift+B` |
| Command Palette | `Ctrl+Shift+P` |

## Advanced: Remote Debugging

For debugging remote servers:

1. **Start Remote Process**:
```bash
python -m debugpy --listen 5678 main.py
```

2. **Select "Python: Attach" configuration**
3. **Set host**: Change `127.0.0.1` to remote IP
4. **Press F5** to attach

## Next Steps

- Set breakpoints in your code
- Run a debug configuration
- Step through code execution
- Inspect variables and expressions
- Practice with the test endpoints

---

Happy debugging! 🐛🚀
