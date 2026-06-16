#!/usr/bin/env python
"""Diagnostic script to check Full App setup."""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    print(f"? Python {sys.version}")
    print(f"? Executable: {sys.executable}")
    return True

def check_env_file():
    """Check .env file."""
    print_header(".env File Configuration")
    env_path = Path(".env")

    if not env_path.exists():
        print("? .env file not found!")
        return False

    print(f"? .env file found at {env_path.absolute()}")

    with open(env_path) as f:
        env_content = f.read()

    # Check for critical variables
    required = {
        "LOCAL_LLM_ENABLED": "Local LLM enabled",
        "LOCAL_LLM_URL": "Local LLM URL",
        "LOCAL_LLM_MODEL": "Local LLM Model",
        "OPENAI_ENABLED": "OpenAI enabled",
        "OPENAI_API_KEY": "OpenAI API key",
    }

    print("\nProvider Configuration:")
    for var, desc in required.items():
        if var in env_content:
            lines = [l for l in env_content.split('\n') if var in l and not l.strip().startswith('#')]
            if lines:
                print(f"  ? {desc}: {lines[0]}")
            else:
                print(f"  ? {desc}: (commented out)")
        else:
            print(f"  ? {desc}: (not found)")

    return True

def check_ports():
    """Check if ports are available."""
    print_header("Port Availability")

    import socket

    ports = {
        8000: "FastAPI",
        8501: "Streamlit",
        11434: "Ollama (Local LLM)"
    }

    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()

        if result == 0:
            print(f"  ? Port {port} ({service}): IN USE")
        else:
            print(f"  ? Port {port} ({service}): Available")

    return True

def check_packages():
    """Check required packages."""
    print_header("Required Packages")

    required = ['fastapi', 'uvicorn', 'streamlit', 'requests', 'pydantic']

    installed = []
    missing = []

    for package in required:
        try:
            __import__(package)
            installed.append(package)
            print(f"  ? {package}")
        except ImportError:
            missing.append(package)
            print(f"  ? {package}: NOT INSTALLED")

    if missing:
        print(f"\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True

def check_ollama():
    """Check if Ollama is running."""
    print_header("Ollama Status (Local LLM)")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)

        if response.status_code == 200:
            models = response.json()
            print(f"? Ollama is running")
            print(f"  Available models: {len(models.get('models', []))}")

            for model in models.get('models', [])[:5]:
                print(f"    - {model['name']}")

            if not models.get('models'):
                print("\n  ? No models installed. Pull one with:")
                print("    ollama pull llama2")

            return True
        else:
            print(f"? Ollama returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("? Cannot connect to Ollama on http://localhost:11434")
        print("  Solutions:")
        print("    1. Check if Ollama is installed: https://ollama.ai")
        print("    2. Start Ollama: ollama serve")
        print("    3. Or set OPENAI_ENABLED=true if using OpenAI instead")
        return False
    except Exception as e:
        print(f"? Error checking Ollama: {e}")
        return False

def check_openai():
    """Check OpenAI configuration."""
    print_header("OpenAI Status")

    api_key = os.getenv("OPENAI_API_KEY", "")
    enabled = os.getenv("OPENAI_ENABLED", "false").lower() == "true"

    if not enabled:
        print("? OpenAI is disabled (OK if using Local LLM)")
        return True

    if not api_key or api_key == "your-openai-api-key":
        print("? OpenAI is enabled but API key is not set")
        print("  Set OPENAI_API_KEY in .env file")
        return False

    print("? OpenAI API key is configured")

    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            print("? OpenAI API is accessible")
            return True
        else:
            print(f"? OpenAI API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"? Cannot reach OpenAI API: {e}")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    print_header("Project Structure")

    required_files = [
        "main.py",
        "streamlit_app.py",
        ".env",
        "app/__init__.py",
        "app/core/config.py",
        "app/api/routes.py",
        "app/services/llm_service.py",
    ]

    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ? {file}")
        else:
            print(f"  ? {file}: NOT FOUND")
            all_exist = False

    return all_exist

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("  GenAI Document Assistant - Full App Diagnostic")
    print("="*60)

    results = {
        "Python Version": check_python_version(),
        "Project Structure": check_project_structure(),
        ".env Configuration": check_env_file(),
        "Required Packages": check_packages(),
        "Port Availability": check_ports(),
        "Ollama (Local LLM)": check_ollama(),
        "OpenAI API": check_openai(),
    }

    print_header("Diagnostic Summary")

    for check, passed in results.items():
        status = "? PASS" if passed else "? FAIL"
        print(f"  {status}: {check}")

    all_passed = all(results.values())

    if all_passed:
        print("\n? All checks passed! You're ready to run the Full App.")
        print("\n  To start:")
        print("    1. Debug ? Select 'FastAPI + Streamlit (Full App)'")
        print("    2. Or manually start FastAPI first, then Streamlit")
    else:
        print("\n? Some checks failed. See above for details.")
        print("\n  Fix the issues and run this diagnostic again.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
