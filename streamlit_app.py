"""Streamlit web UI for GenAI Document Assistant."""

import streamlit as st
import requests
import json
from pathlib import Path
from datetime import datetime

from sympy import false, true

# Configure page
st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
HEALTH_CHECK_URL = f"{API_BASE_URL}/health-check"
UPLOAD_URL = f"{API_BASE_URL}/upload-document"
QUERY_URL = f"{API_BASE_URL}/ask-question"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []

def check_health():
    """Check API health status."""
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_document(file):
    """Upload document to API."""
    try:
        files = {"file": (file.name, file.getbuffer(), file.type)}
        response = requests.post(UPLOAD_URL, files=files, timeout=180)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def ask_question(question: str, use_agents: str = "react"):
    """Ask a question to the RAG pipeline."""
    try:
        payload = {"query": question,
                    "use_agents": use_agents,
                    "include_context": True
                   }
        response = requests.post(QUERY_URL, json=payload, timeout=180)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

# Main UI
st.markdown('<div class="main-header">📚 GenAI Document Assistant</div>', unsafe_allow_html=True)
st.markdown("RAG-powered AI system with local LLM support")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")

    # Health Check
    if st.button("🔍 Check Health", use_container_width=True):
        st.session_state.health_checked = True

    if st.session_state.get("health_checked", False):
        is_healthy, health_data = check_health()
        if is_healthy:
            st.markdown(f'<p class="status-healthy">✓ API is Healthy</p>', unsafe_allow_html=True)
            with st.expander("Health Details"):
                st.json(health_data)
        else:
            st.markdown(f'<p class="status-unhealthy">✗ API is Offline</p>', unsafe_allow_html=True)
            st.error(f"Connection Error: {health_data.get('error', 'Unknown error')}")
            st.info("Make sure the FastAPI server is running on http://localhost:8000")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "❓ Ask Questions", "📊 History"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload PDF Documents")
    st.markdown("Upload PDF files to be processed and indexed for semantic search.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Select a PDF document to upload"
    )

    if uploaded_file is not None:
        st.info(f"📄 Selected: {uploaded_file.name}")
        st.caption(f"Size: {uploaded_file.size / 1024:.2f} KB")

        if st.button("⬆️ Upload Document", use_container_width=True):
            with st.spinner("Uploading and processing document..."):
                success, result = upload_document(uploaded_file)

                if success:
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "timestamp": datetime.now(),
                        "result": result
                    })
                    st.markdown(f"""
                        <div class="success-box">
                        ✓ Document uploaded successfully!<br>
                        <strong>Filename:</strong> {result.get('filename', 'N/A')}<br>
                        <strong>Chunks Created:</strong> {result.get('chunks_created', 0)}<br>
                        <strong>Total Vectors:</strong> {result.get('total_vectors', 0)}<br>
                        <strong>Message:</strong> {result.get('message', '')}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="error-box">
                        ✗ Upload failed<br>
                        <strong>Error:</strong> {result.get('detail', result.get('error', 'Unknown error'))}
                        </div>
                    """, unsafe_allow_html=True)

    # Recently uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Recently Uploaded Files")
        for idx, file_info in enumerate(st.session_state.uploaded_files[-5:], 1):
            st.caption(f"{idx}. {file_info['name']} - {file_info['timestamp'].strftime('%H:%M:%S')}")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions")
    st.markdown("Ask questions about your uploaded documents. The AI will search through the indexed content and provide answers.")

    # Agent mode selector
    agent_mode = st.selectbox(
        "🤖 Agent Mode:",
        options=["react", "sequential", "none"],
        format_func=lambda x: {
            "react": "🔄 ReAct Agents (LLM-driven reasoning loop)",
            "sequential": "📋 Sequential Agents (fixed pipeline)",
            "none": "⚡ RAG Only (fastest, no agents)",
        }.get(x, x),
        index=0,
        help="ReAct: LLM decides which tools to use dynamically. Sequential: fixed 4-step pipeline. RAG Only: direct retrieval + LLM.",
    )

    question = st.text_area(
        "Your Question:",
        placeholder="What is the main topic discussed in the document?",
        height=100
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        submit = st.button("🤖 Ask AI", use_container_width=True)

    with col2:
        pass  # For alignment

    if submit and question:
        with st.spinner("Thinking..."):
            success, result = ask_question(question, use_agents=agent_mode)

            if success:
                st.session_state.query_history.append({
                    "question": question,
                    "answer": result,
                    "timestamp": datetime.now(),
                    "mode": agent_mode,
                })

                st.subheader("📝 Answer")
                st.markdown(f"""
                    <div class="success-box">
                    {result.get('answer', 'No answer generated')}
                    </div>
                """, unsafe_allow_html=True)

                # ReAct steps trace
                react_steps = result.get("react_steps", [])
                if react_steps:
                    with st.expander(f"🔄 ReAct Steps ({len(react_steps)} iterations)"):
                        for i, step in enumerate(react_steps, 1):
                            st.markdown(f"**Step {i}**")
                            st.markdown(f"💭 **Thought:** {step.get('thought', '')}")
                            st.markdown(f"🎯 **Action:** `{step.get('action', '')}`")
                            st.markdown(f"📥 **Input:** `{step.get('action_input', '')}`")
                            obs = step.get('observation', '')
                            if len(obs) > 500:
                                st.text_area(f"Observation {i}", obs, height=150, disabled=True, label_visibility="collapsed")
                            else:
                                st.markdown(f"📋 **Observation:** {obs}")
                            st.markdown("---")

                # Agent pipeline trace
                agent_pipeline = result.get("agent_pipeline", [])
                if agent_pipeline and not react_steps:
                    with st.expander("📋 Agent Pipeline Trace"):
                        st.write(" → ".join(agent_pipeline))

                # Context used
                if "context" in result:
                    with st.expander("Context Retrieved"):
                        st.markdown(result.get("context", "No context available"))

                # Source documents
                if "sources" in result:
                    with st.expander("📚 Source Documents"):
                        sources = result.get("sources", [])
                        if sources:
                            for source in sources:
                                st.caption(f"📖 {source}")
                        else:
                            st.caption("No sources found")

                # Metadata
                if "metadata" in result:
                    with st.expander("ℹ️ Response Metadata"):
                        st.json(result.get("metadata", {}))

                # Provider & model info
                provider = result.get("provider", "")
                model = result.get("model", "")
                if provider or model:
                    st.caption(f"Provider: {provider} | Model: {model} | Mode: {agent_mode}")
            else:
                st.markdown(f"""
                    <div class="error-box">
                    ✗ Query failed<br>
                    <strong>Error:</strong> {result.get('detail', result.get('error', 'Unknown error'))}
                    </div>
                """, unsafe_allow_html=True)
    elif submit and not question:
        st.warning("Please enter a question first.")

# Tab 3: Query History
with tab3:
    st.header("Query History")

    if st.session_state.query_history:
        for idx, history_item in enumerate(st.session_state.query_history[::-1], 1):
            with st.expander(f"Query {len(st.session_state.query_history) - idx + 1}: {history_item['question'][:50]}..."):
                st.subheader("Question:")
                st.write(history_item["question"])
                st.subheader("Answer:")
                st.write(history_item["answer"].get("answer", "No answer"))
                st.caption(f"Time: {history_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No queries yet. Start by asking a question in the 'Ask Questions' tab.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>GenAI Document Assistant • Powered by Local LLM (Llama3) • RAG Pipeline</p>
    <p>API Endpoint: http://localhost:8000 | Docs: http://localhost:8000/docs</p>
    </div>
""", unsafe_allow_html=True)
