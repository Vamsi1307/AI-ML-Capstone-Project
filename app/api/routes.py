"""API routes for document ingestion, query processing, and health checks."""

import os
import tempfile
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from app.core.logging_config import get_logger
from app.core.providers import provider_manager
from app.services.document_ingestion import DocumentIngestionService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retrieval import RetrievalService
from app.services.rag_pipeline import RAGPipeline
from app.services.llm_service import LLMService
from app.agents.agent import (
    AgentOrchestrator,
    ReActOrchestrator,
    PlannerAgent,
    RetrieverAgent,
    ReasoningAgent,
    ResponseAgent,
)

logger = get_logger(__name__)

router = APIRouter()

# Lazy-loaded services (initialized on first use to avoid import-time errors)
_embedding_service = None
_vector_store = None
_retrieval_service = None
_rag_pipeline = None
_agent_orchestrator = None
_react_orchestrator = None
_llm_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or initialize embedding service."""
    global _embedding_service
    if _embedding_service is None:
        logger.info("Initializing embedding service")
        from app.core.config import settings
        _embedding_service = EmbeddingService(model=settings.EMBEDDING_MODEL)
    return _embedding_service


def get_vector_store() -> VectorStoreService:
    """Get or initialize vector store."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing vector store")
        _vector_store = VectorStoreService()
    return _vector_store


def get_retrieval_service() -> RetrievalService:
    """Get or initialize retrieval service."""
    global _retrieval_service
    if _retrieval_service is None:
        logger.info("Initializing retrieval service")
        _retrieval_service = RetrievalService(
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
        )
    return _retrieval_service


def get_rag_pipeline() -> RAGPipeline:
    """Get or initialize RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        logger.info("Initializing RAG pipeline")
        _rag_pipeline = RAGPipeline(retrieval_service=get_retrieval_service())
    return _rag_pipeline


def get_llm_service() -> LLMService:
    """Get or initialize shared LLM service."""
    global _llm_service
    if _llm_service is None:
        logger.info("Initializing LLM service")
        _llm_service = LLMService()
    return _llm_service


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get or initialize agent orchestrator."""
    global _agent_orchestrator
    if _agent_orchestrator is None:
        logger.info("Initializing agent orchestrator")
        _agent_orchestrator = AgentOrchestrator(
            planner=PlannerAgent(),
            retriever=RetrieverAgent(retrieval_service=get_retrieval_service()),
            reasoning=ReasoningAgent(),
            response=ResponseAgent(llm_service=get_llm_service()),
        )
    return _agent_orchestrator


def get_react_orchestrator() -> ReActOrchestrator:
    """Get or initialize ReAct orchestrator."""
    global _react_orchestrator
    if _react_orchestrator is None:
        logger.info("Initializing ReAct orchestrator")
        _react_orchestrator = ReActOrchestrator(
            llm_service=get_llm_service(),
            retrieval_service=get_retrieval_service(),
        )
    return _react_orchestrator


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for asking questions."""

    query: str
    use_agents: str = "react"  # "none", "sequential", or "react"
    include_context: bool = True


class QueryResponse(BaseModel):
    """Response model for query results."""

    query: str
    answer: str
    context_count: int
    confidence: float = 0.0
    agent_pipeline: list = []
    react_steps: list = []
    provider: str = ""
    model: str = ""


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    message: str
    vector_count: int
    providers: Dict[str, bool] = {}


@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and ingest a document.

    Supported formats: PDF, TXT, CSV, Excel, JSON, YAML

    Args:
        file: Document file to upload

    Returns:
        Document ingestion result with chunk information
    """
    logger.info("Document upload initiated", filename=file.filename)

    temp_file = None
    upload_filename = file.filename or "uploaded_file"
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(upload_filename)[1]
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name

        # Validate and ingest document through preprocessing pipeline
        ingestion_service = DocumentIngestionService()
        if not ingestion_service.validate_file(temp_file):
            raise HTTPException(status_code=400, detail="File validation failed")

        # Extract and preprocess document → returns ProcessedDocument
        processed_doc = ingestion_service.ingest_document(
            temp_file, source_filename=upload_filename
        )
        logger.info(
            "Document preprocessing complete",
            filename=file.filename,
            sections=len(processed_doc.sections),
        )

        # Hierarchical chunking with semantic titles, summaries, and keywords
        chunking_service = ChunkingService()
        chunks = chunking_service.chunk_processed_document(
            processed_doc, document_id=upload_filename
        )
        logger.info("Document chunking complete", chunk_count=len(chunks))

        # Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunks]
        logger.info("Generating embeddings", chunk_count=len(chunk_texts))
        embeddings = get_embedding_service().embed_text(chunk_texts)
        logger.info("Embeddings generated", embeddings_shape=embeddings.shape)

        # Prepare enriched metadata for vector store
        metadata_list = [
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "summary": chunk.summary,
                "document_id": upload_filename,
                "chunk_index": chunk.index,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        # Add to vector store
        logger.info("Adding vectors to store", metadata_count=len(metadata_list))
        get_vector_store().add_vectors(embeddings, metadata_list)
        logger.info("Vectors added to store", count=len(chunks))

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "sections_detected": len(processed_doc.sections),
            "total_vectors": get_vector_store().get_vector_count(),
            "file_type": processed_doc.file_type,
            "keywords": processed_doc.metadata.get("keywords", []),
            "message": f"Document processed: {len(processed_doc.sections)} sections, {len(chunks)} chunks created",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document upload failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning("Failed to clean up temp file", error=str(e))


@router.post("/ask-question")
async def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Ask a question about uploaded documents.

    Args:
        request: Query request with question and options

    Returns:
        Answer with context information
    """
    logger.info("Question received", query=request.query, mode=request.use_agents)

    if not request.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Normalise mode value for backward compatibility
    mode = request.use_agents.lower().strip()
    # Support legacy bool values from older clients
    if mode in ("true", "1"):
        mode = "sequential"
    elif mode in ("false", "0"):
        mode = "none"

    try:
        agent_pipeline = []
        react_steps = []

        if mode == "react":
            # --- ReAct pipeline: LLM-driven Thought → Action → Observation loop ---
            context = {"query": request.query}
            result = get_react_orchestrator().orchestrate(context)

            answer = result.get("answer", "")
            confidence = result.get("response", {}).get("answer_confidence", 0.0)
            context_count = result.get("response", {}).get("source_count", 0)
            agent_pipeline = result.get("agent_trace", [])
            react_steps = result.get("react_steps", [])
            provider = result.get("response", {}).get("provider", "")
            model = result.get("response", {}).get("model", "")

        elif mode == "sequential":
            # --- Sequential agent pipeline: Planner → Retriever → Reasoning → Response (LLM) ---
            context = {
                "query": request.query,
                "retrieval_service": get_retrieval_service(),
            }
            result = get_agent_orchestrator().orchestrate(context)

            answer = result.get("answer") or result.get("response", {}).get("status", "Error processing query")
            confidence = result.get("reasoning", {}).get("average_similarity", 0.0)
            context_count = len(result.get("retrieved_documents", []))
            agent_pipeline = result.get("agent_trace", [])
            provider = result.get("response", {}).get("provider", "")
            model = result.get("response", {}).get("model", "")

        else:
            # --- RAG pipeline: direct retrieval + LLM (no agents) ---
            result = get_rag_pipeline().process_query(
                request.query, include_context=request.include_context
            )
            answer = result.get("answer", "")
            confidence = (
                sum(doc.get("similarity_score", 0) for doc in result.get("context", []))
                / len(result.get("context", []))
                if result.get("context")
                else 0.0
            )
            context_count = len(result.get("context", []))
            provider = result.get("provider", "")
            model = result.get("model", "")

        logger.info(
            "Question processed successfully",
            query=request.query,
            confidence=confidence,
            mode=mode,
            provider=provider,
        )

        return QueryResponse(
            query=request.query,
            answer=answer,
            context_count=context_count,
            confidence=float(confidence),
            agent_pipeline=agent_pipeline,
            react_steps=react_steps,
            provider=provider,
            model=model,
        )

    except Exception as e:
        logger.error("Question processing failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/health-check")
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify API is running.

    Returns:
        Health status with system information and provider status
    """
    try:
        vector_count = get_vector_store().get_vector_count()
        provider_status = provider_manager.get_provider_status()
        logger.info("Health check performed", vector_count=vector_count)

        return HealthResponse(
            status="healthy",
            message="GenAI Document Assistant API is running",
            vector_count=vector_count,
            providers=provider_status,
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/providers")
async def get_providers() -> Dict[str, Any]:
    """
    Get LLM provider configuration status.

    Returns:
        Provider information and status
    """
    try:
        enabled_providers = [p for p in provider_manager.get_enabled_providers()]
        primary = provider_manager.get_primary_provider()

        providers_info = {
            "enabled_providers": [p.provider_type.value for p in enabled_providers],
            "primary_provider": primary.provider_type.value,
            "primary_model": primary.model,
            "provider_count": len(enabled_providers),
            "all_providers": provider_manager.get_provider_status(),
        }

        logger.info("Providers information retrieved",
                   enabled_count=len(enabled_providers),
                   primary=primary.provider_type.value)

        return providers_info
    except Exception as e:
        logger.error("Failed to retrieve providers information", error=str(e))
        raise HTTPException(status_code=500, detail=f"Provider info failed: {str(e)}")


@router.post("/debug/test-embedding")
async def test_embedding(request: QueryRequest) -> Dict[str, Any]:
    """
    Debug endpoint to test embedding service directly.

    Args:
        request: Query with text to embed

    Returns:
        Embedding result with shape information
    """
    try:
        logger.info("Testing embedding service", text=request.query)
        service = get_embedding_service()

        # Test single text
        single_embedding = service.embed_text(request.query)
        logger.info("Single embedding result", shape=single_embedding.shape, dtype=single_embedding.dtype)

        # Test multiple texts
        test_texts = [request.query, "another test", "third test"]
        multi_embedding = service.embed_text(test_texts)
        logger.info("Multi embedding result", shape=multi_embedding.shape, dtype=multi_embedding.dtype)

        return {
            "status": "success",
            "provider": service.provider_type.value,
            "model": service.model,
            "single_text_embedding": {
                "shape": single_embedding.shape,
                "dtype": str(single_embedding.dtype),
                "sample_values": single_embedding[:5].tolist() if len(single_embedding) > 0 else [],
            },
            "multi_text_embedding": {
                "shape": multi_embedding.shape,
                "dtype": str(multi_embedding.dtype),
                "sample_first_vector": multi_embedding[0][:5].tolist() if multi_embedding.shape[0] > 0 else [],
            },
        }
    except Exception as e:
        logger.error("Embedding test failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding test failed: {str(e)}")

