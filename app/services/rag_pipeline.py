"""RAG (Retrieval-Augmented Generation) pipeline."""

from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.retrieval import RetrievalService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for grounded LLM responses."""

    def __init__(self, retrieval_service: RetrievalService = None):
        """
        Initialize RAG pipeline.

        Args:
            retrieval_service: Service for document retrieval
        """
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_service = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize LLM service."""
        try:
            self.llm_service = LLMService()
            provider_name = self.llm_service.provider.provider_type.value
            logger.info("LLM service initialized", provider=provider_name, model=self.llm_service.provider.model)
        except Exception as e:
            logger.error("Failed to initialize LLM service", error=str(e))

    def process_query(self, query: str, include_context: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            query: User question
            include_context: Whether to include retrieved context in response

        Returns:
            Dictionary with answer and metadata
        """
        logger.info("Processing query through RAG pipeline", query=query)

        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "context": [],
            "answer": "",
            "model": self.llm_service.provider.model if self.llm_service else "unknown",
            "provider": self.llm_service.provider.provider_type.value if self.llm_service else "unknown",
        }

        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieval_service.retrieve(query)
            result["context"] = retrieved_docs
            logger.info(
                "Retrieved context",
                query=query,
                doc_count=len(retrieved_docs),
            )

            if not retrieved_docs:
                result["answer"] = (
                    "I could not find relevant information in the knowledge base "
                    "to answer your question."
                )
                logger.warning("No relevant documents found for query", query=query)
                return result

            # Step 2: Generate grounded response using LLM
            if self.llm_service and include_context:
                answer = self._generate_answer_with_llm(query, retrieved_docs)
            else:
                answer = self._generate_answer_without_llm(retrieved_docs)

            result["answer"] = answer
            logger.info("Query processing completed", query=query)

        except Exception as e:
            logger.error("Query processing failed", query=query, error=str(e))
            result["answer"] = f"Error processing query: {str(e)}"

        return result

    def _generate_answer_with_llm(
        self, query: str, context: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            query: User question
            context: Retrieved document chunks

        Returns:
            Generated answer
        """
        if not self.llm_service:
            raise Exception("LLM service is not initialized")

        # Build context string
        context_str = "\n\n".join(
            [
                f"[Document {i+1}]\n{doc.get('text', '')}"
                for i, doc in enumerate(context[:3])  # Top 3 documents
            ]
        )

        # Create prompt with guardrails
        system_prompt = """You are a helpful assistant that answers questions based on 
provided documents. Always base your answers on the information in the documents. 
If the documents don't contain relevant information, say so clearly. 
Provide accurate, factual responses and cite the source when appropriate."""

        user_prompt = f"""Based on the following documents, answer the question:

Documents:
{context_str}

Question: {query}

Guidelines:
- Answer only based on the provided documents
- Be specific and concise
- If the question cannot be answered from the documents, say so
- Cite relevant document sources in your answer"""

        try:
            answer = self.llm_service.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            logger.info("LLM answer generated", query=query, provider=self.llm_service.provider.provider_type.value)
            return answer
        except Exception as e:
            logger.error("LLM answer generation failed", error=str(e))
            raise

    def _generate_answer_without_llm(self, context: List[Dict[str, Any]]) -> str:
        """
        Generate answer without LLM (simple context extraction).

        Args:
            context: Retrieved document chunks

        Returns:
            Formatted context response
        """
        if not context:
            return "No relevant information found."

        answer = "Based on the retrieved documents:\n\n"
        for i, doc in enumerate(context[:3], 1):
            text = doc.get("text", "")[:200]  # First 200 chars
            answer += f"{i}. {text}...\n\n"

        return answer
