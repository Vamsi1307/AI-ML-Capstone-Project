"""Tool registry for ReAct agent pattern.

Each tool wraps an existing service as a callable action that the LLM can invoke
during the ReAct Thought → Action → Observation loop.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    """A single tool that the ReAct agent can invoke."""

    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    _execute_fn: Optional[Callable] = field(default=None, repr=False)

    def execute(self, **kwargs) -> str:
        """
        Execute the tool and return a string observation.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            String observation for the LLM
        """
        if self._execute_fn is None:
            return ""
        try:
            result = self._execute_fn(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            return f"Error: {e}"


class ToolRegistry:
    """Registry of tools available to the ReAct agent."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """
        Format all tool descriptions for the LLM system prompt.

        Returns:
            Formatted string listing all tools and their parameters
        """
        lines = []
        for tool in self._tools.values():
            params_str = ", ".join(
                f"{k}: {v}" for k, v in tool.parameters.items()
            )
            lines.append(
                f"- {tool.name}({params_str}): {tool.description}"
            )
        return "\n".join(lines)


def build_tool_registry(retrieval_service) -> ToolRegistry:
    """
    Build the tool registry with all available tools.

    Args:
        retrieval_service: The retrieval service for document search

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    # Tool 1: Search documents
    def search_documents(query: str, top_k: int = 5) -> str:
        """Search indexed documents and return formatted results."""
        try:
            results = retrieval_service.retrieve(query, top_k=int(top_k))
            if not results:
                return "No relevant documents found for this query."

            output_lines = [f"Found {len(results)} relevant document(s):\n"]
            for i, doc in enumerate(results, 1):
                text = doc.get("text", "")[:500]
                score = doc.get("similarity_score", 0)
                doc_id = doc.get("document_id", "unknown")
                output_lines.append(
                    f"[Document {i}] (source: {doc_id}, similarity: {score:.2f})\n{text}\n"
                )
            return "\n".join(output_lines)
        except Exception as e:
            return f"Search failed: {e}"

    registry.register(Tool(
        name="search_documents",
        description="Search indexed documents for relevant information using semantic similarity",
        parameters={"query": "str - the search query", "top_k": "int - number of results (default 5)"},
        _execute_fn=search_documents,
    ))

    # Tool 2: Analyze relevance
    def analyze_relevance(query: str, top_k: int = 5) -> str:
        """Retrieve and analyze document relevance."""
        try:
            results = retrieval_service.retrieve(query, top_k=int(top_k))
            if not results:
                return "No documents to analyze."

            scores = [doc.get("similarity_score", 0) for doc in results]
            avg_score = sum(scores) / len(scores)

            if avg_score > 0.8:
                coherence = "High coherence - documents are highly relevant"
            elif avg_score > 0.5:
                coherence = "Medium coherence - documents are somewhat relevant"
            else:
                coherence = "Low coherence - documents may not be directly relevant"

            themes = set()
            for doc in results:
                doc_id = doc.get("document_id", "")
                if doc_id:
                    themes.add(doc_id.split("_")[0])

            return (
                f"Analysis of {len(results)} documents:\n"
                f"- Average similarity: {avg_score:.3f}\n"
                f"- Coherence: {coherence}\n"
                f"- Key themes: {', '.join(list(themes)[:5]) or 'general'}\n"
                f"- Score range: {min(scores):.3f} to {max(scores):.3f}"
            )
        except Exception as e:
            return f"Analysis failed: {e}"

    registry.register(Tool(
        name="analyze_relevance",
        description="Analyze the relevance and coherence of documents matching a query",
        parameters={"query": "str - the query to analyze against", "top_k": "int - number of docs to analyze (default 5)"},
        _execute_fn=analyze_relevance,
    ))

    # Tool 3: Final answer (terminal action - no function)
    registry.register(Tool(
        name="final_answer",
        description="Provide the final answer to the user's question. Use this when you have enough information to answer.",
        parameters={"answer": "str - the complete answer to return to the user"},
        _execute_fn=None,
    ))

    return registry
