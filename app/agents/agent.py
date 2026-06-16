"""Agentic AI agents for specialized tasks."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from enum import Enum

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Lazy import to avoid circular imports
def _get_llm_service(provider_type=None):
    """Lazy-load LLMService to avoid circular imports."""
    from app.services.llm_service import LLMService
    return LLMService(provider_type=provider_type)


class AgentRole(str, Enum):
    """Enum for agent roles."""

    PLANNER = "planner"
    RETRIEVER = "retriever"
    REASONING = "reasoning"
    RESPONSE = "response"


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, role: AgentRole, name: str = None):
        """
        Initialize agent.

        Args:
            role: Agent role
            name: Agent name
        """
        self.role = role
        self.name = name or role.value
        self.execution_history = []

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent task.

        Args:
            context: Context dictionary with required information

        Returns:
            Modified context with agent outputs
        """
        pass

    def _log_execution(self, context: Dict[str, Any]) -> None:
        """Log agent execution."""
        execution_log = {
            "agent": self.name,
            "role": self.role.value,
            "output": context.get("output"),
        }
        self.execution_history.append(execution_log)
        logger.info(f"Agent {self.name} execution completed", execution=execution_log)


class PlannerAgent(BaseAgent):
    """Agent responsible for planning the query resolution strategy."""

    def __init__(self):
        """Initialize Planner agent."""
        super().__init__(role=AgentRole.PLANNER, name="planner")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan query resolution strategy.

        Args:
            context: Context with user query

        Returns:
            Context with plan
        """
        query = context.get("query", "")
        logger.info("Planner analyzing query", query=query)

        plan = {
            "steps": [
                "retrieve_documents",
                "analyze_relevance",
                "generate_response",
            ],
            "priority": "high" if len(query) > 50 else "normal",
            "requires_reasoning": any(
                word in query.lower()
                for word in ["why", "how", "explain", "compare", "analyze"]
            ),
        }

        context["plan"] = plan
        context["output"] = "Query plan created"
        self._log_execution(context)
        return context


class RetrieverAgent(BaseAgent):
    """Agent responsible for document retrieval."""

    def __init__(self, retrieval_service=None):
        """
        Initialize Retriever agent.

        Args:
            retrieval_service: Service for document retrieval
        """
        super().__init__(role=AgentRole.RETRIEVER, name="retriever")
        self.retrieval_service = retrieval_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant documents.

        Args:
            context: Context with query and retrieval service

        Returns:
            Context with retrieved documents
        """
        query = context.get("query", "")
        retrieval_service = context.get("retrieval_service", self.retrieval_service)

        logger.info("Retriever fetching documents", query=query)

        if retrieval_service:
            try:
                documents = retrieval_service.retrieve(query, top_k=5)
                context["retrieved_documents"] = documents
                context["output"] = f"Retrieved {len(documents)} documents"
                logger.info("Documents retrieved", count=len(documents))
            except Exception as e:
                logger.error("Retrieval failed", error=str(e))
                context["retrieved_documents"] = []
                context["output"] = f"Retrieval failed: {str(e)}"
        else:
            logger.warning("No retrieval service available")
            context["retrieved_documents"] = []
            context["output"] = "No retrieval service available"

        self._log_execution(context)
        return context


class ReasoningAgent(BaseAgent):
    """Agent responsible for analyzing and reasoning over retrieved information."""

    def __init__(self):
        """Initialize Reasoning agent."""
        super().__init__(role=AgentRole.REASONING, name="reasoning")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason over retrieved documents.

        Args:
            context: Context with retrieved documents

        Returns:
            Context with reasoning output
        """
        documents = context.get("retrieved_documents", [])
        requires_reasoning = context.get("plan", {}).get("requires_reasoning", False)

        logger.info("Reasoning agent analyzing documents", doc_count=len(documents))

        reasoning_output = {
            "total_chunks": len(documents),
            "average_similarity": (
                sum(doc.get("similarity_score", 0) for doc in documents)
                / len(documents)
                if documents
                else 0
            ),
            "coherence_analysis": self._analyze_coherence(documents),
            "key_themes": self._extract_themes(documents),
        }

        context["reasoning"] = reasoning_output
        context["output"] = "Reasoning analysis completed"
        self._log_execution(context)
        return context

    @staticmethod
    def _analyze_coherence(documents: List[Dict[str, Any]]) -> str:
        """Analyze coherence of retrieved documents."""
        if not documents:
            return "No documents to analyze"

        avg_score = sum(doc.get("similarity_score", 0) for doc in documents) / len(
            documents
        )
        if avg_score > 0.8:
            return "High coherence - documents are highly relevant"
        elif avg_score > 0.5:
            return "Medium coherence - documents are somewhat relevant"
        else:
            return "Low coherence - documents may not be directly relevant"

    @staticmethod
    def _extract_themes(documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from documents."""
        # Simple theme extraction based on document metadata
        themes = set()
        for doc in documents:
            if "document_id" in doc:
                themes.add(doc["document_id"].split("_")[0])
        return list(themes)[:3]  # Top 3 themes


class ResponseAgent(BaseAgent):
    """Agent responsible for generating the final LLM-powered response."""

    def __init__(self, llm_service=None):
        """
        Initialize Response agent.

        Args:
            llm_service: Optional pre-initialized LLMService instance.
                         If None, it will be created on first execute().
        """
        super().__init__(role=AgentRole.RESPONSE, name="response")
        self._llm_service = llm_service

    def _ensure_llm(self):
        """Lazy-initialize the LLM service."""
        if self._llm_service is None:
            logger.info("Response agent initializing LLM service")
            self._llm_service = _get_llm_service()
        return self._llm_service

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the final answer by calling the LLM with retrieved docs and
        reasoning analysis as context.

        Args:
            context: Context with all agent outputs

        Returns:
            Context updated with 'answer' (LLM-generated text) and 'response' metadata
        """
        query = context.get("query", "")
        documents = context.get("retrieved_documents", [])
        reasoning = context.get("reasoning", {})
        plan = context.get("plan", {})

        logger.info(
            "Response agent generating LLM answer",
            query=query,
            doc_count=len(documents),
            requires_reasoning=plan.get("requires_reasoning", False),
        )

        answer = self._call_llm(query, documents, reasoning, plan)

        response = {
            "status": "success",
            "query": query,
            "answer": answer,
            "answer_confidence": reasoning.get("average_similarity", 0),
            "source_count": len(documents),
            "coherence": reasoning.get("coherence_analysis", ""),
            "key_themes": reasoning.get("key_themes", []),
            "provider": getattr(self._ensure_llm().provider, "provider_type", None) and
                        self._ensure_llm().provider.provider_type.value,
            "model": getattr(self._ensure_llm().provider, "model", "unknown"),
        }

        context["response"] = response
        context["answer"] = answer
        context["output"] = "Response generated"
        self._log_execution(context)
        return context

    def _call_llm(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        reasoning: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> str:
        """
        Build prompts and call the LLM.

        Falls back to a simple text extraction if the LLM is unavailable.
        """
        if not documents:
            return "I could not find relevant information in the knowledge base to answer your question."

        # Build context string from top retrieved documents
        context_str = "\n\n".join(
            f"[Document {i + 1}]\n{doc.get('text', '')}"
            for i, doc in enumerate(documents[:5])
        )

        coherence = reasoning.get("coherence_analysis", "")
        themes = ", ".join(reasoning.get("key_themes", [])) or "general"
        requires_reasoning = plan.get("requires_reasoning", False)

        system_prompt = (
            "You are an expert document assistant. "
            "Answer the user's question strictly based on the provided documents. "
            "If the documents do not contain enough information, say so clearly. "
            "Cite document numbers when appropriate."
        )

        analysis_note = ""
        if requires_reasoning:
            analysis_note = (
                f"\n\nRelevance analysis: {coherence}. "
                f"Key themes detected: {themes}. "
                "Please provide a detailed, analytical answer."
            )

        user_prompt = (
            f"Documents:\n{context_str}"
            f"{analysis_note}\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Base your answer only on the documents above\n"
            "- Be specific and concise\n"
            "- If the question cannot be answered from the documents, say so"
        )

        try:
            llm = self._ensure_llm()
            answer = llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            provider_name = llm.provider.provider_type.value
            logger.info("LLM answer generated by ResponseAgent", provider=provider_name)
            return answer
        except Exception as e:
            logger.error("ResponseAgent LLM call failed, falling back", error=str(e))
            # Graceful fallback: return extracted text
            lines = ["Based on the retrieved documents:\n"]
            for i, doc in enumerate(documents[:3], 1):
                snippet = doc.get("text", "")[:300]
                lines.append(f"{i}. {snippet}...")
            return "\n\n".join(lines)


class AgentOrchestrator:
    """Orchestrates the full agent pipeline: Planner -> Retriever -> Reasoning -> Response."""

    def __init__(
        self,
        planner: PlannerAgent = None,
        retriever: RetrieverAgent = None,
        reasoning: ReasoningAgent = None,
        response: ResponseAgent = None,
        llm_service=None,
    ):
        """
        Initialize orchestrator with agents.

        Args:
            planner: Planning agent
            retriever: Retrieval agent
            reasoning: Reasoning / analysis agent
            response: Response generation agent (calls the LLM)
            llm_service: Optional pre-initialized LLMService shared across agents
        """
        self.planner = planner or PlannerAgent()
        self.retriever = retriever or RetrieverAgent()
        self.reasoning = reasoning or ReasoningAgent()
        # Share the llm_service with the ResponseAgent so it is not re-created
        self.response = response or ResponseAgent(llm_service=llm_service)
        logger.info(
            "AgentOrchestrator initialized",
            agents=["planner", "retriever", "reasoning", "response"],
        )

    def orchestrate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full agent pipeline in sequence.

        Pipeline:
          1. PlannerAgent   - analyses the query and sets execution strategy
          2. RetrieverAgent - fetches relevant document chunks via vector search
          3. ReasoningAgent - scores relevance, extracts themes, assesses coherence
          4. ResponseAgent  - calls the LLM (OpenAI or local Llama) with full context

        Args:
            context (dict): The shared pipeline context. Must contain the key
                "query" with the user's question string.

        Returns:
            dict: The enriched context containing the keys answer, response,
                reasoning, plan, and agent_trace.
        """
        query = context.get("query", "")
        logger.info("Agent orchestration started", query=query)

        agent_trace = []
        try:
            # Step 1 - Plan
            logger.info("[1/4] PlannerAgent running")
            context = self.planner.execute(context)
            agent_trace.append("planner")

            # Step 2 - Retrieve
            logger.info("[2/4] RetrieverAgent running")
            context = self.retriever.execute(context)
            agent_trace.append("retriever")

            # Step 3 - Reason
            logger.info("[3/4] ReasoningAgent running")
            context = self.reasoning.execute(context)
            agent_trace.append("reasoning")

            # Step 4 - Respond (calls LLM)
            logger.info("[4/4] ResponseAgent running (LLM call)")
            context = self.response.execute(context)
            agent_trace.append("response")

            context["agent_trace"] = agent_trace
            logger.info(
                "Agent orchestration completed successfully",
                agents_run=agent_trace,
            )
        except Exception as e:
            logger.error("Agent orchestration failed", error=str(e))
            context["error"] = str(e)
            context["agent_trace"] = agent_trace

        return context


class ReActOrchestrator:
    """Orchestrates the ReAct (Reasoning + Acting) agent loop.

    Instead of running a fixed sequence of agents, the LLM decides which
    tool to use at each step in a Thought → Action → Observation loop.
    The loop continues until the LLM issues a ``final_answer`` action or
    the maximum number of iterations is reached.
    """

    def __init__(
        self,
        llm_service=None,
        retrieval_service=None,
        max_iterations: int = None,
    ):
        """
        Initialize the ReAct orchestrator.

        Args:
            llm_service: Pre-initialized LLMService instance.
                         If None, one is created lazily.
            retrieval_service: RetrievalService for document search.
            max_iterations: Maximum loop iterations (default from config).
        """
        from app.core.config import settings
        from app.agents.tools import build_tool_registry

        self._llm_service = llm_service
        self.max_iterations = max_iterations or settings.MAX_REACT_ITERATIONS
        self.tool_registry = build_tool_registry(retrieval_service)

        logger.info(
            "ReActOrchestrator initialized",
            max_iterations=self.max_iterations,
            tools=[t.name for t in self.tool_registry.get_all()],
        )

    def _ensure_llm(self):
        """Lazy-initialize the LLM service."""
        if self._llm_service is None:
            self._llm_service = _get_llm_service()
        return self._llm_service

    def orchestrate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the ReAct loop.

        Pipeline:
          1. Send query + history to LLM
          2. Parse Thought / Action / Action Input from LLM output
          3. Execute the chosen tool
          4. Record the Observation
          5. Repeat until final_answer or max_iterations

        Args:
            context: Must contain ``query`` (str). Optionally
                ``retrieval_service``.

        Returns:
            Enriched context with ``answer``, ``response``, ``react_steps``,
            and ``agent_trace``.
        """
        from app.agents.prompts import format_react_prompt, format_react_history
        from app.agents.parser import parse_react_output, ReActParseError

        query = context.get("query", "")
        logger.info("ReAct orchestration started", query=query)

        # Build system prompt
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        tool_names = ", ".join(t.name for t in self.tool_registry.get_all())
        system_prompt = format_react_prompt(query, tool_descriptions, tool_names)

        steps: List[Dict[str, Any]] = []
        answer = ""
        agent_trace = ["react_start"]

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"[ReAct {iteration}/{self.max_iterations}] Calling LLM")

            # Build messages with full history
            messages = format_react_history(query, steps, system_prompt)

            # Call LLM
            try:
                llm = self._ensure_llm()
                llm_output = llm.complete(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                )
            except Exception as e:
                logger.error("ReAct LLM call failed", error=str(e), iteration=iteration)
                context["error"] = f"LLM call failed at step {iteration}: {e}"
                break

            logger.debug("ReAct LLM output", output=llm_output[:300])

            # Parse LLM output
            valid_actions = [t.name for t in self.tool_registry.get_all()]
            try:
                parsed = parse_react_output(llm_output, valid_actions)
            except ReActParseError as e:
                logger.warning(
                    "ReAct parse failed, retrying",
                    error=str(e),
                    iteration=iteration,
                )
                # Record the failure and let the LLM try again
                steps.append({
                    "thought": llm_output[:300],
                    "action": "parse_error",
                    "action_input": "{}",
                    "observation": (
                        f"ERROR: Could not parse your response. "
                        f"You MUST use the format: Thought: ... Action: ... Action Input: {{...}}. "
                        f"Valid actions are: {', '.join(valid_actions)}"
                    ),
                })
                agent_trace.append(f"parse_error_{iteration}")
                continue

            # Check for final_answer
            if parsed.action == "final_answer":
                answer = parsed.action_input.get("answer", "")
                steps.append({
                    "thought": parsed.thought,
                    "action": "final_answer",
                    "action_input": str(parsed.action_input),
                    "observation": "Answer delivered.",
                })
                agent_trace.append("final_answer")
                logger.info(
                    "ReAct completed with final_answer",
                    iterations=iteration,
                )
                break

            # Execute tool
            tool = self.tool_registry.get(parsed.action)
            if tool:
                observation = tool.execute(**parsed.action_input)
            else:
                observation = f"Error: Unknown tool '{parsed.action}'"

            steps.append({
                "thought": parsed.thought,
                "action": parsed.action,
                "action_input": str(parsed.action_input),
                "observation": observation,
            })
            agent_trace.append(parsed.action)
            logger.info(
                f"[ReAct {iteration}] Tool executed",
                action=parsed.action,
                observation_len=len(str(observation)),
            )

        else:
            # Max iterations reached without final_answer
            logger.warning(
                "ReAct max iterations reached",
                max_iterations=self.max_iterations,
            )
            # Try to construct answer from last observations
            if steps:
                last_observations = "\n".join(
                    s["observation"] for s in steps
                    if s["action"] not in ("parse_error", "final_answer")
                )
                answer = (
                    f"Based on the information gathered:\n\n{last_observations}\n\n"
                    "(Note: The agent reached its maximum iteration limit.)"
                )
            else:
                answer = "I was unable to find an answer within the allowed steps."
            agent_trace.append("max_iterations_reached")

        # Build response
        llm = self._ensure_llm()
        provider_type = getattr(llm.provider, "provider_type", None)
        response = {
            "status": "success" if answer else "partial",
            "query": query,
            "answer": answer,
            "answer_confidence": 0.0,
            "source_count": sum(
                1 for s in steps if s["action"] == "search_documents"
            ),
            "provider": provider_type.value if provider_type else "",
            "model": getattr(llm.provider, "model", "unknown"),
            "react_steps": steps,
            "iterations": len(steps),
        }

        context["answer"] = answer
        context["response"] = response
        context["react_steps"] = steps
        context["agent_trace"] = agent_trace

        return context

