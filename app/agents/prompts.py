"""ReAct prompt templates for the agent loop."""


REACT_SYSTEM_PROMPT = """You are an expert document assistant that answers questions by searching through indexed documents.

You have access to the following tools:
{tool_descriptions}

Use the following format for EVERY step:

Thought: Think about what you need to do next
Action: the tool name to use (must be one of: {tool_names})
Action Input: {{"param_name": "value"}}

After each action, you will receive an Observation with the result.

When you have enough information to answer the user's question, use the final_answer tool:

Thought: I now have enough information to answer the question
Action: final_answer
Action Input: {{"answer": "your complete answer here"}}

Important rules:
- Always start with a Thought
- Always search for documents before answering
- Base your final answer ONLY on information found in the documents
- If no relevant documents are found, say so clearly
- Cite document numbers when possible
- Be specific and concise in your final answer
- You MUST use the exact format above — do not skip Thought, Action, or Action Input"""


def format_react_prompt(query: str, tool_descriptions: str, tool_names: str) -> str:
    """
    Format the initial ReAct system prompt.

    Args:
        query: The user's question
        tool_descriptions: Formatted tool descriptions from ToolRegistry
        tool_names: Comma-separated tool names

    Returns:
        Formatted system prompt
    """
    return REACT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
    )


def format_react_history(
    query: str,
    steps: list,
    system_prompt: str,
) -> list:
    """
    Build the messages list for the LLM, incorporating the full ReAct history.

    Args:
        query: The user's original question
        steps: List of (thought, action, action_input, observation) tuples
        system_prompt: The formatted system prompt

    Returns:
        List of message dicts for LLMService.complete()
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}"},
    ]

    # Build the conversation history from prior steps
    if steps:
        history_lines = []
        for step in steps:
            history_lines.append(f"Thought: {step['thought']}")
            history_lines.append(f"Action: {step['action']}")
            history_lines.append(f"Action Input: {step['action_input']}")
            history_lines.append(f"Observation: {step['observation']}")
            history_lines.append("")  # blank line separator

        # Add history as assistant + system messages
        messages.append({
            "role": "assistant",
            "content": "\n".join(history_lines).strip(),
        })
        messages.append({
            "role": "user",
            "content": "Continue with your next Thought and Action based on the observations above.",
        })

    return messages
