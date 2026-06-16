"""Parser for ReAct LLM output.

Extracts structured Thought / Action / Action Input from the LLM's freeform
text response.
"""

import json
import re
from typing import Tuple, Optional
from dataclasses import dataclass

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedAction:
    """Parsed result from LLM ReAct output."""

    thought: str
    action: str
    action_input: dict

    def __repr__(self) -> str:
        return f"ParsedAction(action='{self.action}', thought='{self.thought[:50]}...')"


class ReActParseError(Exception):
    """Raised when LLM output cannot be parsed into a valid ReAct step."""
    pass


def parse_react_output(text: str, valid_actions: list) -> ParsedAction:
    """
    Parse the LLM's freeform text into a structured ReAct step.

    Expected format:
        Thought: <reasoning text>
        Action: <tool_name>
        Action Input: {"param": "value"}

    Args:
        text: Raw LLM output text
        valid_actions: List of valid action/tool names

    Returns:
        ParsedAction with thought, action, and action_input

    Raises:
        ReActParseError: If the output cannot be parsed
    """
    text = text.strip()

    # Extract Thought
    thought = _extract_field(text, "Thought")
    if not thought:
        # If no explicit "Thought:" prefix, treat everything before "Action:" as thought
        action_match = re.search(r"Action\s*:", text, re.IGNORECASE)
        if action_match:
            thought = text[:action_match.start()].strip()
        else:
            thought = text

    # Extract Action
    action = _extract_field(text, "Action")
    if not action:
        raise ReActParseError(
            f"Could not find 'Action:' in LLM output. Got: {text[:200]}"
        )

    # Clean action name
    action = action.strip().lower().replace(" ", "_")

    # Validate action
    if action not in [a.lower() for a in valid_actions]:
        raise ReActParseError(
            f"Unknown action '{action}'. Valid actions: {valid_actions}"
        )

    # Extract Action Input
    action_input = _extract_action_input(text)

    logger.debug(
        "Parsed ReAct output",
        thought=thought[:100],
        action=action,
        action_input=action_input,
    )

    return ParsedAction(
        thought=thought,
        action=action,
        action_input=action_input,
    )


def _extract_field(text: str, field_name: str) -> Optional[str]:
    """
    Extract a named field value from the LLM output.

    Handles patterns like:
        Thought: some text here
        Action: tool_name

    Args:
        text: Full LLM output
        field_name: Field to extract (e.g., "Thought", "Action")

    Returns:
        Extracted value or None
    """
    # Match "FieldName:" followed by content, stopping at the next field or end
    pattern = rf"{field_name}\s*:\s*(.+?)(?=\n(?:Thought|Action|Action Input|Observation)\s*:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_action_input(text: str) -> dict:
    """
    Extract the Action Input JSON from the LLM output.

    Handles multiple formats:
        Action Input: {"key": "value"}
        Action Input: key=value
        Action Input: just a string

    Args:
        text: Full LLM output

    Returns:
        Parsed dictionary of parameters
    """
    # Try to find Action Input field
    raw = _extract_field(text, "Action Input")
    if not raw:
        return {}

    # Try JSON parsing first
    json_result = _try_parse_json(raw)
    if json_result is not None:
        return json_result

    # Try to find JSON anywhere in the raw text, matching from first '{' to last '}'
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_result = _try_parse_json(raw[start_idx:end_idx+1])
        if json_result is not None:
            return json_result

    # Fallback: treat as a simple string value
    # Guess the parameter name based on content
    raw_clean = raw.strip().strip('"').strip("'")
    if raw_clean:
        return {"query": raw_clean}

    return {}


def _try_parse_json(text: str) -> Optional[dict]:
    """
    Attempt to parse text as JSON.

    Args:
        text: Potential JSON string

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except (json.JSONDecodeError, TypeError):
        return None
