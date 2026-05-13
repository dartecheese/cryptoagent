"""Narrative Analyst — trending categories, token alignment, rotation signals."""

from cryptoagent.agents.analysts.analyst_base import make_analyst_node
from cryptoagent.agents.utils.agent_utils import get_narrative_data, get_token_metadata


def create_narrative_analyst(llm):
    tools = [get_narrative_data, get_token_metadata]

    system_message = (
        "You are a Narrative Analyst. Fetch data then output ONLY valid JSON:\n"
        '{{"score": 0.0-1.0 (aligned=1, misaligned=0), "confidence": 0.0-1.0, '
        '"phase": "emerging"|"accelerating"|"peak"|"declining", '
        '"rotation": "inflow"|"neutral"|"outflow", '
        '"competitors": ["token1","token2"], '
        '"summary": "1-line finding"}}'
    )
    return make_analyst_node(llm, tools, system_message, "narrative_report")
