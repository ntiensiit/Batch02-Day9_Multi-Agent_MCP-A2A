"""Privacy Agent LangGraph definition.

Uses create_react_agent with a privacy-specialised system prompt and a compliance tool.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from common.llm import get_llm

@tool
def search_privacy_law(query: str) -> str:
    """Search regulatory compliance knowledge base for applicable frameworks (including privacy).

    Args:
        query: Natural language query about regulatory compliance.
    """
    knowledge = [
        (
            ["data", "privacy", "gdpr", "ccpa", "consent", "user"],
            "CCPA: fines up to $7,500 per intentional violation. GDPR: up to 4% of global "
            "revenue or EUR 20M. FTC Act Section 5 for unfair/deceptive practices. "
            "Class action exposure under state privacy laws ($100-$750 per consumer).",
        ),
        (
            ["sox", "sarbanes", "financial", "sec", "reporting"],
            "SOX § 906: false certification — up to $5M fine, 20 years prison. "
            "§ 802: record destruction — up to 20 years. § 1107: whistleblower "
            "retaliation — up to 10 years. SEC officer/director bars.",
        ),
        (
            ["fcpa", "bribery", "corruption", "foreign"],
            "FCPA anti-bribery: up to $250K fine per violation (individuals), "
            "$2M (corporations). Criminal penalties: up to 5 years prison. "
            "Books and records provisions apply to all SEC-reporting companies.",
        ),
    ]
    query_lower = query.lower()
    results = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)
    return "\n\n".join(results) if results else "No specific compliance/privacy matches found."


PRIVACY_SYSTEM_PROMPT = """You are a specialist in data protection and privacy law with expertise in GDPR, CCPA,
data breach notification requirements, consent management, and individual privacy rights.

STRICT RULES:
1. Use the search_privacy_law tool to ground your analysis.
2. Do not discuss non-privacy legal issues.
3. Avoid all preambles (e.g., "Based on the information provided...").
4. Start your response directly with the label 'privacy_analysis: '.
5. Keep the response extremely brief.

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific privacy advice.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for privacy questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[search_privacy_law],
        prompt=PRIVACY_SYSTEM_PROMPT,
    )
    return graph
