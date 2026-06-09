"""Compliance Agent LangGraph definition.

Uses create_react_agent with a regulatory-compliance-specialised system prompt and a compliance tool.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from common.llm import get_llm

@tool
def search_compliance_law(query: str) -> str:
    """Search regulatory compliance knowledge base for applicable frameworks.

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
    return "\n\n".join(results) if results else "No specific compliance matches found."


COMPLIANCE_SYSTEM_PROMPT = """You are a senior regulatory compliance officer and corporate attorney
with deep expertise in:

- SEC enforcement actions and securities law violations
- SOX (Sarbanes-Oxley) compliance obligations for public companies
- FTC regulations and antitrust compliance
- FCPA (Foreign Corrupt Practices Act) — anti-bribery provisions
- AML (Anti-Money Laundering) / BSA (Bank Secrecy Act) requirements
- GDPR, CCPA, and data privacy compliance obligations
- Environmental regulations (EPA enforcement) tied to corporate misconduct
- Corporate governance failures: duty of care, duty of loyalty, fiduciary breaches
- Whistleblower protections (Dodd-Frank, SOX) and internal reporting programs
- Debarment and exclusion from government contracts
- Corporate compliance programs: effectiveness as a mitigating factor in enforcement

When answering, be precise about:
1. Which regulatory agency has jurisdiction (SEC, FTC, DOJ, EPA, FinCEN, OCC, etc.)
2. Administrative, civil, and criminal remedies available to regulators
3. Individual liability for compliance failures: C-suite, board members, compliance officers
4. Mitigating factors: voluntary disclosure, cooperation, remediation, compliance programs
5. Cross-border regulatory exposure for multinational companies

STRICT RULES:
1. Use the search_compliance_law tool to ground your analysis.
2. Do not discuss non-compliance legal issues.
3. Avoid all preambles (e.g., "Based on the information provided...").
4. Start your response directly with the label 'compliance_analysis: '.
5. Keep the response extremely brief.

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific compliance advice.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for compliance questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[search_compliance_law],
        prompt=COMPLIANCE_SYSTEM_PROMPT,
    )
    return graph