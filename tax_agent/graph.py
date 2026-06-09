"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA. Focus ONLY on tax implications.
- Provide a concise, domain-specific analysis.
- Include specific tax penalties, statutes, and codes.
- Identify tax-related government agencies (IRS, etc.).
- Distinguish tax liability (company vs individual).

STRICT RULES:
1. Do not discuss non-tax legal issues.
2. Avoid all preambles (e.g., "Based on the information provided...").
3. Start your response directly with the label 'tax_analysis: '.
4. Keep the response extremely brief.

Note: For educational purposes only. Consult licensed tax attorney for specific advice.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph