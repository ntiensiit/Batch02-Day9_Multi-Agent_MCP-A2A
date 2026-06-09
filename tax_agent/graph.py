"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt and a simulated MCP-compatible tool.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from common.llm import get_llm
from common.mcp_capability import MCPServer

# 1. Local tool implementation (grounding knowledge)
def search_tax_law_local_impl(query: str) -> str:
    """Search tax law knowledge base for relevant statutes and penalties."""
    knowledge = [
        (
            ["tax", "evasion", "fraud", "irs"],
            "Tax evasion (26 U.S.C. § 7201): felony, up to $250K fine and 5 years prison. "
            "Civil fraud penalty: 75% of underpayment (IRC § 6663). Failure to file: up to "
            "$25K fine and 1 year prison.",
        ),
        (
            ["offshore", "overseas", "foreign", "fbar", "fatca"],
            "FBAR penalties: up to $100K or 50% of account balance per violation. "
            "FATCA non-compliance: 30% withholding on US-source payments. "
            "Willful violations may trigger criminal prosecution.",
        ),
        (
            ["transfer", "pricing", "corporate"],
            "Transfer pricing violations (IRC § 482): IRS can reallocate income between "
            "related entities. Penalties: 20-40% of underpayment for substantial/gross "
            "valuation misstatements.",
        ),
    ]
    query_lower = query.lower()
    results = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)
    return "\n\n".join(results) if results else "No specific tax law matches found."

# 2. Set up simulated MCP server (discovery + invocation)
mcp_server = MCPServer("tax_mcp_server")
mcp_server.register_tool(
    name="search_tax_law",
    description="Search tax law knowledge base for relevant statutes and penalties.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query about tax law."
            }
        },
        "required": ["query"]
    },
    func=search_tax_law_local_impl
)

# 3. Create LangChain wrapper tool that invokes the MCP server
@tool
def search_tax_law(query: str) -> str:
    """Search tax law knowledge base for relevant statutes and penalties.

    This tool is invoked via an educational MCP-compatible simulation.
    """
    return mcp_server.call_tool("search_tax_law", {"query": query})


TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA. Focus ONLY on tax implications.
- Use the search_tax_law tool to ground your analysis.
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
        tools=[search_tax_law],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph