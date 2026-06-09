"""Law Agent LangGraph StateGraph definition.

Graph topology:
    analyze_law → check_routing → (parallel) call_tax + call_compliance + call_privacy → aggregate → END

The parallel branches (call_tax / call_compliance / call_privacy) are dispatched via LangGraph's
Send API so that all sub-agent calls happen concurrently.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm
from common.trace import TraceEvent, make_trace_event, summarize_text

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


def _merge_lists(a: list[TraceEvent] | None, b: list[TraceEvent] | None) -> list[TraceEvent]:
    """Reducer: merge lists of trace events, preserving all elements."""
    res = list(a) if a is not None else []
    if b is not None:
        res.extend(b)
    return res


class LawState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool
    # Annotated so parallel branches can both write without conflict
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    privacy_result: Annotated[str, _last_wins]
    trace: Annotated[list[TraceEvent], _merge_lists]
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def analyze_law(state: LawState) -> dict:
    """LLM analysis from a contract / general law perspective."""
    trace_start = make_trace_event("analyze_law", "analyze_start", "Started general legal analysis")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney specialising in contract law, "
                "tort law, and general business law. Analyse the legal aspects of the question "
                "thoroughly, covering relevant statutes, case law principles, and liability exposure."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    content = result.content
    trace_end = make_trace_event(
        "analyze_law",
        "analyze_end",
        f"Completed general legal analysis: {summarize_text(content, 120)}"
    )
    return {"law_analysis": content, "trace": [trace_start, trace_end]}


async def check_routing(state: LawState) -> dict:
    """Determine whether tax, compliance, and/or privacy sub-agents are needed.

    Returns updated state flags so the routing function can read them.
    If delegation depth is already at the max, skip further delegation.
    """
    trace_start = make_trace_event("check_routing", "check_routing_start", "Starting routing check")
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping sub-agents", depth)
        trace_end = make_trace_event(
            "check_routing",
            "check_routing_end",
            f"Max delegation depth ({MAX_DELEGATION_DEPTH}) reached; skipping sub-agents"
        )
        return {
            "needs_tax": False,
            "needs_compliance": False,
            "needs_privacy": False,
            "trace": [trace_start, trace_end]
        }

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                'You are a legal routing expert. Based on the question, decide whether '
                'specialist sub-agents are needed.\n'
                'Reply with ONLY valid JSON — no markdown, no extra text:\n'
                '{"needs_tax": <true|false>, "needs_compliance": <true|false>, "needs_privacy": <true|false>}\n\n'
                'needs_tax = true  → question involves tax law, IRS, tax evasion, penalties\n'
                'needs_compliance = true → question involves regulatory compliance, SEC, SOX, AML, FCPA\n'
                'needs_privacy = true → question involves privacy, data protection, GDPR, personal data'
            )
        ),
        HumanMessage(content=state["question"]),
    ]

    try:
        result = await llm.ainvoke(messages)
        raw = result.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
    except Exception as exc:
        logger.warning("Routing LLM/JSON parse failed: %s — executing deterministic fallback", exc)
        # Step 4 fallback implementation:
        question_lower = state["question"].lower()
        tax_kws = ["tax", "taxation", "irs", "income", "vat", "corporate tax", "thuế"]
        compliance_kws = ["compliance", "regulation", "sec", "aml", "fcpa", "sox", "audit", "policy", "quy định"]
        privacy_kws = ["privacy", "data", "personal data", "gdpr", "data protection", "dữ liệu", "thông tin cá nhân"]

        has_tax = any(kw in question_lower for kw in tax_kws)
        has_compliance = any(kw in question_lower for kw in compliance_kws)
        has_privacy = any(kw in question_lower for kw in privacy_kws)

        if has_tax or has_compliance or has_privacy:
            parsed = {
                "needs_tax": has_tax,
                "needs_compliance": has_compliance,
                "needs_privacy": has_privacy
            }
        else:
            # default to all True if no signals exist
            parsed = {
                "needs_tax": True,
                "needs_compliance": True,
                "needs_privacy": True
            }

    needs_tax = bool(parsed.get("needs_tax", True))
    needs_compliance = bool(parsed.get("needs_compliance", True))
    needs_privacy = bool(parsed.get("needs_privacy", False))
    logger.info("Routing decision: needs_tax=%s needs_compliance=%s needs_privacy=%s", needs_tax, needs_compliance, needs_privacy)

    trace_end = make_trace_event(
        "check_routing",
        "check_routing_end",
        f"Routing decision: needs_tax={needs_tax}, needs_compliance={needs_compliance}, needs_privacy={needs_privacy}"
    )
    return {
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
        "needs_privacy": needs_privacy,
        "trace": [trace_start, trace_end]
    }


def route_to_subagents(state: LawState) -> list[Send]:
    """Routing function: dispatch parallel Send objects based on routing flags.

    This function is used with add_conditional_edges; it returns a list of
    Send objects which LangGraph executes as parallel branches.
    """
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    if state.get("needs_privacy"):
        sends.append(Send("call_privacy", state))
    if not sends:
        # No sub-agents needed — go straight to aggregation
        sends.append(Send("aggregate", state))
    return sends


async def call_tax(state: LawState) -> dict:
    """Delegate to the Tax Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    trace_start = make_trace_event("call_tax", "tax_delegate_start", "Delegating to Tax Agent via A2A")
    try:
        endpoint = await discover("tax_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Tax Agent returned %d chars", len(result))
        trace_end = make_trace_event(
            "call_tax",
            "tax_delegate_end",
            f"Tax Agent completed: {summarize_text(result, 120)}"
        )
        return {"tax_result": result, "trace": [trace_start, trace_end]}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        err_msg = f"[Tax analysis unavailable: {exc}]"
        trace_end = make_trace_event("call_tax", "tax_delegate_failed", f"Tax Agent failed: {exc}")
        return {"tax_result": err_msg, "trace": [trace_start, trace_end]}


async def call_compliance(state: LawState) -> dict:
    """Delegate to the Compliance Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    trace_start = make_trace_event("call_compliance", "compliance_delegate_start", "Delegating to Compliance Agent via A2A")
    try:
        endpoint = await discover("compliance_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Compliance Agent returned %d chars", len(result))
        trace_end = make_trace_event(
            "call_compliance",
            "compliance_delegate_end",
            f"Compliance Agent completed: {summarize_text(result, 120)}"
        )
        return {"compliance_result": result, "trace": [trace_start, trace_end]}
    except Exception as exc:
        logger.exception("call_compliance failed: %s", exc)
        err_msg = f"[Compliance analysis unavailable: {exc}]"
        trace_end = make_trace_event("call_compliance", "compliance_delegate_failed", f"Compliance Agent failed: {exc}")
        return {"compliance_result": err_msg, "trace": [trace_start, trace_end]}


async def call_privacy(state: LawState) -> dict:
    """Delegate to the Privacy Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    trace_start = make_trace_event("call_privacy", "privacy_delegate_start", "Delegating to Privacy Agent via A2A")
    try:
        endpoint = await discover("privacy_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Privacy Agent returned %d chars", len(result))
        trace_end = make_trace_event(
            "call_privacy",
            "privacy_delegate_end",
            f"Privacy Agent completed: {summarize_text(result, 120)}"
        )
        return {"privacy_result": result, "trace": [trace_start, trace_end]}
    except Exception as exc:
        logger.exception("call_privacy failed: %s", exc)
        err_msg = f"[Privacy analysis unavailable: {exc}]"
        trace_end = make_trace_event("call_privacy", "privacy_delegate_failed", f"Privacy Agent failed: {exc}")
        return {"privacy_result": err_msg, "trace": [trace_start, trace_end]}


async def aggregate(state: LawState) -> dict:
    """Combine law_analysis, tax_result, compliance_result, and privacy_result into a final answer."""
    trace_start = make_trace_event("aggregate", "aggregate_start", "Starting synthesis of analyses")
    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance Analysis\n{state['compliance_result']}")
    if state.get("privacy_result"):
        sections.append(f"## Privacy Analysis\n{state['privacy_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response for the client. Combine the following "
                "analyses into a cohesive answer with clear sections. Avoid redundancy. "
                "End with a brief disclaimer that the analysis is educational and the client "
                "should consult licensed attorneys for their specific situation."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    final_content = result.content

    # Append trace log in a readable way at the very end of the final response
    trace_str = "\n\n### Execution Trace Log\n"
    # Sort trace events by timestamp
    sorted_trace = sorted(state.get("trace", []) + [trace_start], key=lambda x: x["timestamp"])
    for idx, event in enumerate(sorted_trace):
        trace_str += f"- **[{event['node']}]** *{event['event']}* @ {event['timestamp']}: {event['details']}\n"

    now_str = datetime.now(timezone.utc).isoformat()
    trace_str += f"- **[aggregate]** *aggregate_end* @ {now_str}: Completed synthesis of analyses\n"

    final_content_with_trace = final_content + trace_str

    return {"final_answer": final_content_with_trace, "trace": [trace_start]}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the Law Agent StateGraph."""
    graph = StateGraph(LawState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("call_privacy", call_privacy)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")

    # Conditional parallel dispatch: after check_routing, route_to_subagents
    # returns a list of Send objects (to call_tax, call_compliance, call_privacy, or aggregate)
    graph.add_conditional_edges(
        "check_routing",
        route_to_subagents,
        ["call_tax", "call_compliance", "call_privacy", "aggregate"],
    )

    graph.add_edge("call_tax", "aggregate")
    graph.add_edge("call_compliance", "aggregate")
    graph.add_edge("call_privacy", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()