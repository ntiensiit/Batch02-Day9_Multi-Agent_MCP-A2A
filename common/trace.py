from datetime import datetime, timezone
from typing import TypedDict

class TraceEvent(TypedDict):
    timestamp: str
    node: str
    event: str
    details: str

def now_iso() -> str:
    """Return the current ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()

def make_trace_event(node: str, event: str, details: str = "") -> TraceEvent:
    """Create a structured trace event."""
    return {
        "timestamp": now_iso(),
        "node": node,
        "event": event,
        "details": details,
    }

def summarize_text(text: str, max_chars: int = 100) -> str:
    """Summarize text by truncating it if it exceeds max_chars."""
    if not text:
        return ""
    text_clean = text.replace("\n", " ").strip()
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars - 3] + "..."
