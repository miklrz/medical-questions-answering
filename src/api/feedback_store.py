"""
Simple in-memory store for human feedback (useful/not useful).
For production, use Redis or DB.
"""

from collections import defaultdict

# {request_id: {"useful": int, "not_useful": int}}
_feedback: dict[str, dict[str, int]] = defaultdict(lambda: {"useful": 0, "not_useful": 0})


def record_feedback(request_id: str, useful: bool) -> None:
    """Record user feedback for a response."""
    key = "useful" if useful else "not_useful"
    _feedback[request_id][key] += 1


def get_feedback_stats(request_id: str) -> dict[str, int]:
    """Get feedback counts for a request."""
    return dict(_feedback.get(request_id, {"useful": 0, "not_useful": 0}))


def get_aggregate_stats() -> dict[str, int]:
    """Get aggregate useful/not_useful across all requests."""
    total = {"useful": 0, "not_useful": 0}
    for v in _feedback.values():
        total["useful"] += v["useful"]
        total["not_useful"] += v["not_useful"]
    return total
