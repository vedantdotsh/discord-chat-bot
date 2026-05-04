"""
memory.py — Per-channel conversation history
Stores the last N message exchanges so the bot remembers context.
"""

from collections import defaultdict
from config import MAX_HISTORY

# { channel_id: [ {"role": "user"|"assistant", "content": str}, ... ] }
_history: dict[int, list[dict]] = defaultdict(list)


def get_history(channel_id: int) -> list[dict]:
    """Return conversation history for a channel."""
    return list(_history[channel_id])


def add_message(channel_id: int, role: str, content: str) -> None:
    """Add a message to history, trimming oldest if over limit."""
    msgs     = _history[channel_id]
    max_msgs = MAX_HISTORY * 2   # each exchange = 1 user + 1 assistant
    msgs.append({"role": role, "content": content})
    while len(msgs) > max_msgs:
        msgs.pop(0)
        if msgs:
            msgs.pop(0)


def clear_history(channel_id: int) -> None:
    """Clear history for a specific channel."""
    _history.pop(channel_id, None)


def stats() -> dict:
    """Return memory usage stats."""
    return {
        "channels_with_history": len(_history),
        "total_messages": sum(len(v) for v in _history.values()),
    }

