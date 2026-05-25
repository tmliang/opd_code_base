"""Shared helpers for teacher dataloader recipes."""

from __future__ import annotations

from typing import Any


def last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last ``role == 'user'`` message in a chat-message list, or None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None
