"""Generation models and utilities."""

from src.generation.user_request import (
    Audience,
    DateRangePreference,
    DeliveryMode,
    DeliveryPreference,
    Depth,
    RequestParseResult,
    SummaryFormat,
    SummaryRequest,
    SummaryRequestSession,
    Tone,
    clarify_request_tool,
    parse_user_request,
    parse_user_request_tool,
)
from src.generation.revision import (
    RevisionOperation,
    RevisionRequest,
    RevisionTarget,
    mark_draft_final_tool,
    parse_revision_request_tool,
    revise_draft_tool,
    rollback_draft_tool,
)

__all__ = [
    "Audience",
    "ContentSynthesizer",
    "DateRangePreference",
    "DeliveryMode",
    "DeliveryPreference",
    "Depth",
    "RequestParseResult",
    "RevisionOperation",
    "RevisionRequest",
    "RevisionTarget",
    "SummaryFormat",
    "SummaryRequest",
    "SummaryRequestSession",
    "Tone",
    "clarify_request_tool",
    "mark_draft_final_tool",
    "parse_revision_request_tool",
    "parse_user_request",
    "parse_user_request_tool",
    "revise_draft_tool",
    "rollback_draft_tool",
]


def __getattr__(name):
    """Lazily expose synthesis classes without creating import cycles."""
    if name == "ContentSynthesizer":
        from src.generation.synthesizer import ContentSynthesizer

        return ContentSynthesizer
    raise AttributeError(name)
