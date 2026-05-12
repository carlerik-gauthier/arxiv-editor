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

__all__ = [
    "Audience",
    "ContentSynthesizer",
    "DateRangePreference",
    "DeliveryMode",
    "DeliveryPreference",
    "Depth",
    "RequestParseResult",
    "SummaryFormat",
    "SummaryRequest",
    "SummaryRequestSession",
    "Tone",
    "clarify_request_tool",
    "parse_user_request",
    "parse_user_request_tool",
]


def __getattr__(name):
    """Lazily expose synthesis classes without creating import cycles."""
    if name == "ContentSynthesizer":
        from src.generation.synthesizer import ContentSynthesizer

        return ContentSynthesizer
    raise AttributeError(name)
