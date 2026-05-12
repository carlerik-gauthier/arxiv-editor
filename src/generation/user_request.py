"""User request intake and preference persistence for Julius.

This module is the phase-6.1 boundary between free-form user messages and the
structured inputs Julius needs before expensive paper fetching, topic modeling,
and specialist hand-offs begin. The parser is intentionally deterministic so it
can run in tests and local CLI flows without API keys; an LLM-backed parser can
later be added behind the same Pydantic contracts.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class Audience(str, Enum):
    """Reader expertise level requested by the user."""

    EXPERT = "expert"
    NON_EXPERT = "non_expert"
    MIXED = "mixed"


class Depth(str, Enum):
    """Requested summary depth."""

    BRIEF = "brief"
    STANDARD = "standard"
    DEEP = "deep"


class Tone(str, Enum):
    """Editorial voice requested for the final summary."""

    EDITORIAL = "editorial"
    TECHNICAL = "technical"
    PEDAGOGICAL = "pedagogical"
    EXECUTIVE = "executive"


class SummaryFormat(str, Enum):
    """Supported summary layouts."""

    ONE_PAGER = "one_pager"
    BULLET_DIGEST = "bullet_digest"
    PAPER_RANKINGS = "paper_rankings"
    CUSTOM = "custom"


class DeliveryMode(str, Enum):
    """Requested delivery channel for the finished document."""

    PREVIEW = "preview"
    FILE = "file"
    EMAIL = "email"


class DateRangePreference(BaseModel):
    """
    User-facing date preference.

    The `label` preserves relative language such as "last week", while
    `start_date` and `end_date` give downstream fetchers concrete bounds.
    """

    label: str = Field(default="last week", description="Human-readable date range label")
    start_date: Optional[date] = Field(default=None, description="Inclusive start date")
    end_date: Optional[date] = Field(default=None, description="Inclusive end date")
    source_text: Optional[str] = Field(
        default=None,
        description="Text fragment that produced this date range",
    )

    @model_validator(mode="after")
    def validate_order(self) -> "DateRangePreference":
        """Reject inverted explicit date ranges."""
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        return self


class DeliveryPreference(BaseModel):
    """Delivery preference with optional email recipient."""

    mode: DeliveryMode = Field(default=DeliveryMode.PREVIEW)
    email_recipient: Optional[str] = Field(default=None)

    @field_validator("email_recipient")
    @classmethod
    def validate_email_recipient(cls, value: Optional[str]) -> Optional[str]:
        """Apply a lightweight email-shape check without adding dependencies."""
        if value is None:
            return value
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("email_recipient must look like an email address")
        return value


class SummaryRequest(BaseModel):
    """
    Structured user preferences for one research-summary workflow.

    Defaults represent the project's normal behavior: a mixed-audience,
    standard-depth one-pager covering last week's recent research, previewed
    before file or email delivery.
    """

    topic_query: Optional[str] = Field(
        default=None,
        description="Free-text research interest such as 'LLM agents' or 'cryptography'",
    )
    date_range: DateRangePreference = Field(default_factory=DateRangePreference)
    audience: Audience = Field(default=Audience.MIXED)
    depth: Depth = Field(default=Depth.STANDARD)
    tone: Tone = Field(default=Tone.EDITORIAL)
    format: SummaryFormat = Field(default=SummaryFormat.ONE_PAGER)
    max_topics: int = Field(default=5, ge=1, le=20)
    max_papers: int = Field(default=10, ge=1, le=50)
    must_include_categories: List[str] = Field(default_factory=list)
    exclude_categories: List[str] = Field(default_factory=list)
    delivery: DeliveryPreference = Field(default_factory=DeliveryPreference)
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Free-text instructions for custom formats or unusual constraints",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("must_include_categories", "exclude_categories")
    @classmethod
    def normalize_categories(cls, values: Sequence[str]) -> List[str]:
        """Normalize and de-duplicate ArXiv category codes while preserving order."""
        seen: Set[str] = set()
        normalized: List[str] = []
        for value in values:
            category = str(value).strip()
            if not category or category in seen:
                continue
            seen.add(category)
            normalized.append(category)
        return normalized

    @model_validator(mode="after")
    def update_timestamp(self) -> "SummaryRequest":
        """Keep `updated_at` populated for persisted session snapshots."""
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at
        return self


class RequestParseResult(BaseModel):
    """Structured output from request intake tools."""

    summary_request: SummaryRequest
    missing_fields: List[str] = Field(default_factory=list)
    ambiguous_fields: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    extracted_fields: List[str] = Field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        """Return true when Julius should ask before starting expensive work."""
        return bool(self.missing_fields or self.ambiguous_fields)

    def to_tool_result(self) -> Dict[str, Any]:
        """Serialize the result using JSON-friendly values for agent tools."""
        payload = self.model_dump(mode="json")
        payload["needs_clarification"] = self.needs_clarification
        return payload


class SummaryRequestSession:
    """
    Persist the latest SummaryRequest across refinement turns.

    The session stores JSON-serializable snapshots so a later JuliusSession can
    save or hydrate the same state without depending on agent internals.
    """

    def __init__(self, initial_request: Optional[Any] = None) -> None:
        self.current_request = _coerce_summary_request(initial_request)
        self.history: List[Dict[str, Any]] = []
        if self.current_request:
            self._record_snapshot("initial", None, self.current_request)

    def apply_message(
        self,
        message: str,
        defaults: Optional[Any] = None,
        reference_date: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Parse a user message, merge it with remembered preferences, and persist it.
        """
        effective_defaults = defaults if defaults is not None else self.current_request
        result = parse_user_request(message, effective_defaults, reference_date)
        self.current_request = result.summary_request
        self._record_snapshot("message", message, self.current_request)
        return result.to_tool_result()

    def remember(self, summary_request: Any, source: str = "manual") -> SummaryRequest:
        """Persist a request supplied by another component."""
        request = _coerce_summary_request(summary_request)
        if request is None:
            raise ValueError("summary_request cannot be empty")
        self.current_request = request
        self._record_snapshot(source, None, request)
        return request

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the persisted request state."""
        return {
            "current_request": (
                self.current_request.model_dump(mode="json") if self.current_request else None
            ),
            "history": list(self.history),
        }

    def _record_snapshot(
        self,
        source: str,
        message: Optional[str],
        request: SummaryRequest,
    ) -> None:
        """Append a session snapshot for auditability and future refinement."""
        self.history.append(
            {
                "source": source,
                "message": message,
                "summary_request": request.model_dump(mode="json"),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def parse_user_request_tool(
    message: str,
    defaults: Optional[Any] = None,
    reference_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Agent-callable wrapper that extracts SummaryRequest preferences from text.

    Args:
        message: User's natural-language request or refinement.
        defaults: Existing request/default fields to preserve unless overridden.
        reference_date: Optional date used to resolve relative ranges in tests
            and deterministic workflows.
    """
    return parse_user_request(message, defaults, reference_date).to_tool_result()


def clarify_request_tool(summary_request: Any) -> Dict[str, Any]:
    """
    Generate at most three focused clarification questions for Julius.

    Defaults are considered sufficient for topic, date range, audience, depth,
    tone, and format. Questions are only generated for fields that block the
    requested delivery or are internally contradictory.
    """
    request = _coerce_summary_request(summary_request)
    if request is None:
        request = SummaryRequest()

    questions: List[str] = []
    if request.delivery.mode == DeliveryMode.EMAIL and not request.delivery.email_recipient:
        questions.append("Which email address should I send the finished summary to?")

    if request.format == SummaryFormat.CUSTOM and not request.custom_instructions:
        questions.append("What custom structure should the summary follow?")

    overlapping_categories = sorted(
        set(request.must_include_categories).intersection(request.exclude_categories)
    )
    if overlapping_categories:
        questions.append(
            "Should I include or exclude these conflicting categories: "
            f"{', '.join(overlapping_categories)}?"
        )

    return {
        "questions": questions[:3],
        "needs_clarification": bool(questions),
        "summary_request": request.model_dump(mode="json"),
    }


def parse_user_request(
    message: str,
    defaults: Optional[Any] = None,
    reference_date: Optional[Any] = None,
) -> RequestParseResult:
    """
    Convert a natural-language request into a complete SummaryRequest.

    The parser treats the supplied defaults as sticky preferences. Any field not
    explicitly mentioned in `message` is preserved from defaults, which lets
    Julius handle refinement turns such as "make it shorter" or "only cs.AI".
    """
    if not message or not message.strip():
        raise ValueError("message cannot be empty")

    reference = _coerce_reference_date(reference_date)
    base_request = _coerce_summary_request(defaults) or _default_summary_request(reference)
    updates: Dict[str, Any] = {}
    extracted_fields: List[str] = []
    assumptions: List[str] = []

    text = " ".join(message.strip().split())
    lowered = text.lower()

    date_range = _extract_date_range(text, reference)
    if date_range:
        updates["date_range"] = date_range
        extracted_fields.append("date_range")
    elif defaults is None or not base_request.date_range.start_date:
        if not base_request.date_range.start_date:
            updates["date_range"] = _default_summary_request(reference).date_range
        assumptions.append("No date range specified; using last week.")

    audience = _extract_choice(lowered, _AUDIENCE_KEYWORDS)
    if audience:
        updates["audience"] = audience
        extracted_fields.append("audience")

    depth = _extract_choice(lowered, _DEPTH_KEYWORDS)
    if depth:
        updates["depth"] = depth
        extracted_fields.append("depth")

    tone = _extract_choice(lowered, _TONE_KEYWORDS)
    if tone:
        updates["tone"] = tone
        extracted_fields.append("tone")
    elif audience == Audience.NON_EXPERT:
        updates["tone"] = Tone.PEDAGOGICAL
        assumptions.append("Using a pedagogical tone for a non-expert audience.")

    output_format = _extract_choice(lowered, _FORMAT_KEYWORDS)
    if output_format:
        updates["format"] = output_format
        extracted_fields.append("format")

    max_topics = _extract_limit(lowered, "topics?")
    if max_topics is not None:
        updates["max_topics"] = max(1, min(max_topics, 20))
        extracted_fields.append("max_topics")

    max_papers = _extract_limit(lowered, "papers?")
    if max_papers is not None:
        updates["max_papers"] = max(1, min(max_papers, 50))
        extracted_fields.append("max_papers")

    delivery = _extract_delivery(text, lowered, base_request.delivery)
    if delivery != base_request.delivery:
        updates["delivery"] = delivery
        extracted_fields.append("delivery")

    include_categories, exclude_categories = _extract_categories(text)
    if include_categories:
        updates["must_include_categories"] = include_categories
        extracted_fields.append("must_include_categories")
    if exclude_categories:
        updates["exclude_categories"] = exclude_categories
        extracted_fields.append("exclude_categories")

    topic_query = _extract_topic_query(text)
    if topic_query:
        updates["topic_query"] = topic_query
        extracted_fields.append("topic_query")
    elif base_request.topic_query is None:
        assumptions.append("No topic specified; Julius will consider all assigned categories.")

    if output_format == SummaryFormat.CUSTOM:
        updates["custom_instructions"] = text

    request_data = base_request.model_dump()
    request_data.update(updates)
    request_data["updated_at"] = datetime.now(timezone.utc)
    summary_request = SummaryRequest.model_validate(request_data)

    missing_fields, ambiguous_fields = _find_clarification_fields(summary_request)

    return RequestParseResult(
        summary_request=summary_request,
        missing_fields=missing_fields,
        ambiguous_fields=ambiguous_fields,
        assumptions=assumptions,
        extracted_fields=_unique_preserving_order(extracted_fields),
    )


def _default_summary_request(reference_date: date) -> SummaryRequest:
    """Build the default request with a concrete last-week range."""
    end_date = reference_date
    start_date = end_date - timedelta(days=7)
    return SummaryRequest(
        date_range=DateRangePreference(
            label="last week",
            start_date=start_date,
            end_date=end_date,
            source_text="default",
        )
    )


def _coerce_summary_request(value: Optional[Any]) -> Optional[SummaryRequest]:
    """Accept SummaryRequest, tool output, or dict defaults."""
    if value is None:
        return None
    if isinstance(value, SummaryRequest):
        return value
    if isinstance(value, RequestParseResult):
        return value.summary_request
    if isinstance(value, dict):
        payload = dict(value)
        if "summary_request" in payload and isinstance(payload["summary_request"], dict):
            payload = payload["summary_request"]
        return SummaryRequest.model_validate(payload)
    raise TypeError("defaults must be a SummaryRequest, RequestParseResult, dict, or None")


def _coerce_reference_date(value: Optional[Any]) -> date:
    """Normalize reference dates supplied by tests, CLI, or agent state."""
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError("reference_date must be a date, datetime, ISO date string, or None")


def _extract_choice(lowered: str, keyword_map: Sequence[Tuple[Any, Sequence[str]]]) -> Optional[Any]:
    """Return the first enum value whose phrase appears in the message."""
    for value, keywords in keyword_map:
        if any(keyword in lowered for keyword in keywords):
            return value
    return None


def _extract_date_range(text: str, reference_date: date) -> Optional[DateRangePreference]:
    """Extract explicit ISO dates or common relative date ranges."""
    lowered = text.lower()
    explicit_dates = [date.fromisoformat(match) for match in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)]
    if len(explicit_dates) >= 2:
        return DateRangePreference(
            label=f"{explicit_dates[0].isoformat()} to {explicit_dates[1].isoformat()}",
            start_date=explicit_dates[0],
            end_date=explicit_dates[1],
            source_text="explicit dates",
        )
    if len(explicit_dates) == 1:
        return DateRangePreference(
            label=explicit_dates[0].isoformat(),
            start_date=explicit_dates[0],
            end_date=explicit_dates[0],
            source_text="explicit date",
        )

    count_match = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(day|days|week|weeks|month|months)\b", lowered)
    if count_match:
        amount = int(count_match.group(1))
        unit = count_match.group(2)
        days = amount
        if unit.startswith("week"):
            days = amount * 7
        elif unit.startswith("month"):
            days = amount * 30
        return DateRangePreference(
            label=count_match.group(0),
            start_date=reference_date - timedelta(days=days),
            end_date=reference_date,
            source_text=count_match.group(0),
        )

    relative_ranges = [
        (("last week", "past week", "previous week", "this week"), 7, "last week"),
        (("last month", "past month", "previous month", "this month"), 30, "last month"),
        (("yesterday",), 1, "yesterday"),
        (("today",), 0, "today"),
    ]
    for phrases, days, label in relative_ranges:
        if any(phrase in lowered for phrase in phrases):
            if label == "yesterday":
                target = reference_date - timedelta(days=1)
                return DateRangePreference(
                    label=label,
                    start_date=target,
                    end_date=target,
                    source_text=label,
                )
            return DateRangePreference(
                label=label,
                start_date=reference_date - timedelta(days=days),
                end_date=reference_date,
                source_text=label,
            )

    return None


def _extract_limit(lowered: str, noun_pattern: str) -> Optional[int]:
    """Extract a numeric limit tied to topics or papers."""
    patterns = [
        rf"\b(?:max|maximum|at most|up to|no more than)\s+(\d+)\s+{noun_pattern}\b",
        rf"\b(\d+)\s+{noun_pattern}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def _extract_delivery(
    text: str,
    lowered: str,
    default: DeliveryPreference,
) -> DeliveryPreference:
    """Extract preview, file, or email delivery preferences."""
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    if "email" in lowered or "send it" in lowered or email_match:
        return DeliveryPreference(
            mode=DeliveryMode.EMAIL,
            email_recipient=email_match.group(0) if email_match else default.email_recipient,
        )
    if any(keyword in lowered for keyword in ("save", "file", "write to disk", "export")):
        return DeliveryPreference(mode=DeliveryMode.FILE, email_recipient=None)
    if "preview" in lowered:
        return DeliveryPreference(mode=DeliveryMode.PREVIEW, email_recipient=None)
    return default


def _extract_categories(text: str) -> Tuple[List[str], List[str]]:
    """Extract ArXiv category include/exclude preferences."""
    categories = re.findall(
        r"\b(?:cs|math|stat|q-bio|q-fin|eess|econ|astro-ph|cond-mat|gr-qc|hep-[a-z]+|nucl-[a-z]+|physics|quant-ph)\.[A-Za-z]{2}\b",
        text,
        flags=re.IGNORECASE,
    )
    include: List[str] = []
    exclude: List[str] = []
    for category in categories:
        start = text.lower().find(category.lower())
        context = text[max(0, start - 40) : start].lower()
        if any(keyword in context for keyword in ("exclude", "without", "remove", "except", "not ")):
            exclude.append(category)
        else:
            include.append(category)
    return _unique_preserving_order(include), _unique_preserving_order(exclude)


def _extract_topic_query(text: str) -> Optional[str]:
    """Extract a compact free-text topic interest from a user message."""
    quoted = re.search(r"['\"]([^'\"]{3,120})['\"]", text)
    if quoted:
        return quoted.group(1).strip()

    candidate = text
    topic_anchor = re.search(
        r"\b(?:about|around|on|focused on|focus on|covering|cover|of)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if topic_anchor:
        candidate = topic_anchor.group(1)

    candidate = re.split(
        r"\b(?:sent to|email|save|as a|in a|with\b|only\b|exclude\b|without\b|to\b|from\b|for\s+(?:last|past|previous|this|\d{4}-\d{2}-\d{2}|experts|non[- ]?experts))\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = _strip_preference_phrases(candidate)
    candidate = re.sub(r"\b[A-Za-z-]+\.[A-Za-z]{2}\b", " ", candidate)
    candidate = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", candidate)
    candidate = re.sub(r"\b\d+\b", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;-")

    if not candidate:
        return None
    lowered = candidate.lower()
    if all(token in _GENERIC_TOPIC_WORDS for token in lowered.split()):
        return None
    if lowered in _GENERIC_TOPIC_WORDS:
        return None
    if len(candidate) < 3:
        return None
    return candidate


def _strip_preference_phrases(value: str) -> str:
    """Remove command, date, format, tone, and generic paper words from a topic."""
    cleaned = value
    phrases = [
        r"\b(?:give me|make|create|write|generate|summarize|summary|digest|brief|one[- ]?pager)\b",
        r"\b(?:latest|recent)\b",
        r"\b(?:last|past|previous|this)\s+(?:week|month|year)\b",
        r"\b(?:last|past|previous)\s+\d+\s+(?:day|days|week|weeks|month|months)\b",
        r"\b(?:non[- ]?technical|non[- ]?expert|mixed|expert|technical|pedagogical|executive|editorial)\b",
        r"\b(?:shorter|longer|deeper|detailed|concise|quick|standard)\b",
        r"\b(?:bullet|bullets|rankings?|top|custom|format|papers?|research|articles?|arxiv)\b",
        r"\b(?:from|to|only|exclude|without|with|for)\b",
        r"\b(?:the|a|an|and|or|me|it|please)\b",
        r"\b(?:today|yesterday)\b",
        r"'s\b",
    ]
    for phrase in phrases:
        cleaned = re.sub(phrase, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _find_clarification_fields(request: SummaryRequest) -> Tuple[List[str], List[str]]:
    """Identify fields that block execution or are contradictory."""
    missing: List[str] = []
    ambiguous: List[str] = []
    if request.delivery.mode == DeliveryMode.EMAIL and not request.delivery.email_recipient:
        missing.append("delivery.email_recipient")
    if request.format == SummaryFormat.CUSTOM and not request.custom_instructions:
        missing.append("custom_instructions")

    overlaps = set(request.must_include_categories).intersection(request.exclude_categories)
    if overlaps:
        ambiguous.append("arxiv_categories")
    return missing, ambiguous


def _unique_preserving_order(values: Sequence[Any]) -> List[Any]:
    """Return a list with duplicates removed in first-seen order."""
    seen: Set[Any] = set()
    unique: List[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


_AUDIENCE_KEYWORDS: Sequence[Tuple[Audience, Sequence[str]]] = (
    (Audience.MIXED, ("mixed audience", "experts and non-experts", "experts and non experts", "both audiences")),
    (Audience.NON_EXPERT, ("non-technical", "non technical", "non-expert", "non expert", "layperson", "general audience", "accessible")),
    (Audience.EXPERT, ("expert", "researcher", "technical audience", "technical summary")),
)

_DEPTH_KEYWORDS: Sequence[Tuple[Depth, Sequence[str]]] = (
    (Depth.BRIEF, ("brief", "short", "shorter", "concise", "quick")),
    (Depth.DEEP, ("deep", "deeper", "detailed", "comprehensive", "in-depth", "in depth")),
    (Depth.STANDARD, ("standard",)),
)

_TONE_KEYWORDS: Sequence[Tuple[Tone, Sequence[str]]] = (
    (Tone.EXECUTIVE, ("executive", "decision maker", "strategic")),
    (Tone.PEDAGOGICAL, ("pedagogical", "intuitive", "intuition", "accessible", "non-technical", "non technical")),
    (Tone.TECHNICAL, ("technical", "formal", "expert tone")),
    (Tone.EDITORIAL, ("editorial", "news", "magazine")),
)

_FORMAT_KEYWORDS: Sequence[Tuple[SummaryFormat, Sequence[str]]] = (
    (SummaryFormat.BULLET_DIGEST, ("bullet digest", "bullet list", "bullets")),
    (SummaryFormat.PAPER_RANKINGS, ("paper ranking", "paper rankings", "rank papers", "ranked papers", "top papers")),
    (SummaryFormat.ONE_PAGER, ("one-pager", "one pager", "one page")),
    (SummaryFormat.CUSTOM, ("custom format", "custom structure")),
)

_GENERIC_TOPIC_WORDS = {
    "summary",
    "digest",
    "brief",
    "papers",
    "research",
    "articles",
    "latest",
    "recent",
    "the",
    "a",
    "an",
    "and",
    "or",
    "me",
    "it",
    "please",
}
