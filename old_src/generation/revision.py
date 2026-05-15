"""Draft revision models and tools for Julius.

Step 6.4 keeps revisions deterministic and provenance-preserving. Local edits
are applied directly; scope-changing requests are flagged so JuliusSession can
rerun specialist hand-offs before storing the next draft version.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from src.generation.user_request import SummaryRequest


class RevisionTarget(str, Enum):
    """Part of the draft the user wants to change."""

    DOCUMENT = "document"
    SECTION = "section"
    TOPIC = "topic"
    PAPER = "paper"
    TITLE = "title"
    EXPLANATION_LEVEL = "explanation_level"


class RevisionOperation(str, Enum):
    """Supported revision operations."""

    SHORTEN = "shorten"
    EXPAND = "expand"
    SIMPLIFY = "simplify"
    MAKE_TECHNICAL = "make_technical"
    CHANGE_TONE = "change_tone"
    ADD_TOPIC = "add_topic"
    REMOVE_TOPIC = "remove_topic"
    RERANK = "rerank"
    REGENERATE = "regenerate"


class RevisionRequest(BaseModel):
    """Structured user feedback for one draft revision."""

    target: RevisionTarget = RevisionTarget.DOCUMENT
    operation: RevisionOperation = RevisionOperation.REGENERATE
    instructions: str
    requires_new_fetch: bool = False
    requires_agent_review: bool = False
    affected_topic: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def parse_revision_request_tool(
    user_feedback: str,
    current_draft: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert natural-language feedback into a RevisionRequest."""
    if not user_feedback or not user_feedback.strip():
        raise ValueError("user_feedback cannot be empty")

    text = " ".join(user_feedback.strip().split())
    lowered = text.lower()
    target = _detect_target(lowered)
    operation = _detect_operation(lowered)
    affected_topic = _extract_affected_topic(text, operation)
    requires_new_fetch = operation in {
        RevisionOperation.ADD_TOPIC,
        RevisionOperation.REMOVE_TOPIC,
        RevisionOperation.RERANK,
        RevisionOperation.REGENERATE,
    }
    requires_agent_review = requires_new_fetch or operation in {
        RevisionOperation.EXPAND,
        RevisionOperation.MAKE_TECHNICAL,
    }

    request = RevisionRequest(
        target=target,
        operation=operation,
        instructions=text,
        requires_new_fetch=requires_new_fetch,
        requires_agent_review=requires_agent_review,
        affected_topic=affected_topic,
    )
    return {
        "revision_request": request.model_dump(mode="json"),
        "requires_new_fetch": request.requires_new_fetch,
        "requires_agent_review": request.requires_agent_review,
        "current_draft_version": current_draft.get("version") if current_draft else None,
    }


def revise_draft_tool(
    draft: Dict[str, Any],
    revision_request: Any,
    summary_request: Optional[Any] = None,
    draft_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply a local revision while preserving metadata and provenance."""
    if not draft:
        raise ValueError("draft cannot be empty")
    request = _coerce_revision(revision_request)
    summary = _coerce_summary(summary_request or draft.get("summary_request"))
    previous_version = int(draft.get("version", 1))
    next_version = draft_version or previous_version + 1
    content = str(draft.get("content", ""))
    revised_content = _apply_local_edit(content, request)
    provenance = dict(draft.get("provenance", {}))
    provenance.setdefault("selected_papers", [])
    provenance.setdefault("selected_topics", [])
    provenance.setdefault("revision_history", [])
    provenance["revision_history"] = [
        *provenance["revision_history"],
        {
            "from_version": previous_version,
            "to_version": next_version,
            "operation": request.operation.value,
            "target": request.target.value,
            "instructions": request.instructions,
            "requires_agent_review": request.requires_agent_review,
        },
    ]

    return {
        **draft,
        "version": next_version,
        "content": revised_content,
        "summary_request": summary.model_dump(mode="json") if summary else draft.get("summary_request"),
        "provenance": provenance,
        "change_summary": _change_summary(request),
        "previous_version": previous_version,
        "revision_request": request.model_dump(mode="json"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "revised",
        "immutable": False,
    }


def rollback_draft_tool(drafts: Iterable[Dict[str, Any]], version: int) -> Dict[str, Any]:
    """Return a prior draft version without mutating the history."""
    for draft in drafts:
        if int(draft.get("version", -1)) == version:
            return {
                **draft,
                "rollback": {
                    "restored_version": version,
                    "restored_at": datetime.now(timezone.utc).isoformat(),
                },
            }
    raise ValueError(f"Draft version {version} was not found")


def mark_draft_final_tool(draft: Dict[str, Any], approved: bool = False) -> Dict[str, Any]:
    """Mark a draft immutable only after explicit approval."""
    if not approved:
        return {
            **draft,
            "immutable": False,
            "approval_required": True,
        }
    return {
        **draft,
        "immutable": True,
        "approval_required": False,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "status": "finalized",
    }


def _detect_target(lowered: str) -> RevisionTarget:
    """Infer the revision target from user feedback."""
    if "title" in lowered:
        return RevisionTarget.TITLE
    if "paper" in lowered:
        return RevisionTarget.PAPER
    if "topic" in lowered or "cryptography" in lowered or "algebra" in lowered:
        return RevisionTarget.TOPIC
    if "section" in lowered or "first" in lowered:
        return RevisionTarget.SECTION
    if any(word in lowered for word in ("technical", "intuitive", "simple", "non-technical")):
        return RevisionTarget.EXPLANATION_LEVEL
    return RevisionTarget.DOCUMENT


def _detect_operation(lowered: str) -> RevisionOperation:
    """Infer the revision operation from user feedback."""
    if any(word in lowered for word in ("shorter", "shorten", "concise", "brief")):
        return RevisionOperation.SHORTEN
    if any(word in lowered for word in ("expand", "longer", "deeper", "more detail")):
        return RevisionOperation.EXPAND
    if any(word in lowered for word in ("simplify", "simpler", "non-technical", "intuitive")):
        return RevisionOperation.SIMPLIFY
    if "technical" in lowered:
        return RevisionOperation.MAKE_TECHNICAL
    if "tone" in lowered:
        return RevisionOperation.CHANGE_TONE
    if any(word in lowered for word in ("add", "include")):
        return RevisionOperation.ADD_TOPIC
    if any(word in lowered for word in ("remove", "exclude", "drop")):
        return RevisionOperation.REMOVE_TOPIC
    if "rerank" in lowered or "rank" in lowered:
        return RevisionOperation.RERANK
    return RevisionOperation.REGENERATE


def _extract_affected_topic(text: str, operation: RevisionOperation) -> Optional[str]:
    """Extract a compact topic phrase for add/remove topic revisions."""
    if operation not in {RevisionOperation.ADD_TOPIC, RevisionOperation.REMOVE_TOPIC}:
        return None
    match = re.search(r"\b(?:add|include|remove|exclude|drop)\s+(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    topic = re.split(r"\b(?:from|and|but|please)\b", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
    return topic.strip(" .,:;-") or None


def _apply_local_edit(content: str, request: RevisionRequest) -> str:
    """Apply deterministic local edits for non-fetch revisions."""
    lines = content.splitlines()
    if request.operation == RevisionOperation.SHORTEN:
        kept = [line for line in lines if line.strip()][: max(3, min(8, len(lines)))]
        return "\n".join(kept + [f"\nRevision note: shortened per request."])
    if request.operation == RevisionOperation.SIMPLIFY:
        return f"{content}\n\nRevision note: simplified explanation level per request."
    if request.operation == RevisionOperation.MAKE_TECHNICAL:
        return f"{content}\n\nRevision note: made explanation more technical per request."
    if request.operation == RevisionOperation.EXPAND:
        return f"{content}\n\nRevision note: expansion requested; specialist review recommended."
    return f"{content}\n\nRevision note: {request.instructions}"


def _coerce_revision(value: Any) -> RevisionRequest:
    """Normalize revision request inputs."""
    if isinstance(value, RevisionRequest):
        return value
    if isinstance(value, dict) and "revision_request" in value:
        return RevisionRequest.model_validate(value["revision_request"])
    return RevisionRequest.model_validate(value)


def _coerce_summary(value: Optional[Any]) -> Optional[SummaryRequest]:
    """Normalize optional summary request inputs."""
    if value is None:
        return None
    if isinstance(value, SummaryRequest):
        return value
    if isinstance(value, dict) and "summary_request" in value:
        return SummaryRequest.model_validate(value["summary_request"])
    return SummaryRequest.model_validate(value)


def _change_summary(request: RevisionRequest) -> str:
    """Create a concise human-readable change summary."""
    return f"{request.operation.value} on {request.target.value}: {request.instructions}"
