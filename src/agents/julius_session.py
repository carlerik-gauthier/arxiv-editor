"""Interactive Julius conversation session.

This module implements phase 6.2: a stateful, deterministic conversation layer
that lets a user refine a SummaryRequest, trigger draft previews, ask why a
draft made certain choices, and finalize the current result. The actual
multi-agent content generation is still owned by later phase-6 steps, so draft
creation here is intentionally a lightweight preview with explicit provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.julius_agent import JuliusAgent
from src.generation.user_request import (
    DeliveryMode,
    SummaryRequest,
    clarify_request_tool,
    parse_user_request_tool,
)


class JuliusSessionState(str, Enum):
    """Conversation states supported by JuliusSession."""

    INTAKE = "INTAKE"
    CLARIFYING = "CLARIFYING"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    REVISING = "REVISING"
    FINALIZED = "FINALIZED"


class JuliusIntent(str, Enum):
    """High-level user intents for session routing."""

    NEW_SUMMARY_REQUEST = "NEW_SUMMARY_REQUEST"
    PREFERENCE_UPDATE = "PREFERENCE_UPDATE"
    SCOPE_UPDATE = "SCOPE_UPDATE"
    DRAFT_QUESTION = "DRAFT_QUESTION"
    GENERATE_DRAFT = "GENERATE_DRAFT"
    REVISION = "REVISION"
    FINALIZATION = "FINALIZATION"
    UNKNOWN = "UNKNOWN"


class JuliusSessionResponse(BaseModel):
    """Structured response returned by `JuliusSession.handle_user_message`."""

    message: str
    state: JuliusSessionState
    summary_request: Optional[Dict[str, Any]] = None
    draft_preview: Optional[str] = None
    actions_taken: List[str] = Field(default_factory=list)
    next_questions: List[str] = Field(default_factory=list)


class JuliusSession:
    """
    Stateful interactive loop for Julius.

    The session owns conversation history, the current SummaryRequest, draft
    previews, user feedback, and progress events. It does not fetch papers or
    run specialist agents yet; those expensive operations are introduced by the
    next phase and can plug into the same response contract.
    """

    def __init__(
        self,
        julius: Optional[JuliusAgent] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        reference_date: Optional[Any] = None,
    ) -> None:
        self.julius = julius or JuliusAgent()
        self.progress_callback = progress_callback
        self.reference_date = reference_date
        self.state = JuliusSessionState.INTAKE
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_request: Optional[SummaryRequest] = None
        self.drafts: List[Dict[str, Any]] = []
        self.user_feedback: List[Dict[str, Any]] = []
        self.progress_events: List[Dict[str, Any]] = []

    def handle_user_message(self, message: str) -> Dict[str, Any]:
        """
        Route one user message and return Julius's structured response.

        Args:
            message: Free-form user message.

        Returns:
            A JSON-serializable response with message, state, current request,
            draft preview, actions taken, and next questions.
        """
        if not message or not message.strip():
            response = self._build_response(
                "Please send a request or a revision instruction.",
                actions_taken=["asked_for_non_empty_message"],
                next_questions=["What research summary should Julius prepare?"],
            )
            self._record_message("assistant", response)
            return response.model_dump(mode="json")

        clean_message = " ".join(message.strip().split())
        self._record_message("user", clean_message)
        intent = classify_user_intent_tool(clean_message, self.state.value)["intent"]

        if intent == JuliusIntent.FINALIZATION.value:
            response = self._finalize_current_draft(clean_message)
        elif intent == JuliusIntent.DRAFT_QUESTION.value:
            response = self._answer_draft_question(clean_message)
        elif intent == JuliusIntent.GENERATE_DRAFT.value:
            response = self._generate_draft_preview()
        elif intent == JuliusIntent.REVISION.value:
            response = self._revise_request_or_draft(clean_message)
        elif intent in {
            JuliusIntent.NEW_SUMMARY_REQUEST.value,
            JuliusIntent.PREFERENCE_UPDATE.value,
            JuliusIntent.SCOPE_UPDATE.value,
        } or self.state in {JuliusSessionState.INTAKE, JuliusSessionState.CLARIFYING}:
            response = self._update_request(clean_message, intent)
        else:
            response = self._build_response(
                "I need one concrete instruction: update the request, generate a draft, ask about the draft, or finalize it.",
                actions_taken=["asked_for_intent_clarification"],
                next_questions=["Should I update preferences, generate a draft, or finalize the current draft?"],
            )

        self._record_message("assistant", response)
        return response.model_dump(mode="json")

    def emit_progress(self, message: str) -> None:
        """Record a short progress event and notify the optional callback."""
        event = {
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.progress_events.append(event)
        if self.progress_callback:
            self.progress_callback(message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session for a future CLI, web, or persistence layer."""
        return {
            "state": self.state.value,
            "conversation_history": list(self.conversation_history),
            "current_request": (
                self.current_request.model_dump(mode="json") if self.current_request else None
            ),
            "drafts": list(self.drafts),
            "user_feedback": list(self.user_feedback),
            "progress_events": list(self.progress_events),
        }

    def _update_request(self, message: str, intent: str) -> JuliusSessionResponse:
        """Parse request preferences and ask only blocking clarifications."""
        parsed = update_summary_request_tool(
            existing_request=self.current_request,
            user_feedback=message,
            reference_date=self.reference_date,
        )
        self.current_request = SummaryRequest.model_validate(parsed["summary_request"])
        self.julius.request_session.remember(self.current_request, source="julius_session")
        clarification = clarify_request_tool(self.current_request)
        next_questions = clarification["questions"]
        self.state = (
            JuliusSessionState.CLARIFYING
            if next_questions
            else JuliusSessionState.PLANNING
        )

        actions = ["updated_summary_request"]
        if intent == JuliusIntent.SCOPE_UPDATE.value:
            actions.append("updated_scope")
        if intent == JuliusIntent.PREFERENCE_UPDATE.value:
            actions.append("updated_preferences")
        if next_questions:
            actions.append("requested_clarification")

        return self._build_response(
            self._request_acknowledgement(next_questions),
            actions_taken=actions,
            next_questions=next_questions,
        )

    def _generate_draft_preview(self) -> JuliusSessionResponse:
        """Create a lightweight draft preview for review."""
        if self.current_request is None:
            self.state = JuliusSessionState.INTAKE
            return self._build_response(
                "I need the summary topic or scope before I can generate a draft.",
                actions_taken=["requested_initial_summary_request"],
                next_questions=["What research area and date range should Julius summarize?"],
            )

        clarification = clarify_request_tool(self.current_request)
        if clarification["needs_clarification"]:
            self.state = JuliusSessionState.CLARIFYING
            return self._build_response(
                "I need one detail before generating the draft.",
                actions_taken=["blocked_generation_for_clarification"],
                next_questions=clarification["questions"],
            )

        self.state = JuliusSessionState.GENERATING
        self.emit_progress("Preparing the paper search scope.")
        self.emit_progress("Modeling candidate topics.")
        self.emit_progress("Preparing specialist review tasks.")
        self.emit_progress("Compiling the draft preview.")

        draft = self._create_draft_stub()
        self.drafts.append(draft)
        self.state = JuliusSessionState.AWAITING_REVIEW
        return self._build_response(
            "Draft preview is ready for review.",
            actions_taken=["generated_draft_preview"],
            draft_preview=draft["content"],
        )

    def _revise_request_or_draft(self, message: str) -> JuliusSessionResponse:
        """Apply feedback to the request and refresh the draft preview if one exists."""
        self.user_feedback.append(
            {
                "message": message,
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.state = JuliusSessionState.REVISING
        parsed = update_summary_request_tool(
            existing_request=self.current_request,
            user_feedback=message,
            reference_date=self.reference_date,
        )
        self.current_request = SummaryRequest.model_validate(parsed["summary_request"])
        self.julius.request_session.remember(self.current_request, source="revision")

        if self.drafts:
            self.emit_progress("Applying the requested revision.")
            draft = self._create_draft_stub(change_summary=message)
            self.drafts.append(draft)
            self.state = JuliusSessionState.AWAITING_REVIEW
            return self._build_response(
                "I updated the draft preview.",
                actions_taken=["recorded_feedback", "updated_summary_request", "revised_draft_preview"],
                draft_preview=draft["content"],
            )

        self.state = JuliusSessionState.PLANNING
        return self._build_response(
            "I updated the request.",
            actions_taken=["recorded_feedback", "updated_summary_request"],
        )

    def _answer_draft_question(self, question: str) -> JuliusSessionResponse:
        """Answer questions about draft choices using stored draft provenance."""
        if not self.drafts:
            return self._build_response(
                "There is no draft yet, so I cannot explain draft choices.",
                actions_taken=["requested_draft_before_explanation"],
                next_questions=["Should I generate the first draft preview now?"],
            )

        answer = explain_draft_choice_tool(self.drafts[-1], question)["answer"]
        return self._build_response(
            answer,
            actions_taken=["answered_draft_question"],
            draft_preview=self.drafts[-1]["content"],
        )

    def _finalize_current_draft(self, message: str) -> JuliusSessionResponse:
        """Mark the current draft as finalized without performing delivery side effects."""
        if not self.drafts:
            return self._build_response(
                "There is no draft to finalize yet.",
                actions_taken=["requested_draft_before_finalization"],
                next_questions=["Should I generate the first draft preview now?"],
            )

        self.state = JuliusSessionState.FINALIZED
        self.user_feedback.append(
            {
                "message": message,
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        delivery = self.current_request.delivery.mode if self.current_request else DeliveryMode.PREVIEW
        action = "finalized_for_review"
        if delivery == DeliveryMode.FILE:
            action = "finalized_for_file_save"
        elif delivery == DeliveryMode.EMAIL:
            action = "finalized_for_email_delivery"

        return self._build_response(
            "Finalized the current draft. Delivery will be handled by the later output workflow.",
            actions_taken=["finalized_draft", action],
            draft_preview=self.drafts[-1]["content"],
        )

    def _create_draft_stub(self, change_summary: Optional[str] = None) -> Dict[str, Any]:
        """Build a deterministic draft preview and provenance record."""
        assert self.current_request is not None
        request = self.current_request
        version = len(self.drafts) + 1
        topic = request.topic_query or "all assigned research areas"
        categories = ", ".join(request.must_include_categories) or "Julius-selected categories"
        excluded = ", ".join(request.exclude_categories) or "none"
        date_range = request.date_range

        content_parts = [
            f"# Draft v{version}: {topic}",
            f"Date range: {date_range.start_date} to {date_range.end_date} ({date_range.label}).",
            f"Audience: {request.audience.value}; depth: {request.depth.value}; tone: {request.tone.value}.",
            f"Format: {request.format.value}; limits: {request.max_topics} topics, {request.max_papers} papers.",
            f"Categories: include {categories}; exclude {excluded}.",
            "This is a session preview. Full paper fetching, specialist analysis, and final synthesis are implemented in later phase-6 steps.",
        ]
        if change_summary:
            content_parts.append(f"Revision note: {change_summary}")

        return {
            "version": version,
            "content": "\n".join(content_parts),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary_request": request.model_dump(mode="json"),
            "provenance": {
                "selected_topics": [topic],
                "selected_papers": [],
                "inclusion_reason": (
                    "The preview follows the user's current topic, date range, audience, "
                    "format, category filters, and depth preferences."
                ),
                "omissions": [
                    "Representative paper selection is deferred to step 6.3.",
                    "Final file/email delivery is deferred to later output steps.",
                ],
            },
            "change_summary": change_summary,
        }

    def _request_acknowledgement(self, next_questions: List[str]) -> str:
        """Summarize the interpreted request in one concise message."""
        if self.current_request is None:
            return "I need a summary request before planning."
        request = self.current_request
        topic = request.topic_query or "all assigned research areas"
        if next_questions:
            return f"I interpreted the request as {topic}, but need one clarification."
        return f"I interpreted the request as {topic}. Say 'generate draft' when ready."

    def _build_response(
        self,
        message: str,
        actions_taken: Optional[List[str]] = None,
        next_questions: Optional[List[str]] = None,
        draft_preview: Optional[str] = None,
    ) -> JuliusSessionResponse:
        """Create the standard response payload."""
        return JuliusSessionResponse(
            message=message,
            state=self.state,
            summary_request=(
                self.current_request.model_dump(mode="json") if self.current_request else None
            ),
            draft_preview=draft_preview,
            actions_taken=actions_taken or [],
            next_questions=next_questions or [],
        )

    def _record_message(self, role: str, content: Any) -> None:
        """Append a user or assistant turn to session history."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "state": self.state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def classify_user_intent_tool(message: str, session_state: str = "INTAKE") -> Dict[str, str]:
    """
    Route a user message to the next session handler.

    The classifier is intentionally keyword-based for deterministic tests and
    local CLI use. An LLM classifier can later replace it behind this contract.
    """
    lowered = message.lower().strip()
    state = session_state.upper()

    if any(keyword in lowered for keyword in ("finalize", "finalise", "approve", "save this", "email it", "send it")):
        return {"intent": JuliusIntent.FINALIZATION.value}
    if lowered in {"generate", "generate draft", "draft", "start", "go ahead"} or "generate draft" in lowered:
        return {"intent": JuliusIntent.GENERATE_DRAFT.value}
    if lowered.endswith("?") or any(
        phrase in lowered
        for phrase in ("why did", "why choose", "why selected", "main result", "omitted", "excluded")
    ):
        return {"intent": JuliusIntent.DRAFT_QUESTION.value}
    if state == JuliusSessionState.AWAITING_REVIEW.value and any(
        keyword in lowered
        for keyword in ("shorter", "longer", "revise", "rewrite", "simplify", "technical", "remove", "add", "focus")
    ):
        return {"intent": JuliusIntent.REVISION.value}
    if any(keyword in lowered for keyword in ("only ", "exclude", "remove ", "include ", "last ", "past ", "cs.", "math.", "stat.")):
        return {"intent": JuliusIntent.SCOPE_UPDATE.value}
    if any(keyword in lowered for keyword in ("shorter", "brief", "deep", "technical", "non-technical", "audience", "tone", "format", "bullet")):
        return {"intent": JuliusIntent.PREFERENCE_UPDATE.value}
    if any(keyword in lowered for keyword in ("summary", "summarize", "digest", "papers", "research", "one-pager", "one pager")):
        return {"intent": JuliusIntent.NEW_SUMMARY_REQUEST.value}
    return {"intent": JuliusIntent.UNKNOWN.value}


def update_summary_request_tool(
    existing_request: Optional[Any],
    user_feedback: str,
    reference_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Convert a user request or refinement into an updated SummaryRequest.

    Existing preferences are sticky: the parser only changes fields mentioned in
    `user_feedback`, so short refinement turns do not erase the prior topic or
    date range.
    """
    return parse_user_request_tool(
        message=user_feedback,
        defaults=existing_request,
        reference_date=reference_date,
    )


def explain_draft_choice_tool(draft: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Explain draft topic, paper, ranking, or omission choices from provenance.
    """
    provenance = draft.get("provenance", {}) if isinstance(draft, dict) else {}
    lowered = question.lower()
    if "paper" in lowered and not provenance.get("selected_papers"):
        answer = (
            "No representative papers have been selected in this session preview. "
            "Paper selection starts in the multi-agent drafting step."
        )
    elif "omit" in lowered or "exclude" in lowered:
        omissions = provenance.get("omissions") or ["No omissions were recorded."]
        answer = "Recorded omissions: " + "; ".join(omissions)
    elif "why" in lowered or "choose" in lowered or "selected" in lowered:
        answer = provenance.get("inclusion_reason", "No inclusion reason was recorded.")
    elif "main result" in lowered:
        answer = (
            "Main results are not available in this preview because full paper analysis "
            "has not run yet."
        )
    else:
        answer = provenance.get("inclusion_reason", "The draft follows the current SummaryRequest.")

    return {
        "answer": answer,
        "question": question,
        "draft_version": draft.get("version") if isinstance(draft, dict) else None,
    }
