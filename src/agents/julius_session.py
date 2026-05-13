"""Interactive Julius conversation session.

This module implements phase 6.2: a stateful, deterministic conversation layer
that lets a user refine a SummaryRequest, trigger draft previews, ask why a
draft made certain choices, and finalize the current result. The actual
multi-agent content generation is still owned by later phase-6 steps, so draft
creation here is intentionally a lightweight preview with explicit provenance.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.julius_agent import JuliusAgent
from src.generation.user_request import (
    DeliveryMode,
    SummaryRequest,
    clarify_request_tool,
    parse_user_request_tool,
)
from src.generation.revision import (
    mark_draft_final_tool,
    parse_revision_request_tool,
    revise_draft_tool,
    rollback_draft_tool,
)
from src.agents.tools.formatting_tool import format_document_tool
from src.agents.tools.quality_check_tool import validate_quality_tool


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
        selected_papers: Optional[List[Dict[str, Any]]] = None,
        analyses: Optional[List[Dict[str, Any]]] = None,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.julius = julius or JuliusAgent()
        self.progress_callback = progress_callback
        self.reference_date = reference_date
        self.selected_papers = selected_papers or []
        self.analyses = analyses or []
        self.output_dir = Path(output_dir)
        self.state = JuliusSessionState.INTAKE
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_request: Optional[SummaryRequest] = None
        self.drafts: List[Dict[str, Any]] = []
        self.draft_versions: Dict[str, Dict[str, Any]] = {}
        self.user_feedback: List[Dict[str, Any]] = []
        self.progress_events: List[Dict[str, Any]] = []
        self.validation_reports: List[Dict[str, Any]] = []
        self.final_output_path: Optional[str] = None

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
        intent = classify_user_intent_tool(
            clean_message,
            self.state.value,
            llm_client=getattr(self.julius, "llm_client", None),
        )["intent"]

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
            "draft_versions": dict(self.draft_versions),
            "user_feedback": list(self.user_feedback),
            "progress_events": list(self.progress_events),
            "validation_reports": list(self.validation_reports),
            "final_output_path": self.final_output_path,
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
        draft_result = self.julius.generate_first_draft_tool(
            summary_request=self.current_request,
            selected_papers=self.selected_papers,
            analyses=self.analyses,
            previous_feedback=self.user_feedback,
        )
        self.emit_progress("Compiling the draft preview.")

        draft = draft_result["draft"]
        validation = self._validate_draft(draft)
        self._store_draft(draft)
        self.state = JuliusSessionState.AWAITING_REVIEW
        return self._build_response(
            self._message_with_warnings("First draft is ready for review.", validation),
            actions_taken=[
                "generated_first_draft",
                "coordinated_specialist_handoffs",
                "validated_draft",
            ],
            draft_preview=draft["content"],
        )

    def _revise_request_or_draft(self, message: str) -> JuliusSessionResponse:
        """Apply feedback to the request and refresh the draft preview if one exists."""
        if message.lower().startswith("rollback"):
            return self._rollback_draft(message)

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
            revision = parse_revision_request_tool(message, self.drafts[-1])
            self.emit_progress("Applying the requested revision.")
            if revision["requires_agent_review"]:
                self.emit_progress("Requesting specialist review for the revision.")
                draft_result = self.julius.generate_first_draft_tool(
                    summary_request=self.current_request,
                    selected_papers=self.selected_papers,
                    analyses=self.analyses,
                    previous_feedback=self.user_feedback,
                    draft_version=len(self.drafts) + 1,
                )
                draft = draft_result["draft"]
            else:
                draft = self.drafts[-1]

            revised = revise_draft_tool(
                draft=draft,
                revision_request=revision,
                summary_request=self.current_request,
                draft_version=len(self.drafts) + 1,
            )
            validation = self._validate_draft(revised)
            self._store_draft(revised)
            self.state = JuliusSessionState.AWAITING_REVIEW
            return self._build_response(
                self._message_with_warnings("I revised the draft.", validation),
                actions_taken=[
                    "recorded_feedback",
                    "updated_summary_request",
                    "parsed_revision_request",
                    "revised_draft",
                    "validated_draft",
                ],
                draft_preview=revised["content"],
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
        final_draft = mark_draft_final_tool(self.drafts[-1], approved=True)
        formatted = self._format_final_document(final_draft)
        validation = self._validate_document(formatted["document"], final_draft)
        output_path = self._save_final_document(formatted)
        self.drafts[-1] = final_draft
        self.draft_versions[f"draft_v{final_draft['version']}"] = final_draft
        self.final_output_path = str(output_path)
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
            self._message_with_warnings(
                f"Finalized and saved the current draft to {output_path}.",
                validation,
            ),
            actions_taken=["finalized_draft", action, "formatted_document", "validated_final_document", "saved_final_document"],
            draft_preview=formatted["document"] if not formatted["is_binary"] else final_draft["content"],
        )

    def _rollback_draft(self, message: str) -> JuliusSessionResponse:
        """Restore a prior draft version by message such as 'rollback to v1'."""
        if not self.drafts:
            return self._build_response(
                "There is no draft history to roll back.",
                actions_taken=["rollback_failed_no_drafts"],
            )
        version = 1
        for token in message.replace("v", " ").split():
            if token.isdigit():
                version = int(token)
                break
        restored = rollback_draft_tool(self.drafts, version)
        self._store_draft(restored)
        self.state = JuliusSessionState.AWAITING_REVIEW
        return self._build_response(
            f"Restored draft v{version}.",
            actions_taken=["rolled_back_draft"],
            draft_preview=restored["content"],
        )

    def _store_draft(self, draft: Dict[str, Any]) -> None:
        """Store draft history under both list and stable version key."""
        self.drafts.append(draft)
        self.draft_versions[f"draft_v{draft['version']}"] = draft

    def _format_final_document(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final draft for the requested delivery mode."""
        output_format = "markdown"
        if self.current_request and self.current_request.delivery.mode == DeliveryMode.EMAIL:
            output_format = "html"
        return format_document_tool(draft, output_format=output_format)

    def _validate_draft(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a draft before showing it to the user."""
        return self._validate_document(draft.get("content", ""), draft)

    def _validate_document(self, document: Any, draft: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a document and store the report."""
        report = validate_quality_tool(
            document=document,
            summary_request=draft.get("summary_request") or self.current_request,
            source_papers=self.selected_papers,
        )
        self.validation_reports.append(report)
        return report

    def _save_final_document(self, formatted: Dict[str, Any]) -> Path:
        """Save the final formatted document under outputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        extension = {"markdown": "md", "html": "html", "pdf": "pdf"}.get(
            formatted["output_format"],
            "md",
        )
        path = self.output_dir / f"julius_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.{extension}"
        document = formatted["document"]
        if formatted["is_binary"]:
            path.write_bytes(document)
        else:
            path.write_text(str(document), encoding="utf-8")
        return path

    def _message_with_warnings(self, message: str, validation: Dict[str, Any]) -> str:
        """Append concise validation warnings to a user-facing message."""
        warnings = validation.get("warnings", [])
        if not warnings:
            return message
        return f"{message} Warnings: {'; '.join(warnings[:2])}"

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


def classify_user_intent_tool(
    message: str,
    session_state: str = "INTAKE",
    llm_client: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Route a user message to the next session handler.

    The classifier is keyword-first for deterministic tests and local CLI use.
    If keywords are inconclusive, it can fall back to an OpenAI-compatible LLM.
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
        for keyword in ("shorter", "longer", "revise", "rewrite", "simplify", "technical", "intuitive", "remove", "add", "focus", "rollback")
    ):
        return {"intent": JuliusIntent.REVISION.value}
    if any(keyword in lowered for keyword in ("only ", "exclude", "remove ", "include ", "last ", "past ", "cs.", "math.", "stat.")):
        return {"intent": JuliusIntent.SCOPE_UPDATE.value}
    if any(keyword in lowered for keyword in ("shorter", "brief", "deep", "technical", "non-technical", "intuitive", "audience", "tone", "format", "bullet")):
        return {"intent": JuliusIntent.PREFERENCE_UPDATE.value}
    if any(keyword in lowered for keyword in ("summary", "summarize", "digest", "papers", "research", "one-pager", "one pager")):
        return {"intent": JuliusIntent.NEW_SUMMARY_REQUEST.value}
    llm_intent = _classify_user_intent_with_llm(
        message=message,
        session_state=state,
        llm_client=llm_client,
    )
    return {"intent": llm_intent}


def _classify_user_intent_with_llm(
    message: str,
    session_state: str,
    llm_client: Optional[Any] = None,
) -> str:
    """Use an OpenAI-compatible LLM when keyword routing returns UNKNOWN."""
    client = llm_client
    if client is None:
        client = _create_openai_client_for_intent()
    if client is None:
        return JuliusIntent.UNKNOWN.value

    model = _intent_classifier_model()
    valid_intents = ", ".join(intent.value for intent in JuliusIntent)
    prompt = (
        "Classify the user message into one Julius intent.\n"
        f"Session state: {session_state}\n"
        f"Valid intents: {valid_intents}\n"
        "Return strict JSON only: {\"intent\":\"<INTENT>\"}.\n"
        "If uncertain, return {\"intent\":\"UNKNOWN\"}.\n"
        f"User message: {message}"
    )
    response = _call_intent_llm(client, prompt, model)
    intent = _extract_intent_from_llm_response(response)
    if intent in {candidate.value for candidate in JuliusIntent}:
        return intent
    return JuliusIntent.UNKNOWN.value


def _call_intent_llm(client: Any, prompt: str, model: str) -> Any:
    """Call common LLM client shapes for intent classification."""
    if callable(client):
        return client(prompt)

    responses_api = getattr(client, "responses", None)
    if responses_api is not None:
        create_method = getattr(responses_api, "create", None)
        if callable(create_method):
            return create_method(
                model=model,
                input=prompt,
                temperature=0,
            )

    chat_api = getattr(client, "chat", None)
    if chat_api is not None:
        completions_api = getattr(chat_api, "completions", None)
        create_method = getattr(completions_api, "create", None)
        if callable(create_method):
            return create_method(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

    for method_name in ("complete", "generate", "chat", "invoke"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                return method(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt="Return only JSON with an intent field.",
                )
            except TypeError:
                return method(prompt)
    return None


def _extract_intent_from_llm_response(response: Any) -> str:
    """Extract the classified intent from common response payload shapes."""
    if response is None:
        return JuliusIntent.UNKNOWN.value
    if isinstance(response, dict):
        if isinstance(response.get("intent"), str):
            return response["intent"].strip().upper()
        text_candidate = response.get("output_text") or response.get("content") or response.get("text")
        if isinstance(text_candidate, str):
            return _intent_from_text_payload(text_candidate)
    if hasattr(response, "output_text") and isinstance(getattr(response, "output_text"), str):
        return _intent_from_text_payload(response.output_text)
    if hasattr(response, "choices"):
        choices = getattr(response, "choices", []) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if isinstance(content, str):
                return _intent_from_text_payload(content)
    if isinstance(response, str):
        return _intent_from_text_payload(response)
    return JuliusIntent.UNKNOWN.value


def _intent_from_text_payload(text: str) -> str:
    """Parse an intent from a text payload."""
    candidate = text.strip()
    if not candidate:
        return JuliusIntent.UNKNOWN.value
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict) and isinstance(parsed.get("intent"), str):
            return parsed["intent"].strip().upper()
    except Exception:
        pass
    normalized = candidate.strip().strip('"').upper()
    if normalized in {intent.value for intent in JuliusIntent}:
        return normalized
    return JuliusIntent.UNKNOWN.value


def _create_openai_client_for_intent() -> Optional[Any]:
    """Create a default OpenAI client for intent classification if configured."""
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or _settings_openai_api_key_for_intent()
        or os.getenv("LLM_API_KEY")
    )
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _settings_openai_api_key_for_intent() -> str:
    """Read OpenAI API key from settings when available."""
    try:
        from config.settings import Settings
    except Exception:
        return ""
    try:
        settings = Settings()
    except Exception:
        return ""
    return settings.openai_api_key or settings.llm_api_key


def _intent_classifier_model() -> str:
    """Resolve the OpenAI model used for intent fallback."""
    model = os.getenv("OPENAI_MODEL")
    if model:
        return model
    try:
        from config.settings import Settings
        settings = Settings()
        if settings.llm_model:
            return settings.llm_model
    except Exception:
        pass
    return "gpt-4o-mini"


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
