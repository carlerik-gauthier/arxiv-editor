"""Backend workflow for the Streamlit Julius app.

The workflow keeps UI code thin: it owns the JuliusSession, explicit action
methods, resumable state, satisfaction signals, and recoverable error messages.
Injected callables let tests simulate data collection or failures without
touching the Streamlit layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.agents.julius_session import JuliusSession
from src.generation.user_request import SummaryRequest


@dataclass
class WorkflowResult:
    """Structured result returned to the Streamlit adapter."""

    message: str
    state: str
    actions_taken: List[str] = field(default_factory=list)
    summary_request: Optional[Dict[str, Any]] = None
    draft_preview: Optional[str] = None
    next_questions: List[str] = field(default_factory=list)
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly result for tests and UI state."""
        return {
            "message": self.message,
            "state": self.state,
            "actions_taken": self.actions_taken,
            "summary_request": self.summary_request,
            "draft_preview": self.draft_preview,
            "next_questions": self.next_questions,
            "recoverable": self.recoverable,
        }


class InteractiveSummaryWorkflow:
    """
    Orchestrate intake, generation, revision, validation, and final output.

    Expensive operations are only invoked through explicit methods such as
    `generate_draft`, `revise_draft`, `validate_current_draft`, and `finalize`.
    """

    def __init__(
        self,
        julius_session: Optional[JuliusSession] = None,
        fetch_papers_fn: Optional[Callable[[SummaryRequest], List[Dict[str, Any]]]] = None,
        analyze_papers_fn: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.session = julius_session or JuliusSession()
        self.fetch_papers_fn = fetch_papers_fn
        self.analyze_papers_fn = analyze_papers_fn
        self.satisfaction_signals: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None

    def handle_message(self, message: str) -> Dict[str, Any]:
        """Handle normal chat intake, preference updates, questions, or finalization."""
        return self._safe_call(lambda: self.session.handle_user_message(message))

    def apply_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sidebar preferences into a normal Julius request update."""
        message = _preference_message(preferences)
        return self.handle_message(message)

    def generate_draft(self) -> Dict[str, Any]:
        """Run optional data hooks, then ask Julius to generate the first draft."""
        def operation() -> Dict[str, Any]:
            self._refresh_data_if_configured()
            return self.session.handle_user_message("generate draft")

        return self._safe_call(operation)

    def revise_draft(self, feedback: str) -> Dict[str, Any]:
        """Apply user feedback and track dissatisfaction/revision signals."""
        self.satisfaction_signals.append({"type": "revision_requested", "feedback": feedback})
        return self._safe_call(lambda: self.session.handle_user_message(feedback))

    def validate_current_draft(self) -> Dict[str, Any]:
        """Validate the latest draft without changing the draft content."""
        def operation() -> Dict[str, Any]:
            if not self.session.drafts:
                return WorkflowResult(
                    message="No draft is available to validate.",
                    state=self.session.state.value,
                    actions_taken=["validation_skipped_no_draft"],
                    recoverable=True,
                ).to_dict()
            report = self.session._validate_draft(self.session.drafts[-1])
            return WorkflowResult(
                message="Validation complete.",
                state=self.session.state.value,
                actions_taken=["validated_current_draft"],
                summary_request=self._current_request_dict(),
                draft_preview=self.session.drafts[-1].get("content"),
                next_questions=[],
                recoverable=False,
            ).to_dict() | {"validation_report": report}

        return self._safe_call(operation)

    def finalize(self) -> Dict[str, Any]:
        """Finalize only when the user explicitly approves."""
        self.satisfaction_signals.append({"type": "accepted_draft"})
        return self._safe_call(lambda: self.session.handle_user_message("finalize this"))

    def retry_last_action(self) -> Dict[str, Any]:
        """Return a retry hint after recoverable failures."""
        return WorkflowResult(
            message="Retry by narrowing the scope, broadening the date range, or using cached partial results.",
            state=self.session.state.value,
            actions_taken=["offered_retry_options"],
            recoverable=True,
        ).to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the workflow so Streamlit reruns can reuse state."""
        return {
            "session": self.session.to_dict(),
            "satisfaction_signals": list(self.satisfaction_signals),
            "last_error": self.last_error,
        }

    def _safe_call(self, operation: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Convert backend failures into recoverable Julius messages."""
        try:
            result = operation()
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = str(exc)
            return WorkflowResult(
                message=(
                    "I could not complete that step. You can retry, narrow the scope, "
                    "broaden categories, or continue with available partial results."
                ),
                state=self.session.state.value,
                actions_taken=["handled_recoverable_workflow_error"],
                summary_request=self._current_request_dict(),
                draft_preview=self.session.drafts[-1].get("content") if self.session.drafts else None,
                next_questions=["Retry with a narrower scope or use the current partial draft?"],
                recoverable=True,
            ).to_dict()

    def _refresh_data_if_configured(self) -> None:
        """Run injected data hooks before draft generation."""
        if not self.session.current_request:
            return
        request = self.session.current_request
        if self.fetch_papers_fn:
            papers = self.fetch_papers_fn(request)
            if not papers:
                raise ValueError("No papers were returned for the current scope.")
            self.session.selected_papers = papers
        if self.analyze_papers_fn and self.session.selected_papers:
            self.session.analyses = self.analyze_papers_fn(self.session.selected_papers)

    def _current_request_dict(self) -> Optional[Dict[str, Any]]:
        """Return the current request as a dict when present."""
        if not self.session.current_request:
            return None
        return self.session.current_request.model_dump(mode="json")


def _preference_message(preferences: Dict[str, Any]) -> str:
    """Build a compact Julius-readable message from sidebar controls."""
    topic = preferences.get("topic_query") or "recent research"
    parts = [
        f"summary of \"{topic}\"",
        f"for {preferences.get('audience', 'mixed')} audience",
        f"with {preferences.get('depth', 'standard')} depth",
        f"in {preferences.get('tone', 'editorial')} tone",
        f"as {preferences.get('format', 'one_pager')}",
    ]
    if preferences.get("date_range"):
        parts.append(str(preferences["date_range"]))
    if preferences.get("must_include_categories"):
        parts.append("include " + ", ".join(preferences["must_include_categories"]))
    if preferences.get("exclude_categories"):
        parts.append("exclude " + ", ".join(preferences["exclude_categories"]))
    if preferences.get("max_topics"):
        parts.append(f"at most {preferences['max_topics']} topics")
    if preferences.get("max_papers"):
        parts.append(f"at most {preferences['max_papers']} papers")
    return " ".join(parts)
