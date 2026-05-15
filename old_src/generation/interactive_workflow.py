"""Backend workflow for the Streamlit Julius app.

The workflow keeps UI code thin: it owns the JuliusSession, explicit action
methods, resumable state, satisfaction signals, and recoverable error messages.
Injected callables let tests simulate data collection or failures without
touching the Streamlit layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
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
        """Ensure enough papers are available before draft generation."""
        if not self.session.current_request:
            return
        request = self.session.current_request
        self._ensure_papers_available(request)
        if self.analyze_papers_fn and self.session.selected_papers:
            self.session.analyses = self.analyze_papers_fn(self.session.selected_papers)

    def _ensure_papers_available(self, request: SummaryRequest) -> None:
        """Ask selected agents to check paper availability and fetch when needed."""
        agent_names = self._selected_agent_names(request)
        required_count = 1
        available_count = self._matching_paper_count(request)
        threshold_met = self._check_agent_thresholds(
            agent_names=agent_names,
            paper_count=available_count,
            min_threshold=required_count,
        )
        if threshold_met:
            return

        papers = self._fetch_papers(request, agent_names, required_count)
        if not papers:
            raise ValueError("No papers were returned for the current scope.")
        self.session.selected_papers = papers

    def _check_agent_thresholds(
        self,
        agent_names: List[str],
        paper_count: int,
        min_threshold: int,
    ) -> bool:
        """Record data availability checks through the selected specialist agents."""
        if not agent_names:
            return paper_count >= min_threshold

        threshold_met = False
        for agent_name in agent_names:
            agent = self.session.julius.specialist_agents[agent_name]
            result = agent.execute_tool(
                "check_threshold_tool",
                {"paper_count": paper_count, "min_threshold": min_threshold},
            )
            if result.success and result.result.get("threshold_met"):
                threshold_met = True
        return threshold_met

    def _fetch_papers(
        self,
        request: SummaryRequest,
        agent_names: List[str],
        min_count: int,
    ) -> List[Dict[str, Any]]:
        """Fetch papers with an injected fetcher or the selected agents' fetch tool."""
        if self.fetch_papers_fn:
            return self.fetch_papers_fn(request)

        start_date, end_date = self._date_bounds(request)
        fetched: List[Dict[str, Any]] = []
        seen_ids = set()
        for agent_name in agent_names:
            agent = self.session.julius.specialist_agents[agent_name]
            categories = self._categories_for_agent(request, agent_name)
            result = agent.execute_tool(
                "fetch_papers_tool",
                {
                    "categories": categories,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "max_results": max(request.max_papers, min_count),
                    "min_count": min_count,
                },
            )
            if not result.success:
                raise ValueError(result.error or f"{agent_name} could not fetch papers.")
            for paper in result.result.get("papers", []):
                paper_id = paper.get("arxiv_id") or paper.get("id") or paper.get("title")
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                fetched.append(paper)
        return fetched

    def _selected_agent_names(self, request: SummaryRequest) -> List[str]:
        """Use Julius's request routing to pick agents responsible for data checks."""
        return self.session.julius._select_agents_for_summary_request(request)

    def _categories_for_agent(self, request: SummaryRequest, agent_name: str) -> List[str]:
        """Prefer explicit category filters, otherwise use the specialist's categories."""
        agent_categories = self.session.julius.specialist_agents[agent_name].categories
        if request.must_include_categories:
            matching = [
                category
                for category in request.must_include_categories
                if category in agent_categories
            ]
            return matching or list(request.must_include_categories)
        return list(agent_categories)

    def _matching_paper_count(self, request: SummaryRequest) -> int:
        """Count available papers that satisfy explicit category filters."""
        if not request.must_include_categories:
            return len(self.session.selected_papers)
        included = set(request.must_include_categories)
        count = 0
        for paper in self.session.selected_papers:
            paper_categories = set(paper.get("categories") or [])
            if included.intersection(paper_categories):
                count += 1
        return count

    def _date_bounds(self, request: SummaryRequest) -> tuple[date, date]:
        """Resolve concrete dates for ArXiv fetching."""
        end = request.date_range.end_date or self._reference_date()
        start = request.date_range.start_date or (end - timedelta(days=7))
        return start, end

    def _reference_date(self) -> date:
        """Return the workflow's deterministic reference date when configured."""
        value = self.session.reference_date
        if value is None:
            return date.today()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value[:10])
        if hasattr(value, "date"):
            return value.date()
        return date.today()

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
