"""Deterministic quality checks for formatted Julius documents."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.generation.user_request import Audience, SummaryFormat, SummaryRequest


class DocumentValidator:
    """Validate completeness, metadata, citations, and audience fit."""

    def validate(
        self,
        document: Any,
        summary_request: Any,
        source_papers: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run deterministic checks before a document is saved or delivered."""
        request = _coerce_request(summary_request)
        text = _document_text(document)
        papers = list(source_papers or [])
        warnings: List[str] = []
        missing: List[str] = []

        if not text.strip():
            missing.append("document")
        if request.topic_query and request.topic_query.lower() not in text.lower():
            warnings.append("Requested topic is not visible in the document.")
        if "Representative Papers" not in text and request.format != SummaryFormat.BULLET_DIGEST:
            warnings.append("Representative paper section is missing.")
        if papers and not _all_papers_cited(text, papers):
            warnings.append("One or more source papers are not cited by title or ArXiv id.")
        if request.audience == Audience.NON_EXPERT and any(word in text.lower() for word in ("lemma", "theorem")):
            warnings.append("Non-expert draft may contain unexplained technical terms.")
        if request.audience == Audience.EXPERT and len(text.split()) < 80:
            warnings.append("Expert draft may be too short.")
        if request.format == SummaryFormat.ONE_PAGER and len(text.split()) > 1200:
            warnings.append("One-pager is likely too long.")

        return {
            "passed": not missing,
            "warnings": warnings,
            "missing": missing,
            "word_count": len(text.split()),
            "source_paper_count": len(papers),
            "audience": request.audience.value,
            "format": request.format.value,
        }

    def generate_improvement_suggestions(self, validation_report: Dict[str, Any]) -> List[str]:
        """Convert validation warnings into concise improvement suggestions."""
        suggestions = []
        for warning in validation_report.get("warnings", []):
            suggestions.append(f"Address: {warning}")
        for missing in validation_report.get("missing", []):
            suggestions.append(f"Add missing {missing}.")
        return suggestions or ["No deterministic improvements required."]


def _coerce_request(value: Any) -> SummaryRequest:
    """Normalize request input."""
    if isinstance(value, SummaryRequest):
        return value
    if isinstance(value, dict):
        return SummaryRequest.model_validate(value)
    return SummaryRequest()


def _document_text(document: Any) -> str:
    """Convert rendered document payloads to text for validation."""
    if isinstance(document, bytes):
        return document.decode("utf-8", errors="ignore")
    if isinstance(document, dict) and "document" in document:
        return _document_text(document["document"])
    return str(document or "")


def _all_papers_cited(text: str, papers: List[Dict[str, Any]]) -> bool:
    """Check whether paper titles or ArXiv ids appear in the document."""
    lowered = text.lower()
    for paper in papers:
        title = str(paper.get("title", "")).lower()
        arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").lower()
        if title and title in lowered:
            continue
        if arxiv_id and arxiv_id in lowered:
            continue
        return False
    return True
