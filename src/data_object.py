"""Domain models used across the arXiv research workflow."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

@dataclass
class Paper:
    """Data model for an ArXiv paper."""

    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    primary_category: str
    pdf_url: str
    entry_id: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate required identifiers after dataclass initialization.

        Returns:
            None: This method validates the instance in place.

        Raises:
            ValueError: If ``arxiv_id`` or ``title`` is empty.
        """
        if not self.arxiv_id:
            raise ValueError("arxiv_id cannot be empty")
        if not self.title:
            raise ValueError("title cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the paper's metadata into JSON-compatible values.

        Returns:
            dict[str, Any]: Paper fields with datetime values in ISO 8601 form.
        """
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "categories": self.categories,
            "primary_category": self.primary_category,
            "pdf_url": self.pdf_url,
            "entry_id": self.entry_id,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }
