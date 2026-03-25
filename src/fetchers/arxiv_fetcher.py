"""
ArXiv API integration module for fetching papers by category and date.

This module provides the ArxivFetcher class for interacting with the ArXiv API
to fetch research papers with proper rate limiting and error handling.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import arxiv

logger = logging.getLogger(__name__)


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

    def __post_init__(self):
        """Validate paper data after initialization."""
        if not self.arxiv_id:
            raise ValueError("arxiv_id cannot be empty")
        if not self.title:
            raise ValueError("title cannot be empty")

    def to_dict(self) -> dict:
        """Convert paper to dictionary representation."""
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


class ArxivFetcherError(Exception):
    """Base exception for ArxivFetcher errors."""

    pass


class RateLimitError(ArxivFetcherError):
    """Raised when rate limit is hit."""

    pass


class ArxivFetcher:
    """
    Fetches papers from ArXiv by category and date range.

    Handles API rate limiting and provides graceful error handling.

    Attributes:
        request_delay: Delay between API requests in seconds.
        max_retries: Maximum number of retry attempts for failed requests.
    """

    def __init__(
        self,
        request_delay: float = 3.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize the ArxivFetcher.

        Args:
            request_delay: Delay between consecutive API requests (seconds).
                          ArXiv recommends at least 3 seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Delay between retry attempts (seconds).
        """
        self.request_delay = max(request_delay, 3.0)  # Minimum 3 seconds per ArXiv guidelines
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_request_time: Optional[float] = None

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_delay:
                sleep_time = self.request_delay - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        self._last_request_time = time.time()

    def parse_paper_metadata(self, result: arxiv.Result) -> Paper:
        """
        Parse an ArXiv API result into a Paper object.

        Args:
            result: An arxiv.Result object from the API.

        Returns:
            A Paper object with extracted metadata.

        Raises:
            ArxivFetcherError: If parsing fails.
        """
        try:
            # Extract arxiv_id from entry_id (format: http://arxiv.org/abs/XXXX.XXXXX)
            arxiv_id = result.entry_id.split("/abs/")[-1]

            # Extract author names
            authors = [author.name for author in result.authors]

            return Paper(
                arxiv_id=arxiv_id,
                title=result.title.replace("\n", " ").strip(),
                authors=authors,
                summary=result.summary.replace("\n", " ").strip(),
                published=result.published,
                updated=result.updated,
                categories=list(result.categories),
                primary_category=result.primary_category,
                pdf_url=result.pdf_url,
                entry_id=result.entry_id,
                comment=result.comment,
                journal_ref=result.journal_ref,
                doi=result.doi,
            )
        except Exception as e:
            logger.error(f"Failed to parse paper metadata: {e}")
            raise ArxivFetcherError(f"Failed to parse paper metadata: {e}") from e

    def fetch_by_category(
        self,
        category: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[Paper]:
        """
        Fetch papers from a specific ArXiv category within a date range.

        Args:
            category: ArXiv category code (e.g., 'math.PR', 'cs.LG').
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive). Defaults to now.
            max_results: Maximum number of papers to fetch.

        Returns:
            List of Paper objects matching the criteria.

        Raises:
            ArxivFetcherError: If the fetch operation fails after retries.
        """
        if end_date is None:
            end_date = datetime.now()

        # Ensure dates are timezone-naive for comparison
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)

        logger.info(
            f"Fetching papers from category '{category}' "
            f"between {start_date.date()} and {end_date.date()}, "
            f"max_results={max_results}"
        )

        # Build the search query
        # ArXiv query syntax: cat:category AND submittedDate:[start TO end]
        date_format = "%Y%m%d%H%M%S"
        query = (
            f"cat:{category} AND "
            f"submittedDate:[{start_date.strftime(date_format)} TO {end_date.strftime(date_format)}]"
        )

        papers: List[Paper] = []
        retries = 0

        while retries <= self.max_retries:
            try:
                self._wait_for_rate_limit()

                # Create search client with sort by submission date
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                # Create client with custom page size for efficiency
                client = arxiv.Client(
                    page_size=100,
                    delay_seconds=self.request_delay,
                    num_retries=self.max_retries,
                )

                # Fetch results
                for result in client.results(search):
                    try:
                        paper = self.parse_paper_metadata(result)

                        # Additional date filtering (ArXiv API date filtering can be imprecise)
                        paper_date = paper.published.replace(tzinfo=None)
                        if start_date <= paper_date <= end_date:
                            papers.append(paper)
                    except ArxivFetcherError as e:
                        logger.warning(f"Skipping paper due to parse error: {e}")
                        continue

                logger.info(f"Successfully fetched {len(papers)} papers from category '{category}'")
                return papers

            except arxiv.UnexpectedEmptyPageError as e:
                logger.warning(f"Unexpected empty page from ArXiv API: {e}")
                # This often indicates we've reached the end of results
                break

            except arxiv.HTTPError as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Max retries exceeded for category '{category}': {e}")
                    raise ArxivFetcherError(
                        f"Failed to fetch papers after {self.max_retries} retries: {e}"
                    ) from e

                logger.warning(
                    f"HTTP error fetching category '{category}' (attempt {retries}/{self.max_retries}): {e}"
                )
                time.sleep(self.retry_delay * retries)  # Exponential backoff

            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Unexpected error fetching category '{category}': {e}")
                    raise ArxivFetcherError(f"Unexpected error: {e}") from e

                logger.warning(
                    f"Error fetching category '{category}' (attempt {retries}/{self.max_retries}): {e}"
                )
                time.sleep(self.retry_delay * retries)

        return papers

    def fetch_multiple_categories(
        self,
        categories: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_results_per_category: int = 100,
    ) -> List[Paper]:
        """
        Fetch papers from multiple ArXiv categories.

        Args:
            categories: List of ArXiv category codes.
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive). Defaults to now.
            max_results_per_category: Maximum papers to fetch per category.

        Returns:
            List of unique Paper objects across all categories.
        """
        all_papers: List[Paper] = []
        seen_ids: set = set()

        for category in categories:
            try:
                papers = self.fetch_by_category(
                    category=category,
                    start_date=start_date,
                    end_date=end_date,
                    max_results=max_results_per_category,
                )

                # Deduplicate papers (some papers appear in multiple categories)
                for paper in papers:
                    if paper.arxiv_id not in seen_ids:
                        seen_ids.add(paper.arxiv_id)
                        all_papers.append(paper)

            except ArxivFetcherError as e:
                logger.error(f"Failed to fetch category '{category}': {e}")
                # Continue with other categories

        logger.info(
            f"Fetched {len(all_papers)} unique papers from {len(categories)} categories"
        )
        return all_papers

    def fetch_with_threshold(
        self,
        categories: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        min_count: int = 100,
        max_results_per_category: int = 200,
        expansion_days: int = 7,
        max_expansions: int = 4,
    ) -> tuple[List[Paper], datetime, datetime]:
        """
        Fetch papers ensuring at least min_count papers are retrieved.

        Automatically expands the date range backwards if the threshold is not met.
        This ensures sufficient data for topic modeling and analysis.

        Args:
            categories: List of ArXiv category codes to fetch from.
            start_date: Initial start of the date range (inclusive).
            end_date: End of the date range (inclusive). Defaults to now.
            min_count: Minimum number of papers required.
            max_results_per_category: Maximum papers to fetch per category per attempt.
            expansion_days: Number of days to expand backwards on each iteration.
            max_expansions: Maximum number of date range expansions to attempt.

        Returns:
            A tuple containing:
                - List of unique Paper objects
                - Actual start_date used (may differ from input if expanded)
                - Actual end_date used

        Raises:
            ArxivFetcherError: If unable to meet threshold after max_expansions.
        """
        if end_date is None:
            end_date = datetime.now()

        # Ensure dates are timezone-naive for consistency
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)

        original_start_date = start_date
        current_start_date = start_date
        expansions = 0

        logger.info(
            f"Fetching papers with threshold of {min_count} from {len(categories)} categories. "
            f"Initial date range: {start_date.date()} to {end_date.date()}"
        )

        while expansions <= max_expansions:
            # Fetch papers with current date range
            papers = self.fetch_multiple_categories(
                categories=categories,
                start_date=current_start_date,
                end_date=end_date,
                max_results_per_category=max_results_per_category,
            )

            paper_count = len(papers)

            logger.info(
                f"Fetched {paper_count} papers with date range "
                f"{current_start_date.date()} to {end_date.date()}"
            )

            # Check if threshold is met
            if paper_count >= min_count:
                if expansions > 0:
                    logger.info(
                        f"Threshold met after {expansions} expansion(s). "
                        f"Expanded date range by {(original_start_date - current_start_date).days} days."
                    )
                else:
                    logger.info(
                        f"Threshold met with original date range. "
                        f"Fetched {paper_count} papers (minimum: {min_count})."
                    )

                return papers, current_start_date, end_date

            # Threshold not met - expand date range backwards
            if expansions >= max_expansions:
                logger.error(
                    f"Failed to meet threshold of {min_count} papers after {max_expansions} expansions. "
                    f"Only fetched {paper_count} papers."
                )
                raise ArxivFetcherError(
                    f"Unable to fetch minimum {min_count} papers. "
                    f"Only found {paper_count} papers after expanding date range "
                    f"{max_expansions} times ({current_start_date.date()} to {end_date.date()})."
                )

            expansions += 1
            days_to_expand = expansion_days * expansions  # Increase expansion on each iteration
            current_start_date = original_start_date - timedelta(days=days_to_expand)

            logger.info(
                f"Threshold not met ({paper_count}/{min_count}). "
                f"Expanding date range backwards by {days_to_expand} days "
                f"(expansion {expansions}/{max_expansions}). "
                f"New start date: {current_start_date.date()}"
            )

        # This should not be reached due to the check above, but included for safety
        raise ArxivFetcherError(
            f"Unexpected error: exceeded max expansions without meeting threshold"
        )
