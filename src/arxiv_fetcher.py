"""
ArXiv API integration module for fetching papers by category and date.

This module provides the ArxivFetcher class for interacting with the ArXiv API
to fetch research papers with proper rate limiting and error handling.
"""

import logging
import re
import time
from io import BytesIO
import tarfile
from datetime import datetime, timedelta
import gzip
from pathlib import Path
from typing import List, Optional

import arxiv
import requests
from pypdf import PdfReader
from src.data_object import Paper

logger = logging.getLogger(__name__)


class ArxivFetcherError(Exception):
    """Base exception for ArxivFetcher errors."""

    pass


class RateLimitError(ArxivFetcherError):
    """Raised when rate limit is hit."""

    pass


class PDFDownloadError(ArxivFetcherError):
    """Raised when PDF download fails."""

    pass


class PDFExtractionError(ArxivFetcherError):
    """Raised when PDF text extraction fails."""

    pass


class SourceDownloadError(ArxivFetcherError):
    """Raised when LaTeX source download fails."""

    pass


class SourceExtractionError(ArxivFetcherError):
    """Raised when LaTeX source extraction fails."""

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
    ) -> None:
        """
        Initialize the ArxivFetcher.

        Args:
            request_delay: Delay between consecutive API requests (seconds).
                          ArXiv recommends at least 3 seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Delay between retry attempts (seconds).

        Returns:
            None: The new fetcher stores normalized rate-limit settings.
        """
        self.request_delay = max(request_delay, 3.0)  # Minimum 3 seconds per ArXiv guidelines
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_request_time: Optional[float] = None

    def _wait_for_rate_limit(self) -> None:
        """Pause as needed to keep requests within the configured rate limit.

        Returns:
            None: Updates the timestamp used to throttle the next request.
        """
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
        max_results: int = 1000,
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
        max_results_per_category: int = 1000,
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

        Raises:
            Exception: If an unexpected error escapes a category fetch.
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
        max_results_per_category: int = 1000,
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

    def download_paper_pdf(
        self,
        paper_id: str,
        output_dir: str = "data/pdfs",
        force_redownload: bool = False,
    ) -> Path:
        """
        Download a paper's PDF file from ArXiv.

        Args:
            paper_id: The ArXiv ID of the paper (e.g., "2301.12345").
            output_dir: Directory to save the PDF. Defaults to "data/pdfs".
            force_redownload: If True, redownload even if file exists.

        Returns:
            Path to the downloaded PDF file.

        Raises:
            PDFDownloadError: If the download fails after retries.
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sanitize paper_id for use as filename (replace '/' with '_')
        safe_paper_id = paper_id.replace("/", "_")
        pdf_file_path = output_path / f"{safe_paper_id}.pdf"

        # Check if file already exists and caching is enabled
        if pdf_file_path.exists() and not force_redownload:
            logger.info(f"PDF already cached for paper {paper_id}: {pdf_file_path}")
            return pdf_file_path

        # Construct PDF URL
        # ArXiv PDF URLs: https://arxiv.org/pdf/{paper_id}.pdf
        # https://arxiv.org/pdf/2603.23460v1.pdf
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        logger.info(f"Downloading PDF for paper {paper_id} from {pdf_url}")

        retries = 0
        while retries <= self.max_retries:
            try:
                # Respect rate limiting
                self._wait_for_rate_limit()

                # Download the PDF
                response = requests.get(
                    pdf_url,
                    timeout=60,  # 60 second timeout
                    stream=True,  # Stream to handle large files
                    headers={"User-Agent": "ArxivFetcher/1.0 (Research Tool)"},
                )

                # Check if request was successful
                response.raise_for_status()

                # Write PDF to file
                with open(pdf_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Successfully downloaded PDF for paper {paper_id} to {pdf_file_path}")
                return pdf_file_path

            except requests.exceptions.HTTPError as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Max retries exceeded downloading PDF for paper {paper_id}: {e}")
                    raise PDFDownloadError(
                        f"Failed to download PDF for paper {paper_id} after {self.max_retries} retries: {e}"
                    ) from e

                logger.warning(
                    f"HTTP error downloading PDF for paper {paper_id} "
                    f"(attempt {retries}/{self.max_retries}): {e}"
                )
                time.sleep(self.retry_delay * retries)

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Network error downloading PDF for paper {paper_id}: {e}")
                    raise PDFDownloadError(
                        f"Network error downloading PDF for paper {paper_id}: {e}"
                    ) from e

                logger.warning(
                    f"Request error downloading PDF for paper {paper_id} "
                    f"(attempt {retries}/{self.max_retries}): {e}"
                )
                time.sleep(self.retry_delay * retries)

            except Exception as e:
                logger.error(f"Unexpected error downloading PDF for paper {paper_id}: {e}")
                raise PDFDownloadError(
                    f"Unexpected error downloading PDF for paper {paper_id}: {e}"
                ) from e

        # This should not be reached
        raise PDFDownloadError(f"Failed to download PDF for paper {paper_id}")

    def fetch_paper_source(self, paper_id: str) -> bytes:
        """
        Fetch a paper's LaTeX source archive from ArXiv without saving it to disk.

        Args:
            paper_id: The ArXiv ID of the paper (e.g., "2301.12345").

        Returns:
            Raw source archive bytes returned by ArXiv.

        Raises:
            SourceDownloadError: If the source is unavailable or the request fails.
        """
        source_url = f"https://arxiv.org/e-print/{paper_id}"
        logger.info(f"Fetching LaTeX source for paper {paper_id} from {source_url}")

        retries = 0
        while retries <= self.max_retries:
            try:
                self._wait_for_rate_limit()

                response = requests.get(
                    source_url,
                    timeout=60,
                    headers={"User-Agent": "ArxivFetcher/1.0 (Research Tool)"},
                )

                if response.status_code in {403, 404, 410}:
                    raise SourceDownloadError(
                        f"LaTeX source unavailable for paper {paper_id}: HTTP {response.status_code}"
                    )

                response.raise_for_status()

                if not response.content:
                    raise SourceDownloadError(f"Empty LaTeX source response for paper {paper_id}")

                # ArXiv may return an HTML page for unavailable sources; treat that as failure.
                content_prefix = response.content[:256].lstrip().lower()
                if content_prefix.startswith(b"<!doctype html") or content_prefix.startswith(b"<html"):
                    raise SourceDownloadError(
                        f"LaTeX source unavailable for paper {paper_id}: received HTML response"
                    )

                return response.content

            except SourceDownloadError:
                raise

            except requests.exceptions.HTTPError as e:
                retries += 1
                if retries > self.max_retries:
                    raise SourceDownloadError(
                        f"Failed to fetch LaTeX source for paper {paper_id} "
                        f"after {self.max_retries} retries: {e}"
                    ) from e
                time.sleep(self.retry_delay * retries)

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > self.max_retries:
                    raise SourceDownloadError(
                        f"Network error fetching LaTeX source for paper {paper_id}: {e}"
                    ) from e
                time.sleep(self.retry_delay * retries)

            except Exception as e:
                raise SourceDownloadError(
                    f"Unexpected error fetching LaTeX source for paper {paper_id}: {e}"
                ) from e

        raise SourceDownloadError(f"Failed to fetch LaTeX source for paper {paper_id}")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text content as a string.

        Raises:
            PDFExtractionError: If text extraction fails.
        """
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting text from PDF: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

            # Combine all text
            full_text = "\n\n".join(text_parts)

            # Check if we got any text
            if not full_text.strip():
                raise PDFExtractionError(f"No text extracted from PDF: {pdf_path}")

            logger.info(
                f"Successfully extracted {len(full_text)} characters from "
                f"{len(reader.pages)} pages in {pdf_path.name}"
            )

            return full_text

        except PDFExtractionError:
            # Re-raise our custom errors
            raise

        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            raise PDFExtractionError(f"Failed to extract text from PDF {pdf_path}: {e}") from e

    def extract_markdown_from_source(self, source_bytes: bytes) -> str:
        """
        Extract markdown text from an ArXiv LaTeX source archive.

        Args:
            source_bytes: Raw bytes returned by the ArXiv e-print endpoint.

        Returns:
            Markdown text extracted from the paper's main LaTeX document.

        Raises:
            SourceExtractionError: If no parseable LaTeX document is found.
        """
        latex_documents = self._extract_latex_documents(source_bytes)
        if not latex_documents:
            raise SourceExtractionError("No LaTeX documents found in source archive")

        _, main_document = self._select_main_latex_document(latex_documents)
        markdown = self._latex_to_markdown(main_document)
        if not markdown.strip():
            raise SourceExtractionError("Extracted LaTeX source did not produce markdown text")
        return markdown

    def fetch_paper_markdown(
        self,
        paper_id: str,
        output_dir: str = "data/pdfs",
        force_redownload: bool = False,
    ) -> str:
        """
        Fetch a paper as markdown text, preferring LaTeX source over PDF extraction.

        The method first requests the ArXiv e-print source. If source retrieval or
        source parsing fails, it falls back to the existing PDF download and text
        extraction pipeline, then normalizes the result into markdown.

        Args:
            paper_id: The ArXiv ID of the paper (e.g., "2301.12345").
            output_dir: Directory to save the fallback PDF. Defaults to "data/pdfs".
            force_redownload: If True, redownload the fallback PDF even if cached.

        Returns:
            Paper content as markdown text.

        Raises:
            PDFDownloadError: If source processing fails and the fallback PDF
                cannot be downloaded.
            PDFExtractionError: If source processing fails and fallback PDF text
                cannot be extracted.
        """
        try:
            source_bytes = self.fetch_paper_source(paper_id)
            markdown = self.extract_markdown_from_source(source_bytes)
            logger.info(f"Extracted markdown from LaTeX source for paper {paper_id}")
            return markdown
        except (SourceDownloadError, SourceExtractionError) as e:
            logger.warning(
                f"Falling back to PDF extraction for paper {paper_id} because source "
                f"processing failed: {e}"
            )

        pdf_path = self.download_paper_pdf(
            paper_id=paper_id,
            output_dir=output_dir,
            force_redownload=force_redownload,
        )
        text = self.extract_text_from_pdf(pdf_path)
        return self._plain_text_to_markdown(text)

    def download_and_extract_paper(
        self,
        paper_id: str,
        output_dir: str = "data/pdfs",
        force_redownload: bool = False,
    ) -> tuple[Path, str]:
        """
        Download a paper's PDF and extract its text content.

        Convenience method that combines download_paper_pdf and extract_text_from_pdf.

        Args:
            paper_id: The ArXiv ID of the paper (e.g., "2301.12345").
            output_dir: Directory to save the PDF. Defaults to "data/pdfs".
            force_redownload: If True, redownload even if file exists.

        Returns:
            A tuple containing:
                - Path to the downloaded PDF file
                - Extracted text content as a string

        Raises:
            PDFDownloadError: If the download fails.
            PDFExtractionError: If text extraction fails.
        """
        pdf_path = self.download_paper_pdf(
            paper_id=paper_id,
            output_dir=output_dir,
            force_redownload=force_redownload,
        )

        text = self.extract_text_from_pdf(pdf_path)

        return pdf_path, text

    def _extract_latex_documents(self, source_bytes: bytes) -> dict[str, str]:
        """Extract decoded LaTeX documents from an arXiv source payload.

        Args:
            source_bytes: Raw e-print archive, gzip stream, or text source.

        Returns:
            dict[str, str]: File names mapped to decoded LaTeX document text;
            empty when no usable document is found.
        """
        documents: dict[str, str] = {}

        try:
            with tarfile.open(fileobj=BytesIO(source_bytes), mode="r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name.lstrip("./")
                    if not name.lower().endswith((".tex", ".ltx")):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    documents[name] = extracted.read().decode("utf-8", errors="ignore")
        except tarfile.TarError:
            pass

        if documents:
            return documents

        for name, payload in self._iter_text_source_candidates(source_bytes):
            if "\\documentclass" in payload or "\\begin{document}" in payload:
                documents[name] = payload

        return documents

    def _iter_text_source_candidates(self, source_bytes: bytes) -> list[tuple[str, str]]:
        """Decode plausible text-source candidates from raw or gzip bytes.

        Args:
            source_bytes: Raw e-print response to inspect for text content.

        Returns:
            list[tuple[str, str]]: Synthetic file names and decoded candidate
            text, excluding PDF payloads.
        """
        candidates = [("source.tex", source_bytes)]

        if source_bytes[:2] == b"\x1f\x8b":
            try:
                candidates.append(("source.tex", gzip.decompress(source_bytes)))
            except OSError:
                logger.debug("Failed to decompress gzipped source payload")

        decoded_candidates: list[tuple[str, str]] = []
        for name, payload in candidates:
            if payload.startswith(b"%PDF"):
                continue
            decoded_candidates.append((name, payload.decode("utf-8", errors="ignore")))

        return decoded_candidates

    def _select_main_latex_document(self, documents: dict[str, str]) -> tuple[str, str]:
        """Choose the most likely main LaTeX document from a source archive.

        Args:
            documents: Decoded LaTeX documents keyed by archive filename.

        Returns:
            tuple[str, str]: Selected filename and its LaTeX content.

        Raises:
            SourceExtractionError: If no usable main document can be selected.
        """
        best_name = ""
        best_content = ""
        best_score = -1

        for name, content in documents.items():
            lowered_name = name.lower()
            score = 0
            if "\\documentclass" in content:
                score += 10
            if "\\begin{document}" in content:
                score += 5
            if "\\title{" in content:
                score += 2
            if any(token in lowered_name for token in ("main", "paper", "ms", "manuscript")):
                score += 3
            score -= len(lowered_name) / 1000

            if score > best_score:
                best_name = name
                best_content = content
                best_score = score

        if not best_content:
            raise SourceExtractionError("No main LaTeX document found in source archive")

        return best_name, best_content

    def _latex_to_markdown(self, latex_text: str) -> str:
        """Convert supported LaTeX structure and inline syntax to markdown.

        Args:
            latex_text: Source text from a selected LaTeX document.

        Returns:
            str: Cleaned markdown containing available title, author, abstract,
            and body content.
        """
        text = latex_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?<!\\)%.*", "", text)

        title = self._extract_latex_command_argument(text, "title")
        author = self._extract_latex_command_argument(text, "author")
        abstract = self._extract_latex_environment(text, "abstract")

        body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
        body = body_match.group(1) if body_match else text
        body = re.sub(r"\\maketitle\b", "", body)
        body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.DOTALL)

        markdown_parts: list[str] = []
        if title:
            markdown_parts.append(f"# {self._latex_inline_to_text(title)}")
        if author:
            markdown_parts.append(self._latex_inline_to_text(author))
        if abstract:
            markdown_parts.append("## Abstract")
            markdown_parts.append(self._latex_block_to_text(abstract))

        body = self._convert_latex_structure_to_markdown(body)
        body = self._latex_block_to_text(body)
        if body:
            markdown_parts.append(body)

        return self._cleanup_markdown("\n\n".join(part for part in markdown_parts if part))

    def _convert_latex_structure_to_markdown(self, text: str) -> str:
        """Convert structural LaTeX commands into markdown markers.

        Args:
            text: LaTeX text whose sections, lists, and equations are converted.

        Returns:
            str: Text with supported structural commands expressed as markdown.
        """
        replacements = (
            (r"\\section\*?\{([^{}]+)\}", r"\n\n## \1\n\n"),
            (r"\\subsection\*?\{([^{}]+)\}", r"\n\n### \1\n\n"),
            (r"\\subsubsection\*?\{([^{}]+)\}", r"\n\n#### \1\n\n"),
            (r"\\paragraph\*?\{([^{}]+)\}", r"\n\n**\1.** "),
            (r"\\begin\{itemize\}", "\n"),
            (r"\\end\{itemize\}", "\n"),
            (r"\\begin\{enumerate\}", "\n"),
            (r"\\end\{enumerate\}", "\n"),
            (r"\\item\s+", "\n- "),
            (r"\\begin\{equation\*?\}", "\n$$\n"),
            (r"\\end\{equation\*?\}", "\n$$\n"),
            (r"\\\[", "\n$$\n"),
            (r"\\\]", "\n$$\n"),
            (r"\\\\", "\n"),
        )

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        return text

    def _latex_block_to_text(self, text: str) -> str:
        """Best-effort conversion of LaTeX text blocks to markdown-friendly text.

        Args:
            text: LaTeX block containing inline markup and commands.

        Returns:
            str: Text with supported inline formatting preserved and unsupported
            commands removed.
        """
        converted = text

        inline_patterns = (
            (r"\\textbf\{([^{}]*)\}", r"**\1**"),
            (r"\\textit\{([^{}]*)\}", r"*\1*"),
            (r"\\emph\{([^{}]*)\}", r"*\1*"),
            (r"\\underline\{([^{}]*)\}", r"\1"),
            (r"\\url\{([^{}]*)\}", r"\1"),
            (r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"[\2](\1)"),
        )

        for pattern, replacement in inline_patterns:
            converted = re.sub(pattern, replacement, converted)

        command_patterns = (
            (r"\\(?:cite|citet|citep|eqref|ref)\*?(?:\[[^\]]*\])?\{[^{}]*\}", "[ref]"),
            (r"\\label\{[^{}]*\}", ""),
            (r"\\(?:footnote|thanks)\{([^{}]*)\}", r" (\1)"),
            (r"\\(?:includegraphics|input|bibliography)\*?(?:\[[^\]]*\])?\{[^{}]*\}", ""),
            (r"\\(?:begin|end)\{[^{}]*\}", ""),
        )

        for pattern, replacement in command_patterns:
            converted = re.sub(pattern, replacement, converted)

        for _ in range(5):
            updated = re.sub(
                r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}",
                r"\1",
                converted,
            )
            if updated == converted:
                break
            converted = updated

        converted = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", "", converted)
        converted = converted.replace("{", "").replace("}", "")
        return converted

    def _plain_text_to_markdown(self, text: str) -> str:
        """Normalize extracted plain text into lightweight markdown paragraphs.

        Args:
            text: Raw text extracted from a fallback PDF.

        Returns:
            str: Cleaned paragraphs with recognized headings promoted to markdown.
        """
        lines = [line.strip() for line in text.splitlines()]
        paragraphs: list[str] = []
        current_paragraph: list[str] = []

        for line in lines:
            if not line:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                continue

            if re.match(r"^(abstract|introduction|conclusion|references)\b", line, re.IGNORECASE):
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(f"## {line}")
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        return self._cleanup_markdown("\n\n".join(paragraphs))

    def _extract_latex_command_argument(self, text: str, command: str) -> str:
        """Extract a simple LaTeX command argument such as ``\\title{...}``.

        Args:
            text: LaTeX source to search.
            command: Command name without the leading backslash.

        Returns:
            str: Trimmed command argument, or an empty string when absent.
        """
        match = re.search(
            rf"\\{command}\*?(?:\[[^\]]*\])?\{{(.*?)\}}",
            text,
            re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _extract_latex_environment(self, text: str, environment: str) -> str:
        """Extract the text contained in a named LaTeX environment.

        Args:
            text: LaTeX source to search.
            environment: Environment name without ``begin`` or ``end`` syntax.

        Returns:
            str: Trimmed environment body, or an empty string when absent.
        """
        match = re.search(
            rf"\\begin\{{{environment}\}}(.*?)\\end\{{{environment}\}}",
            text,
            re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _latex_inline_to_text(self, text: str) -> str:
        """Convert inline LaTeX to text and collapse surplus whitespace.

        Args:
            text: Inline LaTeX fragment to normalize.

        Returns:
            str: Trimmed, markdown-friendly inline text.
        """
        return re.sub(r"\s+", " ", self._latex_block_to_text(text)).strip()

    def _cleanup_markdown(self, text: str) -> str:
        """Normalize whitespace and trim noisy blank lines in markdown output.

        Args:
            text: Markdown content to clean.

        Returns:
            str: Trimmed markdown with normalized spaces and blank lines.
        """
        cleaned = re.sub(r"[ \t]+\n", "\n", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()
