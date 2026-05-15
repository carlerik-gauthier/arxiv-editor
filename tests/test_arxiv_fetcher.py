"""
Tests for the ArXiv fetcher module.

These tests verify the ArxivFetcher class can fetch papers from ArXiv
by category and date range, with proper metadata parsing.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import tempfile
import shutil

from old_src.fetchers.arxiv_fetcher import (
    ArxivFetcher,
    ArxivFetcherError,
    Paper,
    PDFDownloadError,
    PDFExtractionError,
)


class TestPaper:
    """Tests for the Paper data model."""

    def test_paper_creation(self):
        """Test creating a Paper with valid data."""
        paper = Paper(
            arxiv_id="2301.12345",
            title="Test Paper Title",
            authors=["Author One", "Author Two"],
            summary="This is a test summary.",
            published=datetime(2023, 1, 15),
            updated=datetime(2023, 1, 16),
            categories=["math.PR", "stat.TH"],
            primary_category="math.PR",
            pdf_url="https://arxiv.org/pdf/2301.12345",
            entry_id="http://arxiv.org/abs/2301.12345",
        )

        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.primary_category == "math.PR"

    def test_paper_empty_id_raises_error(self):
        """Test that empty arxiv_id raises ValueError."""
        with pytest.raises(ValueError, match="arxiv_id cannot be empty"):
            Paper(
                arxiv_id="",
                title="Test",
                authors=["Author"],
                summary="Summary",
                published=datetime.now(),
                updated=datetime.now(),
                categories=["math.PR"],
                primary_category="math.PR",
                pdf_url="https://arxiv.org/pdf/test",
                entry_id="http://arxiv.org/abs/test",
            )

    def test_paper_empty_title_raises_error(self):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            Paper(
                arxiv_id="2301.12345",
                title="",
                authors=["Author"],
                summary="Summary",
                published=datetime.now(),
                updated=datetime.now(),
                categories=["math.PR"],
                primary_category="math.PR",
                pdf_url="https://arxiv.org/pdf/test",
                entry_id="http://arxiv.org/abs/test",
            )

    def test_paper_to_dict(self):
        """Test Paper serialization to dictionary."""
        paper = Paper(
            arxiv_id="2301.12345",
            title="Test Paper",
            authors=["Author"],
            summary="Summary",
            published=datetime(2023, 1, 15, 10, 30, 0),
            updated=datetime(2023, 1, 16, 11, 0, 0),
            categories=["math.PR"],
            primary_category="math.PR",
            pdf_url="https://arxiv.org/pdf/2301.12345",
            entry_id="http://arxiv.org/abs/2301.12345",
            doi="10.1234/test",
        )

        result = paper.to_dict()

        assert result["arxiv_id"] == "2301.12345"
        assert result["published"] == "2023-01-15T10:30:00"
        assert result["doi"] == "10.1234/test"
        assert isinstance(result["categories"], list)


class TestArxivFetcher:
    """Tests for the ArxivFetcher class."""

    def test_fetcher_initialization(self):
        """Test ArxivFetcher initialization with default values."""
        fetcher = ArxivFetcher()

        assert fetcher.request_delay >= 3.0
        assert fetcher.max_retries == 3

    def test_fetcher_minimum_delay(self):
        """Test that request delay is at least 3 seconds."""
        fetcher = ArxivFetcher(request_delay=1.0)

        # Should enforce minimum of 3 seconds
        assert fetcher.request_delay >= 3.0

    def test_parse_paper_metadata(self):
        """Test parsing ArXiv API result into Paper object."""
        fetcher = ArxivFetcher()

        # Create mock arxiv.Result
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2301.12345"
        mock_result.title = "Test Paper\nWith Newline"
        mock_result.summary = "Test summary\nwith newlines."
        mock_result.published = datetime(2023, 1, 15)
        mock_result.updated = datetime(2023, 1, 16)
        mock_result.categories = ["math.PR", "stat.TH"]
        mock_result.primary_category = "math.PR"
        mock_result.pdf_url = "https://arxiv.org/pdf/2301.12345"
        mock_result.comment = "10 pages"
        mock_result.journal_ref = None
        mock_result.doi = None

        mock_author1 = MagicMock()
        mock_author1.name = "Author One"
        mock_author2 = MagicMock()
        mock_author2.name = "Author Two"
        mock_result.authors = [mock_author1, mock_author2]

        paper = fetcher.parse_paper_metadata(mock_result)

        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Test Paper With Newline"  # Newlines removed
        assert paper.summary == "Test summary with newlines."  # Newlines removed
        assert paper.authors == ["Author One", "Author Two"]
        assert paper.comment == "10 pages"


class TestFetchWithThreshold:
    """Tests for the fetch_with_threshold method."""

    def test_fetch_with_threshold_mock_threshold_met(self):
        """Test fetch_with_threshold when threshold is met immediately."""
        fetcher = ArxivFetcher()

        # Mock the fetch_multiple_categories method to return 100 papers
        mock_papers = [
            Paper(
                arxiv_id=f"2301.{i:05d}",
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                summary=f"Summary {i}",
                published=datetime(2023, 1, 15),
                updated=datetime(2023, 1, 15),
                categories=["math.PR"],
                primary_category="math.PR",
                pdf_url=f"https://arxiv.org/pdf/2301.{i:05d}",
                entry_id=f"http://arxiv.org/abs/2301.{i:05d}",
            )
            for i in range(100)
        ]

        with patch.object(fetcher, "fetch_multiple_categories", return_value=mock_papers):
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 7)

            papers, actual_start, actual_end = fetcher.fetch_with_threshold(
                categories=["math.PR"],
                start_date=start_date,
                end_date=end_date,
                min_count=100,
            )

            assert len(papers) == 100
            assert actual_start == start_date  # No expansion needed
            assert actual_end == end_date

    def test_fetch_with_threshold_mock_needs_expansion(self):
        """Test fetch_with_threshold when date range needs expansion."""
        fetcher = ArxivFetcher()

        # Mock to return fewer papers first, then enough on second call
        call_count = 0

        def mock_fetch_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: not enough papers
                return [
                    Paper(
                        arxiv_id=f"2301.{i:05d}",
                        title=f"Paper {i}",
                        authors=[f"Author {i}"],
                        summary=f"Summary {i}",
                        published=datetime(2023, 1, 15),
                        updated=datetime(2023, 1, 15),
                        categories=["math.PR"],
                        primary_category="math.PR",
                        pdf_url=f"https://arxiv.org/pdf/2301.{i:05d}",
                        entry_id=f"http://arxiv.org/abs/2301.{i:05d}",
                    )
                    for i in range(50)
                ]
            else:
                # Second call: enough papers
                return [
                    Paper(
                        arxiv_id=f"2301.{i:05d}",
                        title=f"Paper {i}",
                        authors=[f"Author {i}"],
                        summary=f"Summary {i}",
                        published=datetime(2023, 1, 15),
                        updated=datetime(2023, 1, 15),
                        categories=["math.PR"],
                        primary_category="math.PR",
                        pdf_url=f"https://arxiv.org/pdf/2301.{i:05d}",
                        entry_id=f"http://arxiv.org/abs/2301.{i:05d}",
                    )
                    for i in range(100)
                ]

        with patch.object(
            fetcher, "fetch_multiple_categories", side_effect=mock_fetch_side_effect
        ):
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 7)

            papers, actual_start, actual_end = fetcher.fetch_with_threshold(
                categories=["math.PR"],
                start_date=start_date,
                end_date=end_date,
                min_count=100,
                expansion_days=7,
            )

            assert len(papers) == 100
            assert actual_start < start_date  # Date was expanded
            assert actual_end == end_date
            assert call_count == 2  # Two attempts were made

    def test_fetch_with_threshold_mock_fails_after_max_expansions(self):
        """Test that fetch_with_threshold raises error after max expansions."""
        fetcher = ArxivFetcher()

        # Mock to always return insufficient papers
        mock_papers = [
            Paper(
                arxiv_id=f"2301.{i:05d}",
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                summary=f"Summary {i}",
                published=datetime(2023, 1, 15),
                updated=datetime(2023, 1, 15),
                categories=["math.PR"],
                primary_category="math.PR",
                pdf_url=f"https://arxiv.org/pdf/2301.{i:05d}",
                entry_id=f"http://arxiv.org/abs/2301.{i:05d}",
            )
            for i in range(50)
        ]

        with patch.object(fetcher, "fetch_multiple_categories", return_value=mock_papers):
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 7)

            with pytest.raises(ArxivFetcherError, match="Unable to fetch minimum 100 papers"):
                fetcher.fetch_with_threshold(
                    categories=["math.PR"],
                    start_date=start_date,
                    end_date=end_date,
                    min_count=100,
                    max_expansions=2,
                )


class TestPDFDownload:
    """Tests for PDF download functionality."""

    def test_download_paper_pdf_success(self):
        """Test successful PDF download."""
        fetcher = ArxivFetcher()

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "2301.12345"

            # Mock requests.get to return fake PDF content
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content = lambda chunk_size: [b"fake pdf content"]
            mock_response.raise_for_status = Mock()

            with patch("requests.get", return_value=mock_response):
                pdf_path = fetcher.download_paper_pdf(
                    paper_id=paper_id,
                    output_dir=temp_dir,
                )

                # Verify the file was created
                assert pdf_path.exists()
                assert pdf_path.name == "2301.12345.pdf"

                # Verify content was written
                with open(pdf_path, "rb") as f:
                    content = f.read()
                    assert content == b"fake pdf content"

    def test_download_paper_pdf_cached(self):
        """Test that cached PDFs are not re-downloaded."""
        fetcher = ArxivFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "2301.12345"

            # Create a fake PDF file
            pdf_path = Path(temp_dir) / f"{paper_id}.pdf"
            pdf_path.write_bytes(b"cached content")

            # Mock requests.get - it should NOT be called
            with patch("requests.get") as mock_get:
                result_path = fetcher.download_paper_pdf(
                    paper_id=paper_id,
                    output_dir=temp_dir,
                    force_redownload=False,
                )

                # Verify the cached file was returned
                assert result_path == pdf_path
                # Verify requests.get was NOT called
                mock_get.assert_not_called()

    def test_download_paper_pdf_force_redownload(self):
        """Test force re-download even when cached."""
        fetcher = ArxivFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "2301.12345"

            # Create a fake cached PDF
            pdf_path = Path(temp_dir) / f"{paper_id}.pdf"
            pdf_path.write_bytes(b"old content")

            # Mock requests.get to return new content
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content = lambda chunk_size: [b"new content"]
            mock_response.raise_for_status = Mock()

            with patch("requests.get", return_value=mock_response):
                result_path = fetcher.download_paper_pdf(
                    paper_id=paper_id,
                    output_dir=temp_dir,
                    force_redownload=True,
                )

                # Verify new content was downloaded
                with open(result_path, "rb") as f:
                    content = f.read()
                    assert content == b"new content"

    def test_download_paper_pdf_http_error(self):
        """Test PDF download with HTTP error."""
        fetcher = ArxivFetcher(max_retries=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "2301.12345"

            # Mock requests.get to raise HTTPError
            import requests

            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

            with patch("requests.get", return_value=mock_response):
                with pytest.raises(PDFDownloadError, match="Failed to download PDF"):
                    fetcher.download_paper_pdf(
                        paper_id=paper_id,
                        output_dir=temp_dir,
                    )

    def test_download_paper_pdf_sanitize_id(self):
        """Test that paper IDs with slashes are sanitized for filenames."""
        fetcher = ArxivFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "math/0601001"  # Old-style ArXiv ID with slash

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content = lambda chunk_size: [b"content"]
            mock_response.raise_for_status = Mock()

            with patch("requests.get", return_value=mock_response):
                pdf_path = fetcher.download_paper_pdf(
                    paper_id=paper_id,
                    output_dir=temp_dir,
                )

                # Verify filename has slash replaced with underscore
                assert pdf_path.name == "math_0601001.pdf"


class TestPDFExtraction:
    """Tests for PDF text extraction functionality."""

    def test_extract_text_from_pdf_success(self):
        """Test successful text extraction from PDF."""
        fetcher = ArxivFetcher()

        # Use a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)

        try:
            # Mock PdfReader
            mock_page = Mock()
            mock_page.extract_text.return_value = "This is page 1 text."

            mock_reader = Mock()
            mock_reader.pages = [mock_page]

            with patch("src.fetchers.arxiv_fetcher.PdfReader", return_value=mock_reader):
                text = fetcher.extract_text_from_pdf(pdf_path)

                assert text == "This is page 1 text."
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_extract_text_from_pdf_multiple_pages(self):
        """Test text extraction from multi-page PDF."""
        fetcher = ArxivFetcher()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)

        try:
            # Mock multiple pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content."

            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content."

            mock_page3 = Mock()
            mock_page3.extract_text.return_value = "Page 3 content."

            mock_reader = Mock()
            mock_reader.pages = [mock_page1, mock_page2, mock_page3]

            with patch("src.fetchers.arxiv_fetcher.PdfReader", return_value=mock_reader):
                text = fetcher.extract_text_from_pdf(pdf_path)

                assert "Page 1 content." in text
                assert "Page 2 content." in text
                assert "Page 3 content." in text
                assert text.count("\n\n") == 2  # Pages separated by double newline
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_extract_text_from_pdf_file_not_found(self):
        """Test extraction when PDF file doesn't exist."""
        fetcher = ArxivFetcher()

        pdf_path = Path("/fake/path/nonexistent.pdf")

        with pytest.raises(PDFExtractionError, match="PDF file not found"):
            fetcher.extract_text_from_pdf(pdf_path)

    def test_extract_text_from_pdf_empty_pdf(self):
        """Test extraction from PDF with no text."""
        fetcher = ArxivFetcher()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)

        try:
            # Mock page with no text
            mock_page = Mock()
            mock_page.extract_text.return_value = ""

            mock_reader = Mock()
            mock_reader.pages = [mock_page]

            with patch("src.fetchers.arxiv_fetcher.PdfReader", return_value=mock_reader):
                with pytest.raises(PDFExtractionError, match="No text extracted"):
                    fetcher.extract_text_from_pdf(pdf_path)
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_extract_text_from_pdf_page_error_continues(self):
        """Test that extraction continues if one page fails."""
        fetcher = ArxivFetcher()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            pdf_path = Path(tmp_file.name)

        try:
            # Mock pages where middle page fails
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1."

            mock_page2 = Mock()
            mock_page2.extract_text.side_effect = Exception("Page extraction failed")

            mock_page3 = Mock()
            mock_page3.extract_text.return_value = "Page 3."

            mock_reader = Mock()
            mock_reader.pages = [mock_page1, mock_page2, mock_page3]

            with patch("src.fetchers.arxiv_fetcher.PdfReader", return_value=mock_reader):
                text = fetcher.extract_text_from_pdf(pdf_path)

                # Should contain text from pages 1 and 3
                assert "Page 1." in text
                assert "Page 3." in text
        finally:
            pdf_path.unlink(missing_ok=True)


class TestDownloadAndExtract:
    """Tests for the combined download and extract functionality."""

    def test_download_and_extract_paper_success(self):
        """Test successful download and extraction."""
        fetcher = ArxivFetcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "2301.12345"

            # Mock download
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content = lambda chunk_size: [b"fake pdf"]
            mock_response.raise_for_status = Mock()

            # Mock extraction
            mock_page = Mock()
            mock_page.extract_text.return_value = "Extracted text content"

            mock_reader = Mock()
            mock_reader.pages = [mock_page]

            with patch("requests.get", return_value=mock_response):
                with patch("src.fetchers.arxiv_fetcher.PdfReader", return_value=mock_reader):
                    pdf_path, text = fetcher.download_and_extract_paper(
                        paper_id=paper_id,
                        output_dir=temp_dir,
                    )

                    assert pdf_path.exists()
                    assert text == "Extracted text content"

    def test_download_and_extract_paper_download_fails(self):
        """Test that extraction is not attempted if download fails."""
        fetcher = ArxivFetcher(max_retries=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper_id = "2301.12345"

            # Mock download failure
            import requests

            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

            with patch("requests.get", return_value=mock_response):
                with pytest.raises(PDFDownloadError):
                    fetcher.download_and_extract_paper(
                        paper_id=paper_id,
                        output_dir=temp_dir,
                    )


class TestArxivFetcherIntegration:
    """Integration tests that actually call the ArXiv API."""

    @pytest.mark.integration
    def test_fetch_papers_from_math_pr(self):
        """
        Test fetching papers from math.PR category.

        This test actually calls the ArXiv API.
        Run with: pytest -m integration
        """
        fetcher = ArxivFetcher(request_delay=3.0)

        # Fetch papers from the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        papers = fetcher.fetch_by_category(
            category="math.PR",
            start_date=start_date,
            end_date=end_date,
            max_results=10,
        )

        # Verify we got papers (may be 0 if no recent submissions)
        assert isinstance(papers, list)

        if papers:
            # Verify paper structure
            paper = papers[0]
            assert paper.arxiv_id
            assert paper.title
            assert paper.authors
            assert paper.summary
            assert paper.published
            assert paper.categories
            assert "math.PR" in paper.categories or paper.primary_category == "math.PR"

            print(f"\nFetched {len(papers)} papers from math.PR")
            print(f"Sample paper: {paper.title[:80]}...")

    @pytest.mark.integration
    def test_fetch_multiple_categories(self):
        """
        Test fetching papers from multiple categories.

        Run with: pytest -m integration
        """
        fetcher = ArxivFetcher(request_delay=3.0)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        papers = fetcher.fetch_multiple_categories(
            categories=["math.PR", "stat.TH"],
            start_date=start_date,
            end_date=end_date,
            max_results_per_category=5,
        )

        assert isinstance(papers, list)
        print(f"\nFetched {len(papers)} unique papers from math.PR and stat.TH")

    @pytest.mark.integration
    def test_fetch_with_threshold_integration(self):
        """
        Test fetch_with_threshold with real ArXiv API.

        This test may expand the date range if there are few recent submissions.
        Run with: pytest -m integration
        """
        fetcher = ArxivFetcher(request_delay=3.0)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Use a lower threshold for testing and categories likely to have papers
        papers, actual_start, actual_end = fetcher.fetch_with_threshold(
            categories=["cs.LG", "math.PR"],
            start_date=start_date,
            end_date=end_date,
            min_count=20,  # Lower threshold for testing
            max_results_per_category=50,
            expansion_days=7,
            max_expansions=3,
        )

        assert isinstance(papers, list)
        assert len(papers) >= 20, f"Expected at least 20 papers, got {len(papers)}"
        assert actual_start <= start_date
        assert actual_end == end_date

        days_expanded = (start_date - actual_start).days
        print(f"\nFetched {len(papers)} papers")
        print(f"Date range: {actual_start.date()} to {actual_end.date()}")
        if days_expanded > 0:
            print(f"Date range was expanded by {days_expanded} days")
        else:
            print("Original date range was sufficient")

    @pytest.mark.integration
    def test_download_and_extract_real_paper(self):
        """
        Test downloading and extracting a real paper from ArXiv.

        Uses a well-known stable paper ID.
        Run with: pytest -m integration
        """
        fetcher = ArxivFetcher(request_delay=3.0)

        # Use a well-known paper (first arXiv paper ever submitted)
        # This paper should always be available
        paper_id = "hep-th/9901001"  # Old-style ArXiv ID

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path, text = fetcher.download_and_extract_paper(
                paper_id=paper_id,
                output_dir=temp_dir,
            )

            # Verify PDF was downloaded
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0

            # Verify text was extracted
            assert len(text) > 0
            assert isinstance(text, str)

            print(f"\nDownloaded and extracted paper: {paper_id}")
            print(f"PDF size: {pdf_path.stat().st_size} bytes")
            print(f"Extracted text length: {len(text)} characters")
            print(f"First 200 characters of text:")
            print(text[:200])

            # Test caching - second download should be instant
            pdf_path_2, text_2 = fetcher.download_and_extract_paper(
                paper_id=paper_id,
                output_dir=temp_dir,
            )

            assert pdf_path_2 == pdf_path
            assert text_2 == text
            print("\nCaching test: Successfully retrieved from cache")


# Quick test runner for development
if __name__ == "__main__":
    """Run a quick integration test."""
    print("Running quick ArXiv fetcher test...")

    fetcher = ArxivFetcher(request_delay=3.0)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"Fetching papers from math.PR between {start_date.date()} and {end_date.date()}")

    papers = fetcher.fetch_by_category(
        category="math.PR",
        start_date=start_date,
        end_date=end_date,
        max_results=10,
    )

    print(f"\nFetched {len(papers)} papers:")
    for i, paper in enumerate(papers[:5], 1):
        print(f"\n{i}. {paper.title[:70]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Published: {paper.published.date()}")
        print(f"   Categories: {paper.categories}")

    print("\nTest completed successfully!")
