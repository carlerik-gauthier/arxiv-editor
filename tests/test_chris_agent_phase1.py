"""Unit tests for Phase 1 ChrisAgent helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
SRC_NEW = ROOT / "src_new"
if str(SRC_NEW) not in sys.path:
    sys.path.insert(0, str(SRC_NEW))

import chris_agent_phase1 as phase1


def _wrapped_fetcher_tool():
    return phase1.arxiv_fetcher_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def test_arxiv_fetcher_tool_defaults_to_allowed_categories(monkeypatch):
    calls: list[str] = []

    class FakeFetcher:
        def fetch_by_category(self, category, start_date, end_date, max_results):
            calls.append(category)
            return [
                phase1.Paper(
                    arxiv_id=f"{category}-1",
                    title=f"title-{category}",
                    authors=["a"],
                    summary="summary",
                    published=start_date,
                    updated=end_date,
                    categories=[category],
                    primary_category=category,
                    pdf_url="https://example.test/paper.pdf",
                    entry_id=f"id-{category}",
                    comment=None,
                    journal_ref=None,
                    doi=None,
                )
            ]

    monkeypatch.setattr(phase1, "ArxivFetcher", FakeFetcher)
    output_dir = ROOT / "data" / "tmp_test_outputs"
    monkeypatch.setenv("ARXIV_FETCH_OUTPUT_DIR", str(output_dir))

    result = _wrapped_fetcher_tool()(
        start_date="2026-01-01",
        end_date="2026-01-31",
        categories=[],
        max_results=phase1.DEFAULT_MAX_RESULTS,
        min_threshold=1,
    )

    assert set(calls) == set(phase1.CHRIS_CATEGORIES)
    assert result["status"] == "success"
    assert result["categories"] == list(phase1.CHRIS_CATEGORIES)
    assert result["rows_saved"] == len(phase1.CHRIS_CATEGORIES)
    assert result["csv_path"] is not None
    csv_path = Path(result["csv_path"])
    assert csv_path.exists()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(phase1.CHRIS_CATEGORIES)
    csv_path.unlink()
    if output_dir.exists() and not any(output_dir.iterdir()):
        output_dir.rmdir()


def test_arxiv_fetcher_tool_returns_failure_below_threshold(monkeypatch):
    class FakeFetcher:
        def fetch_by_category(self, category, start_date, end_date, max_results):
            return []

    monkeypatch.setattr(phase1, "ArxivFetcher", FakeFetcher)
    result = _wrapped_fetcher_tool()(
        start_date="2026-01-01",
        end_date="2026-01-31",
        categories=["math.PR"],
        max_results=phase1.DEFAULT_MAX_RESULTS,
        min_threshold=1,
    )
    assert result["status"] == "failure"
    assert "Not enough papers fetched" in result["reason"]
    assert result["rows_saved"] == 0
    assert result["csv_path"] is None


def test_extract_date_range_invalid_single_date_falls_back(monkeypatch):
    def fake_extractor(_message: str):
        return {"start_date": "2026-99-99", "end_date": None}

    monkeypatch.setattr(phase1, "_extract_date_range_with_llm", fake_extractor)
    start_date, end_date = phase1._extract_date_range("check 2026-99-99 please")

    assert start_date != "2026-99-99"
    assert len(start_date) == 10
    assert len(end_date) == 10


def test_extract_date_range_invalid_pair_falls_back(monkeypatch):
    def fake_extractor(_message: str):
        return {"start_date": "2026-99-99", "end_date": "2026-88-77"}

    monkeypatch.setattr(phase1, "_extract_date_range_with_llm", fake_extractor)
    start_date, end_date = phase1._extract_date_range(
        "between 2026-99-99 and 2026-88-77"
    )

    assert start_date != "2026-99-99"
    assert end_date != "2026-88-77"
    assert len(start_date) == 10
    assert len(end_date) == 10


def test_extract_date_range_with_single_valid_date(monkeypatch):
    def fake_extractor(_message: str):
        return {"start_date": "2026-01-05", "end_date": None}

    monkeypatch.setattr(phase1, "_extract_date_range_with_llm", fake_extractor)
    start_date, end_date = phase1._extract_date_range("from 2026-01-05")
    assert start_date == "2026-01-05"
    assert len(end_date) == 10


def test_extract_date_range_swaps_reversed_dates(monkeypatch):
    def fake_extractor(_message: str):
        return {"start_date": "2026-03-10", "end_date": "2026-01-10"}

    monkeypatch.setattr(phase1, "_extract_date_range_with_llm", fake_extractor)
    start_date, end_date = phase1._extract_date_range("between 2026-03-10 and 2026-01-10")
    assert start_date == "2026-01-10"
    assert end_date == "2026-03-10"


def test_get_arxiv_categories_defaults_to_all(monkeypatch):
    def fake_classifier(_message: str):
        return []

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("recent papers in this field")
    assert categories == list(phase1.CHRIS_CATEGORIES)


def test_get_arxiv_categories_probability_only(monkeypatch):
    def fake_classifier(_message: str):
        return ["math.PR"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("focus on probability and stochastic models")
    assert categories == ["math.PR"]


def test_get_arxiv_categories_statistics_only(monkeypatch):
    def fake_classifier(_message: str):
        return ["math.ST"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("need statistical theory references")
    assert categories == ["math.ST"]


def test_get_arxiv_categories_both(monkeypatch):
    def fake_classifier(_message: str):
        return ["math.PR", "math.ST"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("math.PR and math.ST papers")
    assert categories == ["math.PR", "math.ST"]


def test_get_arxiv_categories_filters_unsupported_and_dedupes(monkeypatch):
    def fake_classifier(_message: str):
        return ["cs.LG", "math.PR", "math.PR", "math.ST"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("something")
    assert categories == ["math.PR", "math.ST"]


def test_extract_tool_parameters_parses_function_call_arguments():
    class RawItem:
        type = "function_call"
        name = "arxiv_fetcher_tool"
        arguments = '{"start_date":"2026-01-01","categories":["math.PR"]}'

    class Item:
        raw_item = RawItem()

    extracted = phase1._extract_tool_parameters([Item()])

    assert extracted == [
        {
            "tool": "arxiv_fetcher_tool",
            "arguments": {"start_date": "2026-01-01", "categories": ["math.PR"]},
        }
    ]


def test_extract_tool_parameters_ignores_non_function_calls():
    class RawItem:
        type = "message"
        name = "ignored"
        arguments = "{}"

    class Item:
        raw_item = RawItem()

    assert phase1._extract_tool_parameters([Item()]) == []
