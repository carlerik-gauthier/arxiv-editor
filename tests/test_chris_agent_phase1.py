"""Unit tests for Phase 1 ChrisAgent helpers."""

from __future__ import annotations

import sys
from pathlib import Path

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

    result = _wrapped_fetcher_tool()(
        start_date="2026-01-01",
        end_date="2026-01-31",
        categories=[],
        max_results=phase1.DEFAULT_MAX_RESULTS,
        min_threshold=1,
    )

    assert set(calls) == set(phase1.CHRIS_CATEGORIES)
    assert set(result["categories"]) == set(phase1.CHRIS_CATEGORIES)
    assert len(result["papers"]) == len(phase1.CHRIS_CATEGORIES)


def test_extract_date_range_invalid_single_date_falls_back():
    start_date, end_date = phase1._extract_date_range("check 2026-99-99 please")

    assert start_date != "2026-99-99"
    assert len(start_date) == 10
    assert len(end_date) == 10


def test_extract_date_range_invalid_pair_falls_back():
    start_date, end_date = phase1._extract_date_range(
        "between 2026-99-99 and 2026-88-77"
    )

    assert start_date != "2026-99-99"
    assert end_date != "2026-88-77"
    assert len(start_date) == 10
    assert len(end_date) == 10


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
        return ["stat.TH"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("need statistical theory references")
    assert categories == ["stat.TH"]


def test_get_arxiv_categories_both(monkeypatch):
    def fake_classifier(_message: str):
        return ["math.PR", "stat.TH"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("math.PR and stat.TH papers")
    assert categories == ["math.PR", "stat.TH"]


def test_get_arxiv_categories_filters_unsupported_and_dedupes(monkeypatch):
    def fake_classifier(_message: str):
        return ["cs.LG", "math.PR", "math.PR", "stat.TH"]

    monkeypatch.setattr(phase1, "_classify_categories_with_llm", fake_classifier)
    categories = phase1._get_arxiv_categories("something")
    assert categories == ["math.PR", "stat.TH"]
