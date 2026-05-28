"""Unit tests for Phase 2 ChrisAgent tools."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NEW = ROOT / "src_new"
if str(SRC_NEW) not in sys.path:
    sys.path.insert(0, str(SRC_NEW))

import chris_agent_phase2 as phase2


def _wrapped_find_topic_tool():
    return phase2.find_topic_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def _wrapped_extract_main_result_tool():
    return phase2.extract_main_result_tool.on_invoke_tool._invoke_tool_impl.__closure__[2].cell_contents


def test_find_topic_tool_fails_when_csv_missing(tmp_path):
    result = _wrapped_find_topic_tool()(csv_path=str(tmp_path / "missing.csv"))
    assert result["status"] == "failure"
    assert result["topics"] == []


def test_find_topic_tool_success(monkeypatch, tmp_path):
    csv_path = tmp_path / "papers.csv"
    csv_path.write_text("arxiv_id,title,summary\n1,t,s\n", encoding="utf-8")

    def fake_compute_topics(path, n_topics, n_papers_per_topic):
        assert path == str(csv_path)
        assert n_topics == 2
        assert n_papers_per_topic == 3
        return [{"topic_title": "T1"}]

    monkeypatch.setattr(phase2, "compute_topics", fake_compute_topics)
    result = _wrapped_find_topic_tool()(csv_path=str(csv_path), n_topics=2, n_papers_per_topic=3)
    assert result["status"] == "success"
    assert result["topics"] == [{"topic_title": "T1"}]


def test_extract_main_result_tool_success(monkeypatch):
    class FakeFetcher:
        def fetch_paper_markdown(self, paper_id, output_dir, force_redownload):
            assert paper_id == "2601.12345"
            return "Main theorem content"

    monkeypatch.setattr(phase2, "ArxivFetcher", FakeFetcher)
    monkeypatch.setattr(phase2, "_extract_main_result_with_llm", lambda arxiv_id, content: "- result")

    result = _wrapped_extract_main_result_tool()(arxiv_id="2601.12345")
    assert result["status"] == "success"
    assert result["main_result"] == "- result"


def test_extract_main_result_tool_failure_on_download(monkeypatch):
    class FakeFetcher:
        def fetch_paper_markdown(self, paper_id, output_dir, force_redownload):
            raise RuntimeError("download failed")

    monkeypatch.setattr(phase2, "ArxivFetcher", FakeFetcher)
    result = _wrapped_extract_main_result_tool()(arxiv_id="2601.12345")
    assert result["status"] == "failure"
    assert result["main_result"] is None
