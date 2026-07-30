"""Shared workflow primitives for arXiv specialist agents."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agents import Agent, Runner, trace
from agents.tool import FunctionTool
from openai import OpenAI

from src.arxiv_fetcher import ArxivFetcher
from src.data_object import Paper
from src.topic_finder import compute_topics

DEFAULT_FETCH_THRESHOLD = 50
DEFAULT_MAX_RESULTS = 1000
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MAX_TOPICS = 5
DEFAULT_OUTPUT_DIR = Path("data/paper")
DEFAULT_PDF_OUTPUT_DIR = Path("data/pdfs")
DEFAULT_MODEL = "gpt-4.1-mini"

TOPIC_SCHEMA = """{
  'topic_title': <title>,
  'topic_description': <description>,
  'topic_count': <paper count>,
  'representative_papers': [
    {'paper_title': <title>, 'paper_arxiv_id': <arxiv id>, 'main_result': <result>}
  ]
}"""


@dataclass(frozen=True)
class SpecialistConfig:
    """Domain-specific settings for an arXiv specialist agent."""

    name: str
    slug: str
    categories: tuple[str, ...]
    category_descriptions: Mapping[str, str]
    personality_and_communication_style: str
    expertise: str
    expertise_domain: str


def check_papers(config: SpecialistConfig, start_date: str, end_date: str, categories: Sequence[str]) -> Dict[str, Any]:
    """Return the expected CSV path when data for a request already exists."""
    selected = _select_categories(config, categories)
    path = _paper_csv_path(config, selected, start_date, end_date)
    return {
        "status": "success",
        "existence": "Data is already available" if path.exists() else "Data is not available. Need to fetch it",
        "csv_path": str(path) if path.exists() else "",
    }


def fetch_papers(
    config: SpecialistConfig,
    start_date: str,
    categories: Sequence[str],
    end_date: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_threshold: int = DEFAULT_FETCH_THRESHOLD,
) -> Dict[str, Any]:
    """Fetch a specialist's papers, deduplicate them, and persist metadata to CSV."""
    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date) if end_date else datetime.now()
    selected = _select_categories(config, categories)
    fetcher = ArxivFetcher()
    papers: List[Paper] = []
    for category in selected:
        papers.extend(fetcher.fetch_by_category(category, start, end, max_results))

    unique_papers = {paper.arxiv_id: paper for paper in papers}
    if len(unique_papers) < min_threshold:
        return {
            "status": "failure",
            "reason": f"Not enough papers fetched to proceed: {len(unique_papers)} < threshold {min_threshold}.",
            "categories": selected,
            "rows_saved": 0,
            "csv_path": None,
        }

    output_path = _paper_csv_path(config, selected, start.date().isoformat(), end.date().isoformat())
    try:
        save_papers_to_csv(list(unique_papers.values()), output_path)
    except OSError as exc:
        return {
            "status": "failure",
            "reason": f"Failed to save fetched papers to CSV: {exc}",
            "categories": selected,
            "rows_saved": 0,
            "csv_path": str(output_path),
        }
    return {
        "status": "success",
        "reason": f"Successfully fetched and saved {len(unique_papers)} papers to CSV.",
        "categories": selected,
        "rows_saved": len(unique_papers),
        "csv_path": str(output_path),
    }


def find_topics(csv_path: str, n_topics: int = DEFAULT_MAX_TOPICS, n_papers_per_topic: int = 3) -> Dict[str, Any]:
    """Extract up to five topics from a persisted paper-metadata CSV."""
    path = Path(csv_path)
    if not path.exists():
        return {"status": "failure", "reason": f"CSV file not found: {csv_path}", "topics": []}
    try:
        topics = compute_topics(str(path), max(1, min(n_topics, DEFAULT_MAX_TOPICS)), max(1, n_papers_per_topic))
    except Exception as exc:
        return {"status": "failure", "reason": f"Topic extraction failed: {exc}", "topics": []}
    return {"status": "success", "reason": f"Extracted {len(topics)} topics from {csv_path}.", "topics": topics}


def extract_main_result(config: SpecialistConfig, arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Download a paper and summarize its principal result with the configured model."""
    try:
        markdown = ArxivFetcher().fetch_paper_markdown(
            paper_id=arxiv_id,
            output_dir=str(Path(os.getenv("ARXIV_PDF_OUTPUT_DIR", str(DEFAULT_PDF_OUTPUT_DIR)))),
            force_redownload=False,
        )
    except Exception as exc:
        return _main_result_failure(arxiv_id, title, f"Failed to download/extract paper {arxiv_id}: {exc}")

    content = (markdown or "").strip()
    if not content:
        return _main_result_failure(arxiv_id, title, f"No content extracted for paper {arxiv_id}, with title {title}.")
    try:
        main_result = _summarize_content(config, arxiv_id, title, content, max(1000, max_chars))
    except Exception as exc:
        return _main_result_failure(arxiv_id, title, f"Main-result extraction failed for {arxiv_id}: {exc}")
    return {
        "status": "success",
        "reason": f"Extracted main result for paper {arxiv_id}, with title {title}.",
        "arxiv_id": arxiv_id,
        "title": title,
        "main_result": main_result,
    }


def classify_categories(config: SpecialistConfig, message: str) -> List[str]:
    """Infer allowed arXiv categories, defaulting to the entire specialty."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return list(config.categories)
    category_definitions = "\n".join(
        f"- {category}: {config.category_descriptions[category]}"
        for category in config.categories
    )
    prompt = (
        "Analyze carefully the message and list all relevant arXiv categories that fit with the message\n"
        f"from this allowed set only: {list(config.categories)}.\n"
        f"Category descriptions:\n{category_definitions}\n"
        "Return JSON only: {\"categories\": [<category>]}\n"
        f"Message: {message}"
    )
    try:
        response = OpenAI(api_key=api_key).responses.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), input=prompt, temperature=0
        )
        payload = json.loads((response.output_text or "").strip())
        predicted = payload.get("categories", [])
    except (Exception, json.JSONDecodeError):
        predicted = []
    return _select_categories(config, predicted)


def extract_date_range(message: str) -> tuple[str, str]:
    """Extract date bounds, using the previous seven days when no valid range is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    extracted: Dict[str, Any] = {}
    if api_key:
        today = datetime.now().date()
        prompt = (
            "Extract a date range. Return JSON only: "
            "{\"start_date\":\"YYYY-MM-DD or null\",\"end_date\":\"YYYY-MM-DD or null\"}.\n"
            f"Today is {today.isoformat()}.\n"
            f"If there is an implicit date range such as 'last week', then end_date is {today.isoformat()}.\n"
            f"If there is not any date intent in the message, then returns null for start_date and end_date.\n"
            f"Message: {message}"
        )
        try:
            response = OpenAI(api_key=api_key).responses.create(
                model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), input=prompt, temperature=0
            )
            extracted = json.loads((response.output_text or "").strip())
        except (Exception, json.JSONDecodeError):
            extracted = {}
    try:
        start = parse_iso_date(str(extracted.get("start_date")))
        end = parse_iso_date(str(extracted.get("end_date"))) if extracted.get("end_date") else datetime.now()
    except (TypeError, ValueError):
        end = datetime.now()
        start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if start > end:
        start, end = end, start
    return start.date().isoformat(), end.date().isoformat()


def build_specialist_agent(config: SpecialistConfig, tools: List[FunctionTool]) -> Agent:
    """Build a configured specialist Agent using the supplied SDK tools."""
    return Agent(
        name=config.name,
        instructions=(
            f"You may work only in these arXiv categories: {', '.join(config.categories)}.\n\n"
            "WHO you are:\n"
            f"- You work in {config.expertise_domain}"
            f"- You are {config.expertise}"
            f"- Your personality and communication style is {config.personality_and_communication_style}\n\n"
            "Tool workflow:\n"
            "1. Call `get_arxiv_categories_tool(message)` first. Pass the user's complete request as `message`. "
            "Use only its returned `categories`; never infer or request a category outside the allowed list.\n"
            "2. Call `check_paper_tool(start_date, end_date, categories)` before fetching. Use ISO dates "
            "(`YYYY-MM-DD`) and the categories returned by the category tool. If it returns an existing `csv_path`, "
            "reuse that path and do not fetch again.\n"
            "3. Only when no CSV is available, call `arxiv_fetcher_tool(start_date, end_date, categories)`. "
            "Pass the requested ISO date range and the inferred categories. Leave `min_threshold` at its default unless "
            "the user explicitly requests another threshold. If it fails because too few papers were found, do not "
            "invent topics or papers: report the reason to JuliusAgent.\n"
            "4. Call `find_topic_tool(csv_path, n_topics, n_papers_per_topic)` only with a CSV path returned by the "
            "check or fetch tool. Set `n_topics` to the number assigned by JuliusAgent, capped at five. Set "
            "`n_papers_per_topic` to the requested number of representative papers, or one when unspecified.\n"
            "5. Call `extract_main_result_tool(arxiv_id, title)` only when the user requests main results. Use the exact "
            "arXiv ID and title of each selected representative paper. Preserve a failure response rather than making "
            "up a result.\n\n"
            "After the tool workflow, return only the requested topics. Each topic must include its title, description, "
            "paper count, and representative papers. Include `main_result` only when it was requested. "
            f"Return valid JSON in this schema: {{\"{config.name}\": [{TOPIC_SCHEMA}]}}. "
            "Return exactly the requested number of topics when enough data exists, never more than five."
            f"The content of your output must reflect your personality and communication style"

        ),
        tools=tools,
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def run_specialist_agent(config: SpecialistConfig, builder: Callable[[], Agent], message: str) -> Dict[str, Any]:
    """Run one traced specialist-agent turn with normalized request context."""
    start_date, end_date = extract_date_range(message)
    categories = classify_categories(config, message)
    enriched_message = (
        f"{message}\n\nRequested date range: start_date={start_date}, end_date={end_date}. "
        f"Use categories={categories} when calling arxiv_fetcher_tool."
    )
    with trace(
        f"{config.slug}-agent-run",
        metadata={"agent": config.name, "start_date": start_date, "end_date": end_date, "categories": ",".join(categories)},
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        result = Runner.run_sync(builder(), enriched_message, max_turns=6)
    return {"reply": str(getattr(result, "final_output", "")), "tool_parameters": extract_tool_parameters(getattr(result, "new_items", []))}


def extract_tool_parameters(new_items: List[Any]) -> List[Dict[str, Any]]:
    """Extract function-call arguments from OpenAI Agents SDK run items."""
    extracted = []
    for item in new_items:
        raw = getattr(item, "raw_item", None)
        item_type = raw.get("type") if isinstance(raw, dict) else getattr(raw, "type", None)
        if item_type != "function_call":
            continue
        arguments = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        extracted.append({"tool": raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", None), "arguments": arguments})
    return extracted


def parse_iso_date(value: str) -> datetime:
    """Parse a required ISO date with a useful validation error."""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def save_papers_to_csv(papers: List[Paper], output_path: Path) -> None:
    """Persist paper metadata in the topic finder's CSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arxiv_id", "title", "authors", "summary", "published", "updated", "categories", "primary_category", "pdf_url", "entry_id", "comment", "journal_ref", "doi"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow({
                "arxiv_id": paper.arxiv_id, "title": paper.title, "authors": "; ".join(paper.authors),
                "summary": paper.summary, "published": paper.published.isoformat(), "updated": paper.updated.isoformat(),
                "categories": "; ".join(paper.categories), "primary_category": paper.primary_category,
                "pdf_url": paper.pdf_url, "entry_id": paper.entry_id, "comment": paper.comment or "",
                "journal_ref": paper.journal_ref or "", "doi": paper.doi or "",
            })


def _select_categories(config: SpecialistConfig, categories: Sequence[str]) -> List[str]:
    requested = set(categories or ())
    selected = [category for category in config.categories if category in requested]
    return selected or list(config.categories)


def _paper_csv_path(config: SpecialistConfig, categories: Sequence[str], start_date: str, end_date: str) -> Path:
    output_dir = Path(os.getenv("ARXIV_FETCH_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    return output_dir / f"arxiv_fetch_{config.slug}_{'_'.join(categories)}_{start_date}_to_{end_date}.csv"


def _main_result_failure(arxiv_id: str, title: str, reason: str) -> Dict[str, Any]:
    return {"status": "failure", "reason": reason, "arxiv_id": arxiv_id, "title": title, "main_result": None}


def _summarize_content(config: SpecialistConfig, arxiv_id: str, title: str, content: str, max_chars: int) -> str:
    chunks = _split_text(content, max_chars)
    summaries = [_summarize_with_llm(config, arxiv_id, title, chunk) for chunk in chunks]
    if len(summaries) == 1:
        return summaries[0]
    return _summarize_with_llm(config, arxiv_id, title, "\n\n".join(summaries))


def _split_text(text: str, chunk_size: int, overlap_ratio: float = 0.15) -> List[str]:
    overlap = max(0, min(int(chunk_size * overlap_ratio), chunk_size - 1))
    step = chunk_size - overlap
    return [text[index:index + chunk_size] for index in range(0, len(text), step)]


def _summarize_with_llm(config: SpecialistConfig, arxiv_id: str, title: str, content: str) -> str:
    prompt = (
        f"Paper ID: {arxiv_id}\nPaper title: {title}\nYou are a {config.expertise}.\n"
        "Extract the main theorem or result in one to three precise bullet points. Do not speculate.\n\n"
        f"Paper content:\n{content}"
    )
    response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).responses.create(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), input=prompt, temperature=0.2
    )
    result = (response.output_text or "").strip()
    if not result:
        raise ValueError("Empty model output")
    return result
