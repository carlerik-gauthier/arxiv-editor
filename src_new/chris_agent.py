"""Phase 1 ChrisAgent implementation with OpenAI Agents SDK."""

from __future__ import annotations

import os
import json
import csv
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, function_tool, trace
from openai import OpenAI

from src_new.arxiv_fetcher import ArxivFetcher
from src_new.data_object import Paper
from src_new.topic_finder import compute_topics


CHRIS_SYSTEM_PROMPT = (
    "Probability/statistics theory expert, focuses on stochastic processes. "
    "You identify key concepts and see application in other fields, such as physics"
)
CHRIS_CATEGORIES = ("math.PR", "math.ST")
DEFAULT_FETCH_THRESHOLD = 50
DEFAULT_MAX_RESULTS = 1000
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_OUTPUT_DIR = Path("data/paper")
DEFAULT_MAX_TOPICS = 5
DEFAULT_PDF_OUTPUT_DIR = Path("data/pdfs")

TOPIC_REPRESENTATION_DICT = """
    {
        'topic_title': <TOPIC TITLE>,
        'topic_description': <topic description>,
        'topic_count':  <topic number of papers>
        'representative_papers': [
            {'paper_title': <paper title>, 'paper_arxiv_id': <paper arxiv_id>, 'main_result': <representative paper main results>},
            {'paper_title': <paper title>, 'paper_arxiv_id': <paper arxiv_id>, 'main_result': <representative paper main results>}
            ]
    }\n
"""
TOPIC_REPRESENTATION_DICT_RULE="""
- The length of the list returned by representative_papers is equal to the number of representative papers.
- Mandatory fields are <TOPIC TITLE>, <topic count>, <topic description>, <paper title> and <paper arxiv_id>
"""
# EXPECTED_FORMAT_OUTPUT_RULE = """
#     For every topic, the expected output structure is:
#         # <TOPIC TITLE>: 
#         <topic description>
#         **Reprensative papers**
#         1. <paper title>, <paper arxiv_id>
#         <reprensative paper main results>
#         2. <paper title>, <paper arxiv_id>
#         <reprensative paper main results>
#     Repeat as many times as there are representative papers.
#     Mandatory elements are <TOPIC TITLE>, <topic description>, <paper title> and <paper arxiv_id>
#     They must be returned regardless of user request
# """

@function_tool(name_override="check_paper_tool")
def check_paper_tool(
    start_date: str,
    end_date: str,
    categories: List[str]
) -> Dict[str, Any]:
    "Use this tool to check if Probability Theory paper in the requested date range have already been collected"
    if len(categories)==0:
        categories = CHRIS_CATEGORIES
    output_dir = Path(os.getenv("ARXIV_FETCH_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    filename = (
        "arxiv_fetch_chris_"
        f"{'_'.join(categories)}_"
        f"{start_date}_to_{end_date}.csv"
    )
    output_path = output_dir / filename
    if os.path.exists(output_path):
        return {
            "status": "success",
            "existence": "Data is already available",
            "csv_path": output_path
        }
    else:
        return {
            "status": "success",
            "existence": "Data is not available. Need to fetch it",
            "csv_path": ''
        }

@function_tool(name_override="arxiv_fetcher_tool")
def arxiv_fetcher_tool(
    start_date: str,
    categories: List[str],
    end_date: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_threshold: int = DEFAULT_FETCH_THRESHOLD,
) -> Dict[str, Any]:
    """
    Fetch papers in ChrisAgent's allowed categories (math.PR, math.ST).

    Use this tool when there is no Probability Theory paper in the requested date range.
    If fetched paper count is below `min_threshold`, return no papers and explain why.
    """
    fetcher = ArxivFetcher()
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date) if end_date else datetime.now()
    categories_set = set(categories).intersection(CHRIS_CATEGORIES)
    if not categories_set:
        categories_set = set(CHRIS_CATEGORIES)
    papers: List[Paper] = []
    for category in categories_set:
        papers.extend(
            fetcher.fetch_by_category(
            category=category,
            start_date=start,
            end_date=end,
            max_results=max_results,
        )
        )
    
    deduped: Dict[str, Paper] = {paper.arxiv_id: paper for paper in papers}
    final_papers = list(deduped.values())

    categories_ordered = [category for category in CHRIS_CATEGORIES if category in categories_set]
    if len(final_papers) < min_threshold:
        return {
            "status": "failure",
            "reason": (
                "Not enough papers fetched to proceed: "
                f"{len(final_papers)} < threshold {min_threshold}."
            ),
            "categories": categories_ordered,
            "rows_saved": 0,
            "csv_path": None,
        }

    output_dir = Path(os.getenv("ARXIV_FETCH_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    filename = (
        "arxiv_fetch_chris_"
        f"{'_'.join(categories)}_"
        f"{start.date().isoformat()}_to_{end.date().isoformat()}.csv"
    )
    output_path = output_dir / filename
    try:
        _save_papers_to_csv(final_papers, output_path)
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Failed to save fetched papers to CSV: {exc}",
            "categories": categories_ordered,
            "rows_saved": 0,
            "csv_path": str(output_path),
        }

    return {
        "status": "success",
        "reason": f"Successfully fetched and saved {len(final_papers)} papers to CSV.",
        "categories": categories_ordered,
        "rows_saved": len(final_papers),
        "csv_path": str(output_path),
    }

@function_tool(name_override="find_topic_tool")
def find_topic_tool(
    csv_path: str,
    n_topics: int = DEFAULT_MAX_TOPICS,
    n_papers_per_topic: int = 3,
) -> Dict[str, Any]:
    """Extract top topics from a CSV file containing fetched paper metadata."""
    path = Path(csv_path)
    if not path.exists():
        return {
            "status": "failure",
            "reason": f"CSV file not found: {csv_path}",
            "topics": [],
        }

    n_topics = max(1, min(n_topics, DEFAULT_MAX_TOPICS))
    try:
        topics = compute_topics(
            path=str(path),
            n_topics=n_topics,
            n_papers_per_topic=max(1, n_papers_per_topic),
        )
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Topic extraction failed: {exc}",
            "topics": [],
        }

    return {
        "status": "success",
        "reason": f"Extracted {len(topics)} topics from {csv_path}.",
        "topics": topics,
    }


@function_tool(name_override="extract_main_result_tool")
def extract_main_result_tool(
    arxiv_id: str,
    title: str = "",
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """Download a paper content from ArXiv and explain its main results."""
    fetcher = ArxivFetcher()
    output_dir = Path(os.getenv("ARXIV_PDF_OUTPUT_DIR", str(DEFAULT_PDF_OUTPUT_DIR)))

    try:
        markdown = fetcher.fetch_paper_markdown(
            paper_id=arxiv_id,
            output_dir=str(output_dir),
            force_redownload=False,
        )
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Failed to download/extract paper {arxiv_id}: {exc}",
            "arxiv_id": arxiv_id,
            "main_result": None,
        }

    content = (markdown or "").strip()
    if not content:
        return {
            "status": "failure",
            "reason": f"No content extracted for paper {arxiv_id}, with title {title}.",
            "arxiv_id": arxiv_id,
            "title": title,
            "main_result": None,
        }

    try:
        effective_max_chars = max(1000, max_chars)
        if len(content) <= effective_max_chars:
            main_result = _extract_main_result_with_llm(arxiv_id=arxiv_id, title=title, content=content)
        else:
            chunks = _split_text_into_chunks(content, effective_max_chars)
            chunk_results: List[str] = []
            for idx, chunk in enumerate(chunks, start=1):
                chunk_result = _extract_main_result_with_llm(
                    arxiv_id=f"{arxiv_id} (chunk {idx}/{len(chunks)})",
                    title=f"{title} (chunk {idx}/{len(chunks)})",
                    content=chunk,
                )
                chunk_results.append(chunk_result)
            main_result = _synthesize_main_result_from_chunks(arxiv_id, title, chunk_results)
    except Exception as exc:
        return {
            "status": "failure",
            "reason": f"Main-result extraction failed for {arxiv_id}: {exc}",
            "arxiv_id": arxiv_id,
            "title": title,
            "main_result": None,
        }

    return {
        "status": "success",
        "reason": f"Extracted main result for paper {arxiv_id}, with title {title}.",
        "arxiv_id": arxiv_id,
        "title": title,
        "main_result": main_result,
    }


@function_tool(name_override="get_arxiv_categories_tool")
def get_arxiv_categories_tool(message: str) -> Dict[str, Any]:
    """Tool wrapper for arXiv category inference."""
    return {
        "categories": _get_arxiv_categories(message),
    }



def build_chris_agent() -> Agent:
    """Create ChrisAgent for Phase 1."""
    return Agent(
        name="ChrisAgent",
        instructions=(
            f"{CHRIS_SYSTEM_PROMPT}\n"
            f"Allowed categories only: {', '.join(CHRIS_CATEGORIES)}.\n"
            "Use `get_arxiv_categories_tool` to get the Arxiv categories to invoke from the user message.\n"
            "Use `check_paper_tool` to verify whether CSV data already exists for the requested date range.\n"
            "Use `arxiv_fetcher_tool` if data does not exist. You must only look at inferred categories. \n"
            "Use `find_topic_tool` only after papers were fetched and a CSV path is available.\n"
            "Use `extract_main_result_tool` when full-paper main results are requested, passing arxiv_id.\n"
            f"When you answer back to JuliusAgent, you **must** return a JSON with this exact schema:"
            f"{{'ChrisAgent': [{TOPIC_REPRESENTATION_DICT}]}}\n"
            "The length of the list to return is equal to the number of requested topics\n"
            f"The rules of the topic representation dictionary are : {TOPIC_REPRESENTATION_DICT_RULE}"
        ),
        tools=[
            get_arxiv_categories_tool,
            check_paper_tool,
            arxiv_fetcher_tool,
            find_topic_tool,
            extract_main_result_tool,
        ],
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )


def run_chris_agent(message: str) -> Dict[str, Any]:
    """Run one ChrisAgent turn with SDK tracing enabled."""
    start_date, end_date = _extract_date_range(message)
    categories = _get_arxiv_categories(message)
    enriched_message = (
        f"{message}\n\n"
        f"Requested date range to use if needed: start_date={start_date}, end_date={end_date}.\n"
        f"Use categories={categories} when calling arxiv_fetcher_tool."
    )

    agent = build_chris_agent()
    with trace(
        "phase3-chris-agent-run",
        metadata={
            "agent": "ChrisAgent",
            "start_date": start_date,
            "end_date": end_date,
            "categories": ",".join(categories),
        },
        disabled=os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "0") == "1",
    ):
        result = Runner.run_sync(agent, enriched_message, max_turns=6)

    return {
        "reply": str(getattr(result, "final_output", "")),
        "tool_parameters": _extract_tool_parameters(getattr(result, "new_items", [])),
    }


def _extract_tool_parameters(new_items: List[Any]) -> List[Dict[str, Any]]:
    """Extract function-tool call arguments from SDK run items."""
    extracted: List[Dict[str, Any]] = []
    for item in new_items:
        raw_item = getattr(item, "raw_item", None)
        if raw_item is None:
            continue

        item_type = raw_item.get("type") if isinstance(raw_item, dict) else getattr(raw_item, "type", None)
        if item_type != "function_call":
            continue

        tool_name = raw_item.get("name") if isinstance(raw_item, dict) else getattr(raw_item, "name", None)
        arguments = raw_item.get("arguments") if isinstance(raw_item, dict) else getattr(raw_item, "arguments", None)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass

        extracted.append(
            {
                "tool": tool_name,
                "arguments": arguments,
            }
        )
    return extracted


def _extract_date_range(message: str) -> tuple[str, str]:
    extracted = _extract_date_range_with_llm(message)
    if extracted:
        start_raw = extracted.get("start_date")
        end_raw = extracted.get("end_date")
        try:
            start = _parse_iso_date(str(start_raw))
            end = _parse_iso_date(str(end_raw)) if end_raw else datetime.now()
        except (ValueError, TypeError):
            pass
        else:
            if start > end:
                start, end = end, start
            return start.date().isoformat(), end.date().isoformat()
    end = datetime.now()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start.date().isoformat(), end.date().isoformat()


def _extract_date_range_with_llm(message: str) -> Dict[str, Optional[str]]:
    """Use an LLM to extract ISO date bounds from a user message."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    today = datetime.now().date()
    day_name = today.strftime("%A")
    month_name = today.strftime("%B")
    prompt = (
        "Extract an explicit date range from the user message.\n"
        "Return JSON only with this exact schema: "
        "{\"start_date\":\"YYYY-MM-DD or null\",\"end_date\":\"YYYY-MM-DD or null\"}.\n"
        f"Today is {day_name}, {month_name} {today.day}, {today.year}"
        "Rule:\n"
        f"-if there is a relative date, then end_date is {today.isoformat()}\n"
        "-if there is not any date intent in the message, then returns null for start_date and end_date\n"
        f"User message: {message}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.05,
    )
    content = (response.output_text or "").strip()
    if not content:
        return {
            "start_date": None,
            "end_date": None,
        }

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {
            "start_date": None,
            "end_date": None,
        }

    return {
        "start_date": payload.get("start_date"),
        "end_date": payload.get("end_date"),
    }

def _classify_categories_with_llm(message: str) -> List[str]:
    """Use an LLM to classify the requested ChrisAgent categories."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = (
        "Classify the user's message into arXiv categories from this allowed set only: "
        f"{list(CHRIS_CATEGORIES)}.\n"
        "Return JSON only with this schema: {\"categories\": <list of predicted categories>}\n" # [<category_prediction>]}\n"
        "Information:\n"
        # "- Include a category only if clearly requested or strongly implied.\n"
        # # "- If neither is requested, return both categories.\n"
        "- 'math.PR' contains papers in proability.\n"
        "- 'math.ST' contains papers in statistics.\n"
        # "- Never return any other category.\n"
        f"User message: {message}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    
    content = (response.output_text or "").strip()
    print(content)
    payload = json.loads(content)
    raw_categories = payload.get("categories", [])
    if not isinstance(raw_categories, list):
        return []
    return [str(category) for category in raw_categories]


def _get_arxiv_categories(message: str) -> List[str]:
    """
    Infer ChrisAgent categories from user message.

    Returns only categories allowed to ChrisAgent and defaults to all allowed
    categories when no explicit/implicit signal is found.
    """
    try:
        raw_categories = _classify_categories_with_llm(message)
    except Exception:
        raw_categories = []

    normalized = [category for category in CHRIS_CATEGORIES if category in raw_categories]
    if not normalized:
        return list(CHRIS_CATEGORIES)
    return normalized

def _save_papers_to_csv(papers: List[Paper], output_path: Path) -> None:
    """Save fetched papers as CSV rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arxiv_id",
        "title",
        "authors",
        "summary",
        "published",
        "updated",
        "categories",
        "primary_category",
        "pdf_url",
        "entry_id",
        "comment",
        "journal_ref",
        "doi",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            writer.writerow(
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": "; ".join(paper.authors),
                    "summary": paper.summary,
                    "published": paper.published.isoformat(),
                    "updated": paper.updated.isoformat(),
                    "categories": "; ".join(paper.categories),
                    "primary_category": paper.primary_category,
                    "pdf_url": paper.pdf_url,
                    "entry_id": paper.entry_id,
                    "comment": paper.comment or "",
                    "journal_ref": paper.journal_ref or "",
                    "doi": paper.doi or "",
                }
            )

# def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
#     data = asdict(paper)
#     data["published"] = paper.published.isoformat()
#     data["updated"] = paper.updated.isoformat()
#     return data


def _parse_iso_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc
    
def _split_text_into_chunks(text: str, chunk_size: int, overlap_ratio: float = 0.15) -> List[str]:
    """Split text into overlapping chunks to preserve context across boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = int(chunk_size * overlap_ratio)
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += step
    return chunks


def _synthesize_main_result_from_chunks(arxiv_id: str, title: str, chunk_results: List[str]) -> str:
    """Combine per-chunk summaries into one final main-result summary."""
    if not chunk_results:
        raise ValueError("No chunk results to synthesize")
    combined = "\n\n".join(
        f"Chunk {idx} summary:\n{result}" for idx, result in enumerate(chunk_results, start=1)
    )
    return _extract_main_result_with_llm(
        arxiv_id=f"{arxiv_id} (final synthesis)",
        title=f"{title} (final synthesis)",
        content=combined,
    )


def _extract_main_result_with_llm(arxiv_id: str, title: str, content: str) -> str:
    """Use the configured OpenAI model to summarize the paper's core contribution."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = (
        f"Paper ID: {arxiv_id}\n"
        f"Paper title: {title}\n"
        "You are a probability/statistics expert.\n"
        "From the paper content below, extract the main theorem or main result and explain in 1-3 concise bullet points why it matters.\n"
        "Be precise and you must not make speculation.\n\n"
        f"Paper content:\n{content}"
    )
    response = client.responses.create(model=model, input=prompt, temperature=0.1)
    text = (response.output_text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    return text
