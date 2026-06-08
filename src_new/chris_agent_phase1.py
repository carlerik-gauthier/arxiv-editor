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

CHRIS_SYSTEM_PROMPT = (
    "Probability theory expert, focuses on stochastic processes. "
    "You identify key concepts and see application in other fields, such as physics"
)
CHRIS_CATEGORIES = ("math.PR", "math.ST")
DEFAULT_FETCH_THRESHOLD = 50
DEFAULT_MAX_RESULTS = 1000
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_OUTPUT_DIR = Path("data/paper")

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



def build_chris_agent() -> Agent:
    """Create ChrisAgent for Phase 1."""
    return Agent(
        name="ChrisAgent",
        instructions=(
            f"{CHRIS_SYSTEM_PROMPT}\n"
            f"Allowed categories only: {', '.join(CHRIS_CATEGORIES)}.\n"
            "To check if robability Theory nor Statistical Theory paper have already beed collected in the requested date range"
            "call `check_paper_tool`"
            "When no Probability Theory nor Statistical Theory paper exists in the requested date range, call "
            "`arxiv_fetcher_tool`." 
        ),
        tools=[check_paper_tool, arxiv_fetcher_tool],
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )


def run_chris_agent(message: str) -> Dict[str, Any]:
    """Run one ChrisAgent turn with SDK tracing enabled."""
    start_date, end_date = _extract_date_range(message) 
    # end_date = datetime.now().date().isoformat()
    # start_date = (datetime.now()-timedelta(days=6)).date().isoformat()
    # categories = CHRIS_CATEGORIES # 
    categories = _get_arxiv_categories(message)
    enriched_message = (
        f"{message}\n\n"
        f"Requested date range to use if needed: start_date={start_date}, end_date={end_date}.\n"
        f"Use categories={categories} when calling arxiv_fetcher_tool."
    )

    agent = build_chris_agent()
    with trace(
        "phase1-chris-agent-run",
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
    # return str(getattr("result_final", ""))

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
    # content = (response.output_text or "{}").strip()
    # payload = json.loads(content)

    content = response.output_text # choices[0].message.content
    print(content)

    if not content or not content.strip():
        return {
        "start_date": None,
        "end_date": None
        }

    content = content.strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {
        "start_date": None,
        "end_date": None
        }
    print(f"start_date: {payload.get("start_date")}; end_date: {payload.get("end_date")}" )
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

    normalized = {category for category in raw_categories if category in CHRIS_CATEGORIES}
    if not normalized:
        return list(CHRIS_CATEGORIES)
    return list(normalized)

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
    
