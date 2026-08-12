"""Shared workflow primitives for arXiv specialist agents."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agents import Agent, Runner, function_tool, trace
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
DEFAULT_MODEL = "gpt-5.4-nano"
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

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

    @property
    def system_prompt(self) -> str:
        """Return this specialist's stable personality guidance.

        The complete runtime instructions are built by
        :func:`build_specialist_agent`; exposing this field keeps the reusable
        configuration easy to inspect without duplicating prompt text.

        Returns:
            str: Personality and communication guidance for this specialist.
        """
        return self.personality_and_communication_style


@dataclass(frozen=True)
class SpecialistToolSet:
    """The configuration-bound SDK tools shared by every specialist agent."""

    check_paper_tool: FunctionTool
    arxiv_fetcher_tool: FunctionTool
    find_topic_tool: FunctionTool
    extract_main_result_tool: FunctionTool
    get_arxiv_categories_tool: FunctionTool

    def as_list(self) -> List[FunctionTool]:
        """Return the tools in their documented workflow order.

        Returns:
            List[FunctionTool]: Category, cache-check, fetch, topic, and
            main-result tools in the order agents should call them.
        """
        return [
            self.get_arxiv_categories_tool,
            self.check_paper_tool,
            self.arxiv_fetcher_tool,
            self.find_topic_tool,
            self.extract_main_result_tool,
        ]


def create_specialist_tools(config: SpecialistConfig) -> SpecialistToolSet:
    """Create shared OpenAI Agents SDK tools bound to one specialist.

    Each specialist has distinct categories and expertise but follows the same
    data workflow. This factory prevents seven copies of equivalent tool code.

    Args:
        config: Domain-specific configuration captured by every created tool.

    Returns:
        SpecialistToolSet: Config-bound tools for category selection, cache
        checks, fetching, topic extraction, and result summaries.
    """

    @function_tool(name_override="check_paper_tool")
    def check_paper_tool(start_date: str, end_date: str, categories: List[str]) -> Dict[str, Any]:
        """Check whether metadata is already cached for a request.

        Args:
            start_date: Inclusive request start date in ISO 8601 format.
            end_date: Inclusive request end date in ISO 8601 format.
            categories: Requested arXiv categories to include in the cache key.

        Returns:
            Dict[str, Any]: Cache status and existing CSV path when available.
        """
        return check_papers(config, start_date, end_date, categories)

    @function_tool(name_override="arxiv_fetcher_tool")
    def arxiv_fetcher_tool(
        start_date: str,
        categories: List[str],
        end_date: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        min_threshold: int = DEFAULT_FETCH_THRESHOLD,
    ) -> Dict[str, Any]:
        """Fetch permitted papers and persist their metadata for topic analysis.

        Args:
            start_date: Inclusive request start date in ISO 8601 format.
            categories: Requested arXiv categories, restricted to this specialist.
            end_date: Inclusive end date; defaults to the current time.
            max_results: Maximum API results to request per category.
            min_threshold: Minimum unique-paper count required before saving.

        Returns:
            Dict[str, Any]: Success or failure status, selected categories, row
            count, reason, and CSV path when data is saved.
        """
        return fetch_papers(config, start_date, categories, end_date, max_results, min_threshold)

    @function_tool(name_override="find_topic_tool")
    def find_topic_tool(
        csv_path: str,
        personality: str,
        communication_style: str,
        n_topics: int = DEFAULT_MAX_TOPICS,
        n_papers_per_topic: int = 3,
    ) -> Dict[str, Any]:
        """Find representative research topics in a paper-metadata CSV.

        Args:
            csv_path: Path to specialist paper metadata in CSV form.
            personality: A brief description of the personality to use.
            communication_style: The style to use to generate the description.
            n_topics: Maximum number of topics to return.
            n_papers_per_topic: Maximum representative papers per topic.

        Returns:
            Dict[str, Any]: Success or failure payload with extracted topics.
        """
        return find_topics(csv_path, personality, communication_style, n_topics, n_papers_per_topic)

    @function_tool(name_override="extract_main_result_tool")
    def extract_main_result_tool(
        arxiv_id: str,
        title: str = "",
        max_chars: int = 12000,
    ) -> Dict[str, Any]:
        """Download one paper and explain its reported principal result.

        Args:
            arxiv_id: ArXiv identifier of the paper to process.
            title: Optional title retained in result metadata and prompts.
            max_chars: Maximum characters per summarization chunk.

        Returns:
            Dict[str, Any]: Success or recoverable failure payload containing
            the paper identity and main-result summary.
        """
        return extract_main_result(config, arxiv_id, title, max_chars)

    @function_tool(name_override="get_arxiv_categories_tool")
    def get_arxiv_categories_tool(message: str) -> Dict[str, List[str]]:
        """Infer allowed arXiv categories relevant to a user request.

        Args:
            message: User request used to select the specialist's categories.

        Returns:
            Dict[str, List[str]]: Mapping containing the selected categories.
        """
        return {"categories": classify_categories(config, message)}

    return SpecialistToolSet(
        check_paper_tool=check_paper_tool,
        arxiv_fetcher_tool=arxiv_fetcher_tool,
        find_topic_tool=find_topic_tool,
        extract_main_result_tool=extract_main_result_tool,
        get_arxiv_categories_tool=get_arxiv_categories_tool,
    )


def check_papers(config: SpecialistConfig, start_date: str, end_date: str, categories: Sequence[str]) -> Dict[str, Any]:
    """Check the deterministic paper-metadata cache for a request.

    Args:
        config: Specialist whose cache namespace and categories are used.
        start_date: Inclusive start date used in the cache key.
        end_date: Inclusive end date used in the cache key.
        categories: Requested categories, filtered to the specialist's allowlist.

    Returns:
        Dict[str, Any]: Success payload identifying whether the expected CSV
        exists and, if so, its path.
    """
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
    """Fetch, deduplicate, and persist a specialist's paper metadata.

    Args:
        config: Specialist configuration that constrains categories and storage.
        start_date: Inclusive fetch start date in ISO 8601 format.
        categories: Requested categories, filtered to the specialist allowlist.
        end_date: Optional inclusive end date; defaults to the current time.
        max_results: Maximum API results requested for each category.
        min_threshold: Minimum number of unique papers required for persistence.

    Returns:
        Dict[str, Any]: Success or recoverable failure payload with selected
        categories, row count, reason, and CSV path.

    Raises:
        ValueError: If an explicitly supplied date is not in ISO 8601 format.
        Exception: If the arXiv client fails while fetching a category.
    """
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


def find_topics(csv_path: str,
                personality: str,
                communication_style: str,
                n_topics: int = DEFAULT_MAX_TOPICS,
                n_papers_per_topic: int = 3
                ) -> Dict[str, Any]:
    """Extract up to five topics from persisted paper metadata.

    Args:
        csv_path: Existing paper-metadata CSV to analyze.
        personality: A brief description of the personality to use.
        communication_style: The style to use to generate the description.
        n_topics: Requested maximum topic count, capped at five.
        n_papers_per_topic: Maximum representative papers per topic.

    Returns:
        Dict[str, Any]: Success payload with topics or a recoverable failure
        payload when the CSV is absent or topic extraction fails.
    """
    if personality.strip() == '':
        personality = "professional"
    if communication_style.strip() == '':
        communication_style = "clear and concise"
    path = Path(csv_path)
    if not path.exists():
        return {"status": "failure", "reason": f"CSV file not found: {csv_path}", "topics": []}
    try:
        topics = compute_topics(str(path), personality, communication_style, max(1, min(n_topics, DEFAULT_MAX_TOPICS)), max(1, n_papers_per_topic))
    except Exception as exc:
        return {"status": "failure", "reason": f"Topic extraction failed: {exc}", "topics": []}
    return {"status": "success", "reason": f"Extracted {len(topics)} topics from {csv_path}.", "topics": topics}


def extract_main_result(config: SpecialistConfig, arxiv_id: str, title: str = "", max_chars: int = 12000) -> Dict[str, Any]:
    """Download a paper and summarize its principal result with the model.

    Args:
        config: Specialist configuration used to ground the summarization prompt.
        arxiv_id: ArXiv identifier of the paper to process.
        title: Optional paper title retained in output and model prompts.
        max_chars: Minimum-bounded maximum characters per summary chunk.

    Returns:
        Dict[str, Any]: Success payload with the main result or recoverable
        failure payload with the paper identity and reason.
    """
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
    """Infer relevant allowed categories, defaulting to the full specialty.

    Args:
        config: Specialist configuration defining the permitted categories.
        message: User request used to choose among permitted categories.

    Returns:
        List[str]: Valid categories inferred from the request, or all permitted
        categories when inference is unavailable or invalid.
    """
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
    """Extract date bounds or fall back to the previous seven calendar days.

    Args:
        message: User request containing explicit or relative date intent.

    Returns:
        tuple[str, str]: Inclusive ISO 8601 start and end dates.
    """
    explicit_dates = ISO_DATE_PATTERN.findall(message)
    if explicit_dates:
        start = parse_iso_date(explicit_dates[0])
        end = parse_iso_date(explicit_dates[1]) if len(explicit_dates) > 1 else start
        if start > end:
            start, end = end, start
        return start.date().isoformat(), end.date().isoformat()

    api_key = os.getenv("OPENAI_API_KEY")
    extracted: Dict[str, Any] = {}
    if api_key:
        today = datetime.now().date()
        prompt = (
            "Read ** very carefully** the message and extract a date range. Return JSON only: "
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
    """Build a specialist agent with the supplied configuration and SDK tools.

    Args:
        config: Specialist identity, expertise, categories, and communication
            guidance.
        tools: Configuration-bound OpenAI Agents SDK tools available to it.

    Returns:
        Agent: Configured specialist agent ready for execution.
    """
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
            "4. Call `find_topic_tool(csv_path, personalit, communication_style, n_topics, n_papers_per_topic)` only"
            "with a CSV path returned by the check or fetch tool. Set `n_topics` to the number assigned by JuliusAgent," 
            "capped at five. Provide your personality to 'personality' and precise your communication style with 'communication_style'."
            "Set `n_papers_per_topic` to the requested number of representative papers, or one when unspecified.\n"
            "5. Call `extract_main_result_tool(arxiv_id, title)` only when the user requests main results. Use the exact "
            "arXiv ID and title of each selected representative paper. Preserve a failure response rather than making "
            "up a result.\n\n"
            "After the tool workflow, return only the requested topics. Each topic must include its title, description, "
            "paper count, and representative papers. Include `main_result` only when it was requested. "
            f"Return valid JSON in this schema: {{\"{config.name}\": [{TOPIC_SCHEMA}]}}. "
            "Return exactly the requested number of topics when enough data exists, never more than five. "
            "The style used for topic title, topic description and main results must meet your personality and communication style."

        ),
        tools=tools,
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def run_specialist_agent(config: SpecialistConfig, builder: Callable[[], Agent], message: str) -> Dict[str, Any]:
    """Run one traced specialist turn with normalized date and category context.

    Args:
        config: Specialist configuration used to infer dates and categories.
        builder: Zero-argument factory that returns the configured specialist.
        message: User request to enrich and send to the specialist.

    Returns:
        Dict[str, Any]: Agent reply and parameters passed to invoked tools.

    Raises:
        Exception: If the OpenAI Agents SDK cannot complete the agent run.
    """
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
    """Extract function-call names and arguments from SDK run items.

    Args:
        new_items: Items emitted during an OpenAI Agents SDK run.

    Returns:
        List[Dict[str, Any]]: One tool-and-arguments mapping per function call.
    """
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
    """Parse one required ISO 8601 date into a datetime value.

    Args:
        value: Date string expected in ``YYYY-MM-DD`` ISO 8601 form.

    Returns:
        datetime: Parsed datetime value.

    Raises:
        ValueError: If ``value`` is absent or not a valid ISO 8601 date.
    """
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def save_papers_to_csv(papers: List[Paper], output_path: Path) -> None:
    """Persist paper metadata in the topic finder's CSV format.

    Args:
        papers: Paper objects to serialize as metadata rows.
        output_path: Destination CSV path; parent directories are created.

    Returns:
        None: The metadata is written to ``output_path``.

    Raises:
        OSError: If the output directory or CSV cannot be created or written.
    """
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
    """Restrict requested categories to a specialist's permitted set.

    Args:
        config: Specialist configuration that defines permitted categories.
        categories: Requested category codes, possibly empty or invalid.

    Returns:
        List[str]: Requested permitted categories, or all permitted categories
        when none of the requested values is valid.
    """
    requested = set(categories or ())
    selected = [category for category in config.categories if category in requested]
    return selected or list(config.categories)


def _paper_csv_path(config: SpecialistConfig, categories: Sequence[str], start_date: str, end_date: str) -> Path:
    """Build the deterministic cache path for a specialist data request.

    Args:
        config: Specialist whose slug namespaces the cache file.
        categories: Selected categories included in the filename.
        start_date: Inclusive start date included in the filename.
        end_date: Inclusive end date included in the filename.

    Returns:
        Path: Expected CSV location under the configured output directory.
    """
    output_dir = Path(os.getenv("ARXIV_FETCH_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    return output_dir / f"arxiv_fetch_{config.slug}_{'_'.join(categories)}_{start_date}_to_{end_date}.csv"


def _main_result_failure(arxiv_id: str, title: str, reason: str) -> Dict[str, Any]:
    """Create a consistent recoverable main-result failure payload.

    Args:
        arxiv_id: Identifier of the paper that could not be processed.
        title: Optional paper title retained for caller context.
        reason: Human-readable explanation of the failure.

    Returns:
        Dict[str, Any]: Failure status, paper identity, reason, and null result.
    """
    return {"status": "failure", "reason": reason, "arxiv_id": arxiv_id, "title": title, "main_result": None}


def _summarize_content(config: SpecialistConfig, arxiv_id: str, title: str, content: str, max_chars: int) -> str:
    """Summarize long paper content in chunks before final synthesis.

    Args:
        config: Specialist configuration used to set model expertise context.
        arxiv_id: Paper identifier included in every model prompt.
        title: Paper title included in every model prompt.
        content: Extracted paper text to summarize.
        max_chars: Maximum number of characters in each overlapping chunk.

    Returns:
        str: One summary or a final synthesis of chunk summaries.

    Raises:
        ValueError: If a model call returns empty output.
        Exception: If the model client cannot produce a summary.
    """
    chunks = _split_text(content, max_chars)
    summaries = [_summarize_with_llm(config, arxiv_id, title, chunk) for chunk in chunks]
    if len(summaries) == 1:
        return summaries[0]
    return _summarize_with_llm(config, arxiv_id, title, "\n\n".join(summaries))


def _split_text(text: str, chunk_size: int, overlap_ratio: float = 0.15) -> List[str]:
    """Split text into overlapping chunks while preserving context.

    Args:
        text: Text to partition.
        chunk_size: Maximum characters included in each chunk.
        overlap_ratio: Fraction of each chunk repeated in the next chunk.

    Returns:
        List[str]: Consecutive chunks, each up to ``chunk_size`` characters.

    Raises:
        ValueError: If ``chunk_size`` produces a non-positive range step.
    """
    overlap = max(0, min(int(chunk_size * overlap_ratio), chunk_size - 1))
    step = chunk_size - overlap
    return [text[index:index + chunk_size] for index in range(0, len(text), step)]


def _summarize_with_llm(config: SpecialistConfig, arxiv_id: str, title: str, content: str) -> str:
    """Ask the configured model for a grounded paper-result summary.

    Args:
        config: Specialist configuration that supplies subject expertise.
        arxiv_id: Paper identifier included in the summary prompt.
        title: Paper title included in the summary prompt.
        content: Paper text or intermediate summaries to synthesize.

    Returns:
        str: Non-empty model-generated main-result summary.

    Raises:
        ValueError: If the model returns no usable text.
        Exception: If the OpenAI client cannot complete the request.
    """
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
