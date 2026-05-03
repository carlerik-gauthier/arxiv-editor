"""Paper analysis helpers for extracting research problems from full text."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_MAX_CHUNK_TOKENS = 1800
DEFAULT_SECTION_SCAN_CHARS = 20000


class PaperAnalyzer:
    """
    Analyze full paper text and metadata for editorial agent workflows.

    The analyzer is intentionally provider-agnostic. When `llm_client` is
    supplied, problem extraction calls a common completion/generation method and
    normalizes the response. Without an LLM, the same public contract is filled
    by deterministic section and keyword heuristics so tests and local workflows
    remain stable.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    ) -> None:
        """
        Initialize a paper analyzer.

        Args:
            llm_client: Optional callable or client object exposing
                `complete`, `generate`, `chat`, or `invoke`.
            max_chunk_tokens: Approximate token budget used by `chunk_text`.

        Raises:
            ValueError: If `max_chunk_tokens` is less than 100.
        """
        if max_chunk_tokens < 100:
            raise ValueError("max_chunk_tokens must be at least 100")
        self.llm_client = llm_client
        self.max_chunk_tokens = max_chunk_tokens

    def extract_problem_statement(
        self,
        paper_text: str,
        paper_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract the problem, motivation, research gap, and context from a paper.

        Args:
            paper_text: Full paper text or a substantial excerpt.
            paper_metadata: Optional title/authors/categories/summary metadata.

        Returns:
            A structured dictionary with `problem`, `motivation`,
            `research_gap`, `context`, `evidence`, `confidence`, `source`,
            `sections_used`, and `chunks_analyzed`.

        Raises:
            ValueError: If `paper_text` is empty.
        """
        normalized_text = _normalize_text(paper_text, parameter_name="paper_text")
        metadata = dict(paper_metadata or {})
        sections = self.extract_sections(normalized_text)
        chunks = self.chunk_text(_analysis_text(sections, normalized_text))

        if self.llm_client is not None:
            try:
                result = self._extract_problem_statement_with_llm(
                    chunks=chunks,
                    paper_metadata=metadata,
                    sections=sections,
                )
                result.setdefault("source", "llm")
                result.setdefault("confidence", "llm")
                result.setdefault("sections_used", _non_empty_section_names(sections))
                result.setdefault("chunks_analyzed", len(chunks))
                return _normalize_problem_result(result, metadata, sections, len(chunks))
            except Exception as exc:
                fallback = self._extract_problem_statement_heuristic(
                    normalized_text,
                    metadata,
                    sections,
                    chunks,
                )
                fallback["source"] = "heuristic_fallback"
                fallback["llm_error"] = str(exc)
                return fallback

        return self._extract_problem_statement_heuristic(
            normalized_text,
            metadata,
            sections,
            chunks,
        )

    def extract_sections(self, paper_text: str) -> Dict[str, str]:
        """
        Identify common paper sections in normalized text.

        The parser recognizes headings such as abstract, introduction,
        background, related work, method, results, discussion, conclusion, and
        references. It is deliberately conservative and returns partial sections
        rather than failing when formatting is unusual.

        Args:
            paper_text: Full paper text or excerpt.

        Returns:
            Mapping from canonical section names to section text. Missing
            sections are omitted.
        """
        text = _normalize_text(paper_text, parameter_name="paper_text")
        scanned_text = text[:DEFAULT_SECTION_SCAN_CHARS]
        matches = list(_iter_section_headings(scanned_text))
        sections: Dict[str, str] = {}

        for index, match in enumerate(matches):
            section_name = _canonical_section_name(match.group("heading"))
            if section_name is None:
                continue
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(scanned_text)
            section_text = scanned_text[start:end].strip(" .\n\t")
            if section_text:
                sections[section_name] = _clean_section_text(section_text)

        if "abstract" not in sections:
            abstract = _extract_inline_abstract(scanned_text)
            if abstract:
                sections["abstract"] = abstract
        if not sections:
            sections["body"] = scanned_text
        return sections

    def chunk_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Split text into approximate token chunks for LLM processing.

        Args:
            text: Input text to split.
            max_tokens: Optional override for approximate tokens per chunk.

        Returns:
            Ordered chunks. Approximation uses words as tokens, which is stable
            enough for prompt sizing without depending on a tokenizer package.

        Raises:
            ValueError: If `max_tokens` is less than 100.
        """
        normalized_text = _normalize_text(text, parameter_name="text")
        token_limit = max_tokens or self.max_chunk_tokens
        if token_limit < 100:
            raise ValueError("max_tokens must be at least 100")

        words = normalized_text.split()
        if len(words) <= token_limit:
            return [normalized_text]

        chunks: List[str] = []
        for start in range(0, len(words), token_limit):
            chunks.append(" ".join(words[start : start + token_limit]))
        return chunks

    def _extract_problem_statement_with_llm(
        self,
        chunks: Sequence[str],
        paper_metadata: Dict[str, Any],
        sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """Call the injected LLM client and parse its structured response."""
        prompt = _build_problem_prompt(
            chunks=chunks[:3],
            paper_metadata=paper_metadata,
            sections=sections,
        )
        response = _call_llm_client(self.llm_client, prompt)
        return _parse_llm_problem_response(response)

    def _extract_problem_statement_heuristic(
        self,
        paper_text: str,
        paper_metadata: Dict[str, Any],
        sections: Dict[str, str],
        chunks: Sequence[str],
    ) -> Dict[str, Any]:
        """Extract problem fields using deterministic section and keyword rules."""
        abstract = sections.get("abstract", "")
        introduction = sections.get("introduction", "")
        related_work = sections.get("related_work", "")
        body = " ".join(
            part
            for part in [
                paper_metadata.get("summary", ""),
                abstract,
                introduction,
                related_work,
                paper_text[:4000],
            ]
            if part
        )
        sentences = _split_sentences(body)

        problem = _first_matching_sentence(
            sentences,
            (
                "address",
                "problem",
                "question",
                "challenge",
                "we study",
                "we investigate",
                "we consider",
                "aim",
            ),
        )
        motivation = _first_matching_sentence(
            sentences,
            (
                "important",
                "motivat",
                "applications",
                "central",
                "fundamental",
                "practical",
                "need",
            ),
        )
        research_gap = _first_matching_sentence(
            sentences,
            (
                "however",
                "open",
                "unknown",
                "limited",
                "lack",
                "gap",
                "not well understood",
                "remain",
            ),
        )

        if not problem:
            problem = _metadata_problem_fallback(paper_metadata, sentences)
        if not motivation:
            motivation = abstract or introduction or problem
        if not research_gap:
            research_gap = _infer_gap_from_problem(problem)

        evidence = [
            sentence
            for sentence in [problem, motivation, research_gap]
            if sentence
        ]
        return _normalize_problem_result(
            {
                "problem": problem,
                "motivation": motivation,
                "research_gap": research_gap,
                "context": _build_context(paper_metadata, sections),
                "evidence": evidence,
                "confidence": "heuristic",
                "source": "heuristic",
                "sections_used": _non_empty_section_names(sections),
                "chunks_analyzed": len(chunks),
            },
            paper_metadata,
            sections,
            len(chunks),
        )


def _normalize_text(text: str, parameter_name: str) -> str:
    """Validate text input and normalize whitespace."""
    if not isinstance(text, str):
        raise TypeError(f"{parameter_name} must be a string")
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError(f"{parameter_name} cannot be empty")
    return normalized


def _iter_section_headings(text: str) -> Iterable[re.Match[str]]:
    """Yield regex matches for likely section headings."""
    pattern = re.compile(
        r"(?:^|\s(?P<number>\d+\.?\s+))(?P<heading>"
        r"abstract|introduction|background|related work|preliminaries|"
        r"methods?|methodology|approach|experiments?|results?|discussion|"
        r"conclusions?|references"
        r")\b\s*[:.]?",
        flags=re.IGNORECASE,
    )
    return pattern.finditer(text)


def _canonical_section_name(heading: str) -> Optional[str]:
    """Map a raw heading to a canonical section name."""
    normalized = heading.lower().strip()
    if normalized == "references":
        return None
    if normalized in {"method", "methods", "methodology", "approach"}:
        return "methods"
    if normalized in {"experiment", "experiments"}:
        return "experiments"
    if normalized in {"result", "results"}:
        return "results"
    if normalized in {"conclusion", "conclusions"}:
        return "conclusion"
    if normalized == "related work":
        return "related_work"
    return normalized.replace(" ", "_")


def _clean_section_text(text: str) -> str:
    """Normalize section text and remove leading numbering artifacts."""
    return re.sub(r"^\d+\.?\s*", "", _normalize_text(text, "section_text"))


def _extract_inline_abstract(text: str) -> str:
    """Extract an abstract when it appears inline without a clear heading boundary."""
    match = re.search(
        r"abstract\s*[:.]?\s+(?P<abstract>.+?)(?:\s+\d+\.?\s+introduction\b|\s+introduction\b)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _clean_section_text(match.group("abstract"))


def _analysis_text(sections: Dict[str, str], paper_text: str) -> str:
    """Build the text slice that should be sent to the LLM or chunker."""
    preferred_sections = [
        sections.get("abstract", ""),
        sections.get("introduction", ""),
        sections.get("background", ""),
        sections.get("related_work", ""),
        sections.get("conclusion", ""),
    ]
    text = " ".join(section for section in preferred_sections if section).strip()
    return text or paper_text[:DEFAULT_SECTION_SCAN_CHARS]


def _build_problem_prompt(
    chunks: Sequence[str],
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
) -> str:
    """Build a concise LLM prompt for structured problem extraction."""
    title = paper_metadata.get("title", "Untitled paper")
    categories = ", ".join(str(category) for category in paper_metadata.get("categories", []))
    section_names = ", ".join(_non_empty_section_names(sections))
    chunk_text = "\n\n".join(f"Chunk {index + 1}: {chunk}" for index, chunk in enumerate(chunks))
    return (
        "Extract the research problem from this paper. Return strict JSON with "
        "keys: problem, motivation, research_gap, context, evidence, confidence.\n"
        f"Title: {title}\n"
        f"Categories: {categories}\n"
        f"Sections available: {section_names}\n\n"
        f"{chunk_text}"
    )


def _call_llm_client(llm_client: Any, prompt: str) -> Any:
    """Call a common LLM client shape and return its raw response."""
    if callable(llm_client):
        return llm_client(prompt)
    for method_name in ("complete", "generate", "chat", "invoke"):
        method = getattr(llm_client, method_name, None)
        if callable(method):
            return method(prompt)
    raise TypeError("llm_client must be callable or expose complete/generate/chat/invoke")


def _parse_llm_problem_response(response: Any) -> Dict[str, Any]:
    """Parse LLM output into a dictionary, accepting common response shapes."""
    if isinstance(response, dict):
        if "choices" in response:
            return _parse_llm_problem_response(response["choices"][0])
        else:
            message = response.get("message")
            if isinstance(message, dict) and "content" in message:
                return _parse_llm_problem_response(message["content"])
            if "content" in response and len(response) <= 3:
                return _parse_llm_problem_response(response["content"])
            return dict(response)
    if hasattr(response, "model_dump"):
        return _parse_llm_problem_response(response.model_dump())
    if hasattr(response, "choices"):
        return _parse_llm_problem_response({"choices": response.choices})
    if hasattr(response, "content"):
        response = response.content
    if isinstance(response, list) and response:
        response = response[0]
    if isinstance(response, dict):
        return dict(response.get("message", response))

    text = str(response).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"problem": text, "confidence": "llm_unstructured"}
    if not isinstance(parsed, dict):
        return {"problem": text, "confidence": "llm_unstructured"}
    return parsed


def _normalize_problem_result(
    result: Dict[str, Any],
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    chunks_analyzed: int,
) -> Dict[str, Any]:
    """Ensure problem extraction output has stable keys and serializable values."""
    evidence = result.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    return {
        "problem": str(result.get("problem") or _metadata_problem_fallback(paper_metadata, [])),
        "motivation": str(result.get("motivation") or result.get("problem") or ""),
        "research_gap": str(result.get("research_gap") or ""),
        "context": str(result.get("context") or _build_context(paper_metadata, sections)),
        "evidence": [str(item) for item in evidence if str(item).strip()],
        "confidence": str(result.get("confidence") or "unknown"),
        "source": str(result.get("source") or "unknown"),
        "sections_used": list(result.get("sections_used") or _non_empty_section_names(sections)),
        "chunks_analyzed": int(result.get("chunks_analyzed") or chunks_analyzed),
        **({"llm_error": result["llm_error"]} if result.get("llm_error") else {}),
    }


def _split_sentences(text: str) -> List[str]:
    """Split normalized text into sentence-like units."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _first_matching_sentence(
    sentences: Sequence[str],
    keywords: Sequence[str],
) -> str:
    """Return the first sentence containing any keyword."""
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            return sentence
    return ""


def _metadata_problem_fallback(
    paper_metadata: Dict[str, Any],
    sentences: Sequence[str],
) -> str:
    """Create a fallback problem statement from metadata or early text."""
    summary = str(paper_metadata.get("summary") or "").strip()
    if summary:
        return _split_sentences(summary)[0]
    title = str(paper_metadata.get("title") or "").strip()
    if title:
        return f"The paper studies the research problem indicated by its title: {title}."
    if sentences:
        return sentences[0]
    return "The paper's research problem could not be determined from the supplied text."


def _infer_gap_from_problem(problem: str) -> str:
    """Create a conservative gap statement when no explicit gap sentence exists."""
    if not problem:
        return ""
    return f"The motivating gap is inferred from the stated focus: {problem}"


def _build_context(
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
) -> str:
    """Build a compact context string from metadata and available sections."""
    title = str(paper_metadata.get("title") or "").strip()
    categories = [str(category) for category in paper_metadata.get("categories", [])]
    context_parts = []
    if title:
        context_parts.append(f"Paper: {title}")
    if categories:
        context_parts.append(f"ArXiv categories: {', '.join(categories)}")
    if sections.get("abstract"):
        context_parts.append(f"Abstract focus: {_first_words(sections['abstract'], 35)}")
    return ". ".join(context_parts)


def _non_empty_section_names(sections: Dict[str, str]) -> List[str]:
    """Return section names whose values contain text."""
    return [name for name, value in sections.items() if value]


def _first_words(text: str, limit: int) -> str:
    """Return the first `limit` words from text with an ellipsis when truncated."""
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "..."
