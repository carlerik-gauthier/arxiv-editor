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

    def extract_key_results(
        self,
        paper_text: str,
        paper_metadata: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract and rank the main findings or contributions from a paper.

        Args:
            paper_text: Full paper text or a substantial excerpt.
            paper_metadata: Optional title/authors/categories/summary metadata.
            domain: Optional domain hint such as `math`, `ml`, `crypto`, or
                `general`. When omitted, the analyzer infers a coarse domain
                from ArXiv categories and paper text.

        Returns:
            A structured dictionary with `results`, `result_count`, `domain`,
            `confidence`, `source`, `sections_used`, and `chunks_analyzed`.
            Each result includes `result_type`, `statement`, `significance`,
            `location`, `evidence`, and `importance_score`.

        Raises:
            ValueError: If `paper_text` is empty.
        """
        normalized_text = _normalize_text(paper_text, parameter_name="paper_text")
        metadata = dict(paper_metadata or {})
        sections = self.extract_sections(normalized_text)
        inferred_domain = _infer_domain(domain, metadata, normalized_text)
        chunks = self.chunk_text(_results_analysis_text(sections, normalized_text))

        if self.llm_client is not None:
            try:
                result = self._extract_key_results_with_llm(
                    chunks=chunks,
                    paper_metadata=metadata,
                    sections=sections,
                    domain=inferred_domain,
                )
                result.setdefault("source", "llm")
                result.setdefault("confidence", "llm")
                result.setdefault("domain", inferred_domain)
                result.setdefault("sections_used", _non_empty_section_names(sections))
                result.setdefault("chunks_analyzed", len(chunks))
                return _normalize_key_results_result(
                    result,
                    domain=inferred_domain,
                    sections=sections,
                    chunks_analyzed=len(chunks),
                )
            except Exception as exc:
                fallback = self._extract_key_results_heuristic(
                    normalized_text,
                    metadata,
                    sections,
                    chunks,
                    inferred_domain,
                )
                fallback["source"] = "heuristic_fallback"
                fallback["llm_error"] = str(exc)
                return fallback

        return self._extract_key_results_heuristic(
            normalized_text,
            metadata,
            sections,
            chunks,
            inferred_domain,
        )

    def rank_results_by_importance(
        self,
        results: Iterable[Dict[str, Any]],
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank extracted result dictionaries by estimated editorial importance.

        Args:
            results: Iterable of result dictionaries. Missing fields are filled
                with conservative defaults.
            domain: Optional domain hint used to weight theorem, empirical, or
                security guarantees appropriately.

        Returns:
            Result dictionaries ordered by descending `importance_score`, with
            stable `rank` fields assigned from 1.
        """
        normalized_results = [
            _normalize_result_item(result, index, domain or "general")
            for index, result in enumerate(results)
        ]
        ranked = sorted(
            normalized_results,
            key=lambda result: (
                result["importance_score"],
                result.get("statement", ""),
            ),
            reverse=True,
        )
        for rank, result in enumerate(ranked, start=1):
            result["rank"] = rank
        return ranked

    def assess_impact(
        self,
        paper: Any,
        results: Optional[Iterable[Dict[str, Any]]] = None,
        field_context: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assess a paper's likely research impact and significance.

        Args:
            paper: Paper metadata dictionary, paper-like object, or raw paper
                text. Dictionaries may include `title`, `summary`, `abstract`,
                `categories`, `text`, `full_text`, or `paper_text`.
            results: Optional extracted key results from `extract_key_results`.
                These can be raw result dictionaries or normalized result items.
            field_context: Optional domain context supplied by a specialist
                agent, such as known open problems or application constraints.
            domain: Optional domain hint such as `math`, `ml`, `crypto`, or
                `general`. When omitted, the analyzer infers it from metadata
                and text.

        Returns:
            A structured assessment with `novelty_score`,
            `solves_open_problem`, `introduces_new_techniques`,
            `potential_applications`, `community_impact`,
            `community_impact_score`, `impact_summary`, `evidence`,
            `confidence`, `source`, `domain`, and `sections_used`.

        Raises:
            ValueError: If no usable paper text or metadata is supplied.
        """
        paper_text, paper_metadata = _paper_text_and_metadata(paper)
        normalized_text = _normalize_text(paper_text, parameter_name="paper")
        sections = self.extract_sections(normalized_text)
        inferred_domain = _infer_domain(domain, paper_metadata, normalized_text)
        normalized_results = self.rank_results_by_importance(
            list(results or []),
            domain=inferred_domain,
        )

        if self.llm_client is not None:
            try:
                result = self._assess_impact_with_llm(
                    paper_metadata=paper_metadata,
                    sections=sections,
                    results=normalized_results,
                    field_context=field_context,
                    domain=inferred_domain,
                )
                result.setdefault("source", "llm")
                result.setdefault("confidence", "llm")
                result.setdefault("domain", inferred_domain)
                result.setdefault("sections_used", _non_empty_section_names(sections))
                return _normalize_impact_assessment(
                    result,
                    paper_metadata=paper_metadata,
                    sections=sections,
                    results=normalized_results,
                    field_context=field_context,
                    domain=inferred_domain,
                )
            except Exception as exc:
                fallback = self._assess_impact_heuristic(
                    paper_text=normalized_text,
                    paper_metadata=paper_metadata,
                    sections=sections,
                    results=normalized_results,
                    field_context=field_context,
                    domain=inferred_domain,
                )
                fallback["source"] = "heuristic_fallback"
                fallback["llm_error"] = str(exc)
                return fallback

        return self._assess_impact_heuristic(
            paper_text=normalized_text,
            paper_metadata=paper_metadata,
            sections=sections,
            results=normalized_results,
            field_context=field_context,
            domain=inferred_domain,
        )

    def generate_impact_narrative(
        self,
        assessment: Dict[str, Any],
    ) -> str:
        """
        Convert an impact assessment into a compact readable narrative.

        Args:
            assessment: Structured assessment from `assess_impact` or an
                equivalent dictionary with novelty, open-problem, technique,
                application, and community impact fields.

        Returns:
            A short paragraph summarizing why the paper may matter.
        """
        return _impact_narrative(assessment)

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

    def _extract_key_results_with_llm(
        self,
        chunks: Sequence[str],
        paper_metadata: Dict[str, Any],
        sections: Dict[str, str],
        domain: str,
    ) -> Dict[str, Any]:
        """Call the injected LLM client and parse key-result output."""
        prompt = _build_key_results_prompt(
            chunks=chunks[:4],
            paper_metadata=paper_metadata,
            sections=sections,
            domain=domain,
        )
        response = _call_llm_client(self.llm_client, prompt)
        return _parse_llm_key_results_response(response)

    def _assess_impact_with_llm(
        self,
        paper_metadata: Dict[str, Any],
        sections: Dict[str, str],
        results: Sequence[Dict[str, Any]],
        field_context: Optional[str],
        domain: str,
    ) -> Dict[str, Any]:
        """Call the injected LLM client and parse impact assessment output."""
        prompt = _build_impact_prompt(
            paper_metadata=paper_metadata,
            sections=sections,
            results=results,
            field_context=field_context,
            domain=domain,
        )
        response = _call_llm_client(self.llm_client, prompt)
        return _parse_llm_impact_response(response)

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

    def _extract_key_results_heuristic(
        self,
        paper_text: str,
        paper_metadata: Dict[str, Any],
        sections: Dict[str, str],
        chunks: Sequence[str],
        domain: str,
    ) -> Dict[str, Any]:
        """Extract key results with deterministic formal-statement and sentence rules."""
        candidate_results = _extract_formal_result_candidates(paper_text, sections)
        candidate_results.extend(
            _extract_sentence_result_candidates(
                paper_metadata=paper_metadata,
                sections=sections,
                paper_text=paper_text,
                domain=domain,
            )
        )
        deduped_results = _deduplicate_results(candidate_results)

        if not deduped_results:
            deduped_results = [
                {
                    "result_type": "summary",
                    "statement": _metadata_problem_fallback(
                        paper_metadata,
                        _split_sentences(paper_text[:2000]),
                    ),
                    "significance": "This is the clearest contribution visible in the supplied text.",
                    "location": "metadata_or_body",
                    "evidence": [],
                }
            ]

        ranked_results = self.rank_results_by_importance(deduped_results, domain=domain)
        return _normalize_key_results_result(
            {
                "results": ranked_results,
                "domain": domain,
                "confidence": "heuristic",
                "source": "heuristic",
                "sections_used": _non_empty_section_names(sections),
                "chunks_analyzed": len(chunks),
            },
            domain=domain,
            sections=sections,
            chunks_analyzed=len(chunks),
        )

    def _assess_impact_heuristic(
        self,
        paper_text: str,
        paper_metadata: Dict[str, Any],
        sections: Dict[str, str],
        results: Sequence[Dict[str, Any]],
        field_context: Optional[str],
        domain: str,
    ) -> Dict[str, Any]:
        """Assess impact with deterministic novelty, technique, and application cues."""
        impact_text = _impact_analysis_text(
            paper_text=paper_text,
            paper_metadata=paper_metadata,
            sections=sections,
            results=results,
            field_context=field_context,
        )
        novelty_score = _estimate_novelty_score(impact_text, results, domain)
        solves_open_problem = _detect_open_problem_claim(impact_text)
        introduces_new_techniques = _detect_new_technique_claim(impact_text, results)
        potential_applications = _extract_potential_applications(
            impact_text,
            field_context=field_context,
            domain=domain,
        )
        evidence = _impact_evidence_sentences(impact_text)
        community_impact_score = _estimate_community_impact_score(
            novelty_score=novelty_score,
            solves_open_problem=solves_open_problem,
            introduces_new_techniques=introduces_new_techniques,
            potential_applications=potential_applications,
            results=results,
            domain=domain,
        )
        assessment = {
            "novelty_score": novelty_score,
            "solves_open_problem": solves_open_problem,
            "introduces_new_techniques": introduces_new_techniques,
            "potential_applications": potential_applications,
            "community_impact": _community_impact_label(community_impact_score),
            "community_impact_score": community_impact_score,
            "evidence": evidence,
            "confidence": "heuristic",
            "source": "heuristic",
            "domain": domain,
            "sections_used": _non_empty_section_names(sections),
        }
        assessment["impact_summary"] = self.generate_impact_narrative(assessment)
        return _normalize_impact_assessment(
            assessment,
            paper_metadata=paper_metadata,
            sections=sections,
            results=results,
            field_context=field_context,
            domain=domain,
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


def _results_analysis_text(sections: Dict[str, str], paper_text: str) -> str:
    """Build the text slice most likely to contain results and contributions."""
    preferred_sections = [
        sections.get("abstract", ""),
        sections.get("results", ""),
        sections.get("experiments", ""),
        sections.get("methods", ""),
        sections.get("discussion", ""),
        sections.get("conclusion", ""),
        sections.get("introduction", ""),
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


def _build_key_results_prompt(
    chunks: Sequence[str],
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    domain: str,
) -> str:
    """Build a concise LLM prompt for domain-aware key result extraction."""
    title = paper_metadata.get("title", "Untitled paper")
    categories = ", ".join(str(category) for category in paper_metadata.get("categories", []))
    section_names = ", ".join(_non_empty_section_names(sections))
    chunk_text = "\n\n".join(f"Chunk {index + 1}: {chunk}" for index, chunk in enumerate(chunks))
    return (
        "Extract the key results from this paper. Return strict JSON with keys: "
        "results, confidence. The results value must be a list of objects with "
        "keys: result_type, statement, significance, location, evidence, "
        "importance_score. For math papers, prioritize theorems and formal "
        "guarantees. For ML papers, prioritize empirical findings, benchmarks, "
        "architectures, ablations, and theoretical guarantees. For crypto papers, "
        "prioritize security guarantees, attacks, protocols, and assumptions.\n"
        f"Domain: {domain}\n"
        f"Title: {title}\n"
        f"Categories: {categories}\n"
        f"Sections available: {section_names}\n\n"
        f"{chunk_text}"
    )


def _build_impact_prompt(
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    results: Sequence[Dict[str, Any]],
    field_context: Optional[str],
    domain: str,
) -> str:
    """Build a concise LLM prompt for structured impact assessment."""
    title = paper_metadata.get("title", "Untitled paper")
    categories = ", ".join(str(category) for category in paper_metadata.get("categories", []))
    result_lines = "\n".join(
        f"- {result.get('result_type', 'result')}: {result.get('statement', '')}"
        for result in results[:6]
    )
    context = field_context or "No extra field context supplied."
    section_text = _first_words(
        " ".join(
            section
            for section in [
                sections.get("abstract", ""),
                sections.get("introduction", ""),
                sections.get("discussion", ""),
                sections.get("conclusion", ""),
            ]
            if section
        ),
        350,
    )
    return (
        "Assess this paper's research impact. Return strict JSON with keys: "
        "novelty_score, solves_open_problem, introduces_new_techniques, "
        "potential_applications, community_impact, community_impact_score, "
        "impact_summary, evidence, confidence. Use scores from 0 to 1.\n"
        f"Domain: {domain}\n"
        f"Title: {title}\n"
        f"Categories: {categories}\n"
        f"Field context: {context}\n"
        f"Key results:\n{result_lines}\n\n"
        f"Relevant paper text:\n{section_text}"
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


def _parse_llm_key_results_response(response: Any) -> Dict[str, Any]:
    """Parse LLM key-result output into a dictionary with a results list."""
    parsed = _parse_llm_problem_response(response)
    if "results" in parsed:
        return parsed
    if "key_results" in parsed:
        parsed["results"] = parsed.pop("key_results")
        return parsed
    if "statement" in parsed:
        return {"results": [parsed], "confidence": parsed.get("confidence", "llm")}
    return {
        "results": [
            {
                "result_type": "llm_unstructured",
                "statement": str(parsed.get("problem") or parsed),
                "significance": "The LLM response did not provide structured result fields.",
                "location": "llm_response",
                "evidence": [],
            }
        ],
        "confidence": parsed.get("confidence", "llm_unstructured"),
    }


def _parse_llm_impact_response(response: Any) -> Dict[str, Any]:
    """Parse LLM impact output into an assessment dictionary."""
    parsed = _parse_llm_problem_response(response)
    if "assessment" in parsed and isinstance(parsed["assessment"], dict):
        parsed = parsed["assessment"]
    if "impact" in parsed and isinstance(parsed["impact"], dict):
        parsed = parsed["impact"]
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


def _normalize_impact_assessment(
    assessment: Dict[str, Any],
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    results: Sequence[Dict[str, Any]],
    field_context: Optional[str],
    domain: str,
) -> Dict[str, Any]:
    """Ensure impact assessment output has stable keys and normalized scores."""
    novelty_score = _bounded_score(assessment.get("novelty_score", 0.0))
    community_impact_score = _bounded_score(
        assessment.get(
            "community_impact_score",
            _estimate_community_impact_score(
                novelty_score=novelty_score,
                solves_open_problem=_to_bool(assessment.get("solves_open_problem")),
                introduces_new_techniques=_to_bool(
                    assessment.get("introduces_new_techniques")
                ),
                potential_applications=_normalize_string_list(
                    assessment.get("potential_applications", [])
                ),
                results=results,
                domain=domain,
            ),
        )
    )
    potential_applications = _normalize_string_list(
        assessment.get("potential_applications", [])
    )
    evidence = _normalize_string_list(assessment.get("evidence", []))
    normalized = {
        "novelty_score": novelty_score,
        "solves_open_problem": _to_bool(assessment.get("solves_open_problem")),
        "introduces_new_techniques": _to_bool(
            assessment.get("introduces_new_techniques")
        ),
        "potential_applications": potential_applications,
        "community_impact": str(
            assessment.get("community_impact")
            or _community_impact_label(community_impact_score)
        ),
        "community_impact_score": community_impact_score,
        "impact_summary": str(assessment.get("impact_summary") or ""),
        "evidence": evidence,
        "confidence": str(assessment.get("confidence") or "unknown"),
        "source": str(assessment.get("source") or "unknown"),
        "domain": str(assessment.get("domain") or domain),
        "field_context": field_context,
        "sections_used": list(
            assessment.get("sections_used") or _non_empty_section_names(sections)
        ),
        "result_count": len(results),
        "paper_title": str(paper_metadata.get("title") or ""),
        **({"llm_error": assessment["llm_error"]} if assessment.get("llm_error") else {}),
    }
    if not normalized["impact_summary"]:
        normalized["impact_summary"] = _impact_narrative(normalized)
    return normalized


def _normalize_key_results_result(
    result: Dict[str, Any],
    domain: str,
    sections: Dict[str, str],
    chunks_analyzed: int,
) -> Dict[str, Any]:
    """Ensure key-result extraction output has stable keys and ranked results."""
    raw_results = result.get("results", [])
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    if isinstance(raw_results, str):
        raw_results = [
            {
                "result_type": "summary",
                "statement": raw_results,
                "significance": "Unstructured result returned by analyzer.",
                "location": "unknown",
                "evidence": [],
            }
        ]

    normalized_results = [
        _normalize_result_item(item, index, str(result.get("domain") or domain))
        for index, item in enumerate(raw_results or [])
        if item
    ]
    ranked_results = sorted(
        normalized_results,
        key=lambda item: (item["importance_score"], item["statement"]),
        reverse=True,
    )
    for rank, item in enumerate(ranked_results, start=1):
        item["rank"] = rank

    return {
        "results": ranked_results,
        "result_count": len(ranked_results),
        "domain": str(result.get("domain") or domain),
        "confidence": str(result.get("confidence") or "unknown"),
        "source": str(result.get("source") or "unknown"),
        "sections_used": list(result.get("sections_used") or _non_empty_section_names(sections)),
        "chunks_analyzed": int(result.get("chunks_analyzed") or chunks_analyzed),
        **({"llm_error": result["llm_error"]} if result.get("llm_error") else {}),
    }


def _normalize_result_item(
    result: Dict[str, Any],
    fallback_index: int,
    domain: str,
) -> Dict[str, Any]:
    """Normalize one extracted result and assign an importance score."""
    result_dict = dict(result)
    statement = str(result_dict.get("statement") or result_dict.get("result") or "").strip()
    if not statement:
        statement = "No result statement was provided."
    evidence = result_dict.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    result_type = str(result_dict.get("result_type") or "result")
    importance_score = result_dict.get("importance_score")
    if importance_score is None:
        importance_score = _estimate_result_importance(result_type, statement, domain)

    try:
        score = float(importance_score)
    except (TypeError, ValueError):
        score = _estimate_result_importance(result_type, statement, domain)

    return {
        "rank": int(result_dict.get("rank") or fallback_index + 1),
        "result_type": result_type,
        "statement": statement,
        "significance": str(
            result_dict.get("significance")
            or _default_result_significance(result_type, domain)
        ),
        "location": str(result_dict.get("location") or "unknown"),
        "evidence": [str(item) for item in evidence if str(item).strip()],
        "importance_score": max(0.0, min(score, 1.0)),
    }


def _infer_domain(
    domain: Optional[str],
    paper_metadata: Dict[str, Any],
    paper_text: str,
) -> str:
    """Infer a coarse result-extraction domain from hint, categories, and text."""
    if domain:
        normalized = domain.lower().strip()
        if normalized in {"machine_learning", "machine learning"}:
            return "ml"
        if normalized in {"cryptography", "security"}:
            return "crypto"
        if normalized in {"mathematics", "math"}:
            return "math"
        return normalized

    categories = " ".join(str(category).lower() for category in paper_metadata.get("categories", []))
    lowered_text = paper_text[:3000].lower()
    if any(category in categories for category in ("cs.lg", "stat.ml", "cs.ai", "cs.cl")):
        return "ml"
    if any(category in categories for category in ("cs.cr", "crypto")):
        return "crypto"
    if "security proof" in lowered_text or "adversary" in lowered_text:
        return "crypto"
    if "benchmark" in lowered_text or "accuracy" in lowered_text or "dataset" in lowered_text:
        return "ml"
    if "theorem" in lowered_text or "lemma" in lowered_text or "proof" in lowered_text:
        return "math"
    return "general"


def _extract_formal_result_candidates(
    paper_text: str,
    sections: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Extract theorem-like formal statements from the supplied paper text."""
    candidates: List[Dict[str, Any]] = []
    formal_pattern = re.compile(
        r"\b(?P<kind>Theorem|Lemma|Proposition|Corollary)\s*"
        r"(?P<number>\d+(?:\.\d+)*)?\s*(?:\([^)]*\))?\s*[:.]?\s+"
        r"(?P<statement>[^.!?]{20,500}[.!?])",
        flags=re.IGNORECASE,
    )
    for match in formal_pattern.finditer(paper_text[:DEFAULT_SECTION_SCAN_CHARS]):
        kind = match.group("kind").lower()
        statement = f"{match.group('kind')} {match.group('number') or ''} {match.group('statement')}"
        statement = " ".join(statement.split())
        candidates.append(
            {
                "result_type": kind,
                "statement": statement,
                "significance": _default_result_significance(kind, "math"),
                "location": _find_statement_location(statement, sections),
                "evidence": [statement],
            }
        )
    return candidates


def _extract_sentence_result_candidates(
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    paper_text: str,
    domain: str,
) -> List[Dict[str, Any]]:
    """Extract result-like sentences using domain-specific keyword patterns."""
    candidate_text = _results_analysis_text(sections, paper_text)
    if paper_metadata.get("summary"):
        candidate_text = f"{paper_metadata['summary']} {candidate_text}"

    sentences = _split_sentences(candidate_text)
    keywords = _domain_result_keywords(domain)
    candidates: List[Dict[str, Any]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        result_type = _classify_result_type(sentence, domain)
        candidates.append(
            {
                "result_type": result_type,
                "statement": sentence,
                "significance": _default_result_significance(result_type, domain),
                "location": _find_statement_location(sentence, sections),
                "evidence": [sentence],
            }
        )
    return candidates


def _domain_result_keywords(domain: str) -> Sequence[str]:
    """Return keywords that indicate result sentences for a coarse domain."""
    common = (
        "we prove",
        "we show",
        "we establish",
        "we introduce",
        "we present",
        "we propose",
        "we demonstrate",
        "our results",
        "main result",
        "result",
    )
    if domain == "ml":
        return common + (
            "outperform",
            "state-of-the-art",
            "accuracy",
            "benchmark",
            "ablation",
            "dataset",
            "improves",
        )
    if domain == "crypto":
        return common + (
            "secure",
            "security",
            "adversary",
            "attack",
            "protocol",
            "proof",
            "guarantee",
        )
    if domain == "math":
        return common + (
            "theorem",
            "lemma",
            "proposition",
            "corollary",
            "bound",
            "characterize",
        )
    return common


def _classify_result_type(sentence: str, domain: str) -> str:
    """Classify a result sentence into a coarse editorial type."""
    lowered = sentence.lower()
    formal_prefix = re.match(
        r"\s*(theorem|lemma|proposition|corollary)\b",
        lowered,
    )
    if formal_prefix or (
        domain != "ml"
        and any(keyword in lowered for keyword in ("theorem", "lemma", "proposition", "corollary"))
    ):
        return "theorem"
    if domain == "ml" and any(
        keyword in lowered
        for keyword in ("outperform", "accuracy", "benchmark", "ablation", "dataset")
    ):
        return "empirical"
    if domain == "crypto" and any(
        keyword in lowered
        for keyword in ("secure", "security", "adversary", "attack", "protocol")
    ):
        return "security_guarantee"
    if any(keyword in lowered for keyword in ("introduce", "present", "propose")):
        return "method"
    if any(keyword in lowered for keyword in ("prove", "establish", "bound", "guarantee")):
        return "guarantee"
    return "finding"


def _deduplicate_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove near-duplicate result statements while preserving first occurrence."""
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()
    for result in results:
        statement = str(result.get("statement") or "").strip()
        key = " ".join(_tokenize_for_dedup(statement)[:18])
        if not statement or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(dict(result))
    return deduped


def _tokenize_for_dedup(text: str) -> List[str]:
    """Tokenize text into lowercase words for simple result deduplication."""
    return re.findall(r"[a-zA-Z0-9-]+", text.lower())


def _find_statement_location(statement: str, sections: Dict[str, str]) -> str:
    """Return the first section name containing the statement or its prefix."""
    statement_prefix = " ".join(statement.lower().split()[:10])
    for section_name, section_text in sections.items():
        if statement_prefix and statement_prefix in section_text.lower():
            return section_name
    lowered_statement = statement.lower()
    for section_name, section_text in sections.items():
        if lowered_statement[:80] in section_text.lower():
            return section_name
    return "body"


def _estimate_result_importance(result_type: str, statement: str, domain: str) -> float:
    """Estimate result importance from type, domain, and statement cues."""
    lowered_statement = statement.lower()
    score = 0.45
    type_weights = {
        "theorem": 0.85,
        "corollary": 0.72,
        "proposition": 0.70,
        "lemma": 0.62,
        "guarantee": 0.76,
        "security_guarantee": 0.84,
        "empirical": 0.74,
        "method": 0.68,
        "finding": 0.58,
        "summary": 0.40,
    }
    score = type_weights.get(result_type.lower(), score)
    if "main result" in lowered_statement or "we prove" in lowered_statement:
        score += 0.08
    if "state-of-the-art" in lowered_statement or "outperform" in lowered_statement:
        score += 0.08
    if "secure" in lowered_statement or "security" in lowered_statement:
        score += 0.06
    if domain == "math" and result_type.lower() in {"theorem", "corollary", "proposition"}:
        score += 0.04
    if domain == "ml" and result_type.lower() == "empirical":
        score += 0.04
    if domain == "crypto" and result_type.lower() == "security_guarantee":
        score += 0.04
    return max(0.0, min(score, 1.0))


def _default_result_significance(result_type: str, domain: str) -> str:
    """Create a conservative significance explanation for a result type."""
    normalized_type = result_type.lower()
    if normalized_type in {"theorem", "lemma", "proposition", "corollary", "guarantee"}:
        return "Formal result that supports the paper's main technical contribution."
    if normalized_type == "empirical":
        return "Empirical result that indicates how the method behaves on data or benchmarks."
    if normalized_type == "security_guarantee":
        return "Security-relevant result about guarantees, attacks, protocols, or assumptions."
    if normalized_type == "method":
        return "Methodological contribution that changes how the problem can be approached."
    if domain == "ml":
        return "Machine learning finding relevant to models, data, evaluation, or deployment."
    if domain == "crypto":
        return "Cryptography or security finding relevant to guarantees or attacks."
    return "Result likely to be important for understanding the paper's contribution."


def _paper_text_and_metadata(paper: Any) -> tuple[str, Dict[str, Any]]:
    """Normalize raw text, dictionaries, or paper-like objects into text and metadata."""
    if isinstance(paper, str):
        return paper, {}
    if isinstance(paper, dict):
        paper_dict = dict(paper)
    elif hasattr(paper, "to_dict") and callable(paper.to_dict):
        paper_dict = dict(paper.to_dict())
    else:
        paper_dict = {
            "title": getattr(paper, "title", None),
            "summary": getattr(paper, "summary", None),
            "abstract": getattr(paper, "abstract", None),
            "categories": getattr(paper, "categories", []),
            "text": getattr(paper, "text", None),
            "full_text": getattr(paper, "full_text", None),
            "paper_text": getattr(paper, "paper_text", None),
        }

    text = str(
        paper_dict.get("paper_text")
        or paper_dict.get("full_text")
        or paper_dict.get("text")
        or " ".join(
            part
            for part in [
                str(paper_dict.get("title") or ""),
                str(paper_dict.get("summary") or paper_dict.get("abstract") or ""),
            ]
            if part
        )
    )
    metadata = {
        key: value
        for key, value in paper_dict.items()
        if key not in {"paper_text", "full_text", "text"}
    }
    return text, metadata


def _impact_analysis_text(
    paper_text: str,
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    results: Sequence[Dict[str, Any]],
    field_context: Optional[str],
) -> str:
    """Build a combined text slice for heuristic impact analysis."""
    result_text = " ".join(str(result.get("statement", "")) for result in results)
    preferred_sections = [
        str(paper_metadata.get("title") or ""),
        str(paper_metadata.get("summary") or paper_metadata.get("abstract") or ""),
        sections.get("abstract", ""),
        sections.get("introduction", ""),
        sections.get("discussion", ""),
        sections.get("conclusion", ""),
        result_text,
        field_context or "",
    ]
    text = " ".join(part for part in preferred_sections if part).strip()
    return text or paper_text[:DEFAULT_SECTION_SCAN_CHARS]


def _estimate_novelty_score(
    impact_text: str,
    results: Sequence[Dict[str, Any]],
    domain: str,
) -> float:
    """Estimate novelty from explicit novelty claims and result strength."""
    lowered = impact_text.lower()
    score = 0.35
    novelty_cues = (
        "novel",
        "new",
        "first",
        "introduce",
        "propose",
        "state-of-the-art",
        "open problem",
        "long-standing",
        "previously unknown",
    )
    score += min(sum(1 for cue in novelty_cues if cue in lowered) * 0.08, 0.36)
    if results:
        score += min(max(result.get("importance_score", 0.0) for result in results) * 0.22, 0.22)
    if domain == "math" and any("theorem" == result.get("result_type") for result in results):
        score += 0.05
    if domain == "ml" and ("state-of-the-art" in lowered or "outperform" in lowered):
        score += 0.06
    if domain == "crypto" and ("secure" in lowered or "attack" in lowered):
        score += 0.05
    return _bounded_score(score)


def _detect_open_problem_claim(impact_text: str) -> bool:
    """Detect whether the paper claims to resolve or address an open problem."""
    lowered = impact_text.lower()
    open_problem_cues = (
        "open problem",
        "long-standing",
        "longstanding",
        "conjecture",
        "previously unknown",
        "settle",
        "resolve",
        "remained open",
    )
    return any(cue in lowered for cue in open_problem_cues)


def _detect_new_technique_claim(
    impact_text: str,
    results: Sequence[Dict[str, Any]],
) -> bool:
    """Detect whether the paper appears to introduce a new method or technique."""
    lowered = impact_text.lower()
    technique_cues = (
        "new method",
        "new technique",
        "novel method",
        "novel framework",
        "we introduce",
        "we propose",
        "we present",
        "algorithm",
        "architecture",
        "protocol",
        "coupling",
    )
    if any(cue in lowered for cue in technique_cues):
        return True
    return any(result.get("result_type") == "method" for result in results)


def _extract_potential_applications(
    impact_text: str,
    field_context: Optional[str],
    domain: str,
) -> List[str]:
    """Extract compact application statements from text and field context."""
    sentences = _split_sentences(impact_text)
    application_keywords = (
        "application",
        "applies",
        "used in",
        "practical",
        "deployment",
        "benchmark",
        "bayesian",
        "cryptography",
        "security",
        "optimization",
        "simulation",
        "private",
    )
    applications = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in application_keywords)
    ][:4]
    if field_context and not applications:
        applications.append(f"Potentially relevant to field context: {field_context}")
    if not applications:
        applications.append(_default_application(domain))
    return applications


def _default_application(domain: str) -> str:
    """Return a conservative default application statement for a domain."""
    if domain == "ml":
        return "Potentially relevant to model design, evaluation, or downstream AI systems."
    if domain == "crypto":
        return "Potentially relevant to security guarantees, protocols, or attack analysis."
    if domain == "math":
        return "Potentially relevant to follow-up theoretical work in the same area."
    return "Potentially relevant to follow-up research in the surrounding field."


def _impact_evidence_sentences(impact_text: str) -> List[str]:
    """Select sentences that support an impact assessment."""
    keywords = (
        "novel",
        "new",
        "first",
        "open problem",
        "long-standing",
        "introduce",
        "propose",
        "application",
        "outperform",
        "secure",
        "result",
    )
    return [
        sentence
        for sentence in _split_sentences(impact_text)
        if any(keyword in sentence.lower() for keyword in keywords)
    ][:6]


def _estimate_community_impact_score(
    novelty_score: float,
    solves_open_problem: bool,
    introduces_new_techniques: bool,
    potential_applications: Sequence[str],
    results: Sequence[Dict[str, Any]],
    domain: str,
) -> float:
    """Estimate broad community impact from novelty, result strength, and applications."""
    score = 0.25 + (0.35 * _bounded_score(novelty_score))
    if solves_open_problem:
        score += 0.18
    if introduces_new_techniques:
        score += 0.12
    if potential_applications:
        score += 0.08
    if results:
        score += min(max(result.get("importance_score", 0.0) for result in results) * 0.18, 0.18)
    if domain in {"ml", "crypto"} and potential_applications:
        score += 0.04
    return _bounded_score(score)


def _community_impact_label(score: float) -> str:
    """Map a community impact score to a readable label."""
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "moderate"
    return "limited"


def _impact_narrative(assessment: Dict[str, Any]) -> str:
    """Create a compact narrative from normalized impact assessment fields."""
    impact = assessment.get("community_impact", "moderate")
    novelty = float(assessment.get("novelty_score", 0.0))
    pieces = [f"Estimated community impact is {impact} with novelty score {novelty:.2f}."]
    if assessment.get("solves_open_problem"):
        pieces.append("The paper appears to address an open or long-standing problem.")
    if assessment.get("introduces_new_techniques"):
        pieces.append("It also appears to introduce a method, technique, or framework.")
    applications = _normalize_string_list(assessment.get("potential_applications", []))
    if applications:
        pieces.append(f"Potential applications include {applications[0]}")
    return " ".join(pieces)


def _bounded_score(value: Any) -> float:
    """Convert a value to a float score clipped to [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 1.0))


def _to_bool(value: Any) -> bool:
    """Normalize booleans from bools, strings, and numeric values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "high"}
    return False


def _normalize_string_list(value: Any) -> List[str]:
    """Normalize strings or iterables into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(item) for item in value if str(item).strip()]
    except TypeError:
        return [str(value)] if str(value).strip() else []


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
