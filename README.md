# ArXiv Research Publishing System

A multi-agent system for generating comprehensive one-pagers about the latest research news from ArXiv. The system features specialized agents named after great researchers, each focusing on specific domains.

## Agents

- **Julius** (Julius Springer) - Editor & Coordinator
- **Michel** (Michel Benaim) - Mathematical Intuition
- **Chris** (Krzystof Burdzy) - Probability Theory
- **Alain** (Alain Valette) - Algebra
- **Bruno** (Bruno Colbois) - Spectral & Riemannian Geometry
- **Elisa** (Elisa Gorla) - Applied Math & Cryptography
- **Felix** (Felix Schlenk) - Dynamical Systems & Symplectic Geometry
- **Abdoulaye** (Abdoulaye Sakho) - Machine Learning
- **JeanBaptiste** - Data Science, NLP, LLM & Agentic AI

## Features

- Automatically fetches papers from ArXiv across multiple mathematical and computer science domains
- Uses BERTopic for intelligent topic modeling
- Selects representative papers for each topic
- Generates summaries suitable for both experts and non-experts
- Emails professional one-pagers to users

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the repository:
```bash
cd arxiv-editor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the application:
```bash
python main.py setup
```

Or manually copy `.env.example` to `.env` and edit it:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Quick Start

Generate a one-pager for the last 7 days:
```bash
python main.py generate --email your-email@example.com
```

### Advanced Usage

**Generate for a specific date range:**
```bash
python main.py generate --start-date 2024-01-01 --end-date 2024-01-07 --email you@example.com
```

**Generate for the last 14 days:**
```bash
python main.py generate --days 14 --email you@example.com
```

**Generate without sending email:**
```bash
python main.py generate --days 7 --no-email
```

**View agent information:**
```bash
python main.py info
```

**Test installation:**
```bash
python main.py test
```

**Chat with Julius interactively:**
```bash
python main.py chat
```

### CLI Commands

- `generate` - Generate a research one-pager
- `chat` - Start an interactive Julius session
- `info` - Display information about agents and the system
- `setup` - Interactive setup wizard for configuration
- `test` - Run installation tests

## Configuration

All configuration is managed through environment variables. Key settings:

- **Email**: SMTP settings for sending one-pagers
- **ArXiv**: Minimum papers threshold, papers per topic, date ranges
- **Topic Modeling**: BERTopic parameters, embedding model
- **Processing**: Batch sizes, parallel workers, API delays
- **Output**: Target word count, max papers in summary

See `.env.example` for all available options.

## Project Structure

```
arxiv-editor/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Configuration template
├── config/                # Configuration management
│   └── settings.py       # Pydantic settings
├── src/                   # Source code
│   ├── agents/           # Agent implementations
│   ├── fetchers/         # ArXiv data fetching
│   ├── processing/       # Text processing & topic modeling
│   ├── analysis/         # Paper analysis
│   ├── generation/       # One-pager generation
│   └── communication/    # Email service
├── tests/                 # Test suite
├── data/                  # Data storage (git-ignored)
│   ├── cache/           # Cached embeddings
│   └── pdfs/            # Downloaded papers
└── outputs/               # Generated one-pagers (git-ignored)
```

## Development Status

**Phase 1: Project Foundation** ✅ COMPLETED
- Project structure
- Configuration management
- CLI interface

**Phase 2: ArXiv Data Fetching** ✅ COMPLETED
- ✅ Step 2.1: Basic API integration (`ArxivFetcher` class with `fetch_by_category`, `parse_paper_metadata`)
- ✅ Step 2.2: Multi-category fetching with minimum threshold
- ✅ Step 2.3: PDF retrieval (`download_paper_pdf`, `extract_text_from_pdf`, `download_and_extract_paper`)

**Phase 3: AI Agent Architecture with Tools and Hand-offs**
- ✅ Step 3.1: Base Agent Class and Tool System
- ✅ Step 3.2: Specialized Agent Implementations with Custom System Prompts
- ✅ Step 3.3: Julius Coordinator Agent with Hand-offs

**Phase 4: Text Processing & Topic Modeling**
- ✅ Step 4.1: Text Embedding Tool
- ✅ Step 4.2: Topic Discovery Tool with BERTopic
- ✅ Step 4.3: Representative Paper Selection Tool

**Phase 5: AI-Powered Paper Analysis Tools**
- ✅ Step 5.1: Problem Statement Extraction Tool
- ✅ Step 5.2: Key Results Extraction Tool
- ✅ Step 5.3: Impact Assessment Tool

**Phase 6: Interactive Julius Summary Generation**
- ✅ Step 6.1: User Request Intake and Preference Model
- ✅ Step 6.2: Julius Conversation Session
- ✅ Step 6.3: Multi-Agent Draft Generation with Hand-offs
- Step 6.4: User-Guided Revision Loop
- Step 6.5: Formatting, Quality Assurance, and Final Output
- Step 6.6: Streamlit App for End-to-End Interactive Summary Workflow

**Phase 7-10**: See CONTEXT.md for full implementation plan
## Ignore 
Phases 7 and 10 must be ignored. They are here for information purpose
## Testing

Run the test suite:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=src tests/
```

## Specialized Agents

Step 3.2 adds concrete specialist classes in `src/agents/specialized_agents.py`.
Each class documents its domain profile, ArXiv category assignments, system
prompt behavior, and tool access. Import agents directly or use the factory:

```python
from src.agents import MichelAgent, create_specialized_agent

michel = MichelAgent()
jean_baptiste = create_specialized_agent("Jean Baptiste")
```

Michel also has `create_metaphor_tool`, a documented deterministic tool for
creating intuitive explanations that can later be replaced by an LLM-backed
implementation without changing its output contract.

## Julius Coordination

Step 3.3 adds `JuliusAgent` in `src/agents/julius_agent.py`. Julius owns the
editorial workflow state machine, specialist hand-offs, result collection, draft
compilation, and optional email delivery.

```python
from src.agents import JuliusAgent

julius = JuliusAgent()
workflow = julius.run_delegated_workflow(
    "Summarize this week's probability and machine learning research",
    agent_names=["Chris", "Abdoulaye"],
)

print(workflow["workflow_state"])  # COMPLETE
print(workflow["one_pager"]["content"])
```

The coordination tools are also available through the normal agent tool system:
`delegate_to_agent_tool`, `request_agent_extension_tool`,
`collect_agent_results_tool`, `compile_one_pager_tool`, `send_email_tool`,
`parse_user_request_tool`, and `clarify_request_tool`. Email sending is
side-effect free unless an `email_sender` callable is injected.

## User Request Intake

Step 6.1 adds `SummaryRequest` in `src/generation/user_request.py`. It captures
the user's topic query, date range, audience, depth, tone, output format, topic
and paper limits, category filters, and delivery preference before Julius starts
fetching papers or delegating specialist work.

```python
from src.generation import parse_user_request_tool, clarify_request_tool

parsed = parse_user_request_tool(
    "Give me a non-technical summary of last week's LLM agent papers",
    reference_date="2026-05-12",
)

print(parsed["summary_request"]["topic_query"])  # LLM agent
print(parsed["needs_clarification"])             # False
```

`SummaryRequestSession` persists the latest preferences across refinement turns,
so follow-ups such as "make it shorter" keep the prior topic, date range, and
delivery choices unless the user overrides them. Julius exposes the same parser
through its normal tool system and stores the current request on
`julius.request_session`.

## Julius Conversation Session

Step 6.2 adds `JuliusSession` in `src/agents/julius_session.py`. It maintains
conversation history, the current `SummaryRequest`, draft previews, feedback,
and progress events. The main entry point is `handle_user_message(message)`,
which returns `message`, `state`, `summary_request`, `draft_preview`,
`actions_taken`, and `next_questions`.

```python
from src.agents import JuliusSession

session = JuliusSession(reference_date="2026-05-12")
session.handle_user_message("Give me a mixed audience summary of LLM agents")
session.handle_user_message("make it shorter and only cs.AI")
response = session.handle_user_message("generate draft")

print(response["state"])         # AWAITING_REVIEW
print(response["draft_preview"]) # deterministic preview for review
```

The session supports intake, clarification, planning, generation, review,
revision, draft-question answering, and finalization states. Draft previews are
generated through Julius's first-draft synthesis workflow.

## Draft Synthesis

Step 6.3 adds synthesis tools in `src/agents/tools/synthesis_tools.py` and
`ContentSynthesizer` in `src/generation/synthesizer.py`. Julius can now select
specialists from the request topic/categories, delegate review tasks, collect
handoff callbacks, synthesize the first draft, and attach provenance for
selected topics, papers, agents, confidence notes, and review warnings.

```python
from src.agents import JuliusAgent
from src.generation import SummaryRequest

julius = JuliusAgent()
result = julius.generate_first_draft_tool(
    summary_request=SummaryRequest(topic_query="LLM agents", must_include_categories=["cs.AI"]),
    selected_papers=[{"title": "Agent Planning Benchmarks", "summary": "We study LLM agents."}],
)

print(result["draft"]["content"])
print(result["draft"]["provenance"]["agent_callbacks"])
```

## Text Embeddings

Step 4.1 adds `TextEmbedder` and `EmbeddingCache` in `src/processing/embedder.py`
plus the agent-callable `embed_text_tool`. The tool uses
`sentence-transformers` with `all-MiniLM-L6-v2`, batches uncached texts, and
stores vectors in a pickle cache keyed by text hash and model name.

```python
from src.agents.tools import embed_text_tool

result = embed_text_tool(
    texts=["We prove a convergence theorem.", "A new language model benchmark."],
    batch_size=32,
)

print(result["embedding_count"], result["dimension"])
```

Specialized agents inherit the tool through `get_base_tools()`, so any
specialist can call `embed_text_tool` before topic modeling or paper selection.

## Topic Discovery

Step 4.2 adds `TopicModeler` in `src/processing/topic_modeler.py` and the
agent-callable `discover_topics_tool` plus `generate_topic_title_tool`. The
modeler embeds title/abstract text with `TextEmbedder`, passes those custom
embeddings into BERTopic configured with UMAP, HDBSCAN, c-TF-IDF, BERTopic's
`MaximalMarginalRelevance` representation for diverse keywords, and BERTopic's
OpenAI `representation_model` using `gpt-4o-mini` for topic labels. It returns
topic titles, keywords, representative papers, outlier counts, representation
metadata, and progress metadata.

```python
from src.agents.tools import discover_topics_tool

topics = discover_topics_tool(
    papers=[
        {"title": "Diffusion models for graphs", "summary": "We study graph generation."},
        {"title": "Convergence of Markov chains", "summary": "We prove mixing bounds."},
    ],
    min_topic_size=2,
)

print(topics["topic_count"])
```

Both topic tools are registered through `get_base_tools()`, so specialized
agents can discover themes after fetching papers and embedding their abstracts.
Set `OPENAI_API_KEY` before running real BERTopic topic discovery, or pass
`use_openai_representation=False` for offline/debug runs.

## License

This project is for research and educational purposes.

## Contributing

See CONTEXT.md for detailed implementation plan and development guidelines.

## Acknowledgments

Named in honor of great researchers who have contributed significantly to their fields.
