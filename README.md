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

### CLI Commands

- `generate` - Generate a research one-pager
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
- Step 3.3: Julius Coordinator Agent with Hand-offs

**Phase 4-10**: See CONTEXT.md for full implementation plan

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

## License

This project is for research and educational purposes.

## Contributing

See CONTEXT.md for detailed implementation plan and development guidelines.

## Acknowledgments

Named in honor of great researchers who have contributed significantly to their fields.
