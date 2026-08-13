# ArXiv Editor

ArXiv Editor is a multi-agent research workflow that turns recent arXiv papers
into a concise, evidence-based one-pager. It supports mathematics and AI,
selects no more than five research topics, and can include one or more
representative papers and their reported main results.


<p align="center">
  <img src="docs/arxiv_edition_team.png"
       alt="The Great 9 — Agentic Arxiv Edition Team"
       width="600">
  <br>
  <em>The Great 9 — Agentic Arxiv Edition Team</em>
</p>

JuliusAgent is the editor and coordinator. It delegates retrieval and topic
analysis to domain specialists, creates a factual first draft, and asks
MichelAgent to review readability and supply any pedagogical explanations.
The factual draft marks required explanations with Michel-owned placeholders;
Julius replaces each required placeholder only with Michel's feedback and sends
the revised one-pager back to Michel for review before delivery. The workflow uses the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
for agents, tools, delegation, runners, and tracing.

## Experimentation outcome
This branch is the result of a successful experiment. Codex coding agent assisted to refine the instruction steps, implementing the code first draft that I reviewed and tested (and of course, understanding it) before approving moving to the next step. When the generated was not fully in line with my expectation, I was able to fix.

## Supported areas

| Family | Specialists | arXiv categories |
| --- | --- | --- |
| Mathematics | Chris (probability/statistics), Alain (algebra), Bruno (geometry), Elisa (applied math/cryptography), Felix (dynamical systems) | `math.PR`, `math.ST`, `math.AG`, `math.RA`, `math.GR`, `math.AT`, `math.DG`, `math.SP`, `cs.CR`, `math.OC`, `math.NA`, `math.DS`, `math.SG` |
| AI | Abdoulaye (machine learning), JeanBaptiste (data science, NLP, LLMs, agentic AI) | `cs.LG`, `stat.ML`, `cs.CL`, `cs.AI`, `cs.MA`, `cs.CE` |

## Quick start

Use Python 3.12 or later and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
```

Run the Phase 10 app:

```bash
streamlit run streamlit_phase10.py
```

The chat history is kept for the active Streamlit session. Ask, for example:

> I want two main topics in probability and one in algebra from 2026-05-25 to 2026-05-29. For every topic, give one representative paper and describe the main result. Write a LinkedIn post.

## Development

Run the offline unit suite with:

```bash
pytest -q
```

The tests do not call arXiv or OpenAI. Mark any future live-network test with
`@pytest.mark.integration`; pytest excludes those by default.

Paper metadata is cached under `data/paper/` and extracted paper files under
`data/pdfs/`. Configure the locations with `ARXIV_FETCH_OUTPUT_DIR` and
`ARXIV_PDF_OUTPUT_DIR` when required. These are generated artifacts and should
not be committed.

See [the workflow guide](docs/WORKFLOW.md) for the system diagram, execution
path, data contracts, and failure behavior.
