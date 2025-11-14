# Project description - User 
## Purpose of Arxiv publishing team

Write a one-pager about the latest (default is last week) research news that is comprehensive for experts and non-experts people. 
It will fetch at most 10 papers that are the most representative, explains the issue it is addressing and the main results. 

To achieve this, there will be different Agents:

- Julius, named in honor of Julius Springer, is the editor and is responsible to answer the user request, plan, calls the other agents, and provide the one-pager
- Michel, named in honor of Michel Benaim, who is an outstanding mathematician skilled in explaining complex mathematical concept to non-experts. He is really good at providing the intuition
- Chris, named in honor of Krzystof Burdzy, who is a one the greatest researcher in probability
- Alain, named in honor of Alain Valette, who is a one the greatest researcher in algebra
- Bruno, named in honor of Bruno Colbois, who is a one the greatest researcher in Spectral and Riemannian Geometry  
- Elisa, named in honor of Elisa Gorla, who is a one the greatest researcher in Applied mathematics, with a focus in cryptography 
- Felix, named in honor of Felix Schlenk, who is a one the greatest researcher in Dynamical systems and symplectic geometry
- Abdoulaye, named in honor of Abdoulaye Sakho, who is a great researcher in Machine Learning

## Agent tools
- Julius interacts with the user, plans, checks and sends the one-pager by email to the user

All other agents' tools are:
- fetch the articles on Arxiv on the right place for the time provided
- check if more data are needed (there must be at leat 100 articles fetched)
- report to Julius and ask him if they can extend th period of time
- embed text data 
- run Bertopic on articles' summary to find out the largest topics, and provide a title for those one
- finds the 5 most representative paper for each topic
- collect the entire paper of the selected representative papers, explain the problem it aims to address and finds the most important results, whether it is for the communauty, or because it is often use in the paper to prove the results.

---

# Claude Suggested Plan

## Overview
This plan breaks down the ArXiv research publishing system into testable, incremental steps. Each phase builds upon the previous one and includes specific testing criteria.

## Plan Highlights

The plan is organized into **10 phases with 30+ testable steps**:

1. **Phase 1-2**: Project foundation and ArXiv data fetching
2. **Phase 3**: Agent architecture (Julius, Michel, Chris, Alain, Bruno, Elisa, Felix, Abdoulaye)
3. **Phase 4**: Text embeddings and BERTopic integration
4. **Phase 5**: Paper analysis (problem extraction, key results, impact assessment)
5. **Phase 6**: One-pager generation with dual-level explanations
6. **Phase 7**: Email integration
7. **Phase 8**: End-to-end integration and optimization
8. **Phase 9**: Testing and documentation
9. **Phase 10**: Deployment and monitoring

## Key Features

- **Testability**: Each step includes specific test criteria
- **Incremental**: Build and validate progressively
- **Clear objectives**: Every step has a defined goal
- **Technology choices**: Specific libraries and approaches recommended
- **Timeline**: 6-week estimate for full implementation

The plan addresses all requirements from the project description including the multi-agent system, BERTopic topic modeling, representative paper selection, and generation of accessible summaries for both experts and non-experts.

## Phase 1: Project Foundation & Setup

### Step 1.1: Initialize Project Structure
**Objective**: Create a clean, organized Python project structure

**Tasks**:
- Create directory structure: `src/`, `tests/`, `config/`, `data/`, `outputs/`
- Initialize Python package with `__init__.py` files
- Create `requirements.txt` with initial dependencies:
  - `arxiv` (for ArXiv API)
  - `bertopic` (for topic modeling)
  - `sentence-transformers` (for embeddings)
  - `pydantic` (for data validation)
  - `python-dotenv` (for environment management)
  - `pytest` (for testing)

**Test**: Verify project structure exists and Python can import the package

### Step 1.2: Configuration Management
**Objective**: Set up configuration system for flexibility

**Tasks**:
- Create `config/settings.py` with Pydantic settings model
- Define configurable parameters:
  - Default time window (7 days)
  - Minimum articles threshold (100)
  - Number of topics to extract
  - Papers per topic (5)
  - ArXiv categories for each agent
  - Email SMTP settings
- Create `.env.example` template
- Add `.env` to `.gitignore`

**Test**: Load configuration successfully, verify all settings have defaults

## Phase 2: ArXiv Data Fetching Module

### Step 2.1: Basic ArXiv API Integration
**Objective**: Fetch papers from ArXiv by category and date

**Tasks**:
- Create `src/fetchers/arxiv_fetcher.py`
- Implement `ArxivFetcher` class with methods:
  - `fetch_by_category(category, start_date, end_date, max_results)`
  - `parse_paper_metadata(paper)` - extract title, authors, summary, date, categories
- Handle API rate limiting and errors gracefully

**Test**: Fetch 10 papers from a specific category (e.g., math.PR) for the last week, verify metadata structure

### Step 2.2: Multi-Category Fetching with Minimum Threshold
**Objective**: Ensure at least 100 papers are fetched

**Tasks**:
- Implement `fetch_with_threshold(categories, start_date, end_date, min_count=100)`
- Auto-expand date range if threshold not met
- Return metadata including actual date range used
- Add logging for transparency

**Test**: Request papers from category with few submissions, verify date range expansion and threshold satisfaction

### Step 2.3: Full Paper Content Retrieval
**Objective**: Download complete PDF content for selected papers

**Tasks**:
- Implement `download_paper_pdf(paper_id, output_dir)`
- Implement PDF text extraction using `pypdf` or `pdfplumber`
- Cache downloaded papers to avoid re-fetching
- Handle download failures gracefully

**Test**: Download 3 papers, extract text, verify content is readable and non-empty

## Phase 3: Agent Architecture Foundation

### Step 3.1: Base Agent Class
**Objective**: Create reusable agent abstraction

**Tasks**:
- Create `src/agents/base_agent.py`
- Define `BaseAgent` class with:
  - `name`: str
  - `expertise`: str (domain description)
  - `categories`: List[str] (ArXiv categories)
  - `fetch_papers(start_date, end_date)` method
  - `check_threshold()` method
  - `request_extension()` method to communicate with Julius
- Add logging and state tracking

**Test**: Instantiate base agent, verify it can fetch papers and check threshold

### Step 3.2: Specialized Agent Implementations
**Objective**: Create domain-specific agents

**Tasks**:
- Create agent classes in `src/agents/`:
  - `MichelAgent` - Mathematics education/intuition (math.HO, math.GM)
  - `ChrisAgent` - Probability (math.PR, stat.TH)
  - `AlainAgent` - Algebra (math.AG, math.RA, math.GR)
  - `BrunoAgent` - Spectral/Riemannian Geometry (math.DG, math.SP)
  - `ElisaAgent` - Applied math/Cryptography (cs.CR, math.OC)
  - `FelixAgent` - Dynamical systems/Symplectic geometry (math.DS, math.SG)
  - `AbdoulayeAgent` - Machine Learning (cs.LG, stat.ML)
- Each agent inherits from `BaseAgent` with specific categories

**Test**: Instantiate each agent, verify they target correct ArXiv categories

### Step 3.3: Julius Coordinator Agent
**Objective**: Implement the orchestrator

**Tasks**:
- Create `src/agents/julius_agent.py`
- Implement `JuliusAgent` class with:
  - `plan_research_summary(user_request)` - parse request, determine date range
  - `coordinate_agents()` - delegate to specialized agents
  - `handle_extension_request(agent, reason)` - approve/deny time extension
  - `compile_summary()` - aggregate results
  - `send_email(recipient, content)` - deliver one-pager
- Add state machine for workflow management

**Test**: Create mock workflow, verify Julius can coordinate multiple agents and handle requests

## Phase 4: Text Processing & Topic Modeling

### Step 4.1: Text Embedding Pipeline
**Objective**: Convert papers to vector representations

**Tasks**:
- Create `src/processing/embedder.py`
- Implement `TextEmbedder` class using `sentence-transformers`
- Use model: `all-MiniLM-L6-v2` (good balance of speed/quality)
- Implement batch processing for efficiency
- Cache embeddings to avoid recomputation

**Test**: Embed 50 paper abstracts, verify output dimensions and similarity calculations work

### Step 4.2: BERTopic Integration
**Objective**: Discover topics in paper collection

**Tasks**:
- Create `src/processing/topic_modeler.py`
- Implement `TopicModeler` class wrapping BERTopic
- Configure BERTopic with:
  - Custom embedding model from Step 4.1
  - UMAP for dimensionality reduction
  - HDBSCAN for clustering
  - c-TF-IDF for topic representation
- Implement `extract_topics(papers, min_topic_size=5)`
- Implement `generate_topic_titles(topic_keywords)` using LLM or rule-based approach

**Test**: Run on 100+ papers, verify topics are coherent and have meaningful titles

### Step 4.3: Representative Paper Selection
**Objective**: Find most representative papers per topic

**Tasks**:
- Implement `select_representative_papers(topic_id, papers, n=5)` in `TopicModeler`
- Use topic probability scores and centrality measures
- Ensure diversity in selection (avoid very similar papers)
- Return papers ranked by representativeness

**Test**: For a topic with 20 papers, select top 5 representatives, manually verify they're central to the topic

## Phase 5: Paper Analysis & Content Extraction

### Step 5.1: Problem Statement Extraction
**Objective**: Identify the research problem each paper addresses

**Tasks**:
- Create `src/analysis/paper_analyzer.py`
- Implement `extract_problem_statement(paper_text)`:
  - Look for introduction, problem setup sections
  - Extract key sentences describing motivation and problem
  - Use regex patterns and NLP (spaCy or transformers)
- Handle different paper structures

**Test**: Extract problem statements from 5 diverse papers, manually verify accuracy

### Step 5.2: Key Results Extraction
**Objective**: Identify main theorems, lemmas, and results

**Tasks**:
- Implement `extract_key_results(paper_text)`:
  - Identify theorem/lemma/proposition environments
  - Extract conclusion section highlights
  - Detect frequently cited results within the paper
  - Rank by importance (citation frequency, emphasis)
- Generate structured output with result type and statement

**Test**: Extract results from 5 papers, verify key theorems are captured

### Step 5.3: Impact Assessment
**Objective**: Determine why results matter

**Tasks**:
- Implement `assess_impact(paper, results)`:
  - Check if results solve long-standing problems
  - Identify novel techniques introduced
  - Assess potential applications
  - Generate impact summary
- Use paper's discussion/conclusion sections

**Test**: Assess impact for 3 papers with known significant contributions, verify quality of assessment

## Phase 6: One-Pager Generation

### Step 6.1: Content Synthesis Engine
**Objective**: Combine all analyses into coherent summaries

**Tasks**:
- Create `src/generation/synthesizer.py`
- Implement `SynthesizeTopicSummary` class:
  - `summarize_topic(topic_title, papers, analyses)` - create topic overview
  - `create_paper_summary(paper, analysis)` - individual paper summary
  - `generate_expert_explanation(content)` - technical version
  - `generate_layperson_explanation(content)` - accessible version
- Use template-based approach or LLM integration (OpenAI/Anthropic API)

**Test**: Generate summaries for 2 topics, verify they're coherent and dual-level (expert/layperson)

### Step 6.2: One-Pager Template & Formatting
**Objective**: Create professional document output

**Tasks**:
- Create `src/generation/formatter.py`
- Design one-pager template structure:
  - Header: Title, date range, agent credits
  - Executive summary (2-3 sentences)
  - Topic sections (3-5 topics max)
    - Topic title and overview
    - Top papers with problem/results/impact
    - Expert vs. non-expert explanations
  - Footer: Methodology note
- Implement formatters: Markdown, HTML, PDF
- Add styling for readability

**Test**: Generate one-pager from sample data, verify formatting and readability

### Step 6.3: Quality Assurance & Validation
**Objective**: Ensure output meets quality standards

**Tasks**:
- Implement `src/generation/validator.py`
- Add quality checks:
  - Readability scores (Flesch-Kincaid)
  - Length constraints (one page ~500-800 words)
  - Citation accuracy
  - Fact consistency across sections
- Implement correction suggestions

**Test**: Validate 3 generated one-pagers, verify quality metrics are reasonable

## Phase 7: Email Integration

### Step 7.1: Email Template System
**Objective**: Create professional email format

**Tasks**:
- Create `src/communication/email_service.py`
- Design email template:
  - Subject line with date range
  - Greeting
  - Brief intro
  - One-pager content (inline HTML)
  - Attachments (PDF version)
  - Signature
- Support HTML and plain text versions

**Test**: Generate email preview, verify formatting in email client

### Step 7.2: SMTP Integration
**Objective**: Send emails reliably

**Tasks**:
- Implement `EmailService` class:
  - `configure_smtp(host, port, username, password, use_tls)`
  - `send_email(recipient, subject, body, attachments)`
  - Error handling and retry logic
  - Support multiple recipients
- Add email validation

**Test**: Send test email to yourself, verify delivery and formatting

## Phase 8: End-to-End Integration

### Step 8.1: Main Application Flow
**Objective**: Connect all components

**Tasks**:
- Create `src/main.py` with CLI interface
- Implement main workflow:
  1. Parse user request (date range, specific topics)
  2. Julius plans and delegates to agents
  3. Each agent fetches papers for their domain
  4. Agents check threshold, request extensions if needed
  5. Embed and cluster all papers
  6. Extract topics and select representatives
  7. Analyze selected papers
  8. Generate one-pager
  9. Send email
- Add progress logging and error recovery

**Test**: Run end-to-end with default settings (last week, all agents), verify complete one-pager is generated and sent

### Step 8.2: Error Handling & Edge Cases
**Objective**: Make system robust

**Tasks**:
- Handle edge cases:
  - No papers in time range
  - API failures
  - Insufficient papers for topic modeling
  - Email delivery failures
- Add graceful degradation
- Implement retry mechanisms
- Create detailed error logs

**Test**: Simulate failures (network issues, empty results), verify system handles gracefully

### Step 8.3: Performance Optimization
**Objective**: Ensure system runs efficiently

**Tasks**:
- Profile bottlenecks (likely: PDF downloads, embeddings, topic modeling)
- Implement optimizations:
  - Parallel paper downloads
  - Batch embedding
  - Caching at multiple levels
  - Async operations where possible
- Monitor memory usage with large datasets

**Test**: Process 500+ papers, measure execution time and memory, verify acceptable performance (<10 min total)

## Phase 9: Testing & Documentation

### Step 9.1: Comprehensive Test Suite
**Objective**: Achieve high test coverage

**Tasks**:
- Write unit tests for each module (target: 80%+ coverage)
- Write integration tests for workflows
- Create fixtures with sample ArXiv data
- Add property-based tests for data validation
- Set up CI/CD with GitHub Actions

**Test**: Run full test suite, verify all tests pass and coverage target is met

### Step 9.2: User Documentation
**Objective**: Enable users to run and configure the system

**Tasks**:
- Create `README.md` with:
  - Project overview
  - Installation instructions
  - Configuration guide
  - Usage examples
  - Troubleshooting
- Create `docs/` directory with:
  - Architecture diagram
  - Agent descriptions
  - API documentation
  - Configuration reference

**Test**: New user follows README to set up and run system successfully

### Step 9.3: Developer Documentation
**Objective**: Enable contributors to extend the system

**Tasks**:
- Add docstrings to all classes and functions
- Create `CONTRIBUTING.md` with development guidelines
- Document agent creation process
- Add architecture decision records (ADRs)
- Create example: adding a new agent

**Test**: Developer can add a new agent following documentation alone

## Phase 10: Deployment & Monitoring

### Step 10.1: Containerization
**Objective**: Make deployment easy and consistent

**Tasks**:
- Create `Dockerfile` for the application
- Create `docker-compose.yml` for local development
- Optimize image size
- Add health checks

**Test**: Build and run container, verify full workflow works in containerized environment

### Step 10.2: Scheduling & Automation
**Objective**: Run automatically on schedule

**Tasks**:
- Add scheduling capability (cron or APScheduler)
- Implement weekly automatic runs
- Add configuration for custom schedules
- Send notifications on completion or failure

**Test**: Schedule weekly run, verify it executes automatically and sends email

### Step 10.3: Monitoring & Logging
**Objective**: Track system health and usage

**Tasks**:
- Implement structured logging (JSON format)
- Add metrics collection:
  - Papers processed
  - Topics discovered
  - Execution time
  - Error rates
- Create dashboard (optional: Grafana/Prometheus)
- Set up alerting for failures

**Test**: Run system, verify logs are generated and metrics are collected

---

## Success Criteria

The project is considered complete when:

1. System can automatically fetch 100+ papers from multiple ArXiv categories for a given time period
2. Topics are discovered and labeled with BERTopic
3. Top 5 representative papers per topic are selected
4. Each paper's problem, results, and impact are extracted
5. A comprehensive one-pager is generated with both expert and non-expert explanations
6. One-pager is emailed to the user in professional format
7. All components have >80% test coverage
8. Documentation is complete and validated by new users
9. System runs end-to-end in <10 minutes for weekly data
10. Error handling is robust with no crashes on common failures

---

## Implementation Order Recommendation

Follow the phases sequentially, but within each phase, steps can sometimes be parallelized:

**Week 1**: Phases 1-2 (Foundation & Data Fetching)
**Week 2**: Phases 3-4 (Agents & Topic Modeling)
**Week 3**: Phase 5 (Paper Analysis)
**Week 4**: Phases 6-7 (Generation & Email)
**Week 5**: Phases 8-9 (Integration & Testing)
**Week 6**: Phase 10 (Deployment & Polish)

Total estimated timeline: **6 weeks** for a single developer working full-time.
