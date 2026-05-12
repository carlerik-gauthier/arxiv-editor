# Project description - User 
## Purpose of Arxiv publishing team

Write a one-pager about the latest (default is last week) research news that is comprehensive for experts and non-experts people. 
It will fetch at most 10 papers that are the most representative, explains the issue it is addressing and the main results. 

To achieve this, there will be different Agents:

- Julius, named in honor of Julius Springer, is the editor and is responsible to answer the user request, plan, calls the other agents, and provide the one-pager
- Michel, named in honor of Michel Benaim, who is an outstanding mathematician skilled in explaining complex mathematical concept to non-experts. He is really good at providing the intuition. When relevant, he uses metaphors
- Chris, named in honor of Krzystof Burdzy, who is a one the greatest researcher in probability
- Alain, named in honor of Alain Valette, who is a one the greatest researcher in algebra
- Bruno, named in honor of Bruno Colbois, who is a one the greatest researcher in Spectral and Riemannian Geometry
- Elisa, named in honor of Elisa Gorla, who is a one the greatest researcher in Applied mathematics, with a focus in cryptography
- Felix, named in honor of Felix Schlenk, who is a one the greatest researcher in Dynamical systems and symplectic geometry
- Abdoulaye, named in honor of Abdoulaye Sakho, who is a great researcher in Machine Learning
- JeanBaptiste, who is a great researcher in Data Science with a focus on NLP, LLM and Agentic AI

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
This plan breaks down the ArXiv research publishing system into testable, incremental steps. The system uses **AI agents with tools and hand-offs** to create intelligent, collaborative workflows for research paper analysis and summarization.

## Plan Highlights

The plan is organized into **10 phases with 30+ testable steps**:

1. **Phase 1-2**: Project foundation and ArXiv data fetching
2. **Phase 3**: **AI Agent Architecture** - LLM-powered agents with tool-calling capabilities and hand-off protocols (Julius, Michel, Chris, Alain, Bruno, Elisa, Felix, Abdoulaye, JeanBaptiste)
3. **Phase 4**: **Agent Tools** - Text embeddings and BERTopic integration as callable tools
4. **Phase 5**: **AI-Powered Analysis Tools** - LLM-driven paper analysis with domain expertise
5. **Phase 6**: **Multi-Agent Content Generation** - Collaborative one-pager creation with hand-offs between specialist agents
6. **Phase 7**: Email integration
7. **Phase 8**: End-to-end integration and optimization
8. **Phase 9**: Testing and documentation
9. **Phase 10**: Deployment and monitoring

## Key Features

- **AI Agents with Tools**: Each agent is powered by an LLM (Claude/GPT) with access to specialized tools
- **Hand-off Protocol**: Agents can delegate tasks to other agents based on expertise
- **Domain Specialization**: Each agent has custom system prompts and domain-specific knowledge
- **Collaborative Workflows**: Multi-agent coordination for complex tasks (e.g., Julius delegates, specialists analyze, Michel refines explanations)
- **Tool-Calling Framework**: Agents reason about which tools to use and when
- **Testability**: Each step includes specific test criteria for agent behavior
- **Incremental**: Build and validate progressively
- **Technology Stack**: LLM APIs (Anthropic Claude/OpenAI), BERTopic, sentence-transformers, tool execution framework
- **Timeline**: 6-8 week estimate for full implementation

The plan addresses all requirements from the project description including the multi-agent system with AI reasoning, BERTopic topic modeling, representative paper selection, and generation of accessible summaries for both experts and non-experts through agent collaboration and hand-offs.

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

## Phase 3: AI Agent Architecture with Tools and Hand-offs

### Step 3.1: Base Agent Class and Tool System
**Objective**: Create reusable AI agent abstraction with tool-calling capabilities

**Tasks**:
- Create `src/agents/base_agent.py`
- Define `BaseAgent` class with:
  - `name`: str
  - `expertise`: str (domain description)
  - `categories`: List[str] (ArXiv categories)
  - `llm_client`: LLM client (Anthropic Claude or OpenAI) for natural language reasoning
  - `system_prompt`: Template defining agent's role, expertise, and behavior
  - `tools`: List of available tools the agent can call
  - `conversation_history`: Maintains context across interactions
- Create `src/agents/tools/` directory for tool definitions
- Implement base tools as callable functions:
  - `fetch_papers_tool(categories, start_date, end_date)` - Fetch papers from ArXiv
  - `check_threshold_tool(paper_count, min_threshold)` - Verify minimum paper count
  - `analyze_paper_tool(paper_text)` - Extract problem/results from paper
  - `generate_summary_tool(papers, topic)` - Create topic summary
- Implement tool execution framework:
  - `execute_tool(tool_name, parameters)` - Safe tool execution with error handling
  - `parse_tool_calls(llm_response)` - Extract tool calls from LLM response
- Add logging and state tracking

**Test**: Instantiate base agent, verify it can reason about tasks and execute tools

### Step 3.2: Specialized Agent Implementations with Custom System Prompts
**Objective**: Create domain-specific AI agents with tailored expertise

**Tasks**:
- Create agent classes in `src/agents/` inheriting from `BaseAgent`:
  - `MichelAgent` - Mathematics education/intuition (math.HO, math.GM)
    - System prompt: Expert at explaining complex math to non-experts using metaphors
    - Custom tool: `create_metaphor_tool(concept)` for generating intuitive explanations
  - `ChrisAgent` - Probability (math.PR, stat.TH)
    - System prompt: Probability theory expert, focuses on stochastic processes
  - `AlainAgent` - Algebra (math.AG, math.RA, math.GR)
    - System prompt: Algebraic structures specialist
  - `BrunoAgent` - Spectral/Riemannian Geometry (math.DG, math.SP)
    - System prompt: Geometry expert, emphasizes geometric intuition
  - `ElisaAgent` - Applied math/Cryptography (cs.CR, math.OC)
    - System prompt: Applied mathematics and cryptography specialist
  - `FelixAgent` - Dynamical systems/Symplectic geometry (math.DS, math.SG)
    - System prompt: Dynamical systems expert, focuses on long-term behavior
  - `AbdoulayeAgent` - Machine Learning (cs.LG, stat.ML)
    - System prompt: ML researcher, explains algorithms and applications
  - `JeanBaptisteAgent` - Data Science/NLP/LLM/Agentic AI (cs.CL, cs.AI, cs.MA, cs.CE)
    - System prompt: Data science expert specializing in NLP, LLMs, and agentic systems
- Each agent has:
  - Domain-specific system prompt defining expertise and communication style
  - Customized tool access based on domain needs
  - ArXiv category assignments

**Test**: Instantiate each agent, verify system prompts are loaded and tools are accessible

### Step 3.3: Julius Coordinator Agent with Hand-offs
**Objective**: Implement AI orchestrator with agent delegation and hand-offs

**Tasks**:
- Create `src/agents/julius_agent.py`
- Implement `JuliusAgent` class (inherits from `BaseAgent`) with:
  - System prompt: Editor and coordinator role, responsible for planning and delegation
  - Specialized coordination tools:
    - `delegate_to_agent_tool(agent_name, task_description)` - Hand off tasks to specialized agents
    - `request_agent_extension_tool(agent_name, reason)` - Request date range extension from agents
    - `collect_agent_results_tool(agent_names)` - Gather results from multiple agents
    - `compile_one_pager_tool(agent_results)` - Synthesize final document
    - `send_email_tool(recipient, content)` - Deliver one-pager
  - Hand-off protocol:
    - Define task context and requirements when delegating
    - Wait for agent completion or status updates
    - Handle partial results and failures gracefully
  - Conversation flow:
    - Parse user request (date range, topics, preferences)
    - Create execution plan with agent assignments
    - Coordinate parallel agent execution where possible
    - Synthesize results into coherent one-pager
- Implement `AgentHandoff` class:
  - `handoff_context`: Contains task description, constraints, and previous results
  - `execute_handoff(from_agent, to_agent, context)` - Transfer control between agents
  - `callback_on_completion()` - Return results to coordinating agent
- Add workflow state machine:
  - States: PLANNING, DELEGATING, COLLECTING, COMPILING, REVIEWING, COMPLETE
  - Track agent task status (PENDING, IN_PROGRESS, COMPLETED, FAILED)

**Test**: Create workflow where Julius delegates to 2+ agents, verify hand-offs work and results are aggregated

## Phase 4: Text Processing & Topic Modeling (Agent Tools)

### Step 4.1: Text Embedding Tool
**Objective**: Create embedding tool for agents to convert papers to vector representations

**Tasks**:
- Create `src/agents/tools/embedding_tool.py`
- Implement `embed_text_tool(texts, batch_size=32)` as an agent-callable tool:
  - Uses `sentence-transformers` with model `all-MiniLM-L6-v2` (good balance of speed/quality)
  - Implements batch processing for efficiency
  - Returns embeddings with proper error handling
- Create `src/processing/embedder.py` as the underlying implementation
- Implement embedding cache:
  - `EmbeddingCache` class with file-based storage (pickle or HDF5)
  - Cache key based on text hash + model version
  - `get_or_create_embedding(text)` for transparent caching
- Register tool with BaseAgent so specialized agents can use it
- Add tool description for LLM to understand when/how to use it

**Test**: Agent calls embed_text_tool with 50 abstracts, verify cached embeddings are reused on second call

### Step 4.2: Topic Discovery Tool with BERTopic
**Objective**: Create topic modeling tool for agents to discover research themes

**Tasks**:
- Create `src/agents/tools/topic_discovery_tool.py`
- Implement `discover_topics_tool(papers, min_topic_size=5, num_topics=None)` as agent-callable tool:
  - Takes list of papers with abstracts/summaries
  - Returns structured topic information with representative papers
  - Includes error handling and progress updates
- Create `src/processing/topic_modeler.py` as underlying implementation
- Implement `TopicModeler` class wrapping BERTopic:
  - Configure BERTopic with:
    - Custom embeddings from embedding_tool
    - UMAP for dimensionality reduction
    - HDBSCAN for clustering
    - c-TF-IDF for topic representation
  - `extract_topics(papers, min_topic_size=5)` - Run topic modeling
  - `get_topic_info(topic_id)` - Get keywords and metadata for a topic
- Implement `generate_topic_title_tool(topic_keywords, sample_papers)`:
  - Uses LLM to create human-readable topic titles
  - Analyzes keywords and representative papers
  - Returns engaging, descriptive title
- Register both tools with BaseAgent for specialist agents to use

**Test**: Agent discovers topics from 100+ papers, verifies topics are coherent with meaningful LLM-generated titles

### Step 4.3: Representative Paper Selection Tool
**Objective**: Create tool for agents to select most representative papers per topic

**Tasks**:
- Create `src/agents/tools/paper_selection_tool.py`
- Implement `select_representative_papers_tool(topic_id, papers, n=5, diversity_threshold=0.7)`:
  - Agent-callable tool that returns top N papers for a topic
  - Tool description explains selection criteria to LLM
  - Returns structured results with ranking scores and justifications
- Implement selection algorithm in `TopicModeler`:
  - `select_representative_papers(topic_id, papers, n=5)` - Core logic
  - Uses topic probability scores from BERTopic
  - Applies centrality measures (distance to topic centroid)
  - Ensures diversity (avoid very similar papers using cosine similarity threshold)
  - Ranks papers by combined score (representativeness + diversity)
- Add `rank_papers_by_relevance_tool(papers, query)`:
  - Allows agents to re-rank papers based on specific criteria
  - Uses semantic similarity between query and papers
- Register tools with specialized agents for paper curation

**Test**: Agent selects top 5 papers from topic with 20 papers, verify diversity and relevance

## Phase 5: AI-Powered Paper Analysis Tools

### Step 5.1: Problem Statement Extraction Tool
**Objective**: Create LLM-powered tool for agents to identify research problems

**Tasks**:
- Create `src/agents/tools/problem_extraction_tool.py`
- Implement `extract_problem_statement_tool(paper_text, paper_metadata)` as agent-callable tool:
  - Uses LLM (Claude/GPT) to analyze paper and extract problem statement
  - Tool description guides LLM to look for: motivation, research gap, specific problem
  - Returns structured output: {problem, motivation, research_gap, context}
  - Handles papers of varying structure and length (with chunking if needed)
- Create `src/analysis/paper_analyzer.py` for underlying logic:
  - `PaperAnalyzer` class with LLM client
  - `extract_sections(paper_text)` - Identify intro, methods, results, conclusion sections
  - `chunk_text(text, max_tokens)` - Split long papers for LLM processing
  - Fallback to heuristic extraction if LLM fails
- Implement prompt engineering:
  - System prompt for paper analysis task
  - Few-shot examples of problem extraction
  - Chain-of-thought reasoning for complex papers
- Register tool with specialized agents (they use domain expertise to guide extraction)

**Test**: Agent extracts problem statements from 5 diverse papers, LLM provides contextualized explanations

### Step 5.2: Key Results Extraction Tool
**Objective**: Create LLM-powered tool for agents to identify main findings

**Tasks**:
- Create `src/agents/tools/results_extraction_tool.py`
- Implement `extract_key_results_tool(paper_text, paper_metadata, domain=None)` as agent-callable tool:
  - Uses LLM with domain-specific prompts (different for math vs ML papers)
  - Tool description explains what constitutes "key results" in different domains
  - Returns structured output: [{result_type, statement, significance, location}]
  - Specialized agents can pass their domain expertise to customize extraction
- Extend `PaperAnalyzer` class:
  - `extract_key_results(paper_text, domain)` - Core extraction logic
  - Hybrid approach:
    - Pattern matching for formal statements (theorems, lemmas, propositions)
    - LLM analysis for significance and informal results
    - Section analysis (abstract, conclusion, results sections prioritized)
  - `rank_results_by_importance(results)` - Order by significance
- Implement domain-specific extraction:
  - Math papers: Look for theorem environments, proofs, corollaries
  - ML papers: Look for performance metrics, novel architectures, ablation studies
  - Crypto papers: Look for security guarantees, attack scenarios, protocols
- Register tool with all specialized agents

**Test**: Different agents extract results from papers in their domains, verify domain-specific insights

### Step 5.3: Impact Assessment Tool
**Objective**: Create LLM-powered tool for agents to evaluate research significance

**Tasks**:
- Create `src/agents/tools/impact_assessment_tool.py`
- Implement `assess_impact_tool(paper, results, field_context=None)` as agent-callable tool:
  - Uses LLM to analyze paper's impact and significance
  - Tool guides agents to evaluate: novelty, applications, theoretical importance, practical value
  - Returns structured assessment: {
      novelty_score,
      solves_open_problem,
      introduces_new_techniques,
      potential_applications,
      community_impact,
      impact_summary
    }
  - Specialized agents can provide field context to improve assessment
- Extend `PaperAnalyzer` class:
  - `assess_impact(paper, results, field_context)` - Core assessment logic
  - LLM-based analysis of:
    - Discussion/conclusion sections for author's claims
    - Novelty of approach compared to related work
    - Theoretical vs practical contributions
    - Potential for follow-up research
  - `generate_impact_narrative(assessment)` - Create readable summary
- Implement agent collaboration for impact assessment:
  - Specialized agent uses domain knowledge to contextualize impact
  - Can request hand-off to other agents for cross-domain assessment
  - Example: ML paper with algebraic techniques → Abdoulaye hands off to Alain for algebraic impact
- Register tool with all agents, enable cross-domain consultation

**Test**: Agent assesses impact for 3 papers, can request hand-offs to domain experts for specialized evaluation

## Phase 6: Interactive Julius Summary Generation

### Step 6.1: User Request Intake and Preference Model
**Objective**: Let the user tell Julius what kind of research summary they want before the agents start writing.

**Tasks**:
- Create `src/generation/user_request.py`
- Define a `SummaryRequest` Pydantic model:
  - `topic_query`: optional free-text research interest such as "LLM agents", "probability", or "cryptography"
  - `date_range`: explicit dates or relative ranges such as "last week" or "last month"
  - `audience`: `expert`, `non_expert`, or `mixed`
  - `depth`: `brief`, `standard`, or `deep`
  - `tone`: `editorial`, `technical`, `pedagogical`, or `executive`
  - `format`: `one_pager`, `bullet_digest`, `paper_rankings`, or `custom`
  - `max_topics`, `max_papers`, `must_include_categories`, `exclude_categories`
  - `delivery`: preview only, save to file, or email
- Implement `parse_user_request_tool(message, defaults)`:
  - Extracts the user's desired scope, audience, date range, and output format
  - Applies sensible defaults when the user is vague
  - Returns missing or ambiguous fields that Julius should clarify
- Implement `clarify_request_tool(summary_request)`:
  - Generates at most 3 focused follow-up questions
  - Avoids asking questions when defaults are enough to proceed
- Add request persistence in the session so Julius can remember user preferences across refinement turns.

**Test**: Julius parses user requests such as "Give me a non-technical summary of last week's LLM agent papers" and produces a complete `SummaryRequest`, asking a clarification only when required.

### Step 6.2: Julius Conversation Session
**Objective**: Build an interactive loop where the user can talk with Julius and steer the desired summary.

**Tasks**:
- Create `src/agents/julius_session.py`
- Implement `JuliusSession`:
  - Maintains conversation history, current `SummaryRequest`, generated drafts, and user feedback
  - Exposes `handle_user_message(message)` as the main entry point
  - Returns structured responses with `message`, `state`, `summary_request`, `draft_preview`, `actions_taken`, and `next_questions`
  - Supports session states: `INTAKE`, `CLARIFYING`, `PLANNING`, `GENERATING`, `AWAITING_REVIEW`, `REVISING`, `FINALIZED`
- Add Julius intents:
  - New summary request
  - Preference update ("make it shorter", "more technical", "focus on applications")
  - Scope update ("only cs.AI", "remove algebra", "include last 14 days")
  - Draft question ("why did you choose this paper?", "what is the main result?")
  - Finalization ("save this", "email it")
- Implement intent classification:
  - `classify_user_intent_tool(message, session_state)` routes each message to request intake, clarification, generation, revision, question answering, or finalization
  - Falls back to asking a concise clarification when Julius cannot infer the intent
- Implement `update_summary_request_tool(existing_request, user_feedback)`:
  - Converts feedback into structured changes
  - Keeps previous preferences unless the user overrides them
- Implement `explain_draft_choice_tool(draft, question)`:
  - Lets Julius answer questions about selected topics, papers, rankings, and omissions
- Implement response streaming/progress hooks:
  - Julius can tell the user when papers are being fetched, topics are being modeled, specialists are reviewing, and the draft is being compiled
  - Status messages remain short and do not expose internal tool noise
- Add a CLI command or interactive mode:
  - `python main.py chat`
  - User can converse with Julius until the summary is finalized
  - Julius shows short status messages when delegating to specialist agents

**Test**: In a single interactive session, the user can request a summary, revise the audience/depth/scope, ask why a paper was selected, and finalize the output without restarting the workflow.

### Step 6.3: Multi-Agent Draft Generation with Hand-offs
**Objective**: Have Julius coordinate specialist agents to produce the first draft according to the user's preferences.

**Tasks**:
- Create `src/agents/tools/synthesis_tools.py`
- Implement synthesis tools for agent collaboration:
  - `create_topic_overview_tool(topic, papers, analyses, summary_request)` - Synthesize topic summary for the requested audience and depth
  - `create_paper_summary_tool(paper, analysis, summary_request)` - Summarize one representative paper according to user preferences
  - `generate_expert_explanation_tool(content, domain)` - Technical version for experts
  - `generate_layperson_explanation_tool(content, metaphors=None)` - Accessible version for non-experts
  - `rank_summary_items_tool(items, ranking_goal)` - Rank papers/topics by relevance, impact, novelty, or user interest
  - `review_and_refine_tool(content, criteria)` - Quality check and improvement
- Create `src/generation/synthesizer.py`:
  - `ContentSynthesizer` class with LLM client
  - Converts agent analyses into structured draft sections
  - Uses `SummaryRequest` to control length, tone, terminology, topic count, and explanation style
- Implement Julius workflow for draft generation:
  - Julius creates an execution plan from `SummaryRequest`
  - Julius delegates topic summaries to specialized agents based on ArXiv categories and detected research area
  - Each specialist creates expert-level findings and flags uncertain claims
  - Michel reviews explanations when the requested audience is `non_expert` or `mixed`
  - Julius compiles the first draft and records why each topic and paper was included
- Implement hand-off context for content generation:
  - `HandoffContext` includes the user request, selected papers, target audience, domain, draft constraints, and previous feedback
  - Agents return structured sections plus confidence notes
- Register synthesis tools with Julius and the specialist agents.

**Test**: Julius generates a first draft that respects requested topic, date range, audience, depth, and format, with visible provenance for selected papers.

### Step 6.4: User-Guided Revision Loop
**Objective**: Let the user refine the draft until it matches what they want.

**Tasks**:
- Create `src/generation/revision.py`
- Implement `RevisionRequest` model:
  - `target`: whole document, section, topic, paper, title, or explanation level
  - `operation`: shorten, expand, simplify, make technical, change tone, add/remove topic, rerank, regenerate
  - `instructions`: free-text user feedback
- Implement `parse_revision_request_tool(user_feedback, current_draft)`:
  - Converts natural language feedback into a structured revision request
  - Detects whether new data fetching or agent re-analysis is required
- Implement `revise_draft_tool(draft, revision_request, summary_request)`:
  - Applies local edits when possible
  - Triggers Julius hand-offs when the feedback requires specialist review
  - Preserves citations, paper metadata, and provenance
- Add draft versioning:
  - Save each draft as `draft_v1`, `draft_v2`, etc.
  - Store change summaries so Julius can explain what changed
  - Allow rollback to a previous draft if the user prefers it
- Add final approval step:
  - Julius asks for confirmation before saving or emailing
  - Final output is marked immutable unless the user starts another revision

**Test**: User feedback such as "make the first topic more intuitive", "remove cryptography", or "give me a deeper expert version" produces a revised draft while preserving paper references and workflow state.

### Step 6.5: Formatting, Quality Assurance, and Final Output
**Objective**: Format, validate, and deliver the approved summary.

**Tasks**:
- Create `src/agents/tools/formatting_tool.py`
- Implement `format_document_tool(content, format="markdown", style="professional")`:
  - Supports Markdown first, then HTML and PDF
  - Formats according to `SummaryRequest.format`
  - Includes title, date range, generated-at timestamp, selected topics, representative papers, and agent credits
- Create `src/generation/formatter.py`:
  - `DocumentFormatter` class with template engine
  - Templates for `one_pager`, `bullet_digest`, `paper_rankings`, and `custom`
  - `apply_template(content, template_name)`
  - `render_to_format(template, output_format)`
- Create `src/agents/tools/quality_check_tool.py`
- Implement `validate_quality_tool(document, summary_request, source_papers)`:
  - Checks completeness against user preferences
  - Checks length constraints for the requested format
  - Verifies citations and paper metadata against fetched papers
  - Checks that expert/non-expert explanations match the requested audience
  - Flags uncertain claims for human review
- Create `src/generation/validator.py`:
  - `DocumentValidator` class with deterministic checks first and LLM checks second
  - `generate_improvement_suggestions(validation_report)`
- Julius quality workflow:
  - Validate the draft before showing it to the user
  - Validate again after each major revision
  - Show concise warnings when confidence is low or a claim needs review
  - Save the final document to `outputs/`
  - Hand off to Phase 7 email delivery if requested

**Test**: Julius formats an approved summary, validates it against the user's preferences, saves it to `outputs/`, and optionally passes it to the email workflow.

### Step 6.6: 
**Objective**: Provide a simple web app where the user can interact with Julius, refine the requested summary conversationally, preview drafts, and finalize the desired result.

**Tasks**:
- Add Streamlit as the first interactive UI:
  - Add `streamlit` to `requirements.txt`
  - Create `app.py` as the local Streamlit entry point
  - Create `src/ui/streamlit_app.py` for UI composition so the app stays thin and testable
  - Run locally with `streamlit run app.py`
- Create `src/generation/interactive_workflow.py`
- Implement `InteractiveSummaryWorkflow`:
  - Accepts a `JuliusSession`, configured agents, fetcher/topic/analysis tools, and output services
  - Orchestrates the full flow: intake → clarification → planning → data collection → specialist analysis → draft generation → user review → revision → final output
  - Keeps workflow state resumable so an interrupted session can continue without losing the request or latest draft
- Build the Streamlit interaction model:
  - Use `st.session_state` to hold `JuliusSession`, `SummaryRequest`, workflow state, draft versions, validation reports, and final output path
  - Use `st.chat_message` and `st.chat_input` for natural conversation with Julius
  - Show Julius progress updates with `st.status` or `st.progress` while fetching, analyzing, drafting, revising, and validating
  - Keep long-running work behind explicit user actions such as "Generate draft", "Revise draft", "Validate", and "Finalize"
- Add sidebar controls for explicit preferences:
  - Topic/query input
  - Date range selector and quick choices such as last week, last month, custom range
  - Audience selector: expert, non-expert, mixed
  - Depth selector: brief, standard, deep
  - Tone selector: editorial, technical, pedagogical, executive
  - Output format selector: one-pager, bullet digest, paper rankings, custom
  - Category include/exclude multiselects
  - Max topics and max papers sliders
- Add draft review UI:
  - Main preview tab for the formatted summary
  - Metadata tab showing selected papers, ArXiv categories, date range, and contributing agents
  - Quality tab showing validation warnings, confidence notes, and missing information
  - Revision history tab showing draft versions and change summaries
  - Buttons for approving, saving, downloading Markdown/HTML, or sending to Phase 7 email later
- Define Julius' user-facing contract:
  - Julius acknowledges the interpreted request before starting expensive work
  - Julius states any assumptions, such as default date range or audience, in plain language
  - Julius asks only necessary clarification questions
  - Julius offers the draft for review before final save/email
  - Julius can explain why topics or papers were included or excluded
- Add summary desire matching:
  - Track `satisfaction_signals` from user feedback, such as accepted draft, requested changes, rejected sections, and repeated preferences
  - Stop revising only when the user approves the summary or explicitly asks to finalize
  - Record final preferences for future sessions when persistence is enabled
- Implement workflow-level error recovery:
  - If fetching fails, Julius explains what failed and offers a retry, narrower scope, or cached/partial summary
  - If too few papers are found, Julius proposes expanding the date range or broadening categories
  - If an agent fails, Julius can continue with available results and flag the limitation
- Add integration tests with mocked tools:
  - Full happy path: user asks for a mixed-audience LLM agent summary, Julius generates a draft, user asks for a shorter version, Julius finalizes it
  - Clarification path: vague request triggers at most 3 questions, then proceeds after answers
  - Revision path: user changes audience, topic scope, and output format without restarting
  - Failure path: paper fetching failure produces a recoverable Julius response
- Add Streamlit-focused tests and checks:
  - Unit-test UI adapter functions without launching a browser
  - Smoke-test that `app.py` imports and builds the page without requiring API keys
  - Verify session state survives reruns for request, draft, and revision history
  - Verify mocked workflow failures appear as recoverable Julius messages in the UI
- Add README/UI documentation:
  - Show example `streamlit run app.py`
  - Show example `python main.py chat`
  - Include a short transcript demonstrating request, clarification, draft review, revision, and finalization
  - Include a screenshot or description of the Streamlit layout once the UI is implemented

**Test**: A mocked Streamlit workflow proves that the user can interact with Julius, adjust preferences, generate and preview a draft, request revisions, ask explanatory questions, validate quality, and finalize/download the summary they want.

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
  9. Save one-pager to data/one_pager folder
- Add progress logging and error recovery

**Test**: Run end-to-end with default settings (last week, all agents), verify complete one-pager is generated and saved

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
