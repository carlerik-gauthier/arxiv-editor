# Goal
- Build an agentic workflow that collects papers from different ArXiv categories to generate a one-pager summary on a field of research specified by the user. If not specified, all fields should be considered
- The one-paper contains at most 5 topics
- Implement it with openai agents SDK

# Rules
- You must use openai agent SDK whenever it is relevant. In particular, Agent, Tools and Tracing must use openai agent SDK
- At each phase, you must create a new Streamlit app to allow to test if the phase is successfully implemented.
    - The  Streamlit apps must allow a chat with the user during a session. In particular, it must have a memory. 
- You must complete a phase, marked with ✅, before moving to the next one
- You must not alter the old_src folder
- refer to src_new/arxiv_fetcher.py and src_new/topic_finder.py whenever it is needed
- You must be concise
- you must ignore old_src

# Plan
## Phase 1: Agent Chris creation,
- Create an Agent called `ChrisAgent` (in honor of Krzystof Burdzy), a mathematician agent specialized in Probability Theory and Statistic Theory.
- `ChrisAgent` has only access to the following ArXiv categories: math.PR and stat.TH
- The system prompt is: "Probability theory expert, focuses on stochastic processes. You identify key concepts and see application in other fields, such as physics"
- `ChrisAgent` has one tool: arxiv_fetcher_tool. It returns a list of Papers
    - It is used whenever there is no Paper from Probability Theory in the date ranges requested by the user
    - If the number of fecthed paper is less than a predined threshold, then returns nothing and explain why 
    - Depending on the request, `ChrisAgent` fetches paper from a subset of available ArXiv or to all 
## Phase 2: Provide new tools to ChrisAgent
- Add a second tool to `ChrisAgent`: find_topic_tools
    - Use it whenever you need to extract topics from an already fetched list of Papers
- Add a third tool to `ChrisAgent`: extract_main_result_tool
    - Use it whenever you need to access the entire Paper content to find and explain the main results
## Phase 3: Agent Julius creation
- Create an Agent called `JuliusAgent` (in honor of Julius Springer), founder of the Springer book edition.
    - System prompt: Editor and coordinator role, responsible for planning, delegation and generating the one-pager
    - Has `ChrisAgent` as a tool. Call it whenever you need content related to Probability or Statistic queries. 
- `JuliusAgent` determines how many topics he need from `ChrisAgent`.
- `JulisAgent` owns the editorial workflow and interacts with the user and is responsible for the Conversation flow:
    - Parse user request (date range, topics, preferences)
    - Create execution plan with agent assignments
    - Coordinate parallel agent execution where possible
    - Synthesize results into coherent one-pager
## Phase 4: Agent Michel Creation
- Create an Agent called `MichelAgent` (in honor of Michel Benaim), a mathematician with outstanding skills to provide mathematical intuition on complex mathematical concepts to non-experts using examples and metaphor
- `MichelAgent` has 3 tools:
    - make_clearer_tool: use it when reformulation is needed ti vulgarize it
    - provide_intuition_tool: use it when examples and intuition are requested to explain a concept
    - metaphor_tool: use it to provide metaphors that are easier to understand when explaining concepts
- Add `MichelAgent` as a tool from `JuliusAgent`
    - `JuliusAgent` calls `MichelAgent` when vulgarization is needed
## Phase 5: Create the other specialized agents
- Create the following remaining specialized agent
    - `AlainAgent` (in honor of Alain Valette) is a mathematician agent specialized in Algebra (math.AG, math.RA, math.GR). He communicates with passion
        - System prompt: Algebraic structures specialist
    - `BrunoAgent` is a mathematician agent specialized in Spectral/Riemannian Geometry (math.DG, math.SP). He is extremely rigorous when communicating.
        - System prompt: Geometry expert, emphasizes geometric intuition
    - `ElisaAgent` is a mathematician agent specialized in Applied math/Cryptography (cs.CR, math.OC). She is dynamic and is result-oriented in her communication
        - System prompt: Applied mathematics and cryptography specialist
    - `FelixAgent` is a mathematician agent specialized in Dynamical systems/Symplectic geometry (math.DS, math.SG). He is the very smart and crazy mathematician
        - System prompt: Dynamical systems expert, focuses on long-term behavior
    - `AbdoulayeAgent` is a ML researcher agent specialized in Machine Learning (cs.LG, stat.ML). He is very enthusiastic and eager in ethic AI, which reflects in his communication
        - System prompt: ML researcher, explains algorithms and applications
    - `JeanBaptisteAgent` is a Data Scientist agent specialized in Data Science/NLP/LLM/Agentic AI (cs.CL, cs.AI, cs.MA, cs.CE). He is very experience in deploying model in production and has a concise communication style
        - System prompt: Data science expert specializing in NLP, LLMs, and agentic systems
- All those agents have the same tools as `ChrisAgent`. The difference lies in their expertise fields
- Add those agents as tools for `JuliusAgent`
## Phase 6: Review, Refactor, Unit test and Document
- Review the code by removing useless line of codes.
- The code must be easy to maintain for future development
- Refactor the code whereever it is needed to prevent code duplication
- generate the unit tests
- Document all functions
- Provide a documentation about the workflow and add a schema about the agentic system