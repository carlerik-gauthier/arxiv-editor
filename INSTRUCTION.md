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
## Phase 1: Agent Chris creation ✅
- Create an Agent called `ChrisAgent` (in honor of Krzystof Burdzy), a mathematician agent specialized in Probability Theory and Statistic Theory.
- `ChrisAgent` has only access to the following ArXiv categories: math.PR and math.ST
- The system prompt is: "Probability theory expert, focuses on stochastic processes. You identify key concepts and see application in other fields, such as physics"
- `ChrisAgent` has one tool: arxiv_fetcher_tool. It returns a list of Papers
    - It is used whenever there is no Paper from Probability Theory in the date ranges requested by the user
    - If the number of fecthed paper is less than a predined threshold, then returns nothing and explain why 
    - Depending on the request, `ChrisAgent` fetches paper from a subset of available ArXiv or to all 
## Phase 2: Provide new tools to ChrisAgent ✅
- Add a second tool to `ChrisAgent`: find_topic_tools
    - Use it whenever you need to extract topics from an already fetched list of Papers. As input, you need a csv location where the papers' metadata are stored.
- Add a third tool to `ChrisAgent`: extract_main_result_tool
    - Use it whenever you need to access the entire paper content to find and explain the main results. You must download it from Arxiv and use arxiv_id to 
## Phase 3: Finalize ChrisAgent ✅
- combine chris_agent_phase1 and chris_agent_phase2 into a unique file chris_agent.py. You are not allowed to import from chris_agent_phase1 or chris_agent_phase2. You must combine them
## Phase 4: Agent Julius creation ✅
- Create an Agent called `JuliusAgent` (in honor of Julius Springer), founder of the Springer book edition.
    - System prompt: Editor and coordinator role, responsible for planning, delegation and generating the one-pager. The one-pager must meet the user request, including tone. The one-pager must be engaging. You can use emojis or speech elevator techniaues to make it appealing. You must remain professional. Unless stated otherwise by the user, the one-pager is aimed for a LinkedIn post. The post must contains between 1 and 5 topics.
    - Has `ChrisAgent` as a tool. Call it whenever you need content related to Probability or Statistic queries. For other domains, reply politely you don't have knowledge about it
- `JuliusAgent` determines how many topics he need from `ChrisAgent`.
- `JulisAgent` owns the editorial workflow and interacts with the user and is responsible for the Conversation flow:
    - Parse user request (date range, topics, preferences)
    - Create execution plan with agent assignments
    - Coordinate parallel agent execution where possible
    - create an engaging one-pager structure
    - Synthesize results into a coherent one-pager
## Phase 5: Agent Michel Creation ✅
- Create an Agent called `MichelAgent` (in honor of Michel Benaim), a mathematician with outstanding skills to provide mathematical intuition on complex mathematical concepts to non-experts using examples and metaphor
- `MichelAgent` has 3 tools:
    - make_clearer_tool: use it when reformulation is needed to vulgarize it
    - provide_intuition_tool: use it when examples and intuition are requested to explain a concept
    - metaphor_tool: use it to provide metaphors that are easier to understand when explaining concepts
- Add `MichelAgent` as a tool from `JuliusAgent`
    - `JuliusAgent` calls `MichelAgent` when vulgarization is needed for a general audience. `JuliusAgent` calls `MichelAgent` as long as the result is not satisfactory  
## Phase 6: Create a second specialized agent ✅
- Create the following remaining specialized agent
    - `AlainAgent` (in honor of Alain Valette) is a mathematician agent specialized in Algebra (math.AG, math.RA, math.GR). He communicates with passion
        - System prompt: Algebraic structures specialist
    - `AlainAgent` has the same tools as `ChrisAgent`. The difference lies in their expertise fields
- Add the second agent as tool for `JuliusAgent`
- Update `JuliusAgent` system prompt by adding "You are responsible to allocate the number of topics to the different specialized agents. If an agent cannot return you requested, you pick another topic from another agent"
- Add a new tool to `JuliusAgent` to allocate the number of topics per specialized agents based on user requests
## Phase 7: Review, Refactor, Unit test
- Review the code by removing useless line of codes.
- Refactor the code whereever it is needed to prevent code duplication. The code must be easy to maintain for future development
- generate the unit tests
## Phase 8 : Add the remaining specialized agents
- Create the following remaining specialized agent
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
- add an helper between the 2 main families of fields : mathematics (probability, statistics, cryptography, applied mathematics, algebra, geometry, dynamical system), and AI (data science, Machine Learning, Generative AI, NLP, LLM)
## Phase 9: Add personality
- Update prompts to make sure answer reflects agents' personalities
    - `ChrisAgent`: TODO
    - `MichelAgent`: TODO
    - `AlainAgent`: TODO
    - `BrunoAgent`: TODO
    - `ElisaAgent`: TODO
    - `FelixAgent`: TODO
    - `AbdoulayeAgent`: TODO
    - `JeanBaptisteAgent`: TODO
## Phase 10: Review, Refactor, Unit test and Document
- Review the code by removing useless line of codes.
- Refactor the code whereever it is needed to prevent code duplication. The code must be easy to maintain for future development
- generate the unit tests
- Document all functions
- Provide a documentation about the workflow and add a schema about the agentic system


get topics of papers from 2026-05-19 to 2026-05-21
get topics of papers from 2026-05-19 to 2026-05-21 and explain the main results for the representative papers
provide the two main topics of probability papers from 2026-05-25 to 2026-05-29. For every topic, get one representative paper and describe the main results. 

**test 1**
I want to get the two main topics of probability papers from 2026-05-25 to 2026-05-29. For every topic, I want to have one representative paper and have the main results described. -> OK

I want to get two main topics in probability and one in algebra from 2026-05-25 to 2026-05-29. For every topic, I want to have one representative paper and have the main results described. I plan to write a LinkedIn post.  -> OK

**test 2**
I want to get three main topics from 2026-05-25 to 2026-05-29. For every topic, I want to have one representative paper and have the main results described. I plan to write a post on a mathematic specialized blog for graduate students. -> OK

I want to get four main topics from 2026-05-25 to 2026-05-29. For every topic, I want to have one representative paper and have the main results described. I plan to write a LinkedIn post. -> OK

**test 3**
I want to get two main topics in group theory from 2026-05-25 to 2026-05-29. For every topic, I want to have one representative paper and have the main results described. I plan to write a LinkedIn post. 

follow-up: extend the time window by one week earlier
-> OK