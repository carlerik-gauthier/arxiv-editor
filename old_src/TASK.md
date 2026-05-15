-- 1. Review the entire repo to wire agents and tools to use Agent and Tools from openai Agent SDK. See https://openai.github.io/openai-agents-python/tools/ and https://openai.github.io/openai-agents-python/agents/.
-- 2. Tracing must be done from openai Agent SDK. See https://openai.github.io/openai-agents-python/tracing/
-- 3. Remove unnecessary line of codes. Be consistent with the different data object
-- 1. Review agents/specialized_agents.py that it uses Agent and Tools from openai Agent SDK. If not, fix it
-- 2. When Julius handoff to one of the specialized agent (except Michel), it must specify the maximum number of topics it wants to get. Make sure that is the case
-- 3. Specialized agent (except Michel) have the same workflow pattern: extract papers from Arxiv to get at least 60 papers (if not enough in the time range provided by the user, then it extends it), run bertopic to extract the topic and return the number of topic requested by Julius (i.e. topic name, description and representative papers, and any extra piece of information needed for Julius). Make sure it is what it is implemented 
-- 4. Topic description in Bertopic must contain LLM call
-- 5. Once Julius received all element, it is responsible for generating the output. Make sure it follows the pattern
-- Topic 1 name + description
-- topic 1 main results and importance
-- references for topic 1
-- Topic 2 name + description
-- topic 2 main results and importance
-- references for topic 2
-- Topic 3 name + description
-- topic 3 main results and importance
-- references for topic 3
-- etc

test: make a summary about probability research papers from the last month