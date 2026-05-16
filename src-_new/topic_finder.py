import openai
import os
import tiktoken
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import OpenAI as OpenAIRepresentation
from bertopic.representation import MaximalMarginalRelevance
from dotenv import load_dotenv

load_dotenv(override=True)


# Load a custom embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

summarization_prompt = """
I have a topic that is described by the following keywords: [KEYWORDS]
In this topic, the following documents are a small but representative subset of all documents in the topic:
[DOCUMENTS]

Based on the information above, give a concise description of this topic in the following format:
topic: <description>
"""


topic_title_prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""

# api_key=os.getenv('OPENAI_API_KEY')
# client = openai.OpenAI(api_key=api_key)
import tiktoken


# Tokenizer
# tokenizer= tiktoken.encoding_for_model("gpt-4o-mini")
#representation_model = [MaximalMarginalRelevance(diversity=0.3),
#                        OpenAIRepresentation(
                            # client,
                            # model="gpt-4o-mini",
                            # prompt=summarization_prompt,
                            # chat=True,
                            # nr_docs=5,
#                             doc_length=200,
#                             tokenizer=tokenizer
#                             )
#                             ]
# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     representation_model=representation_model
#     )
# topics, probs = topic_model.fit_transform(docs)
# topic_model.fit(docs)
# topic_info.loc[1, 'Representation'][0]

# get_representative_docs() returns the most relevant documents per topic.
# Generate topic labels	.generate_topic_labels()
# .representative_docs_	The representative documents for each topic if HDBSCAN is used.

# https://maartengr.github.io/BERTopic/getting_started/representation/representation.html
# https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#prompt-engineering
# https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py
# topic_model.get_topic_info()