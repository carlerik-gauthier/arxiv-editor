import openai
import os
import tiktoken
import pandas as pd
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import OpenAI as OpenAIRepresentation
from bertopic.representation import MaximalMarginalRelevance
from dotenv import load_dotenv
from typing import List
from data_object import Paper

load_dotenv(override=True)

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

def compute_topics(papers: List[Paper], n_topics: int=1, n_papers_per_topic: int=3):
    docs = [f"{p.title} -- {p.summary}" for p in papers]
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    tokenizer= tiktoken.encoding_for_model("gpt-4o-mini")

    topic_summary_representation_model = [
        MaximalMarginalRelevance(diversity=0.3),
        OpenAIRepresentation(
            client,
            model="gpt-4o-mini",
            prompt=summarization_prompt,
            chat=True,
            nr_docs=10,
            doc_length=200,
            diversity=0.3,
            tokenizer=tokenizer
            )
            ]
    
    topic_title_representation_model = [
        MaximalMarginalRelevance(diversity=0.3),
        OpenAIRepresentation(
            client,
            model="gpt-4o-mini",
            prompt=topic_title_prompt,
            chat=True,
            nr_docs=10,
            doc_length=200,
            diversity=0.3,
            tokenizer=tokenizer
            )
            ]
    
    representation_models = {
        "Main": topic_summary_representation_model,
        "topic_title_": topic_title_representation_model
    }
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # uses default parameter of Bertopic for dimension reduction and clustering
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_models
        )
    
    label, prob = topic_model.fit_transform(docs)
    # get topic info and keep top n_topics
    topic_info_base = topic_model.get_topic_info()

    topic_info_base = deepcopy(topic_info_base[topic_info_base.Topic!=-1]).sort_values(by="Count", ascending=False)
    topic_info = topic_info_base.head(n_topics)
    topic_info.reset_index(drop=True, inplace=True)
    topic_info.rename(columns={"Topic": "topic"}, inplace=True)
    # find the most representative papers defined as the top n paper per topic
    df = pd.DataFrame(data={
        # "arxiv_id": [p.arxiv_id for p in papers],
        "paper": papers,
        "topic": label,
        "probabilities": prob
    })

    paper_topic_df = df.merge(topic_info[["topic"]], how="inner", on="topic")
    paper_topic_df['rk'] = paper_topic_df.sort_values(['probabilities'], ascending=[False]) \
             .groupby(['topic']) \
             .cumcount() + 1
    paper_topic_df = deepcopy(paper_topic_df[paper_topic_df.rk<=n_papers_per_topic]).reset_index(drop=True)
    representative_papers = paper_topic_df[["topic", "paper"]].groupby(by="topic").apply(list)

    # combine all infos to return a list of dictionaries. The list length is equal to n_topics and 
    # dictionaries are like
    # {
    #     "topic_title": <title of the topic>,
    #     "nb_papers": <size of the topic>,
    #     "topic_description": <a brief description about the content of the topic>,
    #     "representative_papers": <a list of length n_papers_per_topic of Paper object>
    # }
    final_df = topic_info.merge(representative_papers, on="topic", how="inner")
    final_df["topic_description"] = final_df["Representation"].apply(lambda arr: arr[0])
    final_df["topic_title"] = final_df["topic_title_"].apply(lambda arr: arr[0])
    final_df = deepcopy(final_df[["topic_title", "Count", "topic_description", "paper"]]).rename(
        columns={"Count": "nb_papers", "paper": "representative_papers"})
    
    return final_df.to_dict("records")
