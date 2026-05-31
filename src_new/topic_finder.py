import openai
import os
import tiktoken
import pandas as pd
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import OpenAI as OpenAIRepresentation
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from dotenv import load_dotenv
from typing import List
from src_new.data_object import Paper

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
# FIX papers to paper_path; and papers is now a pandas dataframe
def compute_topics(path: str, n_topics: int=1, n_papers_per_topic: int=3):
    data = pd.read_csv(path)
    data['docs'] = data[["title", "summary"]].apply(
        lambda arr: f"{arr[0]} -- {arr[1]}", raw=True, axis=1
        )
    docs = list(data['docs'].values)
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    tokenizer= tiktoken.encoding_for_model("gpt-4o-mini")
    main_representation = KeyBERTInspired()
    topic_summary_representation_model = [
        MaximalMarginalRelevance(diversity=0.3),
        OpenAIRepresentation(
            client,
            model="gpt-4o-mini",
            prompt=summarization_prompt,
            chat=True,
            nr_docs=10,
            doc_length=2000,
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
        "Main": main_representation,
        "topic_summary_": topic_summary_representation_model,
        "topic_title_": topic_title_representation_model
    }
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # uses default parameter of Bertopic for dimension reduction and clustering
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_models
        )
    
    topic_model.fit(docs)
    
    label, prob = topic_model.transform(docs) # fit_transform(docs)
    data['topic'] = label
    data['probability'] = prob
    # get topic info and keep top n_topics
    topic_info_base = topic_model.get_topic_info()

    topic_info_base = deepcopy(topic_info_base[topic_info_base.Topic!=-1]).sort_values(by="Count", ascending=False)
    topic_info = topic_info_base.head(n_topics)
    topic_info.reset_index(drop=True, inplace=True)
    topic_info.rename(columns={"Topic": "topic"}, inplace=True)
    # find the most representative papers defined as the top n paper per topic
    # df = pd.DataFrame(data={
        # "arxiv_id": [p.arxiv_id for p in papers],
    #     "paper": papers,
    #     "topic": label,
    #     "probabilities": prob
    # })

    paper_topic_df = data.merge(topic_info[["topic"]], how="inner", on="topic")
    paper_topic_df['rk'] = paper_topic_df.sort_values('probability', ascending=False) \
             .groupby(by='topic') \
             .cumcount() + 1
    paper_topic_df = deepcopy(paper_topic_df[paper_topic_df.rk<=n_papers_per_topic]).reset_index(drop=True)
    representative_papers = paper_topic_df[["topic", "arxiv_id"]].groupby(by="topic")['arxiv_id'].apply(list).reset_index()
    representative_papers_title = paper_topic_df[["topic", "title"]].groupby(by="topic")['title'].apply(list).reset_index()
    # combine all infos to return a list of dictionaries. The list length is equal to n_topics and 
    # dictionaries are like
    # {
    #     "topic_title": <title of the topic>,
    #     "nb_papers": <size of the topic>,
    #     "topic_description": <a brief description about the content of the topic>,
    #     "representative_papers": <a list of length n_papers_per_topic of Paper object>
    # }
    final_df = topic_info.merge(representative_papers, on="topic", how="inner")
    final_df = deepcopy(final_df).merge(representative_papers_title, on="topic", how="inner")
    final_df["topic_description"] = final_df["topic_summary_"].apply(lambda arr: arr[0])
    final_df["topic_title"] = final_df["topic_title_"].apply(lambda arr: arr[0])
    #return final_df, paper_topic_df, representative_papers
    final_df = deepcopy(final_df[["topic_title", "Count", "topic_description", "arxiv_id", "title"]]).rename(
        columns={"Count": "nb_papers", "arxiv_id": "representative_papers_arxiv_id", "title": "representative_papers_title"})
    
    return final_df.to_dict("records")
