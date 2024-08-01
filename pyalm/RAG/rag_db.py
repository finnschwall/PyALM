import datetime
import json
import logging
import pickle

import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from sentence_transformers.util import cos_sim
import re
import rixaplugin
import requests
import pandas as pd
import os
import regex as re
import xml.etree.ElementTree as ET
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

knowledge_logger = logging.getLogger("knowledge_db")


model = None
tokenizer = None
device = None
embeddings_db = None
embeddings_list = None
doc_metadata_db=None
current_doc_id = 0

def init():
    """
    Initialize the database and the model

    Usually this file is used in a script to create, modify or query the database.
    This needs to be called before any other function in this file is called.
    """
    global embeddings_db
    global model
    global tokenizer
    global device
    global embeddings_list, doc_metadata_db, current_doc_id

    tokenizer = AutoTokenizer.from_pretrained('Snowflake/snowflake-arctic-embed-m-long')
    model = AutoModel.from_pretrained('Snowflake/snowflake-arctic-embed-m-long', trust_remote_code=True,
                                      add_pooling_layer=False, safe_serialization=True)

    # Alternatively you can use the below. Works great for small datasets but not for large ones
    # It is also much faster.
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-distilbert-base-v4')
    # model = AutoModel.from_pretrained('sentence-transformers/msmarco-distilbert-base-v4', trust_remote_code=True,)

    # Other options I noticed:
    # for different languages: distiluse-base-multilingual-cased-v1
    # For asymmetric similarity (very long entries in the database but short user questions). However its huge (2.3gb): BAAI/bge-m3
    # High on scoreboard: https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/tree/main

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.to(device)
    model.eval()

    if os.path.exists("embeddings_df.pkl"):
        embeddings_db = pd.read_pickle("embeddings_df.pkl")
        doc_metadata_db = pd.read_pickle("doc_metadata_df.pkl")
        with open("embeddings.pkl", "rb") as f:
            embeddings_list = pickle.load(f)
        current_doc_id = doc_metadata_db["doc_id"].max()
    else:
        reset_db()


def reset_db():
    """
    Reset the database to empty state. Purges all data.
    """
    global embeddings_db, embeddings_list, doc_metadata_db, current_doc_id
    current_doc_id = 0
    embeddings_db = pd.DataFrame(columns=["doc_id", "header", "subheader", "location", "url", "tags", "content", ])
    embeddings_list = None
    doc_metadata_db = pd.DataFrame(
        columns=["doc_id", "document_title", "source", "authors", "publisher", "tags", "creation_time", "source_file"])


def query_db_as_string(query, top_k=3, query_tags=None, embd_db=None):
    df, scores = query_db(query, top_k, query_tags, embd_db)
    result = ""
    for i, row in df.iterrows():
        result += f"DOCUMENT TITLE: {row['document_title']}\nDOCUMENT SOURCE: {row['source']}\nSUBTITLE: {row['content']}\n\n"
    return result


def query_db(query, top_k=5, min_score=0.5, query_tags=None, max_chars=3500):
    """
    Query the database for relevant documents

    Converts the query into embedding space and finds the most relevant entries using cosine similarity


    :param query: The query to search for in the database
    :param top_k: The number of top results to return
    :param min_score: The minimum score to consider a document relevant (0-1)
    :param query_tags: The tags to filter the search by
    :param max_chars: The maximum number of characters to return in the content of the documents
    :return: A tuple with the first element being a dataframe of the results and the second element being the list of scores
    """
    global embeddings_db, embeddings_list, doc_metadata_db
    df = embeddings_db
    query_prefix = 'Represent this sentence for searching relevant passages: '
    queries = [query]
    queries_with_prefix = [f"{query_prefix}{i}" for i in queries]
    query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)
    query_tokens.to(device)
    with torch.no_grad():
        query_embeddings = model(**query_tokens)[0][:, 0]
    query_embeddings = query_embeddings.cpu()
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1).numpy()

    if query_tags:
        filtered_df = df[df['tags'].apply(lambda tags: query_tags.issubset(tags))]
    else:
        filtered_df = df
    idx = list(filtered_df.index)
    filtered_embeddings = embeddings_list[idx]
    scores = np.dot(query_embeddings, filtered_embeddings.T).flatten()
    idx = np.argsort(scores)[-top_k:][::-1]
    ret_idx = np.where(scores[idx] > min_score)

    final_docs = filtered_df.iloc[idx].iloc[ret_idx]
    final_scores = scores[idx][ret_idx]

    final_results = pd.merge(final_docs, doc_metadata_db.drop("tags", axis=1), on="doc_id")
    final_results.drop(["creation_time", "source_file"], inplace=True, axis=1)

    if max_chars == 0 or max_chars == -1 or max_chars is None:
        return final_results, final_scores
    else:
        char_count = [len(i) for i in final_results["content"]]
        cumsum = np.cumsum(char_count)
        idx = np.where(cumsum < max_chars)
        return final_results.iloc[idx], final_scores[idx]


def from_json(path):
    """
    Add a document from a json file to the database

    The JSON needs to be in the following format:

    [
        {
            "title": "Title of the document",
            "source": "Source of the document",
            "tags": ["tag1", "tag2"],
            "authors": ["author1", "author2"], #OPTIONAL
            "publisher": "Publisher of the document" #OPTIONAL
        },
        {
            "header": "Header of the content", #OPTIONAL
            "subheader": "Subheader of the content", #OPTIONAL
            "page": "Page number", #OPTIONAL
            "content": "Content"
        },
        ...
    ]

    The content entries should not contain more than ~10 sentences.
    Use the llm_to_doc notebook to split content into smaller chunks if needed.

    :param path: The path to the json file
    """
    global doc_metadata_db, embeddings_db, embeddings_list, current_doc_id
    with open(path, "r") as f:
        data = json.load(f)

    metadata = data[0]
    current_doc_id += 1
    doc_id = current_doc_id

    mod_time = os.path.getmtime(path)
    metadata_entry = {"doc_id": doc_id,
                      "document_title": metadata["title"] if "title" in metadata else metadata["document_title"],
                      "source": metadata["source"] if "source" in metadata else metadata["source_url"],
                      "tags": metadata["tags"],
                      "creation_time": datetime.datetime.utcfromtimestamp(mod_time).strftime('%H:%M %d/%m/%Y'),
                      "source_file": os.path.basename(path),
                      "authors": metadata.get("authors", None),
                      "publisher": metadata.get("publisher", None)}

    content_entries = []
    for entry in data[1:]:
        content_entries.append({"doc_id": doc_id, "content": entry["content"], "tags": metadata["tags"],
                                "header": entry["header"] if "header" in entry else None,
                                "subheader": entry["subheader"] if "subheader" in entry else "",
                                "location": f'Page {entry["page"]}' if "page" in entry else ""})

    add_entry(metadata_entry, content_entries)


def add_entry(metadata_entry, content_entries):
    """
    Add an entry to the database

    Usually you do not call this directly. Use the from_json function instead.

    This expects the entries to already be compatible with the database schema.
    """

    global doc_metadata_db, embeddings_db, embeddings_list

    doc_metadata_db = pd.concat([doc_metadata_db, pd.DataFrame([metadata_entry])], ignore_index=True)
    doc_metadata_db = doc_metadata_db.convert_dtypes()
    df = pd.DataFrame(content_entries)
    additional_embeddings = calculate_embeddings(df)
    if embeddings_list is None:
        embeddings_list = additional_embeddings
    else:
        embeddings_list=np.concatenate((embeddings_list, additional_embeddings,), axis=0)
    # embeddings_list.append(additional_embeddings)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_list, f)
    embeddings_db = pd.concat([embeddings_db, df], ignore_index=True).convert_dtypes()
    embeddings_db.to_pickle("embeddings_df.pkl")
    doc_metadata_db.to_pickle("doc_metadata_df.pkl")



def calculate_embeddings(df):
    """
    Calculate the embeddings for a dataframe

    Usually you do not call this directly. Use the add_entry function instead.
    """
    global tokenizer, model, device
    content_col = df["content"].tolist()
    total_rows = len(content_col)
    # 8245MiB usage for model size 547 MiB with chunk size 100

    embeddings_list_temp = []
    chunk_size = 100
    for start_idx in tqdm(range(0, total_rows, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = content_col[start_idx:end_idx]
        document_tokens = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt',
                                    max_length=512)
        document_tokens.to(device)
        with torch.no_grad():
            doument_embeddings = model(**document_tokens)[0][:, 0]
        embeddings = torch.nn.functional.normalize(doument_embeddings, p=2, dim=1).to("cpu")
        for i in embeddings:
            embeddings_list_temp.append(i)
    return np.array(embeddings_list_temp)


def add_wiki(path_to_xml, tags, name):
    """
    Add a wikipedia document to the database

    This requires an exported XML. You can get export these (for a finite amount) easily from here:
    https://en.wikipedia.org/wiki/Special:Export
    This also allows the export of entire sections.

    :param path_to_xml: The path to the XML file
    :param tags: The tags that all entries will have
    :param name: The name of the document
    """
    global doc_metadata_db, current_doc_id
    current_doc_id += 1
    doc_id = current_doc_id
    entities = get_entities_from_wiki_xml(path_to_xml, tags, doc_id)

    doc_metadata = {"doc_id": doc_id, "authors":None, "publisher": "Wikipedia", "tags": tags, "source": "wikipedia.org",
"creation_time": datetime.datetime.now().strftime('%H:%M %d/%m/%Y'),
                    "source_file": os.path.basename(path_to_xml), "document_title": name}
    add_entry(doc_metadata, entities)


def get_entities_from_wiki_xml(path, tags, doc_id):
    """
    Get JSON entities from a wikipedia XML

    Usually you do not call this directly. Use the add_wiki function instead.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    entities = []
    for page in root[1:]:

        text = page.find("{http://www.mediawiki.org/xml/export-0.10/}revision").find(
            "{http://www.mediawiki.org/xml/export-0.10/}text").text
        title = page.find("{http://www.mediawiki.org/xml/export-0.10/}title").text
        id = page.find("{http://www.mediawiki.org/xml/export-0.10/}id").text
        if "Category:" in title:
            continue

        def repl(matchobj):
            hit = matchobj.groups()[0]
            full = matchobj.group()
            if "|" not in full or "efn|" in full:
                return ""
            elif "math| " in full:
                return f"${re.sub(r'{{((?:[^{}]|(?R))*)}}', repl, hit[6:])}$"
            elif "|" in hit:
                hit = re.sub(r"\|link=y", r"", full)
                if "10^|" in hit:
                    return f"10^{hit[6:-2]}"
                hit = re.sub(r"{{(.*?)\|(.*?)}}", r"\2", hit)
                return hit
            else:
                return full

        sections = re.split(r'={2,5}\s*(.*?)\s*={2,5}', text)
        headers = [title] + sections[1::2]
        section_text = sections[0::2]
        sections = {i: j for i, j in zip(headers, section_text)}
        entries_to_remove = (
            'See also', 'Footnotes', "References", "Sources", "History", "External links", "Bibliography")
        for k in entries_to_remove:
            sections.pop(k, None)

        for i in sections:
            text = sections[i]
            text = text.replace("&lt;", "<")
            text = text.replace("&gt;", ">")
            text = re.sub(r'\[\[(.*?)(?:\|.*?)?\]\]', r'\1', text)
            text = re.sub(r"<ref (.*?)>(.*?)</ref>", '', text)
            text = re.sub(r"<ref>(.*?)</ref>", '', text)
            text = re.sub(r"<ref (.*?)>", '', text)
            text = re.sub(r"<math(.*?)>(.*?)</math>", r'$\2$', text)
            text = re.sub(r"<sub>(.*?)</sub>", r'$\1$', text)
            text = re.sub(r"<sup>(.*?)</sup>", r'^{\1}', text)
            text = re.sub("&nbsp;", " ", text)
            text = re.sub("\t;", "", text)
            text = re.sub(r" {2,20}", "", text)
            text = re.sub(r'{{((?:[^{}]|(?R))*)}}', repl, text)
            text = re.sub("\n", "", text)  # <ref></ref>
            text = re.sub(r"<ref>(.*?)</ref>", '', text)
            text = re.sub(r"\'\'\'(.*?)\'\'\'", r"'\1'", text)
            text = re.sub(r"\'\'(.*?)\'\'", r"'\1'", text)
            entity = {"header": title, "content": i + ":\n" + text,
                      "url": f"https://en.wikipedia.org/?curid={id}#" + "_".join(i.split(" ")),
                      "subheader": i, "tags":tags, "doc_id": doc_id}
            entities.append(entity)
            # sections[i] = text
    return entities

