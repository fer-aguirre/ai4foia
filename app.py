import streamlit as st
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import torch

model = SentenceTransformer("sentence-transformers/gtr-t5-large")


# Read files
train = pd.read_csv('./data/processed/train.csv')
test = pd.read_csv('./data/processed/test.csv')
df_emb = pd.read_csv('./data/processed/embeddings.csv')

dataset = Dataset.from_pandas(df_emb)

dataset_embeddings = torch.from_numpy(dataset.to_pandas().to_numpy()).to(torch.float)

st.markdown("**Inserta una solicitud de información**")
if request := st.text_area("", value=""):
    output = model.encode(request)

    query_embeddings = torch.FloatTensor(output)

    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=3)

    id1 = hits[0][0]['corpus_id']
    id2 = hits[0][1]['corpus_id']
    id3 = hits[0][2]['corpus_id']

    rec1 = train.iloc[id1].str.split(pat="/")[0]
    rec2 = train.iloc[id2].str.split(pat="/")[0]
    rec3 = train.iloc[id3].str.split(pat="/")[0]

    st.markdown(f':green[{rec1[0]}]')
    st.markdown(f':green[{rec2[0]}]')
    st.markdown(f':green[{rec3[0]}]')

st.markdown("""---""")

if st.button('Genera un ejemplo random'):

    test_example = test['combined'].sample(n=1)
    index = test_example.index
    idx = index[0]

    original = test.iloc[idx].str.split(pat="/")[0]

    request = test_example.to_string(index=False)

    st.text(f'{idx}, {request}')

    output = model.encode(request)

    query_embeddings = torch.FloatTensor(output)

    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=3)

    id1 = hits[0][0]['corpus_id']
    id2 = hits[0][1]['corpus_id']
    id3 = hits[0][2]['corpus_id']

    rec1 = train.iloc[id1].str.split(pat="/")[0]
    rec2 = train.iloc[id2].str.split(pat="/")[0]
    rec3 = train.iloc[id3].str.split(pat="/")[0]

    list_rec = [rec1, rec2, rec3]
    unique_list = []
    for string in list_rec:
        if string not in unique_list:
            unique_list.append(string)

    for rec in unique_list:
        st.markdown(f':green[{rec[0]}]')
    st.markdown(f':red[{original[0]}]')