import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("embeddings/faiss.index")

with open("embeddings/answers.json", "r") as f:
    answers = json.load(f)

def get_answer(query):
    query_embedding = model.encode([query])
    _, idx = index.search(np.array(query_embedding), 1)
    return answers[idx[0][0]]