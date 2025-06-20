import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
with open("data/data.json", "r") as f:
    data = json.load(f)

questions = [q for item in data if item["enabled"] for q in item["questions"]]
answers = [item["answer"] for item in data if item["enabled"] for _ in item["questions"]]

# Encode
embeddings = model.encode(questions)

# Create and save FAISS index
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss.index")

# Save answers
with open("embeddings/answers.json", "w") as f:
    json.dump(answers, f)

print("✅ Embeddings and index created.")
