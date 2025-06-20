# app.py
import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from gemini_helper import generate_detailed_response

# Load FAISS index and answers
index = faiss.read_index("embeddings/faiss.index")
with open("embeddings/answers.json", "r") as f:
    answers = json.load(f)

# Initialize the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_answer(user_query):
    # Convert user query into embedding
    embedded_query = model.encode([user_query])
    _, I = index.search(np.array(embedded_query), k=1)
    best_match_idx = I[0][0]
    return answers[best_match_idx]

# Streamlit UI
st.title("Customer Service Chatbot")

st.markdown("""
    This chatbot retrieves answers based on your query.
""")

# User input for the chatbot
user_input = st.text_input("Ask me anything:")

if user_input:
    # Retrieve the most relevant answer based on the user's query
    top_answer = retrieve_answer(user_input)
    
    # Generate a detailed response using Gemini Model
    detailed_response = generate_detailed_response(user_input, top_answer)
    
    # Display the bot's response
    st.subheader("Bot Response:")
    st.write(detailed_response)