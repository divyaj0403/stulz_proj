# gemini_helper.py
import google.generativeai as genai
import os

# Set your API key
genai.configure(api_key="AIzaSyCDmvSvyYdzk7lP5xmW8WGJla-DZw2Lgc0")

chat = None

def generate_detailed_response(user_question, retrieved_answer):
    global chat
    prompt = f"""
Use the following support information to answer the user's question in a helpful and detailed way.

User Question: {user_question}

Support Info: {retrieved_answer}
"""

    if chat is None:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        chat = model.start_chat(history=[])
    
    response = chat.send_message(prompt)
    return response.text.strip()