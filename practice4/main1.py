import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load FAQ data from JSON
with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

# Separate questions and answers
faq_questions = [item['question'] for item in faq_data]
faq_answers = [item['answer'] for item in faq_data]

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the FAQ questions
faq_embeddings = model.encode(faq_questions)

# Chat loop
print("Welcome to the chatbot! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break

    user_embedding = model.encode([user_input])[0]
    similarities = cosine_similarity([user_embedding], faq_embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]

    if best_score < 0.5:
        print("Chatbot: Sorry, I’m not sure how to help with that.")
    else:
        print(f"Chatbot: {faq_answers[best_match_index]}")
