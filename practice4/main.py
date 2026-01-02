from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample FAQ data (you can expand this)
faq_questions = [
    "How can I reset my password?",
    "What are your business hours?",
    "Where is your office located?",
    "How do I contact support?",
    "What is your return policy?",
]

faq_answers = [
    "To reset your password, click 'Forgot Password' on the login page and follow the instructions.",
    "Our business hours are 9 AM to 5 PM, Monday through Friday.",
    "Our office is located at 123 Main Street, Anytown, USA.",
    "You can contact support via email at support@example.com or call 1-800-123-4567.",
    "You can return most items within 30 days for a full refund. See our return policy for details."
]

# Encode all FAQ questions
faq_embeddings = model.encode(faq_questions)

# Chat loop
print("Welcome to the chatbot! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break

    # Encode the user's input
    user_embedding = model.encode([user_input])[0]

    # Compute similarity scores
    similarities = cosine_similarity([user_embedding], faq_embeddings)[0]

    # Find the most similar question
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]

    # Confidence threshold (optional)
    if best_score < 0.5:
        print("Chatbot: Sorry, I’m not sure how to help with that.")
    else:
        response = faq_answers[best_match_index]
        print(f"Chatbot: {response}")
