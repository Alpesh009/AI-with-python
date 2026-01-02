# rag_local_example.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Step 1: Prepare your documents (knowledge base)
documents = [
    "RAG means Retrieval-Augmented Generation.",
    "Sentence Transformers create embeddings from text.",
    "FAISS helps find similar embeddings quickly.",
    "GPT and other LLMs can generate answers based on context.",
    "An assistant can automate searching and answering."
]

# Step 2: Load embedding model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create embeddings for documents
print("Encoding documents...")
embeddings = model.encode(documents, convert_to_numpy=True)

# Step 4: Save embeddings and documents locally (flat files)
print("Saving embeddings and documents...")
np.save('embeddings.npy', embeddings)
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print("Embeddings and documents saved!\n")

# --- Later, load embeddings and documents and create a search index ---

# Step 5: Load saved embeddings and docs
print("Loading embeddings and documents...")
loaded_embeddings = np.load('embeddings.npy')
with open('documents.pkl', 'rb') as f:
    loaded_documents = pickle.load(f)

# Step 6: Create FAISS index for fast search
dimension = loaded_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(loaded_embeddings)

# Step 7: Function to search and get context
def retrieve_context(query, k=2):
    print(f"\nQuery: {query}")
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    print(f"Found top {k} documents:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {loaded_documents[idx]}")
    return [loaded_documents[idx] for idx in indices[0]]

# Step 8: Simulate assistant prompt preparation (for completion API)
def create_prompt(query, contexts):
    context_text = "\n".join(contexts)
    prompt = f"""
You are a helpful assistant. Use the information below to answer the question.

Context:
{context_text}

Question:
{query}
"""
    return prompt.strip()

# --- Example run ---

if __name__ == "__main__":
    user_query = "What is Retrieval-Augmented Generation?"
    relevant_docs = retrieve_context(user_query)
    prompt = create_prompt(user_query, relevant_docs)
    print("\n=== Prompt to send to AI ===\n")
    print(prompt)
    print("\n=== End of prompt ===")

    # Here, you would send `prompt` to your LLM API (e.g., OpenAI GPT) to get the answer.
    # For example:
    # response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    # print(response['choices'][0]['message']['content'])
