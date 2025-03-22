import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai  # Use if calling GPT-based APIs

# Load the CSV file
df = pd.read_csv("C:\\Users\\kashv\\Desktop\\NyaySaarthi\\datasets\\ipc_sections.csv", encoding="ISO-8859-1")

# Fill missing values
df.fillna("", inplace=True)

# Create a column that combines Description, Section, and Offense for embedding
df["combined_text"] = df["Description"] + " " + df["Section"].astype(str) + " " + df["Offense"]

# Initialize Sentence Transformer model for embedding legal text
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate vector embeddings for each law description
embeddings = embedding_model.encode(df["combined_text"].tolist(), convert_to_numpy=True)

# Create FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index and data for later use
faiss.write_index(index, "faiss_index.bin")
df.to_csv("processed_legal_data.csv", index=False)

print("✅ FAISS Index & Data Saved")

def find_relevant_laws(user_input, top_k=3):
    """Finds the most relevant laws based on the user's input scenario."""
    query_embedding = embedding_model.encode([user_input], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        row = df.iloc[indices[0][i]]
        results.append({
            "Offense": row["Offense"],
            "Punishment": row["Punishment"],
            "Section": row["Section"],
            "Cognizable": row["Cognizable"],
            "Bailable": row["Bailable"],
            "Court": row["Court"]
        })
    
    return results

# Example usage
query = "I was attacked and injured in a fight."
laws = find_relevant_laws(query)
for law in laws:
    print(law)

