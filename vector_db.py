from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTOR_DB_PATH, EMBEDDING_MODEL
import pickle
import numpy as np

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)


def retrieve_relevant_docs(query, vector_db, max_k=10):
    relevance_threshold=0.7
    query_embedding = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    query_embedding_ipca = pca.transform(query_embedding)
    results = vector_db.similarity_search_with_score_by_vector(query_embedding_ipca.flatten(), k=max_k)
    relevant_chunks = [doc for doc, score in results if score >= relevance_threshold]
    
    return relevant_chunks


"""
Alternate approach to find k 


def classify_query(query):
    query = query.lower()
    simple_keywords = [
        "what is","what are", "who is","who are", "when", "where", "full name", "age", "birthday", 
        "define", "meaning", "synonym", "antonym", "role of", "did", "was", "is", 
        "does", "did", "list", "mention", "show", "tell me", "first appearance"
    ]

    complex_keywords = [
        "analyze", "character analysis", "explain", "describe", "summary", "summarize",
        "interpret", "compare", "contrast", "significance of", "why", "how", "impact",
        "motivation", "theme", "symbolism", "message", "moral", "philosophy", 
        "historical context", "literary devices", "foreshadowing", "allegory", 
        "metaphor", "satire", "paradox", "hidden meaning", "psychology", "perspective"
    ]

    if any(keyword in query for keyword in simple_keywords):
        return "simple"
    elif any(keyword in query for keyword in complex_keywords):
        return "complex"
    else:
        return "moderate"
"""