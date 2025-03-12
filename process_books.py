import os
import spacy
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, BOOK_FOLDER
from sklearn.decomposition import PCA
import numpy as np
import faiss
import pickle

print("Loading spaCy model...", flush=True)
nlp = spacy.load("en_core_web_sm",disable=["parser","ner"])
if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
print("spaCy model loaded!", flush=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()
    text = text.lower()

    # Process text in smaller chunks to avoid spaCy's max length limit
    max_length = 500000 
    cleaned_tokens = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        doc = nlp(chunk)
        for token in doc:
            if token.is_punct and token.text in {".", "!", "?"}:
                cleaned_tokens.append(token.text)
            elif not token.is_stop and not token.is_punct and not token.is_space:
                cleaned_tokens.append(token.lemma_)

    return " ".join(cleaned_tokens)

def load_books(folder_path):
    books = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            books[file_name] = clean_text(f.read())
    return books

def dynamic_chunking(text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    print("Making dynamic chunks...", flush=True)
    
    # Handle empty or very small text
    if not text or len(text.split()) <= max_chunk_size:
        return [text]
    
    
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', text) 
    
    sentence_lengths = [len(s.split()) for s in sentences]
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, length in zip(sentences, sentence_lengths):
        # If a single sentence exceeds max_chunk_size, split it into smaller parts
        if length > max_chunk_size:
            words = sentence.split()
            for j in range(0, len(words), max_chunk_size):
                sub_sentence = " ".join(words[j:j + max_chunk_size])
                chunks.append(sub_sentence)
            continue

        if current_length + length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += length
        else:
            chunks.append(" ".join(current_chunk))
            if overlap > 0:
                overlap_sentences = current_chunk[-overlap:]
                overlap_length = sum(sentence_lengths[-overlap:]) if overlap <= len(sentence_lengths) else 0
            else:
                overlap_sentences = []
                overlap_length = 0

            current_chunk = overlap_sentences + [sentence]
            current_length = overlap_length + length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Created {len(chunks)} chunks!", flush=True)
    return chunks

def reduce_dimensionality(embeddings, target_dim=256):
    print("Reducing dimensionality", flush=True)
    n_samples, n_features = embeddings.shape
    max_components = min(n_samples, n_features)
    if target_dim > max_components:
        print(f"Warning: target_dim ({target_dim}) is greater than the maximum possible components ({max_components}). PCA cannot be applied. Returning original embeddings.", flush=True)
        return embeddings[:, :target_dim]

    if not os.path.exists("pca_model.pkl"):
        print("Training PCA for dimensionality reduction...", flush=True)
        pca = PCA(n_components=target_dim)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Save PCA model
        with open("pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)

        return reduced_embeddings
    else:
        with open("pca_model.pkl", "rb") as f:
            pca = pickle.load(f)
        return pca.transform(embeddings) 

def batch_embed(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}...", end = "", flush=True)
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
        print(f"Batch embedded", flush=True)
    return np.array(embeddings)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors for HNSW
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)
    return index

def process_books():
    print("Loading Books", flush=True)
    books = load_books(BOOK_FOLDER)
    print("Books Loaded", flush=True)

    if os.path.exists(VECTOR_DB_PATH):
        vector_db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        existing_docs = {doc.metadata["source"] for doc in vector_db.docstore._dict.values()}
    else:
        vector_db = None
        existing_docs = set()
    
    print("Scanning for new books")
    new_docs = []
    new_books = 0
    for book_name, book_text in books.items():
        if book_name in existing_docs:
            print(f"Skipping {book_name} (already indexed)")
            continue
        new_books += 1
        chunks = dynamic_chunking(book_text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for chunk in chunks:
            new_docs.append({"text": chunk, "metadata": {"source": book_name}})
    
    if new_docs:
        texts = [doc["text"] for doc in new_docs]
        embeddings = batch_embed(texts, batch_size=32)  # Adjust batch size based on your system
        print(f"Embeddings shape: {np.array(embeddings).shape}, Type: {type(embeddings)}", flush=True)

        embeddings = reduce_dimensionality(np.array(embeddings))

        print(f"Embeddings shape: {np.array(embeddings).shape}, Type: {type(embeddings)}", flush=True)
        if embeddings.shape[0] > 1:  # Only normalize if more than 1 sample
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            print("Embeddings Normalized", flush = True)
        else:
            print("Skipping FAISS L2 normalization: Only one embedding present", flush=True)


        if vector_db:
            vector_db.index.add(embeddings)
            
            print(type(new_docs), type(new_docs[0]))

            doc_dict = {str(i): doc for i, doc in enumerate(new_docs)}
            vector_db.docstore.add(doc_dict)
        else:
            # Create a new FAISS index with HNSW
            faiss_index = create_faiss_index(embeddings)
            new_docs = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in new_docs]
            vector_db = FAISS.from_documents(new_docs, embedding_model)
            vector_db.index = faiss_index

        vector_db.save_local(VECTOR_DB_PATH)
        print(f"Added {len(new_docs)} new text chunks from {new_books} books to FAISS!", flush=True)
    else:
        print("No new books to process.", flush=True)


if __name__ == "__main__": 
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    process_books()