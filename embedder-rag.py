import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2" # nazwa modelu
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def create_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, kwargs=model_kwargs) # załadowanie modelu embeddingowego
    texts = [doc["text"] for doc in documents] # wartości tekstowe wszystkich dokumentów
    metadata = documents # metadane wszystkich dokumentów, czyli słownik {filename:... , text:...}

    # 1. Zamiana każdego tekstu na wektor liczb
    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    # 2. Konwersja na format NumPy (wymagany przez FAISS) z precyzją float32
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    dimension = embeddings_matrix.shape[1] # długość wektora (np. 768 lub 1536)
    index = faiss.IndexFlatL2(dimension)# ustawienie indeksu przeszukwania
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, kwargs=model_kwargs) # załadowanie modelu embeddingowego
    query_embedding = embeddings.embed_query(query) # embeddowanie zapytania (query)
    # Przygotowujemy wektor do formatu NumPy (FAISS wymaga tablicy 2D, stąd [query_embedding])
    query_embedding_np = np.array([query_embedding]).astype("float32")
    results = faiss_index.similarity_search(query_embedding_np, k) # zwrócenie wyników przeuszkiwania
    return results