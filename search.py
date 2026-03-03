import numpy as np


def search_embeddings(query_embedding, document_embeddings, k=5):
    # cosine similarity between query embedding and document embeddings
    similarity_scores = cosine_similarity(query_embedding, document_embeddings)[0]
    top_k_indices = similarity_scores.argsort()[-k:][::-1]
    return top_k_indices

def cosine_similarity(query_embedding, document_embeddings):
    return np.dot(query_embedding, document_embeddings.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(document_embeddings))
