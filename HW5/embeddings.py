from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts)

def get_similarity_from_embeddings(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def get_similarity_from_texts(text1, text2):
    embedding1 = get_embeddings([text1])[0]
    embedding2 = get_embeddings([text2])[0]
    return get_similarity_from_embeddings(embedding1, embedding2)
