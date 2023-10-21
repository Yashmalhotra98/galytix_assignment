# app/calculate_similarity.py
from gensim.models import KeyedVectors
import numpy as np  # Import NumPy for cosine similarity calculation
import polars as pl
import pandas as pd


def load_word2vec_vectors():
    vectors = KeyedVectors.load_word2vec_format('data/vectors.csv')
    return vectors


# Read phrases from a CSV file
def read_phrases_from_csv(file_path):
    phrases_df = pd.read_csv(file_path)
    return phrases_df['Phrases'].tolist()

def calculate_cosine_similarity(phrase1, phrase2, word_vectors):
    # Calculate the average vector for each phrase
    words1 = phrase1.split()
    words2 = phrase2.split()
    
    vector1 = np.mean([word_vectors[word] for word in words1 if word in word_vectors], axis=0)
    vector2 = np.mean([word_vectors[word] for word in words2 if word in word_vectors], axis=0)
    
    if vector1 is not None and vector2 is not None:
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity
    else:
        return 0.0  # Handle cases where no word vectors were found

def calculate_cosine_similarity_parallel(phrase, phrases, word_vectors):
    similarity_scores = []
    
    for phrase1 in phrases:
        similarity = calculate_cosine_similarity(phrase, phrase1, word_vectors)
        similarity_scores.append((phrase1, similarity))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores

