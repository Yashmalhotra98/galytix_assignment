# app/user_interface.py
from calculate_similarity import load_word2vec_vectors, calculate_cosine_similarity_parallel
import pandas as pd

# Read phrases from a CSV file
def read_phrases_from_csv(file_path):
    phrases_df = pd.read_csv(file_path)
    return phrases_df['Phrases'].tolist()


def main(phrases):
    word_vectors = load_word2vec_vectors()
    
    while True:
        user_input = input("Enter a phrase (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        similarity_scores = calculate_cosine_similarity_parallel(user_input, phrases, word_vectors)
        for phrase, similarity in similarity_scores:
            print(f"Phrase: {phrase}, Similarity: {similarity:.2f}")

if __name__ == '__main__':
    phrases = read_phrases_from_csv('D:/Docs/Galytix_Assessment/data/phrases.csv')
    main(phrases)