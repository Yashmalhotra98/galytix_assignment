# D:\Docs\Galytix_Assessment\data\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin

# app/download_embeddings.py
import gensim
from gensim.models import KeyedVectors

def download_and_save_word2vec_vectors():
    location = 'D:/Docs/Galytix_Assessment/data/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'
    wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000)
    wv.save_word2vec_format('data/vectors.csv')

if __name__ == '__main__':
    download_and_save_word2vec_vectors()
