#%%
from sentence_transformers import SentenceTransformer
sentences = ["apple", "Apple, Inc.", "Applebee's", "McDonald's", "I love apples"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity(embeddings))