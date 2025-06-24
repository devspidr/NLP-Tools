from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example texts
text1 = "I am soundar balajij"
text2 = "I am soundar balaji" 

# Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer() 
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Compute cosine similarity
cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Cosine Similarity: {cos_sim[0][0]:.4f}")
