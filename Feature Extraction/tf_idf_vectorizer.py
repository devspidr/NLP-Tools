# tfidf_vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not done already
nltk.download('stopwords')

def create_tfidf(
    documents: list,
    max_features: int = None,
    remove_stopwords: bool = False,
    ngram_range: tuple = (1, 1)
):
    """
    Create a TF-IDF representation for given documents.

    Args:
        documents (list): List of text documents.
        max_features (int, optional): Max number of features to consider.
        remove_stopwords (bool): If True, removes stopwords.
        ngram_range (tuple): N-gram range (default is (1,1)).

    Returns:
        tuple: (feature names, document-term matrix)
    """
    # Initialize stopword list if requested
    stop_words = stopwords.words('english') if remove_stopwords else None

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range
    )

    # Learn vocabulary and transform documents into vectors
    X = vectorizer.fit_transform(documents)

    # Get the vocabulary (words or n-grams)
    feature_names = vectorizer.get_feature_names_out()

    return feature_names, X.toarray()

def display_top_terms(feature_names, matrix, top_n=3):
    """
    Display the top N weighted terms for each document.

    Args:
        feature_names (array): Feature names.
        matrix (array): TF-IDF matrix.
        top_n (int): Number of top terms to display per document.
    """
    for idx, row in enumerate(matrix):
        print(f"\n📄 Document {idx+1}:")
        sorted_indices = row.argsort()[::-1]  # sort by TF-IDF weight descending
        for i in range(top_n):
            feature_idx = sorted_indices[i]
            print(f"  - {feature_names[feature_idx]}: {row[feature_idx]:.4f}")

if __name__ == "__main__":
    # Example documents
    docs = [
        "I love NLP and machine learning!",
        "NLP is awesome and so much fun.",
        "Machine learning is a part of AI."
    ]

    # 1️⃣ Basic TF-IDF (no stopwords, unigrams only)
    features_basic, matrix_basic = create_tfidf(docs)
    print("🔥 Vocabulary (default):", features_basic)
    print("📊 TF-IDF Matrix (default):\n", matrix_basic)
    display_top_terms(features_basic, matrix_basic)

    # 2️⃣ With stopwords removed
    features_sw, matrix_sw = create_tfidf(docs, remove_stopwords=True)
    print("\n🔥 Vocabulary (stopwords removed):", features_sw)
    print("📊 TF-IDF Matrix (stopwords removed):\n", matrix_sw)
    display_top_terms(features_sw, matrix_sw)

    # 3️⃣ With bigrams and stopwords removed
    features_bigrams, matrix_bigrams = create_tfidf(
        docs,
        remove_stopwords=True,
        ngram_range=(1, 2)
    )
    print("\n🔥 Vocabulary (bigrams + stopwords removed):", features_bigrams)
    print("📊 TF-IDF Matrix (bigrams):\n", matrix_bigrams)
    display_top_terms(features_bigrams, matrix_bigrams)

    # 4️⃣ Limit vocabulary to top 5 features
    features_limited, matrix_limited = create_tfidf(docs, max_features=5)
    print("\n🔥 Vocabulary (top 5 features):", features_limited)
    print("📊 TF-IDF Matrix (limited features):\n", matrix_limited)
    display_top_terms(features_limited, matrix_limited)
