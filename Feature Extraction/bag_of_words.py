# bag_of_words.py

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not done already
nltk.download('stopwords')

def create_bow(
    documents: list,
    max_features: int = None,
    remove_stopwords: bool = False,
    ngram_range: tuple = (1, 1)
):
    """
    Create a Bag of Words representation for given documents.

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

    # Initialize CountVectorizer with desired settings
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range
    )

    # Learn vocabulary and transform documents into vectors
    X = vectorizer.fit_transform(documents)

    # Get the vocabulary (words or n-grams)
    feature_names = vectorizer.get_feature_names_out()

    return feature_names, X.toarray()

if __name__ == "__main__":
    # Example documents
    docs = [
        "I love NLP and machine learning!",
        "NLP is awesome and so much fun.",
        "Machine learning is a part of AI."
    ]

    # Default BoW matrix (no stopwords, no n-grams)
    features_default, matrix_default = create_bow(docs)
    print("🔥 Vocabulary (default):", features_default)
    print("📊 Document-Term Matrix (default):\n", matrix_default)

    # BoW matrix with stopwords removed
    features_sw, matrix_sw = create_bow(docs, remove_stopwords=True)
    print("\n🔥 Vocabulary (stopwords removed):", features_sw)
    print("📊 Document-Term Matrix (stopwords removed):\n", matrix_sw)

    # BoW matrix with bigrams (and stopwords removed)
    features_bigrams, matrix_bigrams = create_bow(
        docs,
        remove_stopwords=True,
        ngram_range=(1, 2)
    )
    print("\n🔥 Vocabulary (bigrams + stopwords removed):", features_bigrams)
    print("📊 Document-Term Matrix (bigrams):\n", matrix_bigrams)

    # BoW matrix with limited features
    features_limited, matrix_limited = create_bow(docs, max_features=5)
    print("\n🔥 Vocabulary (top 5 features):", features_limited)
    print("📊 Document-Term Matrix (limited features):\n", matrix_limited)
