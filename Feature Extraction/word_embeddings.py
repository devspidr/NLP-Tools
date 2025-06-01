# word_embeddings.py

import numpy as np

def load_glove_embeddings(filepath: str) -> dict:
    """
    Load GloVe embeddings from a .txt file.
    Args:
        filepath (str): Path to the GloVe .txt file.
    Returns:
        dict: {word: embedding vector (numpy array)}
    """
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def get_word_vector(word: str, embeddings: dict, default=None) -> np.ndarray:
    """
    Retrieve the vector for a word.
    Args:
        word (str): The word.
        embeddings (dict): Word to vector mapping.
        default: Value to return if word is not in embeddings.
    Returns:
        np.ndarray: The word embedding vector.
    """
    return embeddings.get(word, default)


def get_sentence_embedding(sentence: str, embeddings: dict) -> np.ndarray:
    """
    Compute the average embedding for a sentence.
    Args:
        sentence (str): Input sentence.
        embeddings (dict): Word to vector mapping.
    Returns:
        np.ndarray: Sentence embedding (average of word vectors).
    """
    words = sentence.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if not vectors:
        return np.zeros(list(embeddings.values())[0].shape)  # fallback
    return np.mean(vectors, axis=0)


if __name__ == "__main__":
    # Example usage (assuming you have a GloVe file like glove.6B.50d.txt)
    filepath = "glove.6B.50d.txt"  # Update path to your file
    glove_embeddings = load_glove_embeddings(filepath)

    word = "king"
    vec = get_word_vector(word, glove_embeddings)
    print("Word vector for 'king':", vec)

    sentence = "NLP is amazing"
    sent_vec = get_sentence_embedding(sentence, glove_embeddings)
    print("\nSentence embedding:", sent_vec)
