
import re
from collections import Counter


def cosine_similarity(base: str, query: str) -> float:
    """Using cosine similarity, check to see if the two texts are similar

    Args:
        base (str): base string
        query (str): query string

    Returns:
        float: the percent similarity using cosine
    """
    # first tokenize to vectors
    # split words
    words_regex = re.compile(r"\w+")
    vector1 = Counter(words_regex.findall(base))
    vector2 = Counter(words_regex.findall(query))

    similar_words = set(vector1.keys()) & set(vector2.keys())

    # no similar words
    if len(similar_words) == 0:
        return 0

    # A dot B (numerator)
    top = sum([vector1[x] * vector2[x] for x in similar_words])

    sqrt_summed_sq2_A = sum([vector1[x]**2 for x in vector1.keys()]) ** 0.5
    sqrt_summed_sq2_B = sum([vector2[x]**2 for x in vector2.keys()]) ** 0.5

    # denominator should not be 0, else there is an issue because intersection should be len 0
    return top / (sqrt_summed_sq2_A * sqrt_summed_sq2_B)


def cosine_similarity_match(base: str, query: str, threshold: float) -> bool:
    """Wrapper around cosine similarity

    Args:
        base (str): Base text
        query (str): similarity text to compare to
        threshold (float): threshold of similarity for it to be true

    Returns:
        bool: if similarity is above threshold
    """
    return cosine_similarity(base, query) >= threshold


def test():
    EPSILON = 0.000001

    assert cosine_similarity(
        "cosine", "similarity") == 0, "similarity should be null"
    assert 1 - EPSILON <= cosine_similarity(
        "two text strings", "two text strings") <= 1 + EPSILON, "similarity should be 100%"
    assert 0.20412414523193154 - \
        EPSILON <= cosine_similarity(
            "I love horror movies", "Lights out is a horror movie") <= 0.20412414523193154 + EPSILON, "not the right similarity level"
    assert 0.9486832980505138 - \
        EPSILON <= cosine_similarity(
            "cosine sim", "cosine sim sim") <= 0.9486832980505138 + EPSILON, "is not epsilon within bounds"
    assert 0 - \
        EPSILON <= cosine_similarity(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", "The quick brown fox jumps over the lazy dog") <= 0 + EPSILON, "is not epsilon within 19.2%"

    assert cosine_similarity_match(
        "cosine sim", "cosine sim sim", 0.9), "should expect true"
    assert not cosine_similarity_match(
        "cosine sim", "cosine sim sim", 0.95), "should expect false"

    assert cosine_similarity_match(
        "two text strings", "two text strings", 1), "similarity should be 100%"


if __name__ == "__main__":
    test()
