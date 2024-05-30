"""
Levenshtein edit distance
"""


from typing import List, Tuple, Union
from difflib import SequenceMatcher as SM


def fuzzy_ratio(base: str, query: str) -> float:
    """fuzzy ratio of two strings using difflib

    Args:
        base (str): base str
        query (str): query str

    Returns:
        float: score of the two str similarity
    """
    return SM(None, base, query).ratio()


def fuzzy_search(db: List[str], query: str, threshold=0, match_count=1, score=False) -> Union[List[str], List[Tuple[str, float]]]:
    """Fuzzy searches based on db

    Args:
        db (List[str]): the list of db to search through
        query (str): the string to find item
        threshold (int, optional): the threshold which it will count as a match, 0 means no threshold set. Defaults to 0.
        match_count (int, optional): how many top matches to get. Defaults to 1.
        score (bool, optional): if the score will be transmitted. Defaults to False.

    Returns:
        Union[List[str], List[Tuple[str, float]]]: either returns list of top matches or list of top matches with scores as tuples
    """
    assert match_count >= 1, "match_count must be above or equal to 1"
    # construct similarity with score
    cache = {}
    for item in db:
        tmp = fuzzy_ratio(item, query)
        if threshold == 0 or threshold <= tmp:
            cache[item] = tmp

    # sort by score
    sorted(cache.items(), key=lambda x: x[1], reverse=True)

    return_buff = []
    i = match_count
    for item in sorted(cache.items(), key=lambda x: x[1], reverse=True):
        if i == 0:
            break
        if score:
            return_buff.append(item)
        else:
            return_buff.append(item[0])
        i -= 1

    # if threshold == 0:
    return return_buff


def fuzzy_ratio_match(base: str, query: str, threshold: float) -> bool:
    """Wrapper around fuzzy similarity

    Args:
        base (str): Base text
        query (str): similarity text to compare to
        threshold (float): threshold of similarity for it to be true

    Returns:
        bool: if similarity is above threshold
    """
    return fuzzy_ratio(base, query) >= threshold


def test():
    EPSILON = 0.000001
    assert fuzzy_search(["cosine"], "similarity", score=True)[
        0] == ('cosine', 0.25), "fuzzy search does not work"
    assert fuzzy_search(["cosine", "similarity", ""], "similarity", score=True)[
        0] == ('similarity', 1.0), "fuzzy search does not work"
    assert fuzzy_search(["cosine", "similarity", ""], "similarity")[
        0] == 'similarity', "fuzzy search does not work"
    assert fuzzy_search(["cosine", "similarity", ""], "similarity", score=True, match_count=2) == [
        ('similarity', 1.0), ('cosine', 0.25)], "fuzzy search does not work"
    assert fuzzy_search(["cosine", "similarity", ""], "similarity", score=True, match_count=2, threshold=1) == [
        ('similarity', 1.0)], "fuzzy search does not work"


if __name__ == "__main__":
    test()
