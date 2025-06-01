# text_cleaning_extended.py

import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def basic_cleaning(text: str) -> str:
    """
    Basic cleaning:
    - Lowercasing
    - Removing punctuation
    - Removing extra whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def stem_text(text: str) -> str:
    """
    Apply stemming to each word in the text
    """
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def lemmatize_text(text: str) -> str:
    """
    Apply lemmatization to each word in the text
    """
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def clean_text(text: str, stem=False, lemmatize=True) -> str:
    """
    Complete cleaning pipeline:
    - Basic cleaning
    - Optionally apply stemming or lemmatization
    """
    cleaned_text = basic_cleaning(text)
    
    if stem and lemmatize:
        # By default prioritize lemmatization if both True
        lemmatize = True
        stem = False

    if stem:
        return stem_text(cleaned_text)
    elif lemmatize:
        return lemmatize_text(cleaned_text)
    else:
        return cleaned_text

def compare_cleaning_methods(text: str) -> None:
    """
    Compare original text, basic cleaned text, stemmed, and lemmatized
    """
    print("\n🔹 Original Text:")
    print(text)
    
    cleaned = basic_cleaning(text)
    print("\n🔹 After Basic Cleaning:")
    print(cleaned)
    
    stemmed = stem_text(cleaned)
    print("\n🔹 After Stemming:")
    print(stemmed)
    
    lemmatized = lemmatize_text(cleaned)
    print("\n🔹 After Lemmatization:")
    print(lemmatized)

if __name__ == "__main__":
    example_text = "Running faster than runners run! The studies are studying better than studied."
    compare_cleaning_methods(example_text)

    # Custom usage:
    print("\n🔹 Custom Cleaning (lemmatize):")
    print(clean_text(example_text, lemmatize=True))

    print("\n🔹 Custom Cleaning (stem):")
    print(clean_text(example_text, stem=True, lemmatize=False))
