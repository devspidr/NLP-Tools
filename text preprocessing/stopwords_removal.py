# stopwords_removal.py

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('stopwords')
nltk.download('punkt')

# Load default English stopwords
default_stopwords = set(stopwords.words('english'))

def remove_stopwords(text: str, language: str = 'english', custom_stopwords: set = None, remove_punctuation: bool = False) -> str:
    """
    Remove stopwords from the text, with optional support for:
    - Multiple languages
    - Custom stopwords list
    - Punctuation removal

    Args:
        text (str): Input text
        language (str): Language for stopwords (default: 'english')
        custom_stopwords (set, optional): Additional stopwords to remove
        remove_punctuation (bool): If True, also removes punctuation

    Returns:
        str: Text with stopwords removed
    """
    # Load stopwords for the specified language
    try:
        lang_stopwords = set(stopwords.words(language))
    except OSError:
        print(f"Language '{language}' not supported by NLTK. Using English stopwords instead.")
        lang_stopwords = default_stopwords

    # Merge default and custom stopwords
    if custom_stopwords:
        lang_stopwords = lang_stopwords.union(custom_stopwords)

    # Optionally remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in lang_stopwords]

    return ' '.join(filtered_tokens)


def interactive_demo():
    """
    A simple interactive demo for the user to test stopword removal.
    """
    print("\n🚀 Welcome to Stopwords Removal Interactive Demo!")
    example_text = input("Enter a sentence to process: ")
    lang = input("Enter language code (default: english): ").strip() or "english"
    punctuation_choice = input("Remove punctuation as well? (yes/no): ").strip().lower() == "yes"

    # Custom stopwords (just for demo)
    custom = {"example", "test", "custom"}

    cleaned_text = remove_stopwords(
        example_text,
        language=lang,
        custom_stopwords=custom,
        remove_punctuation=punctuation_choice
    )

    print("\nOriginal Text:\n", example_text)
    print("Cleaned Text:\n", cleaned_text)
    print("Custom stopwords used (example set):", custom)


if __name__ == "__main__":
    # Example usage
    example_text = "This is an example of removing stopwords and punctuation!"
    print("=== Example ===")
    print("Original:", example_text)
    cleaned_text = remove_stopwords(example_text, remove_punctuation=True)
    print("Cleaned (punctuation removed):", cleaned_text)

    # Demo for user
    interactive_demo()
