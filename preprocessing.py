import os
import re


# Common English stop words
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "has", "he", "in", "is", "it", "its",
    "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "or",
    "but", "can", "into", "their", "they", "them",
    "we", "you", "your", "our", "are", "using", "used"
}


def preprocess_text(text):
    """
    Preprocess one sentence.

    Steps:
    1. Convert text to lowercase
    2. Remove numbers and special characters
    3. Tokenize text
    4. Remove stop words
    """

    # Step 1: lowercase
    text = text.lower()

    # Step 2: keep only alphabets and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Step 3: tokenize
    words = text.split()

    # Step 4: remove stop words
    words = [
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    return words


def load_and_preprocess(file_path):
    """
    Read corpus and preprocess all sentences.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split corpus into sentences
    sentences = re.split(r"[.!?]+", text)

    processed_sentences = []

    for sentence in sentences:

        tokens = preprocess_text(sentence)

        if tokens:
            processed_sentences.append(tokens)

    return processed_sentences


# Test the preprocessing module
if __name__ == "__main__":

    # Find project root
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # Corpus path
    corpus_path = os.path.join(
        base_dir,
        "dataset",
        "corpus.txt"
    )

    print("Reading corpus...")
    print()

    sentences = load_and_preprocess(corpus_path)

    print("Preprocessing completed successfully!")
    print("Total sentences:", len(sentences))

    print()
    print("First 5 processed sentences:")
    print("-" * 50)

    for sentence in sentences[:5]:
        print(sentence)