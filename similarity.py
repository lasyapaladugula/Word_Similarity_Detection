import os
import re
import numpy as np
import streamlit as st
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "word2vec_model.model"
)


@st.cache_resource(show_spinner="Loading the trained word model...")
def _load_model(model_path, modified_at):
    """Load one shared model instance and refresh it if the file changes."""
    return Word2Vec.load(model_path)


def load_model():
    """Load the trained model, or return None when it is not installed."""
    if not os.path.isfile(MODEL_PATH):
        return None

    return _load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))


def normalize_word(word):
    """Normalize a single input word like the tokens used during training."""
    tokens = preprocess_text(word)
    if len(tokens) == 1:
        return tokens[0]
    return word.strip().lower()


def get_similar_words(word, top_n=5):
    """
    Find the most similar words for a given word.
    """

    model = load_model()

    if model is None:
        return None

    word = normalize_word(word)

    if word not in model.wv:
        return []

    similar_words = model.wv.most_similar(
        word,
        topn=top_n
    )

    return similar_words


def calculate_word_similarity(word1, word2):
    """
    Calculate similarity between two words.
    """

    model = load_model()

    if model is None:
        return None

    word1 = normalize_word(word1)
    word2 = normalize_word(word2)

    if word1 not in model.wv:
        return None

    if word2 not in model.wv:
        return None

    similarity = model.wv.similarity(
        word1,
        word2
    )

    return similarity


def sentence_to_vector(sentence, model):
    """
    Convert a sentence into one vector
    by taking the average of its word vectors.
    """

    words = preprocess_text(sentence)

    vectors = []

    for word in words:

        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return None

    return np.mean(vectors, axis=0)


def semantic_search(query, top_n=5):
    """
    Find corpus sentences semantically related
    to the user's query.
    """

    model = load_model()

    if model is None:
        return None

    corpus_path = os.path.join(
        BASE_DIR,
        "dataset",
        "corpus.txt"
    )

    if not os.path.isfile(corpus_path):
        return []

    with open(
        corpus_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    # Keep sentence boundaries for common punctuation and line breaks. Splitting
    # only on periods loses documents whose corpus is one sentence per line.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if sentence.strip()
    ]

    query_vector = sentence_to_vector(
        query,
        model
    )

    if query_vector is None:
        return []

    results = []

    for sentence in sentences:

        sentence_vector = sentence_to_vector(
            sentence,
            model
        )

        if sentence_vector is None:
            continue

        score = cosine_similarity(
            [query_vector],
            [sentence_vector]
        )[0][0]

        results.append((sentence, float(score)))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Avoid returning unrelated documents when every candidate has no shared
    # semantic signal with the query.
    results = [result for result in results if result[1] > 0]
    return results[:max(1, int(top_n))]
