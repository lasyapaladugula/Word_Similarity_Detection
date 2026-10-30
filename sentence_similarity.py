import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text
from src.similarity import load_model


# --------------------------------------------------
# CONVERT SENTENCE TO VECTOR
# --------------------------------------------------

def sentence_to_vector(sentence, model):

    words = preprocess_text(sentence)

    vectors = []

    for word in words:

        if word in model.wv:
            vectors.append(
                model.wv[word]
            )

    if len(vectors) == 0:
        return None

    return np.mean(
        vectors,
        axis=0
    )


# --------------------------------------------------
# CALCULATE SENTENCE SIMILARITY
# --------------------------------------------------

def calculate_sentence_similarity(
    sentence1,
    sentence2
):

    model = load_model()

    if model is None:
        return None

    vector1 = sentence_to_vector(
        sentence1,
        model
    )

    vector2 = sentence_to_vector(
        sentence2,
        model
    )

    if vector1 is None or vector2 is None:
        return None

    score = cosine_similarity(
        [vector1],
        [vector2]
    )[0][0]

    return float(score)
