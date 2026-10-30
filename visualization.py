import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.similarity import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------
# CREATE PCA VISUALIZATION
# --------------------------------------------------

def create_visualization(
    word=None,
    top_n=10
):

    model = load_model()

    if model is None:
        return None

    words = []
    vectors = []

    # If a word is provided
    if word and word.lower() in model.wv:

        word = word.lower()

        similar_words = model.wv.most_similar(
            word,
            topn=top_n
        )

        words.append(word)

        vectors.append(
            model.wv[word]
        )

        for similar_word, score in similar_words:

            words.append(similar_word)

            vectors.append(
                model.wv[similar_word]
            )

    else:

        # Use first words from vocabulary
        vocabulary = list(
            model.wv.index_to_key
        )

        selected_words = vocabulary[:top_n]

        for selected_word in selected_words:

            words.append(selected_word)

            vectors.append(
                model.wv[selected_word]
            )

    if len(vectors) < 2:
        return None

    # Convert to PCA
    pca = PCA(
        n_components=2
    )

    result = pca.fit_transform(
        vectors
    )

    # Create graph
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        result[:, 0],
        result[:, 1]
    )

    for i, word in enumerate(words):

        ax.annotate(
            word,
            (
                result[i, 0],
                result[i, 1]
            )
        )

    ax.set_title(
        "Word2Vec Word Vector Visualization"
    )

    ax.set_xlabel(
        "PCA Component 1"
    )

    ax.set_ylabel(
        "PCA Component 2"
    )

    return fig


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    figure = create_visualization(
        "computer",
        10
    )

    if figure is not None:

        output_path = os.path.join(
            BASE_DIR,
            "model",
            "word_vectors.png"
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            "Visualization saved at:"
        )

        print(output_path)

        plt.show()

    else:

        print(
            "Unable to create visualization."
        )
