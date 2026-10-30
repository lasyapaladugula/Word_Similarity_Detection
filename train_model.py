import os
from gensim.models import Word2Vec

from src.preprocessing import load_and_preprocess


# Find project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Dataset path
CORPUS_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "corpus.txt"
)

# Model folder
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

# Create model folder if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 60)
print("WORD SIMILARITY DETECTION USING WORD2VEC")
print("=" * 60)

print()
print("Loading and preprocessing corpus...")

sentences = load_and_preprocess(CORPUS_PATH)

print("Total sentences:", len(sentences))

print()
print("Training CBOW model...")

# CBOW model
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=100,
    seed=42
)

# Save CBOW model
cbow_path = os.path.join(
    MODEL_DIR,
    "cbow_model.model"
)

cbow_model.save(cbow_path)

print("CBOW model saved successfully!")


print()
print("Training Skip-Gram model...")

# Skip-Gram model
skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100,
    seed=42
)

# Save Skip-Gram model
skipgram_path = os.path.join(
    MODEL_DIR,
    "skipgram_model.model"
)

skipgram_model.save(skipgram_path)

# Save Skip-Gram as the main project model
main_model_path = os.path.join(
    MODEL_DIR,
    "word2vec_model.model"
)

skipgram_model.save(main_model_path)

print("Skip-Gram model saved successfully!")

print()
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print()
print("Generated model files:")
print("1. cbow_model.model")
print("2. skipgram_model.model")
print("3. word2vec_model.model")