from sentence_transformers import SentenceTransformer

from redrob_ranker.neural_retrieval import MODEL_NAME, MODEL_PATH


def main() -> None:
    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public model {MODEL_NAME} to {MODEL_PATH}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.save_pretrained(str(MODEL_PATH))
    print(f"Saved {model.get_embedding_dimension()}-dimensional model")


if __name__ == "__main__":
    main()
