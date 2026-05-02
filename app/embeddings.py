from langchain_huggingface import HuggingFaceEmbeddings

# all-mpnet-base-v2 produces 768-dimensional embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

def embed_text_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return embedding_model.embed_documents(texts)

def embed_text(text: str) -> list[float]:
    return embedding_model.embed_query(text)