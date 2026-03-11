from langchain_huggingface import HuggingFaceEmbeddings

# 384-dimensional embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def embed_text_batch(texts: list[str]) -> list[list[float]]:
    return embedding_model.embed_documents(texts)


def embed_text(text: str) -> list[float]:
    return embedding_model.embed_query(text)