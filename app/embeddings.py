from langchain_huggingface import HuggingFaceEmbeddings

# 384-dimensional embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def embed_text(text: str):
    return embedding_model.embed_query(text)

vec = embed_text("What is acne?")
print(len(vec))