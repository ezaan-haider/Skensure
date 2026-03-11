import os
import re
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app import models
from app.embeddings import embed_text_batch   # assumes batch embedding function

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)       # remove citations [12]
    text = re.sub(r"\s+", " ", text)          # normalize whitespace
    return text.strip()


# -----------------------------
# INGESTION FUNCTION
# -----------------------------
def ingest_pdfs_from_folder(folder_path: str):

    db: Session = None

    try:
        db = SessionLocal()

        # Parent chunk splitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        # Child chunk splitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80
        )

        chunk_counter = 0
        batch_size = 32

        print("Starting PDF ingestion...")

        for file in os.listdir(folder_path):

            if not file.endswith(".pdf"):
                continue

            file_path = os.path.join(folder_path, file)

            print(f"\nProcessing file: {file}")

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            # Clean text before splitting
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)

            # Parent chunks using split_documents to preserve metadata
            parent_chunks = parent_splitter.split_documents(documents)

            for parent_chunk in parent_chunks:

                # Store parent document
                parent_doc = models.ParentDocument(
                    content=parent_chunk.page_content,
                    source=file,
                    page=parent_chunk.metadata.get("page")
                )

                db.add(parent_doc)
                db.flush()  # get parent ID

                # Child chunks from parent
                child_docs = child_splitter.create_documents(
                    [parent_chunk.page_content]
                )

                child_texts = []
                child_metadata = []

                for child in child_docs:

                    text = child.page_content.strip()

                    if len(text) < 50:
                        continue

                    metadata = {
                        "source": file,
                        "page": parent_chunk.metadata.get("page"),
                        "parent_id": parent_doc.id
                    }

                    child_texts.append(text)
                    child_metadata.append(metadata)

                if not child_texts:
                    continue

                # -----------------------------
                # BATCH EMBEDDING (BAAI)
                # -----------------------------
                vectors = embed_text_batch(child_texts)

                for text, vector, meta in zip(child_texts, vectors, child_metadata):

                    child_chunk = models.ChildChunk(
                        parent_id=meta["parent_id"],
                        content=text,
                        embedding=vector,
                        source=meta["source"],
                        page=meta["page"]
                    )

                    db.add(child_chunk)

                    chunk_counter += 1

                    if chunk_counter % batch_size == 0:
                        db.commit()
                        print(f"{chunk_counter} chunks stored...")

        db.commit()

        print("\nIngestion completed successfully.")
        print(f"Total chunks stored: {chunk_counter}")

    except Exception as e:
        print("Error during ingestion:", str(e))
        if db:
            db.rollback()

    finally:
        if db:
            db.close()


# -----------------------------
# RUN INGESTION
# -----------------------------
if __name__ == "__main__":
    ingest_pdfs_from_folder("data")