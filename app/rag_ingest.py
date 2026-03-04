from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import models
from app.embeddings import embed_text

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os


def ingest_pdfs_from_folder(folder_path: str):
    db: Session = SessionLocal()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    all_chunks = []

    # Load PDFs
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents = loader.load()

            for doc in documents:
                chunks = splitter.split_text(doc.page_content)

                for chunk in chunks:
                    all_chunks.append({
                        "content": chunk
                    })

    print(f"Total chunks created: {len(all_chunks)}")

    # Store in DB
    for item in all_chunks:
        vector = embed_text(item["content"])

        doc = models.Document(
            content=item["content"],
            embedding=vector       # make sure column exists
        )

        db.add(doc)

    db.commit()
    db.close()

    print("PDF ingestion complete.")

ingest_pdfs_from_folder("data")