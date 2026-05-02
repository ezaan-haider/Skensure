import os
import re
import hashlib
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app import models
from app.embeddings import embed_text_batch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# FILE HASHING
# -----------------------------
def calculate_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Remove excessive whitespace but preserve sentence flow
    text = re.sub(r"\s+", " ", text)

    # Do NOT aggressively remove all [12] references.
    # In medical texts, some bracketed numbers may be meaningful.
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    return text.strip()


def infer_condition_from_filename(filename: str) -> str:
    name = filename.lower()

    if "acne" in name:
        return "acne"
    if "rosacea" in name:
        return "rosacea"
    if "shingles" in name or "zoster" in name:
        return "shingles"
    if "chickenpox" in name or "varicella" in name:
        return "chickenpox"
    if "eczema" in name or "atopic" in name or "dermatitis" in name:
        return "atopic_dermatitis"

    return "general"

# -----------------------------
# DUPLICATE CHECK
# -----------------------------
def pdf_already_ingested(db: Session, source: str, file_hash: str) -> bool:
    """
    Best version: checks file_hash if your model has it.
    Fallback: checks source filename only.
    """

    if hasattr(models.ParentDocument, "file_hash"):
        existing = (
            db.query(models.ParentDocument)
            .filter(models.ParentDocument.file_hash == file_hash)
            .first()
        )
    else:
        existing = (
            db.query(models.ParentDocument)
            .filter(models.ParentDocument.source == source)
            .first()
        )

    return existing is not None


# -----------------------------
# SAFE PARENT DOCUMENT CREATION
# -----------------------------
def create_parent_document(
    content: str,
    source: str,
    page: Optional[int],
    file_hash: str,
    condition: str
):
    data = {
        "content": content,
        "source": source,
        "page": page,
        "condition": condition,
    }

    if hasattr(models.ParentDocument, "file_hash"):
        data["file_hash"] = file_hash

    return models.ParentDocument(**data)


# -----------------------------
# INGESTION FUNCTION
# -----------------------------
def ingest_pdfs_from_folder(folder_path: str):
    db: Session | None = None

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_child_chunks = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0

    try:
        db = SessionLocal()

        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")

        print("Starting PDF ingestion...")

        pdf_files = sorted(
            [file for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]
        )

        if not pdf_files:
            print("No PDF files found.")
            return

        for file in pdf_files:
            file_path = os.path.join(folder_path, file)
            file_hash = calculate_file_hash(file_path)
            condition = infer_condition_from_filename(file)

            print(f"\nProcessing file: {file}")

            try:
                if pdf_already_ingested(db, source=file, file_hash=file_hash):
                    print(f"Skipped duplicate/already ingested file: {file}")
                    skipped_files += 1
                    continue

                loader = PyPDFLoader(file_path)
                documents = loader.load()

                if not documents:
                    print(f"No text extracted from: {file}")
                    skipped_files += 1
                    continue

                cleaned_documents = []

                for doc in documents:
                    cleaned_text = clean_text(doc.page_content)

                    if len(cleaned_text) < 100:
                        continue

                    doc.page_content = cleaned_text
                    doc.metadata["source"] = file
                    doc.metadata["file_hash"] = file_hash

                    cleaned_documents.append(doc)

                if not cleaned_documents:
                    print(f"No usable text after cleaning: {file}")
                    skipped_files += 1
                    continue

                parent_chunks = parent_splitter.split_documents(cleaned_documents)

                if not parent_chunks:
                    print(f"No parent chunks created for: {file}")
                    skipped_files += 1
                    continue

                file_child_count = 0

                for parent_chunk in parent_chunks:
                    parent_text = parent_chunk.page_content.strip()

                    if len(parent_text) < 200:
                        continue

                    page = parent_chunk.metadata.get("page")

                    parent_doc = create_parent_document(
                        content=parent_text,
                        source=file,
                        page=page,
                        file_hash=file_hash,
                        condition=condition,
                    )

                    db.add(parent_doc)
                    db.flush()

                    child_docs = child_splitter.create_documents(
                        [parent_text],
                        metadatas=[parent_chunk.metadata],
                    )

                    child_texts = []

                    for child in child_docs:
                        child_text = child.page_content.strip()

                        if len(child_text) < 80:
                            continue

                        child_texts.append(child_text)

                    if not child_texts:
                        continue

                    vectors = embed_text_batch(child_texts)

                    if len(vectors) != len(child_texts):
                        raise ValueError(
                            f"Embedding mismatch in {file}: "
                            f"{len(child_texts)} texts but {len(vectors)} vectors"
                        )

                    for child_text, vector in zip(child_texts, vectors):
                        child_chunk = models.ChildChunk(
                        parent_id=parent_doc.id,
                        content=child_text,
                        embedding=vector,
                        source=file,
                        page=page,
                        condition=condition,
                    )

                        db.add(child_chunk)
                        file_child_count += 1
                        total_child_chunks += 1

                db.commit()

                processed_files += 1
                print(f"Completed: {file}")
                print(f"Child chunks stored from this file: {file_child_count}")

            except Exception as file_error:
                db.rollback()
                failed_files += 1
                print(f"Failed to ingest {file}: {str(file_error)}")

        print("\nIngestion completed.")
        print(f"Files processed: {processed_files}")
        print(f"Files skipped: {skipped_files}")
        print(f"Files failed: {failed_files}")
        print(f"Total child chunks stored: {total_child_chunks}")

    except Exception as e:
        print("Fatal ingestion error:", str(e))
        if db:
            db.rollback()

    finally:
        if db:
            db.close()


if __name__ == "__main__":
    ingest_pdfs_from_folder("D:\skensureV1\data")