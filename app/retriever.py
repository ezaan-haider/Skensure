from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import SessionLocal
from app.embeddings import embed_text


def retrieve_relevant_chunks(query: str, k: int = 3):
    db: Session = SessionLocal()

    query_vector = embed_text(query)

    sql = text("""
        SELECT id, content
        FROM documents
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :k
    """)

    results = db.execute(
        sql,
        {
            "query_embedding": query_vector,
            "k": k
        }
    ).fetchall()

    db.close()

    return [row.content for row in results]