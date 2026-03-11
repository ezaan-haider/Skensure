from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import SessionLocal
from app.embeddings import embed_text


def retrieve_relevant_chunks(query: str, k: int = 3):

    db: Session = None

    try:
        db = SessionLocal()

        # Embed the user query
        query_vector = embed_text(query)

        # Search CHILD chunks but return PARENT documents
        sql = text("""
            SELECT
                p.id AS parent_id,
                p.content AS parent_content,
                c.content AS child_content,
                p.source,
                p.page
            FROM child_chunks c
            JOIN parent_documents p
                ON c.parent_id = p.id
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :k
        """)

        results = db.execute(
            sql,
            {
                "query_embedding": query_vector,
                "k": k
            }
        ).fetchall()

        # Return parent context
        contexts = []

        for row in results:
            contexts.append({
                "content": row.parent_content,
                "source": row.source,
                "page": row.page
            })

        return contexts

    except Exception as e:
        print("Retrieval error:", str(e))
        return []

    finally:
        if db:
            db.close()