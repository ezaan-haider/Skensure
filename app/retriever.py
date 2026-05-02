from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import SessionLocal
from app.embeddings import embed_text


def retrieve_relevant_chunks(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    condition: str | None = None,
    min_similarity: float = 0.35
):
    db: Session | None = None

    try:
        db = SessionLocal()
        query_vector = embed_text(query)

        if condition:
            sql = text("""
                WITH ranked_chunks AS (
                    SELECT
                        c.id AS child_id,
                        c.parent_id,
                        c.content AS child_content,
                        c.source,
                        c.page,
                        c.condition,
                        c.embedding <=> CAST(:query_embedding AS vector) AS distance
                    FROM child_chunks c
                    WHERE c.condition = :condition
                    ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :fetch_k
                ),
                best_parent_chunks AS (
                    SELECT DISTINCT ON (rc.parent_id)
                        rc.parent_id,
                        rc.child_id,
                        rc.child_content,
                        rc.source,
                        rc.page,
                        rc.condition,
                        rc.distance
                    FROM ranked_chunks rc
                    ORDER BY rc.parent_id, rc.distance
                )
                SELECT
                    bpc.parent_id,
                    p.content AS parent_content,
                    bpc.child_content,
                    bpc.source,
                    bpc.page,
                    bpc.condition,
                    bpc.distance,
                    1 - bpc.distance AS similarity
                FROM best_parent_chunks bpc
                JOIN parent_documents p ON p.id = bpc.parent_id
                WHERE 1 - bpc.distance >= :min_similarity
                ORDER BY bpc.distance
                LIMIT :k
            """)

            params = {
                "query_embedding": query_vector,
                "condition": condition,
                "k": k,
                "fetch_k": fetch_k,
                "min_similarity": min_similarity,
            }

        else:
            sql = text("""
                WITH ranked_chunks AS (
                    SELECT
                        c.id AS child_id,
                        c.parent_id,
                        c.content AS child_content,
                        c.source,
                        c.page,
                        c.condition,
                        c.embedding <=> CAST(:query_embedding AS vector) AS distance
                    FROM child_chunks c
                    ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :fetch_k
                ),
                best_parent_chunks AS (
                    SELECT DISTINCT ON (rc.parent_id)
                        rc.parent_id,
                        rc.child_id,
                        rc.child_content,
                        rc.source,
                        rc.page,
                        rc.condition,
                        rc.distance
                    FROM ranked_chunks rc
                    ORDER BY rc.parent_id, rc.distance
                )
                SELECT
                    bpc.parent_id,
                    p.content AS parent_content,
                    bpc.child_content,
                    bpc.source,
                    bpc.page,
                    bpc.condition,
                    bpc.distance,
                    1 - bpc.distance AS similarity
                FROM best_parent_chunks bpc
                JOIN parent_documents p ON p.id = bpc.parent_id
                WHERE 1 - bpc.distance >= :min_similarity
                ORDER BY bpc.distance
                LIMIT :k
            """)

            params = {
                "query_embedding": query_vector,
                "k": k,
                "fetch_k": fetch_k,
                "min_similarity": min_similarity,
            }

        rows = db.execute(sql, params).fetchall()

        return [
            {
                "content": row.parent_content,
                "matched_chunk": row.child_content,
                "source": row.source,
                "page": row.page,
                "condition": row.condition,
                "similarity": round(float(row.similarity), 4),
            }
            for row in rows
        ]

    except Exception as e:
        print("Retrieval error:", str(e))
        return []

    finally:
        if db:
            db.close()