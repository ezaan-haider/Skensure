from app.rag_ingest import ingest_documents

medical_docs = [
    """
    Acne is a common skin condition that occurs when hair follicles become clogged
    with oil and dead skin cells. It often causes whiteheads, blackheads, or pimples.
    Acne is most common among teenagers, though it affects people of all ages.
    """,
    """
    Melanoma is a serious form of skin cancer that begins in cells known as melanocytes.
    Though less common than other skin cancers, melanoma is more dangerous because
    it is more likely to spread to other parts of the body if not detected early.
    """
]

ingest_documents(medical_docs)