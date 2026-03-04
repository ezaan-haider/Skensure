from app.retriever import retrieve_relevant_chunks

results = retrieve_relevant_chunks("What is acne?")

for r in results:
    print("----")
    print(r)