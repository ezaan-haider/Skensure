from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.llm import model
from app.database import SessionLocal
from app import models
from app.retriever import retrieve_relevant_chunks
import re

import logging

logger = logging.getLogger("skensure.langgraph")


class ChatState(TypedDict):
    session_id: int
    user_id: int
    message: str
    image_id: Optional[int]

    messages: List[dict]

    retrieved_chunks: List[dict]
    knowledge_context: str

    prediction: Optional[str]
    confidence: Optional[float]

    prediction_context: str
    prediction_info: Optional[str]
    summary: Optional[str]

    reply: str

def fetch_prediction(state: ChatState):
    if not state.get("image_id"):
        return {"prediction": None, "confidence": None}

    db = SessionLocal()

    try:
        image = db.query(models.Image).filter(
            models.Image.id == state["image_id"],
            models.Image.user_id == state["user_id"]
        ).first()

        if not image or not image.prediction:
            return {"prediction": None, "confidence": None}

        return {
            "prediction": image.prediction,
            "confidence": image.confidence
        }

    finally:
        db.close()


def retrieve_prediction_info(state: ChatState):
    if not state.get("prediction"):
        return {"prediction_context": ""}

    disease = state["prediction"]

    chunks = retrieve_relevant_chunks(
        query=disease,
        k=5,
        condition=disease,
        min_similarity=0.30
    )

    if not chunks:
        context = "No reliable information found."
    else:
        context = "\n\n".join(
            f"Source: {c['source']} (page {c['page']})\n{c['content']}"
            for c in chunks
        )

    return {
        "prediction_context": context
    }


def disease_explainer(state: ChatState):
    if not state.get("prediction"):
        return {}

    disease = state["prediction"]

    prompt = f"""
        The model predicted that the user may have: {disease}.

        Using ONLY the medical knowledge provided below, explain:
        - what it is
        - common appearance
        - possible causes
        - general prevention tips

        Do NOT diagnose or confirm.

        Medical Knowledge:
        {state["prediction_context"]}
        """

    response = model.invoke([HumanMessage(content=prompt)]).content

    return {
        "prediction_info": response   # ✅ store separately
    }

def route_prediction(state: ChatState):
    confidence = state.get("confidence")

    if state.get("prediction") and confidence is not None and confidence >= 0.60:
        return "retrieve_prediction"

    return "load_history"



def load_history(state: ChatState):
    db = SessionLocal()

    try:
        session = db.query(models.ChatSession).filter(
            models.ChatSession.id == state["session_id"],
            models.ChatSession.user_id == state["user_id"]
        ).first()

        summary = session.summary if session else ""

        chats = (
            db.query(models.Chat)
            .filter(
                models.Chat.session_id == state["session_id"],
                models.Chat.user_id == state["user_id"]
            )
            .order_by(models.Chat.created_at.desc())
            .limit(4)
            .all()
        )

        chats = list(reversed(chats))

        messages = []

        if summary:
            messages.append({
                "role": "system",
                "content": f"Conversation Summary:\n{summary}"
            })

        for chat in chats:
            if chat.role == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": chat.message[:300]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": chat.message[:1000]
                })

        messages.append({
            "role": "user",
            "content": f"Question: {state['message']}"
        })

        return {"messages": messages}

    finally:
        db.close()

def retrieve_node(state: ChatState):
    prediction = state.get("prediction")
    user_message = state["message"]

    if prediction:
        retrieval_query = f"{prediction} {user_message}"

        chunks = retrieve_relevant_chunks(
            query=retrieval_query,
            k=4,
            condition=prediction,
            min_similarity=0.30
        )

        if not chunks:
            chunks = retrieve_relevant_chunks(
                query=retrieval_query,
                k=4,
                condition=None,
                min_similarity=0.35
            )
    else:
        retrieval_query = user_message

        chunks = retrieve_relevant_chunks(
            query=retrieval_query,
            k=4,
            condition=None,
            min_similarity=0.35
        )

    if not chunks:
        context = "No reliable medical knowledge was retrieved."
    else:
        context = "\n\n".join(
            f"Source: {c['source']} (page {c['page']})\n{c['content']}"
            for c in chunks
        )

    return {
        "retrieved_chunks": chunks,
        "knowledge_context": context
    }

def build_prompt(state: ChatState):
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not retrieved_chunks:
        safety_context = (
            "No reliable medical knowledge was retrieved. "
            "You must not answer medically beyond saying that the system does not have enough information."
        )
    else:
        safety_context = state["knowledge_context"]

    system_message = {
        "role": "system",
        "content": (
            "You are Skensure, an AI-powered dermatology assistant for educational purposes only.\n\n"

            "STRICT RULES:\n"
            "1. Use ONLY the medical knowledge provided below.\n"
            "2. Do NOT use outside knowledge.\n"
            "3. Do NOT diagnose, confirm diseases, or prescribe treatment.\n"
            "4. If retrieved knowledge is missing or weak, say: 'I am not sure based on the provided information.'\n"
            "5. Do NOT make assumptions about the user's condition.\n"
            "6. Always advise professional medical review for concerning, worsening, painful, spreading, infected, or uncertain skin symptoms.\n\n"

            "RESPONSE FORMAT:\n"
            "- Answer the latest user question only.\n"
            "- Keep it concise and clear.\n"
            "- Mention uncertainty where appropriate.\n"
            "- End with a 'Sources used' line listing source filenames and pages if sources are available.\n\n"

            "MEDICAL KNOWLEDGE:\n"
            f"{safety_context}"
        )
    }

    return {
        "messages": [system_message] + state["messages"]
    }

def llm_node(state: ChatState):
    lc_messages = []

    for msg in state["messages"]:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    logger.info("Calling LLM for session_id=%s user_id=%s", state["session_id"], state["user_id"])
    response = model.invoke(lc_messages).content

    return {
        "reply": response,
        "messages": state["messages"] + [
            {"role": "assistant", "content": response}
        ]
    }

def summarize_node(state: ChatState):
    db = SessionLocal()

    try:
        session = db.query(models.ChatSession).filter(
            models.ChatSession.id == state["session_id"],
            models.ChatSession.user_id == state["user_id"]
        ).first()

        if not session:
            return {}

        old_summary = session.summary or ""

        prompt = f"""
Previous summary:
{old_summary}

Latest user message:
{state["message"]}

Latest assistant reply:
{state["reply"]}

Update the summary in a concise way.
Focus on the user's dermatology concern and key advice already given.
"""

        new_summary = model.invoke([HumanMessage(content=prompt)]).content

        session.summary = new_summary
        db.commit()

        return {"summary": new_summary}

    finally:
        db.close()

def save_chat(state: ChatState):
    db = SessionLocal()

    try:
        db.add(models.Chat(
            user_id=state["user_id"],
            session_id=state["session_id"],
            image_id=state["image_id"],
            role="user",
            message=state["message"]
        ))

        assistant_message = state["reply"]

        if state.get("prediction_info"):
            assistant_message = (
                f"{state['prediction_info']}\n\n"
                f"---\n\n"
                f"{state['reply']}"
            )

        db.add(models.Chat(
            user_id=state["user_id"],
            session_id=state["session_id"],
            image_id=state["image_id"],
            role="assistant",
            message=assistant_message
        ))

        db.commit()

        return {}

    finally:
        db.close()

def generate_title(state: ChatState):
    db = SessionLocal()

    try:
        session = db.query(models.ChatSession).filter(
            models.ChatSession.id == state["session_id"],
            models.ChatSession.user_id == state["user_id"]
        ).first()

        if not session or session.title != "New Chat":
            return {}

        prompt = f"""
Generate a short 3-5 word title for this conversation.

User message:
{state["message"]}

Return only the title.
"""

        title = model.invoke([HumanMessage(content=prompt)]).content.strip()
        title = title.replace('"', '').replace("'", "")

        session.title = title[:60]
        db.commit()

        return {}

    finally:
        db.close()

def rerank_chunks(state: ChatState):
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {}

    # Use existing similarity score from retriever instead of another LLM call
    filtered = sorted(
        chunks,
        key=lambda x: x.get("similarity", 0),
        reverse=True
    )[:2]

    context = "\n\n".join(
        f"Source: {c['source']} (page {c['page']})\n{c['content']}"
        for c in filtered
    )

    return {
        "retrieved_chunks": filtered,
        "knowledge_context": context
    }

from langgraph.graph import StateGraph, END

graph = StateGraph(ChatState)


graph.add_node("fetch_prediction", fetch_prediction)
graph.add_node("retrieve_prediction", retrieve_prediction_info)
graph.add_node("disease_explainer", disease_explainer)
graph.add_node("load_history", load_history)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_chunks)
graph.add_node("build_prompt", build_prompt)
graph.add_node("llm", llm_node)
graph.add_node("save", save_chat)
graph.add_node("summarize", summarize_node)
graph.add_node("generate_title", generate_title)

graph.set_entry_point("fetch_prediction")

graph.add_conditional_edges(
    "fetch_prediction",
    route_prediction,
    {
        "retrieve_prediction": "retrieve_prediction",
        "load_history": "load_history"
    }
)

graph.add_edge("retrieve_prediction", "disease_explainer")
graph.add_edge("disease_explainer", "load_history")
graph.add_edge("load_history", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "build_prompt")
graph.add_edge("build_prompt", "llm")
graph.add_edge("llm", "save")
graph.add_edge("save", "generate_title")
graph.add_edge("generate_title", "summarize")
graph.add_edge("summarize", END)

chat_graph = graph.compile()