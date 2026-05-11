from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.llm import model
from app.database import SessionLocal
from app import models
from app.retriever import retrieve_relevant_chunks
import json
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

    condition_context: Optional[str]
    mentioned_conditions: List[str]
    rewritten_query: Optional[str]
    user_language: Optional[str]

    reply: str



URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

def detect_user_language(text: str) -> str:
    """
    Detects whether the user is writing in Urdu script.
    Returns:
    - 'urdu'
    - 'english'
    """

    if not text:
        return "english"

    if URDU_SCRIPT_RE.search(text):
        return "urdu"

    return "english"

def detect_language_node(state: ChatState):
    language = detect_user_language(state.get("message", ""))

    return {
        "user_language": language
    }

CONDITION_STRONG_KEYWORDS = {
    "acne": ["acne"],
    "rosacea": ["rosacea"],
    "shingles": ["shingles", "zoster", "herpes zoster"],
    "chickenpox": ["chickenpox", "varicella"],
    "atopic_dermatitis": ["atopic dermatitis", "atopic eczema", "eczema"],
}

# Weak words are only used when NO explicit disease name is found.
# Do not include vague words like "spots", "rash", "bumps", "marks" here.
CONDITION_WEAK_KEYWORDS = {
    "acne": ["pimples", "blackheads", "whiteheads", "comedones"],
    "rosacea": ["facial redness", "flushing", "red face"],
    "atopic_dermatitis": ["dermatitis"],
}


COMPARISON_TERMS = [
    "different", "difference", "compare", "compared", "versus", "vs",
    "similar", "same as", "like", "unlike", "distinguish",
    "develop into", "turn into", "become"
]


def keyword_found(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def is_comparison_query(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()
    return any(term in lowered for term in COMPARISON_TERMS)


def normalize_condition(value: str | None) -> Optional[str]:
    if not value:
        return None

    lowered = value.lower().strip()

    for condition, keywords in CONDITION_STRONG_KEYWORDS.items():
        if any(keyword_found(lowered, keyword) for keyword in keywords):
            return condition

    for condition, keywords in CONDITION_WEAK_KEYWORDS.items():
        if any(keyword_found(lowered, keyword) for keyword in keywords):
            return condition

    return None



def detect_all_conditions_from_text(text: str) -> list[str]:
    if not text:
        return []

    found = []

    # First detect explicit disease names.
    for condition, keywords in CONDITION_STRONG_KEYWORDS.items():
        if any(keyword_found(text, keyword) for keyword in keywords):
            found.append(condition)

    # If explicit disease names exist, ignore weak/vague symptom words.
    if found:
        return found

    # Only use weak condition hints if no clear disease was mentioned.
    for condition, keywords in CONDITION_WEAK_KEYWORDS.items():
        if any(keyword_found(text, keyword) for keyword in keywords):
            found.append(condition)

    return found

def detect_condition_from_text(text: str) -> Optional[str]:
    conditions = detect_all_conditions_from_text(text)
    return conditions[0] if conditions else None

def get_previous_conditions_from_messages(messages: list[dict]) -> list[str]:
    """
    Finds the most recent true active condition from previous conversation messages.

    Important:
    - Skips comparison questions because the mentioned condition may be only a comparison target.
      Example: "how is this different from acne" should not make acne the active condition.
    - Prefers user messages.
    - Falls back to assistant/system only if needed.
    """

    # 1. Prefer previous user messages, but skip comparison queries
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")

            if is_comparison_query(content):
                continue

            found = detect_all_conditions_from_text(content)
            if found:
                return found

    # 2. Fallback: inspect non-comparison assistant/system messages
    for msg in reversed(messages):
        content = msg.get("content", "")

        if is_comparison_query(content):
            continue

        found = detect_all_conditions_from_text(content)
        if found:
            return found

    return []

def fallback_rewrite_query(user_message: str, condition: Optional[str]) -> str:
    msg = user_message.lower()

    if not condition:
        return user_message

    if any(word in msg for word in ["why", "cause", "causes", "happening", "trigger", "triggers"]):
        return f"{condition} causes triggers risk factors flare factors"

    if any(word in msg for word in ["contagious", "spread", "catch", "infectious", "transmit", "transmission"]):
        return f"{condition} contagious transmission infectious spread"

    if any(word in msg for word in ["treat", "treatment", "manage", "management", "what should i do", "help"]):
        return f"{condition} treatment management self care medical advice"

    if any(word in msg for word in ["prevent", "avoid", "stop it happening"]):
        return f"{condition} prevention avoidance triggers"

    if any(word in msg for word in ["symptom", "symptoms", "look like", "appearance", "signs"]):
        return f"{condition} symptoms clinical features appearance"

    return f"{condition} {user_message}"

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

def detect_condition_context(state: ChatState):
    latest_message = state.get("message", "")
    latest_conditions = detect_all_conditions_from_text(latest_message)

    prediction_condition = normalize_condition(state.get("prediction"))

    messages = state.get("messages", [])
    previous_conditions = get_previous_conditions_from_messages(messages)

    comparison_query = is_comparison_query(latest_message)

    # Case 1: latest message explicitly/weakly mentions condition(s)
    if latest_conditions:

        # Important fix:
        # If user asks "how is this different from acne",
        # acne is the comparison target, not necessarily the active condition.
        if comparison_query and previous_conditions:
            combined_conditions = []

            for condition in previous_conditions:
                if condition not in combined_conditions:
                    combined_conditions.append(condition)

            for condition in latest_conditions:
                if condition not in combined_conditions:
                    combined_conditions.append(condition)

            return {
                "condition_context": previous_conditions[0],
                "mentioned_conditions": combined_conditions
            }

        # Normal case: user directly names a condition.
        return {
            "condition_context": latest_conditions[0],
            "mentioned_conditions": latest_conditions
        }

    # Case 2: no latest condition, use image prediction if available
    if prediction_condition:
        return {
            "condition_context": prediction_condition,
            "mentioned_conditions": [prediction_condition]
        }

    # Case 3: no latest condition, use previous conversation context
    if previous_conditions:
        return {
            "condition_context": previous_conditions[0],
            "mentioned_conditions": previous_conditions
        }

    # Case 4: vague symptom-only query
    # Do not force acne/rosacea/chickenpox.
    return {
        "condition_context": None,
        "mentioned_conditions": []
    }

def rewrite_query_node(state: ChatState):
    user_message = state["message"]
    condition = state.get("condition_context")
    mentioned_conditions = state.get("mentioned_conditions", [])
    user_language = state.get("user_language", "english")

    fallback_query = fallback_rewrite_query(user_message, condition)

    prompt = f"""
You are rewriting a dermatology user question into an English retrieval search query.

The medical documents are in English.
Even if the user writes in Urdu, the retrieval query MUST be English.

Do not answer the user.
Do not diagnose.
Create a short English retrieval query.

User language:
{user_language}

Active condition:
{condition}

All mentioned conditions:
{mentioned_conditions}

User question:
{user_message}

Return ONLY valid JSON:
{{
  "retrieval_query": "english retrieval query"
}}
"""

    try:
        response = model.invoke([HumanMessage(content=prompt)]).content.strip()

        match = re.search(r"\{.*\}", response, re.DOTALL)

        if not match:
            return {"rewritten_query": fallback_query}

        data = json.loads(match.group(0))

        rewritten_query = data.get("retrieval_query", "").strip()

        if not rewritten_query:
            rewritten_query = fallback_query

        for cond in mentioned_conditions:
            if cond not in rewritten_query.lower():
                rewritten_query = f"{cond} {rewritten_query}"

        if len(rewritten_query.split()) < 2:
            rewritten_query = fallback_query

        return {
            "rewritten_query": rewritten_query
        }

    except Exception:
        return {
            "rewritten_query": fallback_query
        }

def retrieve_node(state: ChatState):
    condition_context = state.get("condition_context")
    mentioned_conditions = state.get("mentioned_conditions", [])
    user_message = state["message"]

    retrieval_query = state.get("rewritten_query") or fallback_rewrite_query(
        user_message,
        condition_context
    )

    chunks = []

    if mentioned_conditions:
        per_condition_k = 2 if len(mentioned_conditions) > 1 else 4

        for condition in mentioned_conditions:
            condition_query = f"{condition} {retrieval_query}"

            condition_chunks = retrieve_relevant_chunks(
                query=condition_query,
                k=per_condition_k,
                condition=condition,
                min_similarity=0.30
            )

            chunks.extend(condition_chunks)

    # 2. If no condition-specific chunks found, fallback to active condition
    if not chunks and condition_context:
        chunks = retrieve_relevant_chunks(
            query=retrieval_query,
            k=4,
            condition=condition_context,
            min_similarity=0.30
        )

    # 3. Final fallback: whole corpus
    if not chunks:
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

    user_language = state.get("user_language", "english")

    language_instruction = ""

    if user_language == "urdu":
        language_instruction = (
            "7. The user is speaking in Urdu. "
            "Respond fully in Urdu script while keeping medical terminology clear.\n"
        )

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
            f"{language_instruction}"

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
    )[:4]

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
graph.add_node("detect_condition_context", detect_condition_context)
graph.add_node("detect_language", detect_language_node)
graph.add_node("rewrite_query", rewrite_query_node)
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
graph.add_edge("load_history", "detect_condition_context")
graph.add_edge("detect_condition_context", "detect_language")
graph.add_edge("detect_language", "rewrite_query")
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "build_prompt")
graph.add_edge("build_prompt", "llm")
graph.add_edge("llm", "save")
graph.add_edge("save", "generate_title")
graph.add_edge("generate_title", "summarize")
graph.add_edge("summarize", END)

chat_graph = graph.compile()