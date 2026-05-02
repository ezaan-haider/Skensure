from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import engine, Base, SessionLocal
from app import models
from app.security import hash_password
from fastapi import File, UploadFile
import shutil
import os
from vision.inference import predict_image
from app.retriever import retrieve_relevant_chunks
from app.langgraph import chat_graph
from dotenv import load_dotenv
import os
from uuid import uuid4
from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skensure")

logger.info("Hugging Face token loaded: %s", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again shortly."}
    )

# Create tables
Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"message": "SkensureV1 backend running"}


@app.post("/signup")
def signup(email: str, password: str, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_pw = hash_password(password)

    # Create new user
    new_user = models.User(
        email=email,
        hashed_password=hashed_pw
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User created successfully"}

@app.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    from app.security import verify_password

    if not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    return {"message": "Login successful"}

@app.post("/upload-image")
@limiter.limit("10/minute")
def upload_image(
    request: Request,
    user_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    ext = Path(file.filename).suffix.lower()

    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    safe_filename = f"{uuid4()}{ext}"
    file_path = os.path.join(upload_dir, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_image = models.Image(
        user_id=user_id,
        image_path=file_path
    )

    db.add(new_image)
    db.commit()
    db.refresh(new_image)

    return {
        "message": "Image uploaded successfully",
        "image_id": new_image.id,
        "file_path": file_path
    }

@app.post("/predict/{image_id}")
def predict(image_id: int, db: Session = Depends(get_db)):
    # Get image record
    image_record = db.query(models.Image).filter(models.Image.id == image_id).first()

    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")

    # Run model prediction
    label, confidence = predict_image(image_record.image_path)

    # Update database
    image_record.prediction = label
    image_record.confidence = confidence
    db.commit()

    return {
        "prediction": label,
        "confidence": confidence
    }

@app.post("/create-session")
def create_session(user_id: int, db: Session = Depends(get_db)):
    session = models.ChatSession(user_id=user_id)

    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "session_id": session.id
    }

@app.get("/sessions/{user_id}")
def get_sessions(user_id: int, db: Session = Depends(get_db)):
    sessions = db.query(models.ChatSession).filter(
        models.ChatSession.user_id == user_id
    ).order_by(models.ChatSession.created_at.desc()).all()

    return [
        {
            "session_id": s.id,
            "title": s.title,
            "created_at": s.created_at
        }
        for s in sessions
    ]

@app.put("/session/{session_id}/rename")
def rename_session(session_id: int, title: str, db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.title = title
    db.commit()

    return {"message": "Session renamed"}

@app.get("/session/{session_id}/messages")
def get_session_messages(session_id: int, db: Session = Depends(get_db)):
    chats = db.query(models.Chat).filter(
        models.Chat.session_id == session_id
    ).order_by(models.Chat.created_at).all()

    return chats

@app.post("/chat")
@limiter.limit("20/minute")
def create_chat(
    request: Request,
    user_id: int,
    session_id: int,
    message: str,
    image_id: int = None,
    db: Session = Depends(get_db)
):
    if not message or len(message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Message is too long")

    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == user_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found for this user")

    if image_id is not None:
        image = db.query(models.Image).filter(
            models.Image.id == image_id,
            models.Image.user_id == user_id
        ).first()

        if not image:
            raise HTTPException(status_code=404, detail="Image not found for this user")

    result = chat_graph.invoke({
        "user_id": user_id,
        "session_id": session_id,
        "message": message.strip(),
        "image_id": image_id
    })

    return {
        "prediction_info": result.get("prediction_info"),
        "reply": result.get("reply"),
        "sources": result.get("retrieved_chunks")
    }

@app.delete("/session/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # delete chats first (important)
    db.query(models.Chat).filter(
        models.Chat.session_id == session_id
    ).delete()

    db.delete(session)
    db.commit()

    return {"message": "Session deleted"}

@app.get("/chat-history/{user_id}")
def get_chat_history(user_id: int, image_id: int = None, db: Session = Depends(get_db)):
    query = db.query(models.Chat).filter(models.Chat.user_id == user_id)

    if image_id:
        query = query.filter(models.Chat.image_id == image_id)

    chats = query.order_by(models.Chat.created_at).all()

    return [
        {
            "role": chat.role,
            "message": chat.message,
            "created_at": chat.created_at
        }
        for chat in chats
    ]