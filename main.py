from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import engine, Base, SessionLocal
from app import models
from app.security import hash_password
from fastapi import File, UploadFile
import shutil
import os
from vision.inference import predict_image
from app.llm import generate_reply
from app.retriever import retrieve_relevant_chunks


app = FastAPI()

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
def upload_image(
    user_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Check if user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create uploads folder path
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create image record
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


@app.post("/chat")
def create_chat(
    user_id: int,
    message: str,
    image_id: int = None,
    db: Session = Depends(get_db)
):
    # 1️⃣ Check user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2️⃣ Store user message
    user_chat = models.Chat(
        user_id=user_id,
        image_id=image_id,
        role="user",
        message=message
    )
    db.add(user_chat)
    db.commit()

    # 3️⃣ Fetch last 8 messages for context
    recent_chats = (
        db.query(models.Chat)
        .filter(models.Chat.user_id == user_id)
        .order_by(models.Chat.created_at.desc())
        .limit(8)
        .all()
    )

    recent_chats = list(reversed(recent_chats))

    # 🔎 Retrieve relevant medical knowledge
    retrieved_chunks = retrieve_relevant_chunks(message, k=3)

    knowledge_context = "\n\n".join(retrieved_chunks)

    # 4️⃣ Convert to message format
    messages = [
    {
        "role": "system",
        "content": (
            "You are Skensure, an AI dermatology educational assistant.\n\n"
            "Use ONLY the medical knowledge provided below to answer.\n"
            "If the answer is not in the provided knowledge, say you are unsure.\n"
            "Do NOT provide prescriptions or definitive diagnoses.\n"
            "Encourage consulting a licensed dermatologist.\n\n"
            "Medical Knowledge:\n"
            f"{knowledge_context}"
        )
    }
]

    for chat in recent_chats:
        messages.append({
            "role": chat.role,
            "content": chat.message
        })

    print("==== MESSAGES SENT TO LLM ====")
    for m in messages:
        print(m)

    # 5️⃣ Call LLM
    assistant_reply = generate_reply(messages)

    # 6️⃣ Store assistant reply
    assistant_chat = models.Chat(
        user_id=user_id,
        image_id=image_id,
        role="assistant",
        message=assistant_reply
    )
    db.add(assistant_chat)
    db.commit()

    # 7️⃣ Return reply
    return {
        "reply": assistant_reply
    }

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