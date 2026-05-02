from sqlalchemy import Column, Integer, String, TIMESTAMP, text
from app.database import Base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from datetime import datetime
from sqlalchemy import Text  # add this import
from pgvector.sqlalchemy import Vector

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from sqlalchemy import ForeignKey, Float
from sqlalchemy.orm import relationship



class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )



class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String, nullable=False)
    prediction = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )

    user = relationship("User")

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)

    role = Column(String, nullable=False)  # "user", "assistant", "system"
    message = Column(String, nullable=False)

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )

    user = relationship("User")
    image = relationship("Image")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New Chat")
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)




class ParentDocument(Base):
    __tablename__ = "parent_documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)

    source = Column(String, nullable=False)
    page = Column(Integer, nullable=True)
    condition = Column(String, nullable=True, index=True)

    file_hash = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    children = relationship(
        "ChildChunk",
        back_populates="parent",
        cascade="all, delete-orphan"
    )


class ChildChunk(Base):
    __tablename__ = "child_chunks"

    id = Column(Integer, primary_key=True, index=True)

    parent_id = Column(
        Integer,
        ForeignKey("parent_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False)

    source = Column(String, nullable=False)
    page = Column(Integer, nullable=True)
    condition = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    parent = relationship("ParentDocument", back_populates="children")