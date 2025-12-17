"""
Database connection module for the RAG Chatbot application.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = os.getenv("NEON_DB_URL", "postgresql://neondb_owner:npg_mwfiIl78yMoW@ep-shiny-silence-ab4yvadr-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from the .env file


settings = Settings()

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()