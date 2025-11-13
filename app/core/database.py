from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOSTNAME = os.getenv("DB_HOSTNAME")
DB_PORT = os.getenv("DB_PORT")

encoded_password = quote_plus(DB_PASSWORD)

DATABASE_URL = f"postgresql://{DB_USERNAME}:{encoded_password}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}"

def init_db():
    """Initialize the database connection and test it."""
    try:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"options": "-c search_path=public"}
        )

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            connection.commit()
        
        logger.info("Database connection successful.")
        return engine
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

engine = init_db()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


def get_db():
    """Provide a new session for each request."""
    db = SessionLocal()
    try:
        yield db

    finally:
        db.close()