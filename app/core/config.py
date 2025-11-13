from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    """Configuration settings for the application."""
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CUSTOM_GOOGLE_SEARCH = os.getenv('CUSTOM_GOOGLE_SEARCH')
    CX_ID = os.getenv('CX_ID')
    
    @staticmethod
    def validate():
        """Validate the configuration settings."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        if not Config.CUSTOM_GOOGLE_SEARCH:
            raise ValueError("Missing CUSTOM_GOOGLE_SEARCH in environment variables.")
        if not Config.CX_ID:
            raise ValueError("Missing CX_ID in environment variables.")