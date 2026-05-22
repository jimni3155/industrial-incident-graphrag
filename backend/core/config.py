from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    EMBEDDING_MODEL = "text-embedding-3-small"

    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"


settings = Settings()