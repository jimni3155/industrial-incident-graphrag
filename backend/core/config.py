from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # LLM gateway
    GATEWAY_API_KEY   = os.getenv("GATEWAY_API_KEY")
    GATEWAY_BASE_URL  = os.getenv("GATEWAY_BASE_URL")
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-sonnet-4-6")

    EMBEDDING_MODEL = "text-embedding-3-small"

    MULTIMODAL_MODEL      = os.getenv("MULTIMODAL_MODEL",      "gemini-2.5-pro")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"


settings = Settings()