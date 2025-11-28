from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# Look for .env in the backend directory or project root
_backend_dir = Path(__file__).parent.parent
_project_root = _backend_dir.parent
_env_file = _backend_dir / ".env"
if not _env_file.exists():
    _env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
    print(f"✓ Loaded environment variables from {_env_file}")

# Determine default data directory based on environment
# If running in Docker, use /app/data, otherwise use ../data relative to backend
_default_data_dir = "/app/data" if os.path.exists("/app/data") else str(Path(__file__).parent.parent.parent / "data")

class Settings(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local-384")
    llm_provider: str = os.getenv("LLM_PROVIDER", "stub")  # stub | openai | ollama
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")  # qdrant | memory
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333" if os.path.exists("/app/data") else "http://localhost:6333")
    collection_name: str = os.getenv("COLLECTION_NAME", "policy_helper")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    data_dir: str = os.getenv("DATA_DIR", _default_data_dir)

settings = Settings()

# Debug logging (without exposing the full API key)
if settings.openai_api_key:
    key_preview = settings.openai_api_key[:8] + "..." if len(settings.openai_api_key) > 8 else "***"
    print(f"✓ OpenAI API key found: {key_preview}")
else:
    print("⚠ OpenAI API key not found in environment variable OPENAI_API_KEY")
