import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")

DB_PATH = "./db/embeddings.db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
