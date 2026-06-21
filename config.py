import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
QWEN_MODEL = "qwen/qwen3-32b"           # synthesis only (httpx call in coordinator.py)
GROQ_MODEL = "openai/gpt-oss-20b"       # coordinator shell — consistent JSON tool calls

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_ROUTING_MODEL = "phi3.5"

DB_PATH = "./db/embeddings.db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
