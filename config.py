import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
QWEN_MODEL = "qwen/qwen3-32b"           # Node 5 synthesis only
GROQ_MODEL = "openai/gpt-oss-20b"       # AgentOS coordinator shell (JSON tool calls)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_ROUTING_MODEL = "phi3.5"
OLLAMA_KEEP_ALIVE = "5m"                # Keep model hot between Node 1 and Node 3 calls

SIMILARITY_THRESHOLD = 0.25             # Node 2: override SEARCH → RAG if score exceeds this

DB_PATH = "./db/embeddings.db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Maximum character length for individual child chunks in markdown paths
# If a paragraph exceeds this, it is split further on sentence boundaries
# PDF and TXT paths use paragraph-count grouping instead and ignore this
CHUNK_SIZE = 500
TOP_K = 5
