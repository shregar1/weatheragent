import os
import sys

from dotenv import load_dotenv
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, colorize=True, format="<green>{time:MMMM-D-YYYY}</green> | <black>{time:HH:mm:ss}</black> | <level>{level}</level> | <cyan>{message}</cyan> | <magenta>{name}:{function}:{line}</magenta> | <yellow>{extra}</yellow>")

logger.info("Loading Environment Variables")
load_dotenv()

logger.info("Loading API Configuration")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
logger.info("Loaded API Configuration")

logger.info("Loading LLM Configuration")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
logger.info("Loaded LLM Configuration")

logger.info("Loading Vector Database Configuration")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "document_embeddings")
logger.info("Loaded Vector Database Configuration")

logger.info("Loading Application Configuration")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP")
logger.info("Loaded Application Configuration")

logger.info("Loaded Environment Variables")