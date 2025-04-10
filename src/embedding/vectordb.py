from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_COLLECTION_NAME, OPENAI_API_KEY, EMBEDDING_MODEL, logger


class VectorDatabase:

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
    def create_collection_if_not_exists(self, collection_name=QDRANT_COLLECTION_NAME):
        """
        Create collection if it doesn't exist
        """

        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 1536,  # Dimension of OpenAI embeddings
                    "distance": "Cosine"
                }
            )
    
    def store_documents(self, documents, collection_name=QDRANT_COLLECTION_NAME):
        """
        Store documents in the vector database
        """

        logger.debug("Create collection if does not exist")
        self.create_collection_if_not_exists(collection_name)
        logger.debug("Created collection")
        
        logger.debug("Initializing vector store")
        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        logger.debug("Initialized vector store")
        
        logger.debug("Adding documents to the collection")
        vectorstore.add_documents(documents)
        logger.debug("Added documents to the collection")

        return vectorstore
    
    def get_retriever(self, collection_name=QDRANT_COLLECTION_NAME):
        """
        Get a retriever for the vector database
        """

        logger.debug("Initializing vector store")
        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        logger.debug("Initialized vector store")
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )