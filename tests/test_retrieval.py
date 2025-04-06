import unittest

from unittest.mock import patch, MagicMock
from src.embedding.vectordb import VectorDatabase


class TestVectorDatabase(unittest.TestCase):

    @patch('qdrant_client.QdrantClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def setUp(self, mock_embeddings, mock_client):
        self.vector_db = VectorDatabase()
        self.vector_db.client = mock_client
        self.vector_db.embeddings = mock_embeddings
    
    def test_create_collection_if_not_exists(self):
        # Mock the client response
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        self.vector_db.client.get_collections.return_value.collections = [mock_collection]
        
        # Test creating a collection that doesn't exist
        self.vector_db.create_collection_if_not_exists("new_collection")
        
        # Verify that create_collection was called
        self.vector_db.client.create_collection.assert_called_once()
        
        # Reset mock
        self.vector_db.client.create_collection.reset_mock()
        
        # Test creating a collection that exists
        self.vector_db.create_collection_if_not_exists("test_collection")
        
        # Verify that create_collection was not called
        self.vector_db.client.create_collection.assert_not_called()
    
    @patch('langchain_qdrant.Qdrant')
    def test_store_documents(self, mock_qdrant):
        # Mock documents
        mock_documents = [MagicMock(), MagicMock()]
        
        # Mock Qdrant instance
        mock_qdrant_instance = MagicMock()
        mock_qdrant.return_value = mock_qdrant_instance
        
        # Test storing documents
        self.vector_db.store_documents(mock_documents)
        
        # Verify that add_documents was called
        mock_qdrant_instance.add_documents.assert_called_once_with(mock_documents)
    
    @patch('langchain_qdrant.Qdrant')
    def test_get_retriever(self, mock_qdrant):
        # Mock Qdrant instance
        mock_qdrant_instance = MagicMock()
        mock_qdrant.return_value = mock_qdrant_instance
        
        # Test getting a retriever
        self.vector_db.get_retriever()
        
        # Verify that as_retriever was called
        mock_qdrant_instance.as_retriever.assert_called_once()
        # Check if search_kwargs is set correctly
        mock_qdrant_instance.as_retriever.assert_called_with(
            search_type="similarity",
            search_kwargs={"k": 5}
        )