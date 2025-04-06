import unittest

from unittest.mock import patch, MagicMock
from src.llm.chain import LLMChain


class TestLLMChain(unittest.TestCase):

    def setUp(self):
        self.llm_chain = LLMChain()
    
    def test_create_weather_chain(self):
        # Test creating a weather chain
        weather_prompt = self.llm_chain.create_weather_chain()
        
        # Verify the prompt is created correctly
        self.assertIn("weather_data", weather_prompt.input_variables)
        self.assertIn("question", weather_prompt.input_variables)
        self.assertIn("weather data", weather_prompt.template.lower())
    
    @patch('langchain_openai.ChatOpenAI')
    def test_create_document_chain(self, mock_llm):
        # Mock the retriever
        mock_retriever = MagicMock()
        
        # Test creating a document chain
        document_chain = self.llm_chain.create_document_chain(mock_retriever)
        
        # Verify the chain is created
        self.assertIsNotNone(document_chain)