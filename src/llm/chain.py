from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import LLM_MODEL, OPENAI_API_KEY


class LLMChain:

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.2
        )
    
    def create_weather_chain(self):
        """
        Create a chain for processing weather data
        """
        weather_template = """
        You are a helpful assistant that provides weather information.
        Below is the weather data for a city:
        
        {weather_data}
        
        Based on the above weather data, please answer the following question:
        {question}
        """
        
        weather_prompt = PromptTemplate(
            input_variables=["weather_data", "question"],
            template=weather_template
        )
        
        return weather_prompt
    
    def create_document_chain(self, retriever):
        """
        Create a chain for answering questions from documents
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain