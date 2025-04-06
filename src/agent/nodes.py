import json

from typing import List, TypedDict, Literal
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, LLM_MODEL, logger

from src.api.weather import WeatherAPI
from src.embedding.vectordb import VectorDatabase
from src.llm.chain import LLMChain


class AgentState(TypedDict):

    messages: List[BaseMessage]
    query_type: Literal["weather", "document", "unknown"]
    response: str
    city: str
    documents: List[str]


def classify_query(state: AgentState) -> AgentState:
    """
    Classify whether the query is about weather or documents
    """

    logger.debug("Extracting the latest message")
    last_message = state["messages"][-1].content
    
    logger.debug("Defining classifier prompt")
    prompt = f"""
    Please determine if the following query is asking about weather or information from a document:
    
    Query: {last_message}
    
    If the query is asking about weather in a specific city, respond with "weather".
    If the query is asking for information from a document, respond with "document".
    If you're not sure, respond with "unknown".
    
    Also, if it's a weather query, extract the city name from the query.
    
    Format your response as a JSON object with two fields:
    - type: Either "weather", "document", or "unknown"
    - city: The city name (only if type is "weather")
    """
    
    logger.debug("Using OpenAI to classify")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )
    
    response = llm.invoke(prompt)
    
    logger.debug("Parse the response to extract type and city")

    try:
        classification = json.loads(response.content)
        state["query_type"] = classification.get("type", "unknown")
        state["city"] = classification.get("city", "") if state["query_type"] == "weather" else ""

    except Exception as err:

        logger.error(f"{err.__class__} Exception occured. {err}")
        state["query_type"] = "unknown"
        state["city"] = ""
    
    return state

def get_weather(state: AgentState) -> AgentState:
    """
    Fetch weather data for the specified city
    """

    if state["query_type"] != "weather":
        return state
    
    logger.debug("Getting the city from the state")
    city = state["city"]
    
    logger.debug("Getting weather data")
    weather_api = WeatherAPI()
    weather_data = weather_api.get_weather(city)
    formatted_weather = weather_api.format_weather_data(weather_data)
    
    logger.debug("Processing with LLM")
    llm_chain = LLMChain()
    weather_prompt = llm_chain.create_weather_chain()
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", weather_prompt.template),
        ("human", "{question}")
    ])
    
    chain = chat_prompt | llm_chain.llm
    
    logger.debug("Getting the question from the latest message")
    question = state["messages"][-1].content
    
    logger.debug("Getting response")
    result = chain.invoke({
        "weather_data": formatted_weather,
        "question": question
    })
    
    state["response"] = result.content
    
    return state

def query_document(state: AgentState) -> AgentState:
    """
    Query documents based on the input
    """
    
    if state["query_type"] != "document":
        return state
    
    logger.debug("Get the question from the latest message")
    question = state["messages"][-1].content
    
    logger.debug("Getting vector database retriever")
    vector_db = VectorDatabase()
    retriever = vector_db.get_retriever()
    
    logger.debug("Creating QA chain")
    llm_chain = LLMChain()
    qa_chain = llm_chain.create_document_chain(retriever)
    
    logger.debug("Getting response")
    result = qa_chain({"query": question})
    
    state["response"] = result["result"]
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """
    Generate a response based on the query type
    """

    if state["query_type"] == "unknown":
        state["response"] = """I'm not sure if you're asking about weather or about information from a document. 
        
Could you please clarify your question? 
- For weather information, please ask about the weather in a specific city.
- For document information, please ask about the content of the available documents."""
    
    logger.debug("Adding the response as an AI message")
    state["messages"].append(AIMessage(content=state["response"]))
    
    return state