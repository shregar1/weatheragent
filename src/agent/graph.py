from langgraph.graph import StateGraph, END
from typing import List
from langchain_core.messages import BaseMessage

from src.agent.nodes import AgentState, classify_query, get_weather, query_document, generate_response

from config import logger


def create_agent_graph():
    """
    Create a LangGraph for the agent
    """

    logger.debug("Creating workflow")
    workflow = StateGraph(AgentState)
    
    logger.debug("Adding nodes")
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("get_weather", get_weather)
    workflow.add_node("query_document", query_document)
    workflow.add_node("generate_response", generate_response)
    
    logger.debug("Add edges")
    workflow.add_conditional_edges(
        "classify_query",
        lambda state: state["query_type"],
        {
            "weather": "get_weather",
            "document": "query_document",
            "unknown": "generate_response"
        }
    )
    workflow.add_edge("get_weather", "generate_response")
    workflow.add_edge("query_document", "generate_response")
    workflow.add_edge("generate_response", END)
    
    logger.debug("Setting entry point")
    workflow.set_entry_point("classify_query")
    
    logger.debug("Compiling the graph")
    agent_graph = workflow.compile()
    
    return agent_graph

def create_agent_executor():
    """
    Create an agent executor for the graph
    """
    agent_graph = create_agent_graph()
    
    def agent_executor(messages: List[BaseMessage]) -> AgentState:
        """
        Execute the agent with the given messages
        """
        logger.debug("Initializing state")
        state = AgentState(
            messages=messages,
            query_type="unknown",
            response="",
            city="",
            documents=[]
        )
        logger.debug("Initialized state")
        
        logger.debug("Executing the graph")
        result = agent_graph.invoke(state)
        logger.debug("Executed the graph")
        
        return result
    
    return agent_executor