from langgraph.graph import StateGraph, END
from typing import List
from langchain_core.messages import BaseMessage

from src.agent.nodes import AgentState, classify_query, get_weather, query_document, generate_response


def create_agent_graph():
    """
    Create a LangGraph for the agent
    """
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("get_weather", get_weather)
    workflow.add_node("query_document", query_document)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("classify_query", "get_weather")
    workflow.add_edge("get_weather", "query_document")
    workflow.add_edge("query_document", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Compile the graph
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
        # Initialize state
        state = AgentState(
            messages=messages,
            query_type="unknown",
            response="",
            city="",
            documents=[]
        )
        
        # Execute the graph
        result = agent_graph.invoke(state)
        
        return result
    
    return agent_executor