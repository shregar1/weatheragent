import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import create_agent_executor
from src.document.loader import DocumentLoader
from src.document.processor import DocumentProcessor
from src.embedding.vectordb import VectorDatabase
import os

def initialize_app():
    """
    Initialize the application, load documents, and create vector database
    """
    # Initialize document loader
    doc_loader = DocumentLoader()
    
    # Check if PDF directory exists, if not create it
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Get available PDFs
    available_pdfs = doc_loader.list_available_pdfs()
    
    # Display available PDFs
    st.sidebar.title("Available Documents")
    if available_pdfs:
        for pdf in available_pdfs:
            st.sidebar.write(f"- {pdf}")
    else:
        st.sidebar.write("No PDFs available. Please upload some.")
    
    # PDF uploader
    uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        # Save the uploaded file
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the uploaded file
        documents = doc_loader.load_pdf(uploaded_file.name)
        
        # Process documents
        processor = DocumentProcessor()
        chunks = processor.split_documents(documents)
        
        # Store in vector database
        vector_db = VectorDatabase()
        vector_db.store_documents(chunks)
        
        st.sidebar.success(f"File {uploaded_file.name} uploaded and processed successfully!")
        
        # Rerun the app to refresh the list
        st.experimental_rerun()

def main():
    """
    Main function to run the Streamlit app
    """
    st.title("Weather & Document Q&A Agent")
    
    # Initialize the app
    initialize_app()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hello! I'm your Weather & Document assistant. You can ask me about weather in a specific city or ask questions about documents you've uploaded.")
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)
    
    # Get user input
    user_input = st.chat_input("Ask a question about weather or documents")
    if user_input:
        # Add user message to state
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").write(user_input)
        
        # Create agent executor
        agent_executor = create_agent_executor()
        
        # Execute agent
        with st.spinner("Thinking..."):
            result = agent_executor(st.session_state.messages)
        
        # Display assistant response
        ai_message = result["messages"][-1]
        st.session_state.messages.append(ai_message)
        st.chat_message("assistant").write(ai_message.content)

if __name__ == "__main__":
    main()