
import os
import json
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import create_agent_executor
from src.document.loader import DocumentLoader
from src.document.processor import DocumentProcessor
from src.embedding.vectordb import VectorDatabase

from config import logger, PROCESSED_FILE_PATH

def load_processed_files():
    """
    Load the list of processed files and ensure they still exist in the data/ folder.
    Clean up any entries for missing files.
    """
    processed = []
    if os.path.exists(PROCESSED_FILE_PATH):
        with open(PROCESSED_FILE_PATH, "r") as f:
            try:
                processed = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Processed file tracking JSON was empty or malformed. Resetting.")
                processed = []

    logger.debug("Filter to include only files that still exist in the `data/` directory")
    processed = [file for file in processed if os.path.exists(os.path.join("data", file))]

    logger.debug("Rewrite the JSON to reflect accurate file existence")
    with open(PROCESSED_FILE_PATH, "w") as f:
        json.dump(processed, f)

    return processed

def save_processed_file(filename):
    processed = load_processed_files()
    if filename not in processed:
        processed.append(filename)
        with open(PROCESSED_FILE_PATH, "w") as f:
            json.dump(processed, f)

def initialize_app():
    logger.debug("Initialize document loader")
    doc_loader = DocumentLoader()

    if not os.path.exists("data"):
        os.makedirs("data")
    
    logger.debug("Loading processed files")
    processed_files = load_processed_files()
    logger.debug("Loaded processed files")

    logger.debug("Listing available documents")
    st.sidebar.title("Available Documents")
    if processed_files:
        for pdf in processed_files:
            st.sidebar.write(f"- {pdf}")
    else:
        st.sidebar.write("No PDFs available. Please upload some.")
    logger.debug("Listed available documents")

    logger.debug("Uploading new pdf file")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:

        logger.debug("Building file path")
        file_path = os.path.join("data", uploaded_file.name)
        logger.debug("Built file path")

        if uploaded_file.name in processed_files:

            logger.debug(f"{uploaded_file.name} already processed")
            st.sidebar.info(f"File {uploaded_file.name} already processed.")

        else:
            
            logger.debug("Sving uploaded file")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.debug(f"Saved file: {uploaded_file.name}")

            logger.debug("Loading uploading pdf file")
            documents = doc_loader.load_pdf(uploaded_file.name)
            logger.debug("Loaded uploading pdf file")

            logger.debug("Processing uploaded pdf file")
            processor = DocumentProcessor()
            chunks = processor.split_documents(documents)
            logger.debug("Processed uploaded pdf file")

            logger.debug(f"Storing {len(chunks)} chunks in vector DB")
            vector_db = VectorDatabase()
            vector_db.store_documents(chunks)
            logger.debug("Stored chunks in vecot DB")

            logger.debug("Saving processed file metadata")
            save_processed_file(uploaded_file.name)
            logger.debug("Saved processed file metadata")

            st.sidebar.success(f"File {uploaded_file.name} uploaded and processed successfully!")
            logger.debug("Rerunning the app to refresh the list")
            st.rerun()

def main():
    """
    Main function to run the Streamlit app
    """
    st.title("Weather & Document Q&A Agent")
    
    logger.debug("Initializing the app")
    initialize_app()
    
    logger.debug("Initializing session state")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hello! I'm your Weather & Document assistant. You can ask me about weather in a specific city or ask questions about documents you've uploaded.")
        ]
    logger.debug("Initialised session state")
    
    logger.debug("Displaying chat messages")
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)
    logger.debug("Displayed chat messages")
    
    logger.debug("Getting user input")
    user_input = st.chat_input("Ask a question about weather or documents")
    if user_input:

        logger.debug("Got user input")

        logger.debug("Adding user message to state")
        message = HumanMessage(content=user_input)
        st.session_state.messages.append(message)
        st.chat_message("user").write(message.content)
        logger.debug("Added user message to state")
        
        logger.debug("Creating agent executor")
        agent_executor = create_agent_executor()
        logger.debug("Created agent executor")
        
        logger.debug("Executing agent")
        with st.spinner("Thinking..."):
            result = agent_executor(st.session_state.messages)
        logger.debug("Executed agent")
        
        logger.debug("Displaying assistant response")
        ai_message: AIMessage = result["messages"][-1]
        st.session_state.messages.append(ai_message)
        st.chat_message("assistant").write(ai_message.content)
        logger.debug("Displayed assistant response")

if __name__ == "__main__":
    main()