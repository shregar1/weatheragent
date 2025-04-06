import os
from langchain_community.document_loaders import PyPDFLoader

from config import PDF_DIRECTORY


class DocumentLoader:

    def __init__(self):
        self.pdf_directory = PDF_DIRECTORY
    
    def list_available_pdfs(self):
        """
        List all available PDF files in the directory
        """
        return [file for file in os.listdir(self.pdf_directory) if file.endswith('.pdf')]
    
    def load_pdf(self, filename):
        """
        Load a PDF file and return a document object
        """
        filepath = os.path.join(self.pdf_directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        loader = PyPDFLoader(filepath)
        return loader.load()