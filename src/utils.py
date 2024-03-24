from src.logger import logging
from src.exception import Chatbot
import os,sys
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from src.config import model_name,chunk_overlap,chunk_size

def load_pdf(data):
    """
    Definition:This function load document.

    Return:object of pdf document.
    
    """
    try:
        loader = DirectoryLoader(data,
                        glob="*.pdf",
                        loader_cls=PyPDFLoader)

        documents = loader.load()
        logging.info(f"document: {documents[:10]}")
        return documents
    except Exception as e:
        raise Chatbot(e,sys)


def text_split(extracted_data):
    """
    Definition:This function split text into chucks with some overlap.

    Return:List of Object of text in chunks.
    
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        text_chunks = text_splitter.split_documents(extracted_data)
        logging.info(f"length of chucks: {len(text_chunks)}")
        logging.info(f"chuck[0]: {text_chunks[0]}")
        return text_chunks
    
    except Exception as e:
        raise Chatbot(e,sys)


def download_hugging_face_embeddings():
    """
    Definition:download embedding from hugging face.

    Return:embedding object.
    
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logging.info(f"embedding downloaded")
        return embeddings
    except Exception as e:
        raise Chatbot(e,sys)


