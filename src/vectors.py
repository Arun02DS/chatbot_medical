from src.logger import logging
from src.exception import Chatbot
import os,sys
import pinecone
from pinecone import Pinecone, ServerlessSpec


def vector(texts,embeddings):
    """
    Definition: This function create vectors embedding as a list from texts chunks.

    Return: list of vectors
    
    """
    try:
        vectors = []
        for i in range(len(texts)):
            vector = embeddings.embed_query(texts[i].page_content)
            if isinstance(vector,list):
                vector_obj={
                'id':str(i),
                'values':vector,
                'metadata':{'text':texts[i].page_content}
                }
                vectors.append(vector_obj)
            else:
                logging.info(f"Invalid texts")
                print("Invalid texts")
        return vectors
    except Exception as e:
        raise Chatbot(e,sys)

def create_or_get_index(pc, index_name,dimension):
    """
    Definition:This fuction creates index if not exist already.

    Return:None
    
    """
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='gcp-starter',
                    region='us-central1'
                )
            )
            print(f"Index '{index_name}' created successfully.")
            logging.info(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
            logging.info(f"Index '{index_name}' already exists.")
    except Exception as e:
        raise Chatbot(e,sys)

def process_vectors_in_batches(vectors, batch_size, index):
    """
    Definition:This function upsert vector embedding to the pinecone database.

    Return:None
    
    """
    try:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            sample_vector = batch[0]
            index.upsert(vectors=batch)
    except Exception as e:
        raise Chatbot(e,sys)
