from src.utils import text_split,load_pdf,download_hugging_face_embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.config import *
from src.vectors import create_or_get_index,process_vectors_in_batches,vector
import os


extracted_data = load_pdf(FILE_PATH)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(
      api_key=PINECONE_API_KEY
    )

create_or_get_index(pc=pc,index_name=index_name,dimension=dimension)

index=pc.Index(index_name)
vectors = vector(text_chunks=text_chunks,embeddings=embeddings)

process_vectors_in_batches(vector=vectors, batch_size=batch_size, index=index)