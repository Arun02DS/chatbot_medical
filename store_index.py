from src.logger import logging
from src.utils import text_split,load_pdf,download_hugging_face_embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.config import *
from src.vectors import create_or_get_index,process_vectors_in_batches,vector
import os


logging.info(f"{'..>==>..'*4} PINECONE VECTOR DATABASE {'..<==<..'*4}")


logging.info(f"Data is Extracted From the pdf files")
extracted_data = load_pdf(FILE_PATH)
logging.info(f"Extraction completed from {FILE_PATH}")
logging.info(f"corpus is splitted into chunks")
text_chunks = text_split(extracted_data)
logging.info(f"downloading hugging face embedding {model_name}")
embeddings = download_hugging_face_embeddings()

logging.info(f"making pinecone connection")
pc = Pinecone(
      api_key=PINECONE_API_KEY
    )

logging.info(f"creating or geeting index from pipecone")
create_or_get_index(pc=pc,index_name=index_name,dimension=dimension)

logging.info(f"index name : {index_name}")
index=pc.Index(index_name)


logging.info(f"creating vecotrs to upsert into pinecone")
vectors = vector(text_chunks=text_chunks,embeddings=embeddings)
logging.info(f"vectors:{len(vectors)}")


logging.info(f"upserting vectors into pinecone")
process_vectors_in_batches(vector=vectors, batch_size=batch_size, index=index)

logging.info(f"{'..>==>..'*4} VECTORS INSERTED INTO PINECONE VECTOR DATABASE {'..<==<..'*4}")