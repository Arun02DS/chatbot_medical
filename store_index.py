from src.utils import *
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('Pinecone_api_key')


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(
      api_key=PINECONE_API_KEY
    )

if 'medical-chatbot' not in pc.list_indexes().names():
  pc.create_index(
    name='medical-chatbot',
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
      cloud='gcp-starter',
      region='us-central1'
        )
  )

  vectors = vector(text_chunks)

batch_size=50
for i in range(0, len(vectors), batch_size):
  batch = vectors[i:i + batch_size]
  # Print the type of the batch and the type of a sample vector
  #print(f'Type of batch: {type(batch)}')
  sample_vector = batch[0]
  #print(f"Type of vector for id {sample_vector['id']} : {type(sample_vector['values'])}")
  #print(batch) # print the batch to verify the format
  index.upsert(vectors=batch)