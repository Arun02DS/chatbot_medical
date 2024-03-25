from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

PINECONE_API_KEY = os.environ.get('Pinecone_api_key')

FILE_PATH = Path(r"C:\Users\Arun Singh Negi\Documents\medical_bot\data")

model_name="sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 500
chunk_overlap = 20

index_name = "medical-chatbot"
batch_size=50
dimension=384