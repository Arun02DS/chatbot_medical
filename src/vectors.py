# vectors.py
import pinecone
from pinecone import Pinecone, ServerlessSpec


def vector(texts,embeddings):
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
          print("Invalid")
    return vectors

def create_or_get_index(pc, index_name,dimension):
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
    else:
        print(f"Index '{index_name}' already exists.")

def process_vectors_in_batches(vectors, batch_size, index):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        sample_vector = batch[0]
        index.upsert(vectors=batch)
