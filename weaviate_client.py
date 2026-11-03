import weaviate
import weaviate.classes.config as wvcc

def get_weaviate_client():
    client = weaviate.connect_to_local(
        host="127.0.0.1",
        port=8080,
        grpc_port=50051
    )     # connect to weaviate docker container on local host
    return client

def setup_schema(client):
    collection_name="RAG_Chunks"

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    # define schema
    client.collections.create(
        name=collection_name,
        properties=[
            # BM25 sparse index
            wvcc.Property(name="text",data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="document_context",data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="source",data_type=wvcc.DataType.TEXT),
        ],
        # NO vectorization needed
        vectorizer_config=wvcc.Configure.Vectorizer.none(),
    )
    print(f"Collection {collection_name} created successfully for Hybrid Search.")
    return client.collections.get(collection_name)