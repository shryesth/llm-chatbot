import os
from utils.prepare_vectordb import PrepareVectorDB

persist_directory = "data/vectordb/processed/chroma/"

def upload_data_manually() -> None:
    """
    Uploads data manually to the VectorDB.

    This function initializes a PrepareVectorDB instance with configuration parameters
    and chunk_overlap. It then checks if the VectorDB already exists in the specified
    persist_directory. If not, it calls the prepare_and_save_vectordb method to
    create and save the VectorDB. If the VectorDB already exists, a message is printed
    indicating its presence.

    Returns:
        None
    """
    os.makedirs(persist_directory, exist_ok=True)
    
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory="data/docs",
        persist_directory=persist_directory,
        chunk_size=1500,
        chunk_overlap=250,
    )
    if len(os.listdir("data/vectordb/processed/chroma/")) == 0:
        prepare_vectordb_instance.prepare_and_save_vectordb()
    else:
        print(f"VectorDB already exists in {persist_directory}")
    return None


if __name__ == "__main__":
    upload_data_manually()
