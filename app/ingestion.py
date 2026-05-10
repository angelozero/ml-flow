import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from llm_service import get_embeddings

load_dotenv()


def main():

    print("Inicializando serviço de inserção de dados no Pinecone")
    
    loader = TextLoader(os.getenv("PINECONE_DATA_FILE"))
    document = loader.load()
    embeddings = get_embeddings()
    pinecone_index_name = os.getenv("INDEX_NAME")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    texts = text_splitter.split_documents(documents=document)
    
    PineconeVectorStore.from_documents(texts, embeddings, index_name=pinecone_index_name)

    print("Dados inseridos no Pinecone com sucesso")

if __name__ == "__main__":
    main()
