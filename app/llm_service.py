import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_EMBEDDING_NAME = os.getenv("MODEL_EMBEDDING_NAME")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


def get_chat_model(model_name=MODEL_NAME):
    """
    Configuração baseada na imagem do VSCode/LiteLLM.
    """
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )


def get_embeddings(model_embedding_name=MODEL_EMBEDDING_NAME):
    """
    Initializes and returns the Embeddings instance.
    Configured to handle float encoding format for Ollama compatibility.
    """
    return OllamaEmbeddings(
        model=model_embedding_name, base_url="http://localhost:11434"
    )
