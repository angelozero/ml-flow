import os
from dotenv import load_dotenv
import mlflow
from typing import Annotated, Literal, TypedDict
from llm_service import get_chat_model, get_embeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configura o endereço do servidor MLflow que subimos no Docker
mlflow.set_tracking_uri(os.getenv("ML_FLOW_TRACKING_URI"))
mlflow.set_experiment("AngeloZero_Experiment_v1")

# Ativa o rastreamento automático para LangChain/LangGraph (MLflow 3.0)
mlflow.langchain.autolog()


# 1. Definimos o Estado: o que o grafo "lembra" durante a execução
class AgentState(TypedDict):
    # add_messages faz com que as novas mensagens sejam anexadas ao histórico
    messages: Annotated[list, add_messages]


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# 2. Inicializamos o modelo (Ollama local)


# 3. Funções dos Especialistas (Nós)
def billing_agent(state: AgentState):
    print("--- CONSULTANDO BASE DE DADOS (RAG) ---")

    user_msg = state["messages"][-1].content

    embeddings = get_embeddings()

    pinecone_vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )

    retriever = pinecone_vectorstore.as_retriever(searh_kwargs={"k": 3})
    docs = retriever.invoke(user_msg)

    context = format_docs(docs)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                - Você é o Agente de Faturamento.
                - Use as informações abaixo para responder ao cliente.
                - Se a informação não estiver no contexto, diga que não tem acesso a esse dado.
             """,
            ),
            ("human", "{question}\n\nContexto: {context}\n\nResposta: "),
        ]
    )

    llm = get_chat_model()
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": user_msg, "context": context})

    return {"messages": [response]}


def tech_support_agent(state: AgentState):
    print("--- CHAMANDO SUPORTE TÉCNICO ---")
    return {
        "messages": ["Sou o suporte técnico. Qual problema seu dispositivo apresenta?"]
    }


# 4. O Supervisor (Lógica de Roteamento)
def supervisor(state: AgentState) -> Literal["billing", "tech"]:
    last_message = state["messages"][-1].content.lower()

    if (
        "conta" in last_message
        or "pagamento" in last_message
        or "fatura" in last_message
    ):
        return "billing"
    return "tech"


# 5. Construção do Grafo (API moderna do LangGraph com START explícito)
workflow = StateGraph(AgentState)

# Adicionamos os nós especialistas
workflow.add_node("billing", billing_agent)
workflow.add_node("tech", tech_support_agent)

# Ponto de entrada condicional usando add_conditional_edges a partir de START
workflow.add_conditional_edges(
    START,
    supervisor,
    {
        "billing": "billing",
        "tech": "tech",
    },
)

# Após o especialista responder, o fluxo termina
workflow.add_edge("billing", END)
workflow.add_edge("tech", END)

app = workflow.compile()

# =========================
# ========= TEST =========
# =========================

if __name__ == "__main__":
    # Teste 1: Faturamento
    print("=== Teste 1: Faturamento ===")
    inputs = {"messages": [("user", "Minha fatura veio errada")]}
    for output in app.stream(inputs):
        print(output)

    print("\n" + "=" * 30 + "\n")

    # Teste 2: Suporte Técnico
    print("=== Teste 2: Suporte Técnico ===")
    inputs = {"messages": [("user", "Meu Wi-Fi não conecta")]}
    for output in app.stream(inputs):
        print(output)
