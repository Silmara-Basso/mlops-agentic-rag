# Pipeline de Automação de Testes Para Agentes de IA

# Imports
import os
import numpy as np
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Evita problema de compatibilidade entre Streamlit e PyTorch
import torch
torch.classes.__path__ = []

# Evita problema de compatibilidade com o SO
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Inicialização do LLM
llm = ChatOpenAI(model_name = "hermes-3-llama-3.2-3b@q6_k", 
                 openai_api_base = "http://192.168.2.8:1234/v1",
                 openai_api_key = "lm-studio",
                 temperature = 0.3,
                 max_tokens = 256)

# Define o modelo de embeddings
embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en")

# Carrega o banco vetorial do RAG
vector_db = Chroma(persist_directory = "vectordb", embedding_function = embedding_model)

# Cria o retriever para recuperar os dados do RAG
retriever = vector_db.as_retriever()

# Cria o prompt template com placeholders
prompt = PromptTemplate.from_template(
    "Você é um assistente especializado em logística e supply chain. Responda em português do Brasil com base em:\n{context}\nPergunta: {input}"
)

# Cria a chain de execução
log_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

# Cria a chain de recuperação
qa_chain = create_retrieval_chain(retriever, log_chain)

# Define o mecanismo de busca
search = DuckDuckGoSearchAPIWrapper(region = "br-pt", max_results = 5)

# Cria a ferramenta do Agente de IA
web_search_tool = Tool(name = "WebSearch", func = search.run, description = "Busca web.")

# Inicializa o agente de pesquisa na web
log_agent_executor = initialize_agent(tools = [web_search_tool],
                                      llm = llm,
                                      agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose = True,
                                      handle_parsing_errors = True)

# Classe do estado do agente principal
class AgentState(BaseModel):
    query: str
    next_step: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0

# Função para o passo de decisão do agente
def log_passo_decisao_agente(state: AgentState) -> AgentState:
    query = state.query.lower()
    if any(palavra in query for palavra in ["explique", "resuma", "defina", "conceito", "geral", "o que é"]):
        state.next_step = "gerar"
    elif any(palavra in query for palavra in ["busque na web", "notícias", "atualizado", "recente", "últimas informações"]):
        state.next_step = "usar_web"
    else:
        state.next_step = "retrieve"
    return state

# Função para usar busca na web
def log_usar_ferramenta_web(state: AgentState) -> AgentState:
    resultado = log_agent_executor.invoke(state.query)
    state.ranked_response = resultado.get("output", "Nenhuma informação obtida pela busca web.")
    state.confidence_score = 0.0  
    return state

# Função para recuperar documentos do RAG
def log_retrieve_info(state: AgentState) -> AgentState:
    retrieved_docs = retriever.invoke(state.query)
    state.retrieved_info = retrieved_docs
    return state

# Função para gerar múltiplas respostas do LLM
def log_gera_multiplas_respostas(state: AgentState) -> AgentState:
    responses = [qa_chain.invoke({"input": state.query}) for _ in range(5)]
    state.possible_responses = responses
    return state

# Função que avalia a similaridade entre respostas do LLM e documentos recuperados do RAG
def log_avalia_similaridade(state: AgentState) -> AgentState:
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses
    retrieved_embeddings = embedding_model.embed_documents(retrieved_texts) if retrieved_texts else []
    response_texts = [response["answer"] if isinstance(response, dict) and "answer" in response else str(response) for response in responses]
    response_embeddings = embedding_model.embed_documents(response_texts) if response_texts else []

    if not retrieved_embeddings or not response_embeddings:
        state.similarity_scores = [0.0] * len(response_texts)
        return state

    similarities = [
        np.mean([cosine_similarity([response_embedding], [doc_embedding])[0][0] for doc_embedding in retrieved_embeddings])
        for response_embedding in response_embeddings
    ]

    state.similarity_scores = similarities
    return state

# Função para criar um rank das respostas (somente a melhor resposta será mostrada ao usuário final)
def log_rank_respostas(state: AgentState) -> AgentState:
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores))
    if response_with_scores:
        ranked_responses = sorted(response_with_scores, key=lambda x: x[1], reverse=True)
        state.ranked_response = ranked_responses[0][0]
        state.confidence_score = ranked_responses[0][1]
    else:
        state.ranked_response = "Desculpe, não encontrei informações relevantes."
        state.confidence_score = 0.0
    return state

# Criação do Agente de IA com LangGraph

# Cria o workflow de execução do Agente de IA
workflow = StateGraph(AgentState)
workflow.add_node("decision", log_passo_decisao_agente)
workflow.add_node("retrieve", log_retrieve_info)
workflow.add_node("generate_multiple", log_gera_multiplas_respostas)
workflow.add_node("evaluate_similarity", log_avalia_similaridade)
workflow.add_node("rank_responses", log_rank_respostas)
workflow.add_node("usar_web", log_usar_ferramenta_web)

# Ponto de entrada (início) da execução
workflow.set_entry_point("decision")

# Cria arestas condicionais
workflow.add_conditional_edges(
    "decision",
    lambda state: {
        "retrieve": "retrieve",
        "gerar": "generate_multiple",
        "usar_web": "usar_web"
    }[state.next_step]
)

# Adiciona as demais arestas
workflow.add_edge("retrieve", "generate_multiple")
workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")

# Compila o workflow para execução
agent_workflow = workflow.compile()





