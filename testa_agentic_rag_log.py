# Pipeline de Automação de Testes Para Agentes de IA

# Imports
import unittest
from unittest.mock import patch, MagicMock
from langchain.schema import Document
from agentic_rag_log import (
    log_passo_decisao_agente,
    log_usar_ferramenta_web,
    log_retrieve_info,
    log_gera_multiplas_respostas,
    log_avalia_similaridade,
    log_rank_respostas,
    AgentState,
)

class TestAgenticRAG(unittest.TestCase):

    def teste_log_1_passo_decisao_agente_retrieve(self):
        state = AgentState(query = "Quais as fases da cadeia de suprimentos?")
        new_state = log_passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "retrieve")

    def teste_log_2_passo_decisao_agente_usar_web(self):
        state = AgentState(query = "Pesquise sobre as novidades recentes em logistica e supply chain.")
        new_state = log_passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "usar_web")

    def teste_log_3__passo_decisao_agente_gerar(self):
        state = AgentState(query = "Resuma o conceito de supply chain.")
        new_state = log_passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "gerar")

    @patch("agentic_rag_log.log_agent_executor")
    def teste_log_4_usar_ferramenta_web(self, mock_executor):
        mock_executor.invoke.return_value = {"output": "Resposta da web."}
        state = AgentState(query = "notícias recentes sobre supply chain")
        new_state = log_usar_ferramenta_web(state)
        mock_executor.invoke.assert_called_once_with("notícias recentes sobre supply chain")
        self.assertEqual(new_state.ranked_response, "Resposta da web.")

    @patch("agentic_rag_log.retriever")
    def teste_log_5_retrieve_info(self, mock_retriever):
        mock_retriever.invoke.return_value = [Document(page_content = "Documento sobre cadeia de suprimentos")]
        state = AgentState(query = "fases da cadeia de suprimentos")
        new_state = log_retrieve_info(state)
        mock_retriever.invoke.assert_called_once_with("fases da cadeia de suprimentos")
        self.assertEqual(len(new_state.retrieved_info), 1)

    @patch("agentic_rag_log.qa_chain")
    def teste_log_6_gera_multiplas_respostas(self, mock_chain):
        mock_chain.invoke.return_value = "Resposta gerada"
        state = AgentState(query = "fases da cadeia de suprimentos")
        new_state = log_gera_multiplas_respostas(state)
        self.assertEqual(len(new_state.possible_responses), 5)

    @patch("agentic_rag_log.embedding_model")
    def teste_log_7_avalia_similaridade(self, mock_embedding_model):
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        state = AgentState(
            query = "cadeia de suprimentos",
            retrieved_info = [Document(page_content = "doc")],
            possible_responses = ["resposta"]
        )
        new_state = log_avalia_similaridade(state)
        self.assertEqual(len(new_state.similarity_scores), 1)

    def teste_log_8_rank_respostas(self):
        state = AgentState(
            query = "cadeia de suprimentos",
            possible_responses = ["resp1", "resp2"],
            similarity_scores = [0.8, 0.9]
        )
        new_state = log_rank_respostas(state)
        self.assertEqual(new_state.ranked_response, "resp2")
        self.assertEqual(new_state.confidence_score, 0.9)

if __name__ == '__main__':
    print("\nIniciando os testes. Aguarde...\n")
    unittest.main(verbosity = 2)


