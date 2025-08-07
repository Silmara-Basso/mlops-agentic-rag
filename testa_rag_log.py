# Pipeline de Automação de Testes em Módulo de RAG Para Aplicações de IA

# O unittest.mock é um módulo da biblioteca padrão Python que serve para criar objetos "falsos" (mocks) durante os testes, 
# ou seja, ele permite simular o comportamento de objetos reais para testar unidades de código isoladamente.

# Ele é muito útil quando:

# 1- Você quer testar um pedaço do seu código que depende de alguma coisa externa (como uma API, banco de dados, leitura de arquivos, etc.), 
# mas não quer realmente fazer a chamada externa durante o teste.

# 2- Você quer "forçar" certos comportamentos para ver como sua função reage (por exemplo: forçar uma exceção ou um valor de retorno específico).

# Imports necessários para os testes
import os
import shutil
import unittest
from unittest.mock import patch
from langchain.schema import Document
from rag_log import log_func_cria_vectordb, log_diretorio_dados, log_diretorio_vectordb

# Classe de testes para verificar funcionalidades relacionadas ao banco vetorial (VectorDB)
class TestRAGVectorDB(unittest.TestCase):

    # Método executado uma vez antes de todos os testes; cria diretório necessário para testes
    @classmethod
    def setUpClass(cls):
        os.makedirs(log_diretorio_vectordb, exist_ok = True)

    # Método executado uma vez após todos os testes; remove diretório criado
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(log_diretorio_vectordb, ignore_errors = True)

    # Prepara o ambiente antes de cada teste individual garantindo diretório limpo
    def setUp(self):
        if os.path.exists(log_diretorio_vectordb):
            for filename in os.listdir(log_diretorio_vectordb):
                file_path = os.path.join(log_diretorio_vectordb, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

    # Testa a criação do banco vetorial com documentos válidos, utilizando mocks
    # A anotação @patch('rag_log.DirectoryLoader') faz parte do módulo unittest.mock e ela serve para substituir temporariamente 
    # a classe ou função DirectoryLoader que está dentro do módulo rag_log por um mock, durante o teste.
    @patch('rag_log.DirectoryLoader')
    @patch('rag_log.Chroma')
    @patch('rag_log.HuggingFaceEmbeddings')
    def teste_log_1_criacao_vectordb_com_documentos(self, mock_embedding, mock_chroma, mock_loader):

        # Verifica se os documentos retornados são válidos
        mock_loader.return_value.load.return_value = [
            Document(page_content = 'conteúdo do documento 1'),
            Document(page_content = 'conteúdo do documento 2')
        ]

        # Executa a função
        log_func_cria_vectordb()

        # mock verificando se o método load() foi chamado exatamente uma vez durante o teste.
        mock_loader.return_value.load.assert_called_once()
        
        # Essa linha está verificando se o mock mock_embedding foi chamado exatamente uma vez com os argumentos específicos.
        mock_embedding.assert_called_once_with(
            model_name = 'BAAI/bge-base-en',
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': True}
        )
        
        # mock verificando se o método from_documents() foi chamado exatamente uma vez durante o teste.
        mock_chroma.from_documents.assert_called_once()

    # Testa a criação do banco vetorial sem documentos disponíveis
    @patch('rag_log.DirectoryLoader')
    def teste_log_2_criacao_vectordb_sem_documentos(self, mock_loader):

        # Cria lista vazia de arquivos para testar a função
        mock_loader.return_value.load.return_value = []

        # Executa a função
        with patch('builtins.print') as mock_print:

            # Chamada à função
            log_func_cria_vectordb()

            # Verifica se a função executou corretamente quando nenhum documento é encontrado
            mock_print.assert_any_call("Nenhum documento encontrado.")

    # Testa o tratamento de exceções ao carregar documentos
    @patch('rag_log.DirectoryLoader', side_effect = Exception("Erro ao carregar documentos"))
    def teste_log_3_tratamento_excecao_loader(self, mock_loader):

        # Coloca a execução da função no contexto de exceção
        with self.assertRaises(Exception) as context:
            log_func_cria_vectordb()

        # Testa a exceção
        self.assertIn("Erro ao carregar documentos", str(context.exception))

    # Testa se a estrutura de chunks (divisão dos documentos) está correta
    @patch('rag_log.DirectoryLoader')
    def teste_log_4_verifica_estrutura_dos_chunks(self, mock_loader):

        # Documento de teste
        documento_conteudo = 'Este é um documento teste longo que será dividido em chunks pelo splitter.'

        # Carrega o documento de teste
        mock_loader.return_value.load.return_value = [Document(page_content = documento_conteudo)]

        # Testa a divisão em chunks
        with patch('rag_log.RecursiveCharacterTextSplitter.split_documents', 
                return_value = [Document(page_content = 'chunk 1'), Document(page_content = 'chunk 2')]) as mock_split:
            with patch('rag_log.Chroma.from_documents') as mock_chroma:
                log_func_cria_vectordb()
                mock_split.assert_called_once()
                args, kwargs = mock_split.call_args
                self.assertEqual(len(args[0]), 1)
                self.assertEqual(args[0][0].page_content, documento_conteudo)
                mock_chroma.assert_called_once()

    # Testa o tratamento de exceções durante a criação do banco vetorial pelo Chroma
    @patch('rag_log.DirectoryLoader')
    @patch('rag_log.Chroma.from_documents', side_effect = Exception("Erro ao criar banco vetorial"))
    def teste_log_5_tratamento_excecao_chroma(self, mock_chroma, mock_loader):

        # Carrega um documento de exemplo
        mock_loader.return_value.load.return_value = [Document(page_content = 'conteúdo válido')]

        # Coloca em modo de exceção
        with self.assertRaises(Exception) as context:
            log_func_cria_vectordb()

        # Testa a exceção
        self.assertIn("Erro ao criar banco vetorial", str(context.exception))

# Executa a suite de testes se o script for rodado diretamente
if __name__ == '__main__':
    print("\nIniciando os Testes do Módulo de RAG...\n")
    unittest.main(verbosity = 2)


