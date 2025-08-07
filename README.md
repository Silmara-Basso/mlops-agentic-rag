# mlops-agentic-rag
Validação  da funcionalidade e confiabilidade de Agentes de Inteligência Artificial (Agentic RAG)


# Pipeline de Automação de Testes em Módulo de RAG Para Aplicações de IA

### Para executar a app:

1) Ative o ambiente virtual:
````
python3 -m venv logvenv
source logvenv/bin/activate
````

2) Instale o pip e as dependências:
````
pip install pip
pip install -U duckduckgo-search
pip install -r requirements.txt 
````

3) Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute: 
'''
python rag_log.py 
python testa_rag_log.py -v
'''
Obs: O teste vai apagar o banco vetor

4) Intale o LMStudio
https://lmstudio.ai/

5) Faça o download do Hermes 3 - Llama-3.2 3B e escolha a Q6_K

6) Na opção developer, ative e configure "Serve on Local Network"

![LMStudio](/images/LMStudio.png)

7) Ajuste openai_api_base no script agentic_rag_log.py com o endereço da API e adicione /v1 no final


```
python rag_log.py 
streamlit run app_log.py
python testa_agentic_rag_log.py -v
deactivate
```

## Exemplos de perguntas:

### - Resuma o conceito de supply chain.
### - Quais as fases da cadeia de suprimentos?
### - Pesquise sobre as novidades recentes em logistica e supply chain.




## Resposta da App usando somente o Modelo Hernes 3
![Modelo](/images/somenteModelo.png)

## Resposta da App usando RAG
![RAG1](/images/RAG.png)
![RAG2](/images/documento%20RAG.png)

## Resposta indo na Web
![Wab](/images/internet.png)

# Resultado dos testes para Agentic RAG
![Testes](/images/Testes%20RAG.png)
