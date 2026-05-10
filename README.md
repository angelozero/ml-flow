# 🧪 MLflow + LLM Judges — Estudo de Avaliação de Agentes

> Projeto de estudo para entender como usar **MLflow** para rastrear experimentos e como criar **juízes LLM** para avaliar a qualidade das respostas de agentes de IA.

---

## 📚 Índice

1. [O que é MLflow?](#1-o-que-é-mlflow)
2. [O que são LLM Judges?](#2-o-que-são-llm-judges)
3. [Como as peças se encaixam](#3-como-as-peças-se-encaixam)
4. [Estrutura do Projeto](#4-estrutura-do-projeto)
5. [Pré-requisitos](#5-pré-requisitos)
6. [Configuração do Ambiente](#6-configuração-do-ambiente)
7. [Subindo o MLflow com Docker](#7-subindo-o-mlflow-com-docker)
8. [Executando os Scripts](#8-executando-os-scripts)
9. [Visualizando Experimentos no MLflow UI](#9-visualizando-experimentos-no-mlflow-ui)
10. [Comparando Múltiplos Experimentos](#10-comparando-múltiplos-experimentos)

---

## 1. O que é MLflow?

**MLflow** é uma plataforma open-source para gerenciar o ciclo de vida de modelos de Machine Learning e IA. Ele resolve um problema clássico: *"Rodei vários experimentos, mas não lembro qual configuração deu o melhor resultado."*

Com MLflow você consegue:

| Funcionalidade | O que faz |
|---|---|
| **Tracking** | Registra parâmetros, métricas e artefatos de cada execução |
| **Experiments** | Agrupa execuções relacionadas em um mesmo experimento |
| **Runs** | Cada execução individual dentro de um experimento |
| **UI** | Dashboard web para visualizar e comparar tudo |
| **GenAI Evaluate** | API específica para avaliar respostas de LLMs com juízes |

### Conceito central: Experiment → Run → Metrics

```
Experiment: "AngeloZero_Experiment"
│
├── Run #1 (mlflow_judge_test.py)
│   ├── scorer: correctness_ollama
│   └── metrics: correctness/yes_rate = 0.5
│
├── Run #2 (mlflow_judge.py)
│   ├── scorer: judge_qwen
│   ├── scorer: judge_llama
│   └── metrics: Corretude_Qwen/yes_rate = 0.8
│              Corretude_Llama/yes_rate = 0.6
│
└── Run #3 (nova configuração)
    └── ...
```

---

## 2. O que são LLM Judges?

Um **LLM Judge** (Juiz LLM) é um modelo de linguagem usado para **avaliar a resposta de outro modelo**. Em vez de comparar strings exatas (o que seria frágil), você usa uma LLM para julgar se a resposta está correta, completa ou relevante.

### Por que usar um juiz LLM?

Imagine que seu agente respondeu:

> *"O valor da sua fatura é trezentos e cinquenta reais, vencendo no dia cinco de maio."*

A resposta esperada era:

> *"R$ 350,00 com vencimento em 05/05/2026."*

Uma comparação de string direta diria que está **errado**. Um juiz LLM entende que semanticamente está **correto**.

### Como funciona o fluxo de avaliação

```
┌─────────────────────────────────────────────────────────┐
│                    mlflow.genai.evaluate                │
│                                                         │
│  Dataset de Avaliação                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │ inputs:       {"query": "Qual o valor da fatura?"}│  │
│  │ outputs:      "O valor é R$ 350,00..."            │  │
│  │ expectations: {"expected_facts": ["R$ 350,00"...]}│  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│              ┌───────────────────────┐                  │
│              │    @scorer (Juiz)     │                  │
│              │                       │                  │
│              │  1. Monta o prompt    │                  │
│              │  2. Chama a LLM       │                  │
│              │  3. Retorna Feedback  │                  │
│              └───────────────────────┘                  │
│                          │                              │
│                          ▼                              │
│              Feedback { value: "yes", rationale: "..." }│
│                          │                              │
│                          ▼                              │
│              MLflow registra a métrica automaticamente  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Como as peças se encaixam

Este projeto simula um **agente de atendimento** da empresa fictícia *ConnectMax Telecom* e avalia suas respostas usando juízes LLM.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLUXO COMPLETO                           │
│                                                                 │
│  data/data.txt          app/ingestion.py                        │
│  (Manual ConnectMax) ──► (Carrega no Pinecone)                  │
│                                    │                            │
│                                    ▼                            │
│                          app/main.py                            │
│                          (Agente LangGraph)                     │
│                          ┌─────────────────┐                    │
│                          │   Supervisor    │                    │
│                          │  ┌───────────┐  │                    │
│                          │  │ Billing   │  │ ← RAG (Pinecone)   │
│                          │  │  Agent    │  │                    │
│                          │  └───────────┘  │                    │
│                          │  ┌───────────┐  │                    │
│                          │  │   Tech    │  │                    │
│                          │  │  Support  │  │                    │
│                          │  └───────────┘  │                    │
│                          └─────────────────┘                    │
│                                    │                            │
│                          respostas do agente                    │
│                                    │                            │
│                                    ▼                            │
│  app/eval_data.py        app/mlflow_judge.py                    │
│  (Dataset de testes) ──► (Avalia com 2 juízes LLM)              │
│                          ┌─────────────────┐                    │
│                          │  judge_qwen     │ ← Qwen3:8b         │
│                          │  judge_llama    │ ← Llama3.2         │
│                          └─────────────────┘                    │
│                                    │                            │
│                                    ▼                            │
│                          MLflow Tracking Server                 │
│                          (http://localhost:5050)                │
│                          Dashboard com métricas comparativas    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Estrutura do Projeto

```
ml-flow/
│
├── docker/
│   └── docker-compose.yml      # Sobe o servidor MLflow
│
├── app/
│   ├── llm_service.py          # Configura o modelo LLM e embeddings (Ollama)
│   ├── ingestion.py            # Carrega data/data.txt no Pinecone (RAG)
│   ├── main.py                 # Agente LangGraph com 2 especialistas
│   ├── eval_data.py            # Dataset de avaliação (perguntas + respostas esperadas)
│   ├── mlflow_judge_test.py    # Avaliação simples com 1 juiz (Qwen)
│   └── mlflow_judge.py         # Avaliação avançada com 2 juízes (Qwen + Llama)
│
├── data/
│   └── data.txt                # Manual da ConnectMax Telecom (base de conhecimento RAG)
│
├── .env_example                # Variáveis de ambiente necessárias
└── pyproject.toml              # Dependências do projeto
```

### Papel de cada arquivo

| Arquivo | Papel |
|---|---|
| [`llm_service.py`](app/llm_service.py) | Factory para criar o modelo de chat e embeddings via Ollama |
| [`ingestion.py`](app/ingestion.py) | Lê `data.txt`, divide em chunks e indexa no Pinecone |
| [`main.py`](app/main.py) | Grafo LangGraph: supervisor roteia para agente de faturamento ou suporte técnico |
| [`eval_data.py`](app/eval_data.py) | 5 casos de teste com `inputs`, `outputs` e `expected_facts` |
| [`mlflow_judge_test.py`](app/mlflow_judge_test.py) | Avaliação básica: 1 juiz, dataset inline, bom para começar |
| [`mlflow_judge.py`](app/mlflow_judge.py) | Avaliação avançada: 2 juízes diferentes para comparar modelos |

---

## 5. Pré-requisitos

- **Python 3.14+** (gerenciado pelo `uv`)
- **Docker** (para o servidor MLflow)
- **Ollama** instalado localmente com os modelos:
  - `qwen3:8b` — modelo principal e juiz
  - `llama3.2` — segundo juiz para comparação
  - `rjmalagon/gte-qwen2-1.5b-instruct-embed-f16` — embeddings
- **Pinecone** — conta gratuita para o vector store (RAG)

### Instalando os modelos no Ollama

```bash
ollama pull qwen3:8b
ollama pull llama3.2
ollama pull rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest
```

---

## 6. Configuração do Ambiente

```bash
# 1. Clone o repositório e entre na pasta
cd ml-flow

# 2. Copie o arquivo de variáveis de ambiente
cp .env_example .env

# 3. Edite o .env com suas chaves
#    Obrigatório: PINECONE_API_KEY
#    Opcional: ajuste MODEL_NAME se quiser outro modelo
```

Conteúdo do [`.env_example`](.env_example):

```env
OPENAI_API_KEY="ollama"       # Placeholder — Ollama não precisa de chave real

# LLM (via Ollama local)
MODEL_NAME=qwen3:8b
MODEL_EMBEDDING_NAME=rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest
API_KEY=ollama
BASE_URL=http://localhost:11434/v1/
OLLAMA_BASE_URL=http://localhost:11434/v1/chat/completions
JUDGE_MODEL=openai:/qwen3:8b

# Pinecone (vector store para RAG)
INDEX_NAME=rag-mlflow-llm-judges
PINECONE_DATA_FILE=data/data.txt
PINECONE_API_KEY=YOUR_PINECONE_API_KEY

# MLflow
ML_FLOW_TRACKING_URI=http://localhost:5050
```

```bash
# 4. Instale as dependências com uv
uv sync
```

---

## 7. Subindo o MLflow com Docker

O servidor MLflow roda em um container Docker e persiste os dados em um volume local.

```bash
# Sobe o servidor MLflow em background
cd docker
docker compose up -d

# Verifique se está rodando
docker ps
```

Acesse o dashboard em: **http://localhost:5050**

> **Como funciona internamente:**
> - O container expõe a porta `5000` internamente, mapeada para `5050` no host
> - Os dados são persistidos em `docker/mlflow_data/` (SQLite + artefatos)
> - Cada vez que você roda um script de avaliação, um novo **Run** é criado automaticamente

---

## 8. Executando os Scripts

### Passo 1 — Indexar os dados no Pinecone (RAG)

Só precisa rodar uma vez para popular o vector store:

```bash
cd app
uv run python ingestion.py
```

Isso lê [`data/data.txt`](data/data.txt) (manual da ConnectMax), divide em chunks de 1000 caracteres e indexa no Pinecone com embeddings do Ollama.

---

### Passo 2 — Testar o agente

```bash
cd app
uv run python main.py
```

O agente LangGraph vai processar duas perguntas de teste e o MLflow vai registrar automaticamente as traces via `mlflow.langchain.autolog()`.

---

### Passo 3 — Avaliação simples (1 juiz)

```bash
cd app
uv run python mlflow_judge_test.py
```

Roda a avaliação com **1 juiz** (Qwen3) em um dataset de 2 perguntas. Bom para entender o fluxo básico.

**O que acontece:**
1. Para cada item do dataset, o juiz recebe `inputs + outputs + expected_facts`
2. O juiz LLM decide se a resposta contém os fatos esperados (`yes` / `no`)
3. O MLflow registra o resultado como métrica `correctness/yes_rate`

---

### Passo 4 — Avaliação avançada (2 juízes)

```bash
cd app
uv run python mlflow_judge.py
```

Roda a avaliação com **2 juízes diferentes** (Qwen3:8b e Llama3.2) no dataset completo de 5 perguntas do [`eval_data.py`](app/eval_data.py).

**O que acontece:**
1. Cada item do dataset é avaliado pelos dois juízes independentemente
2. O `judge_llama` tem lógica extra: normaliza respostas no formato `yes|no` (Llama avalia fato por fato)
3. O MLflow registra duas métricas separadas: `Corretude_Qwen/yes_rate` e `Corretude_Llama/yes_rate`
4. Você pode comparar se os dois juízes concordam entre si

---

## 9. Visualizando Experimentos no MLflow UI

Acesse **http://localhost:5050** após rodar qualquer script.

### Tela inicial — Lista de Experimentos

```
┌─────────────────────────────────────────────────────┐
│  MLflow                                             │
│                                                     │
│  Experiments                                        │
│  ┌─────────────────────────────────────────────┐    │
│  │ 📁 AngeloZero_Experiment          3 runs    │    │
│  │ 📁 Angelo_Evaluations_v3          1 run     │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Tela de Runs — Dentro de um Experimento

Clique no experimento para ver todos os runs:

```
┌──────────────────────────────────────────────────────────────────┐
│  AngeloZero_Experiment                                           │
│                                                                  │
│  Run Name          │ Date       │ Corretude_Qwen │ Corretude_Llama│
│  ──────────────────┼────────────┼────────────────┼────────────── │
│  run_abc123        │ 2026-05-10 │ 0.80           │ 0.60          │
│  run_def456        │ 2026-05-09 │ 0.60           │ 0.40          │
│  run_ghi789        │ 2026-05-08 │ 1.00           │ 0.80          │
└──────────────────────────────────────────────────────────────────┘
```

### Tela de Detalhes de um Run

Clique em um run específico para ver:

- **Overview**: parâmetros, tags, duração
- **Metrics**: gráficos das métricas ao longo do tempo
- **Artifacts**: artefatos salvos (traces, tabelas de avaliação)
- **Traces** (MLflow 3.0): rastreamento completo das chamadas LLM

### Aba Traces — Rastreamento de Chamadas LLM

O `mlflow.langchain.autolog()` em [`main.py`](app/main.py) captura automaticamente:

```
Trace: billing_agent
│
├── ChatPromptTemplate.invoke()
│   └── input: {"question": "Minha fatura veio errada", "context": "..."}
│
├── ChatOllama.invoke()
│   ├── input: [mensagens formatadas]
│   ├── output: "O valor da sua fatura é R$ 350,00..."
│   └── latency: 2.3s
│
└── StrOutputParser.invoke()
    └── output: "O valor da sua fatura é R$ 350,00..."
```

---

## 10. Comparando Múltiplos Experimentos

Esta é a parte mais poderosa do MLflow para estudos: **comparar runs lado a lado**.

### Como gerar múltiplos runs para comparar

Execute o script de avaliação várias vezes, alterando algo entre as execuções:

```bash
# Run 1: configuração padrão
uv run python mlflow_judge.py

# Altere o dataset em eval_data.py (adicione mais casos de teste)
# Run 2: dataset expandido
uv run python mlflow_judge.py

# Altere o modelo do agente em .env (MODEL_NAME=llama3.2)
# Run 3: agente diferente
uv run python mlflow_judge.py
```

### Comparando no Dashboard

1. Acesse **http://localhost:5050**
2. Clique no experimento `AngeloZero_Experiment`
3. **Selecione múltiplos runs** usando as checkboxes
4. Clique em **"Compare"**

Você verá uma tela como esta:

```
┌──────────────────────────────────────────────────────────────────┐
│  Compare Runs                                                    │
│                                                                  │
│  Metric                  │ Run #1  │ Run #2  │ Run #3            │
│  ────────────────────────┼─────────┼─────────┼──────────         │
│  Corretude_Qwen/yes_rate │  0.80   │  0.60   │  1.00             │
│  Corretude_Llama/yes_rate│  0.60   │  0.40   │  0.80             │
│                                                                  │
│  [Parallel Coordinates Chart]                                    │
│  [Bar Chart]                                                     │
│  [Scatter Plot]                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### O que observar na comparação

| O que comparar | Como fazer | O que aprender |
|---|---|---|
| **Dois juízes diferentes** | Rodar `mlflow_judge.py` | Qwen e Llama concordam? Qual é mais rigoroso? |
| **Mesmo juiz, datasets diferentes** | Alterar `eval_data.py` | Quais perguntas o agente erra mais? |
| **Agentes diferentes** | Alterar `MODEL_NAME` no `.env` | Qual modelo de agente responde melhor? |
| **Prompts diferentes** | Alterar o prompt em `main.py` | O prompt afeta a qualidade das respostas? |

### Exemplo prático: entendendo a discordância entre juízes

Quando `Corretude_Qwen/yes_rate = 0.80` e `Corretude_Llama/yes_rate = 0.60`:

- **Qwen aprovou 4/5 respostas**, Llama aprovou apenas 3/5
- Isso indica que **Llama é mais rigoroso** ou interpreta os fatos de forma diferente
- Clique no run e veja a aba **"Evaluation Results"** para ver qual pergunta específica gerou discordância
- O campo `rationale` de cada avaliação explica o raciocínio do juiz

### Dica: Adicionando tags para facilitar a comparação

Você pode adicionar tags manualmente no código para identificar os runs:

```python
# No início do seu script de avaliação
with mlflow.start_run(tags={"agente": "qwen3:8b", "dataset": "v2", "juiz": "dual"}):
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        scorers=[judge_qwen, judge_llama],
    )
```

No dashboard, você pode filtrar e ordenar por essas tags.

---

## 🔑 Conceitos-chave para fixar

| Conceito | Definição simples |
|---|---|
| **Experiment** | Pasta que agrupa runs relacionados |
| **Run** | Uma execução única (um `python script.py`) |
| **Metric** | Número registrado em um run (ex: `yes_rate = 0.8`) |
| **Scorer / Judge** | Função decorada com `@scorer` que avalia uma resposta |
| **Feedback** | Resultado de um scorer: `value` (yes/no) + `rationale` (explicação) |
| **Trace** | Rastreamento completo de uma chamada LLM (inputs, outputs, latência) |
| **autolog** | `mlflow.langchain.autolog()` captura traces automaticamente sem código extra |

---

## 🚀 Próximos passos sugeridos

1. **Adicione mais casos de teste** em [`eval_data.py`](app/eval_data.py) e compare os runs
2. **Crie um terceiro juiz** com critério diferente (ex: avaliar concisão em vez de corretude)
3. **Troque o modelo do agente** no `.env` e compare se respostas melhoram
4. **Adicione métricas customizadas** como latência ou número de tokens usados
5. **Explore a aba Traces** no MLflow UI para ver o rastreamento completo das chamadas LLM
