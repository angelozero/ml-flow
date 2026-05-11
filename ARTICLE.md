# Avaliação de Agentes IA com MLflow 3.0 e LLM Judges

Este guia documenta a jornada de criação de um sistema de auditoria para agentes de IA, utilizando **LangGraph** para orquestração e **MLflow** para governança e métricas.

---

## 1. O Ponto de Partida: Por que avaliar?

Em sistemas tradicionais, `2 + 2` é sempre `4`. Na IA generativa, a resposta pode variar. Como garantir que um Agente de Faturamento não dê um desconto indevido ou invente uma data de vencimento?

A solução não é olhar os logs um por um, mas criar uma **Esteira de Avaliação Automática**.

---

## 2. O Objeto de Estudo: O Grafo (LangGraph)

Antes de avaliar, precisamos de algo para rodar. No arquivo `main.py`, utilizamos o **LangGraph** para criar um fluxo onde um "Supervisor" decide quem resolve o problema.

### Conceito Didático: O Roteador Inteligente
O Grafo funciona como uma central telefônica. Ele recebe a pergunta e direciona para o especialista.

```python
# Trecho do main.py: Definindo a lógica de roteamento
def supervisor(state: AgentState) -> Literal["billing", "tech"]:
    last_message = state["messages"][-1].content.lower()

    if any(word in last_message for word in ["conta", "pagamento", "fatura"]):
        return "billing" # Manda para o especialista em RAG
    return "tech"    # Manda para o suporte técnico
```

**O que aprendemos aqui:** Nesta etapa, o MLflow faz **Monitoria (Observability)** através do `autolog()`. Ele registra o caminho (Trace) e o tempo de resposta, mas ainda não diz se a resposta está certa.

---

## 3. O Gabarito: Ground Truth (eval_data.py)

Para julgar, precisamos de uma "verdade". No arquivo `eval_data.py`, definimos o que esperamos do modelo.

```python
eval_dataset = [
    {
        "inputs": {"query": "Qual o valor da fatura do Angelo?"},
        "outputs": "O valor é de R$ 350,00...", # Resposta gerada pelo modelo
        "expectations": {
            "expected_facts": [
                "Valor de R$ 350,00",
                "Nome do cliente: Angelo"
            ]
        }
    }
]
```

**Dúvida comum:** *"Preciso olhar o banco de dados (Pinecone) aqui?"*
**Resposta:** Não necessariamente. Nesta etapa, o `expected_facts` é o seu **Contrato de Negócio**. Se o dado não estiver aí, o juiz dirá que está errado, independentemente do que o banco de dados entregou.

---

## 4. Os Juízes: LLM-as-a-Judge

Aqui entra a sofisticação. Usamos uma LLM para ler a resposta de outra. No projeto, usamos dois perfis: **Qwen** (Generalista) e **Llama** (Auditor Rigoroso).

### Juiz 1: Qwen (O Professor de Redação)
Ele usa o prompt padrão do MLflow para entender se, semanticamente, a resposta faz sentido.

### Juiz 2: Llama (O Auditor Fiscal)
Aqui implementamos uma lógica customizada. Ele analisa fato por fato e só aprova se **todos** estiverem presentes.

```python
@scorer
def judge_llama(inputs, outputs, expectations):
    # ... lógica de captura do prompt ...
    
    # Normalização: O Llama avalia fato por fato (ex: yes|no|yes)
    verdicts = [v.strip() for v in raw_result.split("|")]
    
    # Lógica "All or Nothing": Só é "yes" se TUDO for "yes"
    final_value = "yes" if all(v == "yes" for v in verdicts) else "no"
    
    return Feedback(value=final_value, rationale=rationale, ...)
```

---

## 5. Comparação e Visualização (Dashboard)

Com o MLflow rodando em Docker, acessamos o dashboard para comparar os juízes.

### Tom vs. Precisão
* **Se o Qwen der "Nota 5" e o Llama der "No":** A resposta está bem escrita e educada, mas falta um dado técnico essencial (erro de precisão).
* **Se o Llama der "Yes" e o Qwen der "Nota 2":** Os dados estão corretos, mas a resposta está mal formatada ou confusa (erro de tom).

### O Papel da Temperatura 0.0
Para todos os juízes, configuramos `temperature=0.0`. 
> **Por quê?** Queremos determinismo. Um juiz não pode "acordar de bom humor" e dar notas diferentes para o mesmo texto. Ele deve ser um validador constante.

---

## 6. Resolvendo Problemas Locais (Monkey-Patching)

Como rodamos modelos locais via **Ollama**, o tempo de processamento pode exceder o padrão de 60 segundos das bibliotecas. Para evitar que o sistema caia, aplicamos um "remendo" técnico:

```python
# Aumentando o timeout para 5 minutos (300s) para suportar o Ollama local
def _patched_send_request(endpoint, headers, payload):
    response = _requests.post(..., timeout=300)
    return response.json()

_model_utils._send_request = _patched_send_request
```

---

## Conclusão: O Ciclo de Desenvolvimento Sênior

Ao final deste estudo, o fluxo de trabalho deixa de ser baseado em "tentativa e erro" e passa a ser baseado em dados:

1.  **Desenvolve** o Grafo no LangGraph.
2.  **Monitora** a execução com Traces automáticos.
3.  **Avalia** o resultado final contra o Ground Truth.
4.  **Audita** a base de dados (RAG) se os juízes apontarem falhas.