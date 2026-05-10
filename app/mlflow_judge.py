import os

# IMPORTANTE: definir OPENAI_API_KEY antes de importar mlflow.
# O Ollama aceita qualquer valor como chave — usamos "ollama" como placeholder.
os.environ["OPENAI_API_KEY"] = "ollama"

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import scorer
from mlflow.genai.judges.utils.invocation_utils import invoke_judge_model
from mlflow.genai.judges.prompts.correctness import get_prompt

from eval_data import eval_dataset

# ---------------------------------------------------------------------------
# Monkey-patch: aumenta o timeout do _send_request para 300s.
# O MLflow hardcoda timeout=60 em mlflow.metrics.genai.model_utils._send_request,
# o que causa falhas com modelos locais (Ollama) que precisam de mais tempo.
# Substituímos a função diretamente no módulo após o import.
# ---------------------------------------------------------------------------
import mlflow.metrics.genai.model_utils as _model_utils
import requests as _requests

_original_send_request = _model_utils._send_request

def _patched_send_request(endpoint, headers, payload):
    try:
        response = _requests.post(
            url=endpoint,
            headers=headers,
            json=payload,
            timeout=300,  # 5 minutos — suficiente para modelos locais grandes
        )
        response.raise_for_status()
    
    except _requests.exceptions.HTTPError as e:
        body = getattr(e.response, "text", "")
        from mlflow.exceptions import MlflowException
        raise MlflowException(
            f"Failed to call LLM endpoint at {endpoint}.\n- Error: {e}\n"
            f"- Response body: {body}\n- Input payload: {payload}."
        ) from e
    
    except _requests.exceptions.Timeout as e:
        from mlflow.exceptions import MlflowException
        raise MlflowException(
            f"Request to LLM endpoint timed out after 300s: {endpoint}"
        ) from e
    return response.json()

_model_utils._send_request = _patched_send_request

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Configuração do MLflow (MLflow 3.0)
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(os.getenv("ML_FLOW_TRACKING_URI", "http://localhost:5050"))
mlflow.set_experiment("AngeloZero_Experiment_v2")

# ---------------------------------------------------------------------------
# 2. Configurações dos juízes LLM via Ollama (API OpenAI-compatible)
#    base_url roteia as chamadas para o Ollama local em vez da API OpenAI.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1/chat/completions"
)
MODEL_QWEN = "openai:/qwen3:8b"
MODEL_LLAMA = "openai:/llama3.2"


# ---------------------------------------------------------------------------
# 3. Definição dos Juízes como @scorer (MLflow 3.0)
#    Cada @scorer é um juiz LLM independente que avalia a corretude da resposta
#    do agente em relação aos fatos esperados do domínio ConnectMax.
# ---------------------------------------------------------------------------

@scorer
def judge_qwen(inputs, outputs, expectations):
    """
    Juiz de Corretude usando Qwen3:8b via Ollama.
    Avalia se a resposta do agente contém os fatos esperados do manual ConnectMax.
    """
    expected_facts = expectations.get("expected_facts") if expectations else None
    expected_response = expectations.get("expected_response") if expectations else None

    prompt = get_prompt(
        request=str(inputs),
        response=str(outputs),
        expected_facts=expected_facts,
        expected_response=expected_response,
    )

    return invoke_judge_model(
        model_uri=MODEL_QWEN,
        prompt=prompt,
        assessment_name="Corretude_Qwen",
        base_url=OLLAMA_BASE_URL,
    )


@scorer
def judge_llama(inputs, outputs, expectations):
    """
    Juiz de Corretude usando Llama3.2 via Ollama.
    Avalia se a resposta do agente contém os fatos esperados do manual ConnectMax.

    Nota: Llama3.2 avalia cada fato individualmente e retorna "yes|no" por fato.
    Normalizamos o resultado: "yes" se TODOS os fatos forem suportados, "no" caso contrário.
    """
    import json
    from mlflow.entities.assessment import Feedback, FeedbackValue
    from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType

    expected_facts = expectations.get("expected_facts") if expectations else None
    expected_response = expectations.get("expected_response") if expectations else None

    prompt = get_prompt(
        request=str(inputs),
        response=str(outputs),
        expected_facts=expected_facts,
        expected_response=expected_response,
    )

    # Chama o endpoint diretamente para capturar a resposta bruta do Llama
    raw_response = _model_utils.score_model_on_payload(
        model_uri=MODEL_LLAMA,
        payload=prompt,
        eval_parameters={"temperature": 0.0},
        proxy_url=OLLAMA_BASE_URL,
    )

    # Normaliza o resultado: Llama pode retornar "yes|no", "no|yes", etc.
    # Consideramos "yes" somente se TODOS os veredictos forem "yes".
    try:
        parsed = json.loads(raw_response)
        raw_result = str(parsed.get("result", "no")).lower()
        rationale = parsed.get("rationale", "")
        verdicts = [v.strip() for v in raw_result.split("|")]
        final_value = "yes" if all(v == "yes" for v in verdicts) else "no"
    except (json.JSONDecodeError, AttributeError):
        final_value = "no"
        rationale = f"Falha ao parsear resposta do Llama: {str(raw_response)[:200]}"

    return Feedback(
        name="Corretude_Llama",
        value=final_value,
        rationale=rationale,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=MODEL_LLAMA,
        ),
    )


# ---------------------------------------------------------------------------
# 4. Execução da Avaliação com mlflow.genai.evaluate (MLflow 3.0)
#    O dataset já está no formato correto: lista de dicts com
#    "inputs", "outputs" e "expectations".
# ---------------------------------------------------------------------------
results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[judge_qwen, judge_llama],
)

print("🚀 Avaliação concluída! Confira no Dashboard.")
print(f"Run ID: {results.run_id}")
print(f"Métricas: {results.metrics}")
print(f"Verifique em {os.getenv('ML_FLOW_TRACKING_URI', 'http://localhost:5050')}")
