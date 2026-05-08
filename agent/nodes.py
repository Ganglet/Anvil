import time
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from agent.retriever import PaperRetriever
from agent.schema import ClusterExplanation

_MAX_RETRIES = 4
_RETRY_DELAYS = [5, 15, 30, 60]


def _invoke_with_retry(llm, messages):
    """Invoke LLM with exponential backoff on 503/429 errors."""
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e)
            if attempt == len(_RETRY_DELAYS) - 1:
                raise
            if "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "quota" in msg.lower():
                print(f"      [retry {attempt + 1}/{_MAX_RETRIES}] Gemini unavailable, waiting {delay}s...")
                time.sleep(delay)
            else:
                raise

PATCH_STRATEGIES = [
    "adversarial_training",
    "targeted_augmentation",
    "counterfactual_generation",
    "stylized_augmentation",
]

_EXPLAIN_PROMPT = """\
You are an adversarial ML security analyst writing a section of an audit report.

VULNERABILITY CLUSTER:
- Name: {name}
- Dominant attack: {dominant_attack}
- Size: {size} failures
- Attack distribution: {attack_distribution}

RELEVANT RESEARCH (retrieved from adversarial ML papers):
{context}

Write a concise technical explanation (3-5 sentences) of:
1. What vulnerability this cluster represents
2. Why the model is failing in this specific way
3. Which paper/concept best explains it

Be specific. Reference the paper if relevant. Do not use bullet points — write in prose.
"""

_RECOMMEND_PROMPT = """\
You are an adversarial ML security analyst recommending a remediation strategy.

VULNERABILITY CLUSTER:
- Name: {name}
- Dominant attack: {dominant_attack}
- Explanation: {explanation}

AVAILABLE PATCH STRATEGIES:
- adversarial_training: fine-tune on adversarially perturbed examples (best for gradient sensitivity)
- targeted_augmentation: augment training data targeting the identified weakness (best for edge-case underrepresentation)
- counterfactual_generation: generate examples varying spurious features while holding label-relevant features fixed (best for shortcut/background dependency)
- stylized_augmentation: train on style-transferred images to reduce texture bias (best for texture/semantic sensitivity)

Respond in this exact format — nothing else:
STRATEGY: <one of the four strategy names>
PARAM_LAYERS: <comma-separated layer names to target, or "all">
PARAM_STRENGTH: <low|medium|high>
PARAM_STEPS: <integer number of fine-tuning steps>
"""


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever: PaperRetriever = state["retriever"]
    cluster = state["cluster"]

    query = f"{cluster.name} {cluster.dominant_attack} adversarial vulnerability"
    chunks = retriever.retrieve(query)

    state["chunks"] = chunks
    return state


def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm: ChatGoogleGenerativeAI = state["llm"]
    cluster = state["cluster"]
    chunks = state["chunks"]

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = _EXPLAIN_PROMPT.format(
        name=cluster.name,
        dominant_attack=cluster.dominant_attack,
        size=cluster.size,
        attack_distribution=cluster.attack_distribution,
        context=context,
    )

    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
    state["explanation"] = response.content.strip()
    return state


def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm: ChatGoogleGenerativeAI = state["llm"]
    cluster = state["cluster"]
    explanation = state["explanation"]

    prompt = _RECOMMEND_PROMPT.format(
        name=cluster.name,
        dominant_attack=cluster.dominant_attack,
        explanation=explanation,
    )

    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
    patch_params = _parse_recommendation(response.content.strip())

    sources = list({c["source"] for c in state["chunks"]})

    state["result"] = ClusterExplanation(
        cluster_id=cluster.cluster_id,
        cluster_name=cluster.name,
        explanation=explanation,
        patch_strategy=patch_params.pop("strategy"),
        patch_params=patch_params,
        sources=sources,
    )
    return state


def _parse_recommendation(text: str) -> dict:
    """Parse the structured recommendation response into a dict."""
    result = {
        "strategy": "adversarial_training",
        "layers": "all",
        "strength": "medium",
        "steps": 100,
    }
    for line in text.splitlines():
        if line.startswith("STRATEGY:"):
            val = line.split(":", 1)[1].strip()
            if val in PATCH_STRATEGIES:
                result["strategy"] = val
        elif line.startswith("PARAM_LAYERS:"):
            result["layers"] = line.split(":", 1)[1].strip()
        elif line.startswith("PARAM_STRENGTH:"):
            val = line.split(":", 1)[1].strip()
            if val in ("low", "medium", "high"):
                result["strength"] = val
        elif line.startswith("PARAM_STEPS:"):
            try:
                result["steps"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return result
