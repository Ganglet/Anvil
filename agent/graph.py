import os
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from agent.nodes import retrieve_node, explain_node, recommend_node
from agent.retriever import PaperRetriever
from agent.schema import ClusterExplanation, ExplanationReport
from clustering.taxonomy import VulnerabilityCluster, VulnerabilityTaxonomy


def _build_graph():
    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("explain", explain_node)
    graph.add_node("recommend", recommend_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


_COMPILED_GRAPH = _build_graph()


def run_agent(taxonomy: VulnerabilityTaxonomy) -> ExplanationReport:
    """
    Run the LangGraph explanation agent over every cluster in the taxonomy.

    Args:
        taxonomy: VulnerabilityTaxonomy from Phase 4

    Returns:
        ExplanationReport with one ClusterExplanation per cluster
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    retriever = PaperRetriever(top_k=5)

    explanations: List[ClusterExplanation] = []

    for cluster in taxonomy.clusters:
        print(f"  Explaining cluster [{cluster.cluster_id}]: {cluster.name}")
        state = {
            "llm": llm,
            "retriever": retriever,
            "cluster": cluster,
            "chunks": [],
            "explanation": "",
            "result": None,
        }
        final_state = _COMPILED_GRAPH.invoke(state)
        explanations.append(final_state["result"])

    return ExplanationReport(
        model_name=taxonomy.model_name,
        explanations=explanations,
    )
