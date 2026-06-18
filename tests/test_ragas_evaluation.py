"""RAGAS evaluation hooks for retrieval quality."""

from __future__ import annotations

import os

import pytest

from agent import LegalContractReviewAgent
from tests.conftest import SAMPLE_NDA_TEXT


@pytest.mark.ragas
def test_ragas_faithfulness_and_answer_relevancy() -> None:
    """Evaluate RAG output with RAGAS when evaluator credentials are configured."""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
        pytest.skip("RAGAS evaluation needs an API-backed evaluator key.")
    ragas = pytest.importorskip("ragas")
    datasets = pytest.importorskip("datasets")
    metrics = pytest.importorskip("ragas.metrics")

    agent = LegalContractReviewAgent()
    agent.rag_store.index_contract("ragas_eval_doc", SAMPLE_NDA_TEXT)
    retrieved = agent.rag_store.retrieve("Does this NDA contain one-sided indemnity?")
    flags = agent.risk_analyser_node(
        {"raw_text": SAMPLE_NDA_TEXT, "retrieved_clauses": retrieved}
    )["flags"]
    dataset = datasets.Dataset.from_dict(
        {
            "question": ["Does the NDA contain one-sided indemnity risk?"],
            "answer": [flags[0].plain_english_explanation],
            "contexts": [[item.text for item in retrieved]],
        }
    )
    result = ragas.evaluate(
        dataset,
        metrics=[metrics.faithfulness, metrics.answer_relevancy],
    )
    assert float(result["faithfulness"]) >= 0.5
    assert float(result["answer_relevancy"]) >= 0.5
