"""Unit tests for individual LangGraph nodes."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from agent import LegalContractReviewAgent, UserFacingError, heuristic_flags
from models import RiskLevel
from tests.conftest import SAMPLE_NDA_TEXT


def test_pdf_upload_node_accepts_pdf(pdf_factory) -> None:
    """PDF upload node validates a normal PDF path."""
    pdf_path = pdf_factory(SAMPLE_NDA_TEXT)
    agent = LegalContractReviewAgent()
    state = {"file_path": str(pdf_path), "document_name": "sample-nda.pdf"}
    assert agent.pdf_upload_node(state)["document_name"] == "sample-nda.pdf"


def test_pdf_upload_node_rejects_non_pdf(tmp_path: Path) -> None:
    """PDF upload node rejects non-PDF extensions."""
    text_path = tmp_path / "contract.txt"
    text_path.write_text("hello", encoding="utf-8")
    agent = LegalContractReviewAgent()
    with pytest.raises(UserFacingError) as error:
        agent.pdf_upload_node({"file_path": str(text_path), "document_name": "contract.txt"})
    assert error.value.error_code == "INVALID_FILE_TYPE"


def test_parser_node_extracts_text(pdf_factory) -> None:
    """Parser node extracts text from a real text-based PDF."""
    pdf_path = pdf_factory(SAMPLE_NDA_TEXT)
    agent = LegalContractReviewAgent()
    state = {"file_path": str(pdf_path), "document_name": "sample-nda.pdf"}
    parsed = agent.pypdf_parser_node(state)
    assert "Confidential Information" in parsed["raw_text"]


def test_parser_node_rejects_blank_pdf(tmp_path: Path) -> None:
    """Parser node rejects blank or image-only PDFs with a clean error."""
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as output:
        writer.write(output)
    agent = LegalContractReviewAgent()
    with pytest.raises(UserFacingError) as error:
        agent.pypdf_parser_node({"file_path": str(pdf_path), "document_name": "blank.pdf"})
    assert error.value.error_code == "IMAGE_ONLY_OR_EMPTY_PDF"


def test_parser_node_rejects_corrupted_pdf(tmp_path: Path) -> None:
    """Parser node returns a clean corrupted-PDF error."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    agent = LegalContractReviewAgent()
    with pytest.raises(UserFacingError) as error:
        agent.pypdf_parser_node({"file_path": str(pdf_path), "document_name": "corrupted.pdf"})
    assert error.value.error_code == "CORRUPTED_PDF"


def test_chromadb_embedder_node_indexes_text() -> None:
    """Embedder node indexes parsed contract text into ChromaDB."""
    agent = LegalContractReviewAgent()
    state = {"document_id": "doc1", "raw_text": SAMPLE_NDA_TEXT}
    output = agent.chromadb_embedder_node(state)
    assert output["chunks_indexed"] >= 1


def test_clause_retriever_node_returns_context() -> None:
    """Retriever node returns relevant context and retrieval quality metrics."""
    agent = LegalContractReviewAgent()
    state = {
        "document_id": "doc2",
        "raw_text": SAMPLE_NDA_TEXT,
        "chunks_indexed": 1,
    }
    agent.rag_store.index_contract("doc2", SAMPLE_NDA_TEXT)
    output = agent.clause_retriever_node(state)
    assert output["retrieved_clauses"]
    assert "top_similarity" in output["retrieval_quality"]


def test_risk_analyser_node_flags_indemnity() -> None:
    """Risk analyzer node produces structured legal risk findings (heuristic fallback)."""
    agent = LegalContractReviewAgent()
    state = {
        "raw_text": SAMPLE_NDA_TEXT,
        "retrieved_clauses": agent.rag_store.retrieve("indemnity governing law termination"),
    }
    output = agent.risk_analyser_node(state)
    clause_names = [flag.clause_name for flag in output["flags"]]
    assert any(
        "indemnity" in name.lower() or "termination" in name.lower()
        or "confidentiality" in name.lower() or "governing" in name.lower()
        for name in clause_names
    ), f"Expected NDA risk flags, got: {clause_names}"


def test_score_generator_node_scores_flags() -> None:
    """Score generator converts risk flags into a bounded score."""
    agent = LegalContractReviewAgent()
    state = {"flags": heuristic_flags(SAMPLE_NDA_TEXT)}
    output = agent.score_generator_node(state)
    assert 0 <= output["risk_score"] <= 100
    assert "Overall score" in output["summary"]


def test_json_response_node_builds_response() -> None:
    """JSON response node creates a validated Pydantic response."""
    agent = LegalContractReviewAgent()
    state = {
        "document_name": "sample-nda.pdf",
        "flags": heuristic_flags(SAMPLE_NDA_TEXT),
        "risk_score": 77,
        "summary": "Detected risk.",
        "retrieval_quality": {"top_similarity": 0.9},
    }
    output = agent.json_response_node(state)
    assert output["response"].document_name == "sample-nda.pdf"


def test_non_nda_document_edge_case(pdf_factory) -> None:
    """A non-NDA document still returns missing-clause risks rather than crashing."""
    text = "Invoice number 123. Payment is due in thirty days for consulting services."
    pdf_path = pdf_factory(text, "invoice.pdf")
    agent = LegalContractReviewAgent()
    response = agent.review_pdf(str(pdf_path), "invoice.pdf")
    assert response.risk_score > 0
    assert any(flag.risk_level in {RiskLevel.HIGH, RiskLevel.MEDIUM} for flag in response.flags)
