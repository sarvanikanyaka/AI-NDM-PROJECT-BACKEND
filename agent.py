"""LangGraph legal contract review agent."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError
from pypdf import PdfReader

from models import ReviewResponse, RiskFlag, RiskLevel
from rag import ContractRagStore, RetrievedClause, estimate_retrieval_quality

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - handled in environments without LangChain.
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - keeps local tests independent of LangSmith.

    def traceable(*_args: Any, **_kwargs: Any) -> Any:
        """No-op trace decorator when LangSmith is not installed."""

        def decorator(function: Any) -> Any:
            return function

        return decorator


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a legal clause risk detection agent for NDA review.
Detect these risks:
- one-sided indemnity
- missing termination dates
- vague confidentiality scope
- missing governing law
Return ONLY valid JSON with this shape:
{
  "flags": [
    {
      "clause_name": "string",
      "risk_level": "HIGH|MEDIUM|LOW",
      "plain_english_explanation": "string"
    }
  ]
}

Few-shot example 1:
Input: "Recipient shall indemnify, defend, and hold harmless Discloser from all claims.
This Agreement is governed by New York law."
Output:
{
  "flags": [
    {
      "clause_name": "One-sided indemnity",
      "risk_level": "HIGH",
      "plain_english_explanation": "Only the recipient appears to indemnify the discloser."
    },
    {
      "clause_name": "Termination date",
      "risk_level": "MEDIUM",
      "plain_english_explanation": "The excerpt does not define when the NDA expires."
    }
  ]
}

Few-shot example 2:
Input: "Confidential Information means any information disclosed by either party,
excluding public information, prior knowledge, independently developed information,
and legally obtained third-party information. This NDA expires two years after signature."
Output:
{
  "flags": [
    {
      "clause_name": "Confidentiality scope",
      "risk_level": "LOW",
      "plain_english_explanation": "The scope has standard exclusions, reducing ambiguity."
    },
    {
      "clause_name": "Termination date",
      "risk_level": "LOW",
      "plain_english_explanation": "The NDA includes a clear two-year duration."
    }
  ]
}
"""


class UserFacingError(Exception):
    """Exception type for errors safe to show to users."""

    def __init__(self, message: str, error_code: str) -> None:
        """Create a user-facing exception."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ContractState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes."""

    file_path: str
    document_name: str
    document_id: str
    raw_text: str
    chunks_indexed: int
    retrieved_clauses: List[RetrievedClause]
    flags: List[RiskFlag]
    risk_score: int
    summary: str
    response: ReviewResponse
    retrieval_quality: Dict[str, float]


class LegalContractReviewAgent:
    """End-to-end LangGraph agent for NDA risk detection."""

    def __init__(self, rag_store: Optional[ContractRagStore] = None) -> None:
        """Initialize RAG, LLM, and the compiled LangGraph workflow."""
        self.rag_store = rag_store or ContractRagStore()
        self.llm = self._build_llm()
        self.graph = self._build_graph()

    @traceable(name="legal_contract_review_run")
    def review_pdf(self, file_path: str, document_name: str) -> ReviewResponse:
        """Review a PDF contract and return a structured risk report."""
        state: ContractState = {
            "file_path": file_path,
            "document_name": document_name,
            "document_id": f"{int(time.time() * 1000)}_{sanitize_name(document_name)}",
        }
        final_state = self.graph.invoke(state)
        return final_state["response"]

    def pdf_upload_node(self, state: ContractState) -> ContractState:
        """Validate the uploaded PDF before parsing."""
        log_node_io("PDF Upload Node", "input", state)
        file_path = state.get("file_path", "")
        if not file_path.lower().endswith(".pdf"):
            raise UserFacingError("Please upload a valid PDF file.", "INVALID_FILE_TYPE")
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise UserFacingError("The uploaded PDF is empty.", "EMPTY_FILE")
        log_node_io("PDF Upload Node", "output", state)
        return state

    def pypdf_parser_node(self, state: ContractState) -> ContractState:
        """Extract searchable text from the uploaded PDF."""
        log_node_io("PyPDF Parser Node", "input", state)
        try:
            reader = PdfReader(state["file_path"])
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("PDF parsing failed: %s", exc)
            raise UserFacingError(
                "We could not read this PDF. Please upload a non-corrupted text PDF.",
                "CORRUPTED_PDF",
            ) from exc
        raw_text = "\n".join(pages).strip()
        if not raw_text:
            raise UserFacingError(
                "This PDF does not contain extractable text. Please upload a text-based NDA PDF.",
                "IMAGE_ONLY_OR_EMPTY_PDF",
            )
        if not looks_english(raw_text):
            raise UserFacingError(
                "This reviewer currently supports English NDA documents only.",
                "NON_ENGLISH_CONTRACT",
            )
        state["raw_text"] = raw_text
        log_node_io("PyPDF Parser Node", "output", state)
        return state

    def chromadb_embedder_node(self, state: ContractState) -> ContractState:
        """Index the parsed contract text into ChromaDB."""
        log_node_io("ChromaDB Embedder Node", "input", state)
        chunks_indexed = self.rag_store.index_contract(
            state["document_id"], state["raw_text"]
        )
        if chunks_indexed == 0:
            raise UserFacingError("The contract text is empty after parsing.", "EMPTY_CONTRACT")
        state["chunks_indexed"] = chunks_indexed
        log_node_io("ChromaDB Embedder Node", "output", state)
        return state

    def clause_retriever_node(self, state: ContractState) -> ContractState:
        """Retrieve contract evidence and risk patterns for the analyzer."""
        log_node_io("Clause Retriever Node", "input", state)
        query = (
            "NDA indemnity termination confidentiality scope governing law risky clauses "
            f"{state['raw_text'][:1200]}"
        )
        retrieved = self.rag_store.retrieve(query, top_k=8)
        state["retrieved_clauses"] = retrieved
        state["retrieval_quality"] = estimate_retrieval_quality(retrieved)
        log_node_io("Clause Retriever Node", "output", state)
        return state

    def risk_analyser_node(self, state: ContractState) -> ContractState:
        """Analyze retrieved clauses and produce structured risk flags."""
        log_node_io("Risk Analyser Node", "input", state)
        context = "\n\n".join(item.text for item in state.get("retrieved_clauses", []))
        prompt = (
            "Review this NDA text and retrieved context. Return JSON only.\n\n"
            f"Contract text:\n{state['raw_text'][:6000]}\n\nRetrieved context:\n{context[:3000]}"
        )
        flags = self._llm_flags(prompt)
        if not flags:
            flags = heuristic_flags(state["raw_text"])
        state["flags"] = attach_evidence(flags, state.get("retrieved_clauses", []))
        log_node_io("Risk Analyser Node", "output", state)
        return state

    def score_generator_node(self, state: ContractState) -> ContractState:
        """Calculate the overall contract risk score and summary."""
        log_node_io("Score Generator Node", "input", state)
        weights = {RiskLevel.HIGH: 30, RiskLevel.MEDIUM: 16, RiskLevel.LOW: 5}
        risk_score = min(100, sum(weights[flag.risk_level] for flag in state["flags"]))
        high_count = sum(1 for flag in state["flags"] if flag.risk_level == RiskLevel.HIGH)
        medium_count = sum(
            1 for flag in state["flags"] if flag.risk_level == RiskLevel.MEDIUM
        )
        state["risk_score"] = risk_score
        state["summary"] = (
            f"Detected {len(state['flags'])} clause findings: {high_count} high risk, "
            f"{medium_count} medium risk. Overall score is {risk_score}/100."
        )
        log_node_io("Score Generator Node", "output", state)
        return state

    def json_response_node(self, state: ContractState) -> ContractState:
        """Build the final Pydantic response object for FastAPI."""
        log_node_io("JSON Response Node", "input", state)
        state["response"] = ReviewResponse(
            document_name=state["document_name"],
            risk_score=state["risk_score"],
            flags=state["flags"],
            summary=state["summary"],
            retrieval_quality=state.get("retrieval_quality", {}),
        )
        log_node_io("JSON Response Node", "output", state)
        return state

    def _build_graph(self) -> Any:
        """Compile the LangGraph multi-node workflow."""
        workflow = StateGraph(ContractState)
        # PDF Upload Node: validates file type, presence, and basic upload integrity.
        workflow.add_node("pdf_upload", self.pdf_upload_node)
        # PyPDF Parser Node: extracts text and rejects unreadable or unsupported PDFs.
        workflow.add_node("pypdf_parser", self.pypdf_parser_node)
        # ChromaDB Embedder Node: chunks and stores parsed contract text for retrieval.
        workflow.add_node("chromadb_embedder", self.chromadb_embedder_node)
        # Clause Retriever Node: retrieves risky clause patterns and matching contract evidence.
        workflow.add_node("clause_retriever", self.clause_retriever_node)
        # Risk Analyser Node: calls Groq or falls back to deterministic legal heuristics.
        workflow.add_node("risk_analyser", self.risk_analyser_node)
        # Score Generator Node: converts clause severities into a 0-100 risk score.
        workflow.add_node("score_generator", self.score_generator_node)
        # JSON Response Node: validates the final API payload with Pydantic models.
        workflow.add_node("json_response", self.json_response_node)
        workflow.set_entry_point("pdf_upload")
        workflow.add_edge("pdf_upload", "pypdf_parser")
        workflow.add_edge("pypdf_parser", "chromadb_embedder")
        workflow.add_edge("chromadb_embedder", "clause_retriever")
        workflow.add_edge("clause_retriever", "risk_analyser")
        workflow.add_edge("risk_analyser", "score_generator")
        workflow.add_edge("score_generator", "json_response")
        workflow.add_edge("json_response", END)
        return workflow.compile()

    def _build_llm(self) -> Any:
        """Create the Groq LLM client when credentials are configured."""
        if ChatGroq is None or not os.getenv("GROQ_API_KEY"):
            LOGGER.info("GROQ_API_KEY not configured; using deterministic risk fallback.")
            return None
        return ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
            max_retries=1,
        )

    def _llm_flags(self, prompt: str) -> List[RiskFlag]:
        """Call the LLM and parse risk flags, returning an empty list on timeout."""
        if self.llm is None or HumanMessage is None or SystemMessage is None:
            return []
        try:
            response = self.llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            payload = parse_json_object(str(response.content))
            return [RiskFlag(**item) for item in payload.get("flags", [])]
        except (TimeoutError, ValidationError, json.JSONDecodeError) as exc:
            LOGGER.warning("LLM response could not be used: %s", exc)
            return []
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("LLM call failed, falling back to heuristics: %s", exc)
            return []


def heuristic_flags(text: str) -> List[RiskFlag]:
    """Deterministically detect core NDA risks when the LLM is unavailable."""
    lower_text = text.lower()
    flags = [
        detect_indemnity(lower_text),
        detect_termination(lower_text),
        detect_confidentiality_scope(lower_text),
        detect_governing_law(lower_text),
    ]
    return [flag for flag in flags if flag is not None]


def detect_indemnity(lower_text: str) -> Optional[RiskFlag]:
    """Detect one-sided or broad indemnity language."""
    has_indemnity = any(word in lower_text for word in ["indemnify", "hold harmless"])
    mutual_language = any(word in lower_text for word in ["mutual", "each party", "both parties"])
    if has_indemnity and not mutual_language:
        return RiskFlag(
            clause_name="One-sided indemnity",
            risk_level=RiskLevel.HIGH,
            plain_english_explanation=(
                "The NDA appears to place indemnity obligations on only one side, "
                "which can create uneven liability."
            ),
        )
    if has_indemnity:
        return RiskFlag(
            clause_name="Indemnity",
            risk_level=RiskLevel.LOW,
            plain_english_explanation=(
                "The NDA includes indemnity language, but it appears more balanced "
                "because mutual wording is present."
            ),
        )
    return None


def detect_termination(lower_text: str) -> RiskFlag:
    """Detect whether a termination date or duration is missing."""
    duration_pattern = r"\b(\d+\s*(year|month|day)s?|expires?|terminat|effective until|survive)\b"
    if re.search(duration_pattern, lower_text):
        return RiskFlag(
            clause_name="Termination date",
            risk_level=RiskLevel.LOW,
            plain_english_explanation=(
                "The NDA includes duration, expiry, termination, or survival language."
            ),
        )
    return RiskFlag(
        clause_name="Missing termination date",
        risk_level=RiskLevel.MEDIUM,
        plain_english_explanation=(
            "The NDA does not clearly say when confidentiality duties end or when "
            "the agreement terminates."
        ),
    )


def detect_confidentiality_scope(lower_text: str) -> RiskFlag:
    """Detect vague confidentiality scope and missing common exclusions."""
    exclusions = ["public", "prior knowledge", "independently developed", "third party"]
    present_exclusions = sum(1 for exclusion in exclusions if exclusion in lower_text)
    if "confidential information" not in lower_text:
        return RiskFlag(
            clause_name="Vague confidentiality scope",
            risk_level=RiskLevel.HIGH,
            plain_english_explanation=(
                "The NDA does not clearly define what counts as confidential information."
            ),
        )
    if present_exclusions < 2:
        return RiskFlag(
            clause_name="Vague confidentiality scope",
            risk_level=RiskLevel.MEDIUM,
            plain_english_explanation=(
                "The confidentiality definition is present but lacks standard exclusions, "
                "making the scope broad."
            ),
        )
    return RiskFlag(
        clause_name="Confidentiality scope",
        risk_level=RiskLevel.LOW,
        plain_english_explanation=(
            "The NDA defines confidential information and includes common exclusions."
        ),
    )


def detect_governing_law(lower_text: str) -> RiskFlag:
    """Detect whether governing law is missing."""
    if "governing law" in lower_text or "laws of" in lower_text or "jurisdiction" in lower_text:
        return RiskFlag(
            clause_name="Governing law",
            risk_level=RiskLevel.LOW,
            plain_english_explanation="The NDA identifies governing law or jurisdiction language.",
        )
    return RiskFlag(
        clause_name="Missing governing law",
        risk_level=RiskLevel.MEDIUM,
        plain_english_explanation=(
            "The NDA does not identify which law controls disputes, which can create uncertainty."
        ),
    )


def attach_evidence(flags: List[RiskFlag], clauses: List[RetrievedClause]) -> List[RiskFlag]:
    """Attach short retrieved excerpts to each risk flag."""
    for flag in flags:
        matching = next(
            (
                clause
                for clause in clauses
                if flag.clause_name.lower().split()[0] in clause.text.lower()
                or flag.clause_name.lower().split()[0] in clause.clause_type.lower()
            ),
            clauses[0] if clauses else None,
        )
        if matching:
            flag.evidence = matching.text[:280]
    return flags


def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from model output."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(match.group(0))


def looks_english(text: str) -> bool:
    """Use a conservative character heuristic to reject non-English documents."""
    letters = re.findall(r"[A-Za-z]", text)
    non_ascii = [character for character in text if ord(character) > 127]
    if len(letters) < 30:
        return False
    return len(non_ascii) / max(len(text), 1) < 0.2


def log_node_io(node_name: str, direction: str, state: ContractState) -> None:
    """Log sanitized node input/output for auditing and LangSmith correlation."""
    safe_state = {
        key: summarize_value(value)
        for key, value in state.items()
        if key not in {"raw_text", "retrieved_clauses"}
    }
    if "raw_text" in state:
        safe_state["raw_text_chars"] = len(state["raw_text"])
    if "retrieved_clauses" in state:
        safe_state["retrieved_count"] = len(state["retrieved_clauses"])
    LOGGER.info("%s %s: %s", node_name, direction, safe_state)


def summarize_value(value: Any) -> Any:
    """Summarize verbose values before logging."""
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return value
    return value


def sanitize_name(name: str) -> str:
    """Return a filesystem-safe identifier from a document name."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "contract"


def most_common_clause(flags: List[RiskFlag]) -> Optional[str]:
    """Return the most common clause name from a list of flags."""
    if not flags:
        return None
    return Counter(flag.clause_name for flag in flags).most_common(1)[0][0]
