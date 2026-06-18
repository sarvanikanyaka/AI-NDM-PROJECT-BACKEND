"""ChromaDB-backed retrieval helpers for NDA clause review."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Iterable, List

import chromadb

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedClause:
    """A retrieved clause chunk with similarity metadata."""

    text: str
    score: float
    clause_type: str


RISK_KNOWLEDGE_BASE = [
    {
        "id": "indemnity_one_sided",
        "clause_type": "One-sided indemnity",
        "text": (
            "Risky NDAs may require only the receiving party to indemnify, defend, "
            "and hold harmless "
            "the disclosing party for broad losses, third-party claims, damages, fees, or expenses."
        ),
    },
    {
        "id": "termination_missing",
        "clause_type": "Missing termination date",
        "text": (
            "A balanced NDA should specify a term, expiration date, or termination right. "
            "Missing duration "
            "can create indefinite duties and operational uncertainty."
        ),
    },
    {
        "id": "confidentiality_vague",
        "clause_type": "Vague confidentiality scope",
        "text": (
            "Confidential information should be clearly scoped, with exclusions for public "
            "information, prior knowledge, independently developed information, and information "
            "received lawfully."
        ),
    },
    {
        "id": "governing_law_missing",
        "clause_type": "Missing governing law",
        "text": (
            "Contracts should identify the governing law and preferably the forum or jurisdiction. "
            "Missing "
            "governing law increases dispute uncertainty."
        ),
    },
]


class HashEmbeddingFunction:
    """Small deterministic embedding function for local Chroma retrieval."""

    def name(self) -> str:
        """Return embedding function name for ChromaDB compatibility."""
        return "hash_embedding_function"

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """Embed query texts for ChromaDB compatibility."""
        return [self.embed_text(text) for text in input]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """Embed document texts for ChromaDB compatibility."""
        return [self.embed_text(text) for text in input]

    def __call__(self, input: List[str]) -> List[List[float]]:  # pylint: disable=redefined-builtin
        """Return stable bag-of-token hash embeddings for ChromaDB."""
        return [self.embed_text(text) for text in input]

    @staticmethod
    def embed_text(text: str) -> List[float]:
        """Embed text into a normalized 64-dimensional hash vector."""
        vector = [0.0] * 64
        tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % len(vector)
            vector[index] += 1.0
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]


class ContractRagStore:
    """ChromaDB store used to index contract chunks and retrieve risky clauses."""

    def __init__(self, collection_name: str = "legal_contract_review") -> None:
        """Create an in-memory Chroma collection with deterministic embeddings."""
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=HashEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_knowledge_base()

    def index_contract(self, document_id: str, text: str) -> int:
        """Chunk and index a contract, returning the number of chunks stored."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{document_id}_chunk_{index}" for index, _ in enumerate(chunks)]
        metadatas = [
            {"source": document_id, "clause_type": "Uploaded contract"} for _ in chunks
        ]
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        LOGGER.info("Indexed %s contract chunks for %s", len(chunks), document_id)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 6) -> List[RetrievedClause]:
        """Retrieve the most relevant clause chunks for a query."""
        results = self._collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        retrieved = []
        for document, distance, metadata in zip(documents, distances, metadatas):
            retrieved.append(
                RetrievedClause(
                    text=document,
                    score=max(0.0, 1.0 - float(distance)),
                    clause_type=str(metadata.get("clause_type", "Unknown")),
                )
            )
        LOGGER.info("Retrieved %s clauses for query: %s", len(retrieved), query)
        return retrieved

    def _seed_knowledge_base(self) -> None:
        """Insert canonical NDA risk descriptions into the collection."""
        count = self._collection.count()
        if count:
            return
        self._collection.add(
            ids=[item["id"] for item in RISK_KNOWLEDGE_BASE],
            documents=[item["text"] for item in RISK_KNOWLEDGE_BASE],
            metadatas=[
                {"source": "risk_knowledge_base", "clause_type": item["clause_type"]}
                for item in RISK_KNOWLEDGE_BASE
            ],
        )


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> List[str]:
    """Split contract text into overlapping chunks suitable for retrieval."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []
    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = max(0, end - overlap)
    return chunks


def estimate_retrieval_quality(results: Iterable[RetrievedClause]) -> dict[str, float]:
    """Calculate lightweight retrieval quality metrics for observability."""
    scores = [result.score for result in results]
    if not scores:
        return {"mean_similarity": 0.0, "top_similarity": 0.0}
    return {
        "mean_similarity": round(sum(scores) / len(scores), 3),
        "top_similarity": round(max(scores), 3),
    }
