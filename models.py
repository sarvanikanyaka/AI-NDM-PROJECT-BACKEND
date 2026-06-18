"""Pydantic models for Legal Contract Review API shapes."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Supported risk levels for detected clauses."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskFlag(BaseModel):
    """A single clause risk finding returned by the agent."""

    clause_name: str = Field(..., examples=["One-sided indemnity"])
    risk_level: RiskLevel = Field(..., examples=["HIGH"])
    plain_english_explanation: str = Field(
        ...,
        examples=[
            "The NDA requires only one party to indemnify the other, "
            "which can create an uneven liability burden."
        ],
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Short retrieved excerpt that supports the finding.",
    )


class ReviewResponse(BaseModel):
    """Successful contract review response."""

    document_name: str
    risk_score: int = Field(..., ge=0, le=100)
    flags: List[RiskFlag]
    summary: str
    retrieval_quality: Dict[str, float] = Field(default_factory=dict)
    authenticity_status: str = Field(default="UNKNOWN")
    authenticity_score: int = Field(default=0, ge=0, le=100)
    authenticity_explanation: str = Field(default="")


class ErrorResponse(BaseModel):
    """Clean error response returned to the frontend."""

    detail: str
    error_code: str


class MetricsResponse(BaseModel):
    """Operational metrics shown in the dashboard."""

    total_contracts_reviewed: int
    average_risk_score: float
    most_flagged_clause_type: Optional[str]
    clause_counts: Dict[str, int] = Field(default_factory=dict)


class HistoryItem(BaseModel):
    """Review history item shown in the frontend."""

    document_name: str
    risk_score: int
    authenticity_status: str
    reviewed_at: str
    summary: str
