"""FastAPI application for the Legal Contract Review Agent."""

from __future__ import annotations

import logging
import json
import os
import secrets
import smtplib
import tempfile
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from hashlib import sha256
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from agent import LegalContractReviewAgent, UserFacingError
from models import ErrorResponse, HistoryItem, MetricsResponse, ReviewResponse

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("APP_DATA_DIR", ".data"))
USERS_FILE = DATA_DIR / "users.json"

app = FastAPI(
    title="Legal Contract Review & Risk Detection Agent",
    description="RAG-powered NDA risk analysis using LangGraph, ChromaDB, LangSmith, and Groq.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:5173,"
            "http://127.0.0.1:5173,"
            "https://ai-ndm-project-frontend-production.up.railway.app"
        ),
    ).replace(" ", "").split(","),
    allow_origin_regex=os.getenv(
        "CORS_ORIGIN_REGEX",
        r"https://.*\.(up\.railway\.app|vercel\.app|netlify\.app)",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = LegalContractReviewAgent()
review_history: List[ReviewResponse] = []
history_items: List[HistoryItem] = []
users: dict[str, dict[str, str | bool]] = {}
sessions: dict[str, str] = {}


class ChatRequest(BaseModel):
    """Legal chat request sent from the frontend assistant."""

    message: str = Field(..., min_length=1, max_length=1200)
    contract_context: str = Field(default="", max_length=12000)


class ChatResponse(BaseModel):
    """Legal chat response returned to the frontend assistant."""

    response: str


class RegisterRequest(BaseModel):
    """Registration request for demo email verification."""

    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=160)
    password: str = Field(..., min_length=8, max_length=80)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject numeric or symbol-only names."""
        clean_value = value.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,79}", clean_value):
            raise ValueError("Name must contain letters only, not numbers or symbols.")
        return clean_value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Validate email format without requiring extra dependencies."""
        clean_value = normalize_email(value)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_value):
            raise ValueError("Enter a valid email address.")
        return clean_value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """Require a basic strong password for demo auth."""
        if len(value) < 8 or not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
            raise ValueError("Password must be at least 8 characters and include letters and numbers.")
        return value


class RegisterResponse(BaseModel):
    """Registration response with verification status."""

    message: str
    email: str
    otp_required: bool = True


class OtpVerifyRequest(BaseModel):
    """OTP verification request."""

    email: str = Field(..., min_length=5, max_length=160)
    otp: str = Field(..., min_length=6, max_length=6)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Validate email format for OTP verification."""
        clean_value = normalize_email(value)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_value):
            raise ValueError("Enter a valid email address.")
        return clean_value

    @field_validator("otp")
    @classmethod
    def validate_otp(cls, value: str) -> str:
        """Validate OTP contains exactly six digits."""
        if not re.fullmatch(r"\d{6}", value.strip()):
            raise ValueError("OTP must be a 6-digit number.")
        return value.strip()


class LoginRequest(BaseModel):
    """Login request for verified users."""

    email: str = Field(..., min_length=5, max_length=160)
    password: str = Field(..., min_length=8, max_length=80)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Validate email format for login."""
        clean_value = normalize_email(value)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_value):
            raise ValueError("Enter a valid email address.")
        return clean_value


class LoginResponse(BaseModel):
    """Login response containing a session token."""

    message: str
    token: str
    name: str
    email: str


class VerifyResponse(BaseModel):
    """Email verification response."""

    message: str
    verified: bool


@app.exception_handler(UserFacingError)
async def user_error_handler(_request, exc: UserFacingError) -> JSONResponse:
    """Return clean validation and agent errors to the frontend."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=exc.message, error_code=exc.error_code).model_dump(),
    )


@app.on_event("startup")
def load_persisted_users() -> None:
    """Load verified users from disk when the API process starts."""
    users.update(read_users_from_disk())


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.post(
    "/review",
    response_model=ReviewResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def review_contract(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> ReviewResponse:
    """Upload an NDA PDF and return AI-generated risk flags."""
    require_session(authorization)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise UserFacingError("Please upload an NDA PDF file.", "INVALID_FILE_TYPE")
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(await file.read())
        temporary_path = temporary_file.name
    try:
        result = agent.review_pdf(temporary_path, file.filename)
        authenticity = detect_fake_nda(result)
        result.authenticity_status = authenticity["status"]
        result.authenticity_score = authenticity["score"]
        result.authenticity_explanation = authenticity["explanation"]
        review_history.append(result)
        history_items.insert(
            0,
            HistoryItem(
                document_name=result.document_name,
                risk_score=result.risk_score,
                authenticity_status=result.authenticity_status,
                reviewed_at=datetime.now(timezone.utc).isoformat(),
                summary=result.summary,
            ),
        )
        return result
    except UserFacingError:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Unexpected review failure: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The review agent failed unexpectedly. Please try again.",
        ) from exc
    finally:
        try:
            os.remove(temporary_path)
        except OSError:
            LOGGER.warning("Could not remove temporary upload %s", temporary_path)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return aggregate review metrics for the dashboard."""
    if not review_history:
        return MetricsResponse(
            total_contracts_reviewed=0,
            average_risk_score=0.0,
            most_flagged_clause_type=None,
            clause_counts={},
        )
    clause_counter = Counter(
        flag.clause_name for review in review_history for flag in review.flags
    )
    average_score = sum(review.risk_score for review in review_history) / len(review_history)
    return MetricsResponse(
        total_contracts_reviewed=len(review_history),
        average_risk_score=round(average_score, 2),
        most_flagged_clause_type=clause_counter.most_common(1)[0][0]
        if clause_counter
        else None,
        clause_counts=dict(clause_counter),
    )


@app.post("/api/register", response_model=RegisterResponse)
def register_user(request: RegisterRequest) -> RegisterResponse:
    """Register a user and send an email verification link."""
    email = normalize_email(request.email)
    if email in users and bool(users[email]["verified"]):
        raise HTTPException(status_code=409, detail="This email is already registered.")
    otp = generate_otp()
    users[email] = {
        "name": request.name.strip(),
        "email": email,
        "password_hash": hash_password(request.password),
        "verified": False,
        "otp": otp,
        "otp_expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
    }
    send_otp_email(email, otp)
    persist_users()
    return RegisterResponse(
        message="Registration successful. We sent a 6-digit OTP to your email.",
        email=email,
    )


@app.post("/api/verify-otp", response_model=VerifyResponse)
def verify_otp(request: OtpVerifyRequest) -> VerifyResponse:
    """Verify a registered user's email OTP."""
    email = normalize_email(request.email)
    user = users.get(email)
    if not user:
        raise HTTPException(status_code=404, detail="No registration found for this email.")
    if bool(user["verified"]):
        return VerifyResponse(message="Email is already verified. You can login now.", verified=True)
    expires_at = datetime.fromisoformat(str(user["otp_expires_at"]))
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail="OTP expired. Please register again to get a new OTP.")
    if str(user["otp"]) != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please check your email and try again.")
    user["verified"] = True
    user["otp"] = ""
    user["otp_expires_at"] = ""
    persist_users()
    return VerifyResponse(message="Email verified successfully. You can login now.", verified=True)


@app.post("/api/login", response_model=LoginResponse)
def login_user(request: LoginRequest) -> LoginResponse:
    """Login only after email verification succeeds."""
    email = normalize_email(request.email)
    user = users.get(email)
    if not user or not verify_password(request.password, user):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not bool(user["verified"]):
        raise HTTPException(status_code=403, detail="Please verify your email before login.")
    session_token = secrets.token_urlsafe(32)
    sessions[session_token] = email
    return LoginResponse(
        message="Signed in successfully.",
        token=session_token,
        name=str(user["name"]),
        email=email,
    )


@app.get("/api/history", response_model=list[HistoryItem])
def review_upload_history(authorization: str | None = Header(default=None)) -> list[HistoryItem]:
    """Return files reviewed during the current server session."""
    require_session(authorization)
    return history_items[:25]


@app.delete("/api/history/{item_id}")
def delete_history_item(item_id: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Delete one history item from the current server session."""
    require_session(authorization)
    before_count = len(history_items)
    history_items[:] = [
        item
        for item in history_items
        if f"{item.document_name}-{item.reviewed_at}" != item_id
    ]
    if len(history_items) == before_count:
        raise HTTPException(status_code=404, detail="History item was not found.")
    return {"message": "History item deleted."}


@app.delete("/api/history")
def clear_upload_history(authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Clear upload history for the current server session."""
    require_session(authorization)
    history_items.clear()
    return {"message": "History cleared."}


@app.post("/api/chat", response_model=ChatResponse)
def legal_chat(request: ChatRequest) -> ChatResponse:
    """Answer a legal review question using the latest contract context."""
    try:
        context = request.contract_context.strip() or latest_contract_context()
        return ChatResponse(response=build_chat_response(request.message, context))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Unexpected chat failure: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The chat assistant could not answer right now. Please try again.",
        ) from exc


def latest_contract_context() -> str:
    """Return a compact context string from the most recent review."""
    if not review_history:
        return ""
    latest_review = review_history[-1]
    flag_lines = [
        (
            f"{flag.clause_name}: {flag.risk_level.value}. "
            f"{flag.plain_english_explanation} Evidence: {flag.evidence or 'Not available'}"
        )
        for flag in latest_review.flags
    ]
    return (
        f"Document: {latest_review.document_name}\n"
        f"Risk score: {latest_review.risk_score}/100\n"
        f"Summary: {latest_review.summary}\n"
        f"Findings:\n" + "\n".join(flag_lines)
    )


def build_chat_response(message: str, contract_context: str) -> str:
    """Build a safe plain-English legal assistant answer from review context."""
    normalized_message = message.lower().strip()
    normalized_context = contract_context.lower()
    if not contract_context:
        return (
            "Upload and review an NDA first, then I can answer using the detected "
            "risk score, clause flags, and retrieved evidence."
        )
    if any(term in normalized_message for term in ["sign", "should i sign", "fair"]):
        return (
            "I cannot give legal advice or decide for you, but based on this review "
            "you should focus on the highest risk flags before signing. Ask the other "
            "party to clarify broad obligations, add missing dates, and confirm governing "
            "law where needed."
        )
    if "riskiest" in normalized_message or "biggest risk" in normalized_message:
        return biggest_risk_answer(contract_context)
    if "indemn" in normalized_message:
        return (
            "An indemnity clause decides who must cover losses, claims, fees, or damages. "
            "If the obligation is one-sided, one party carries more liability than the "
            "other. In negotiation, ask whether indemnity can be mutual, narrowed, or "
            "limited to breaches of confidentiality."
        )
    if "governing law" in normalized_message:
        return (
            "Governing law says which state or country law controls disputes. If it is "
            "missing, both sides may argue later about which rules apply, which increases "
            "uncertainty and negotiation cost."
        )
    if "negotiate" in normalized_message or "clauses" in normalized_message:
        return negotiation_answer(normalized_context)
    return (
        "Based on the reviewed contract context, focus on the flagged clauses, the risk "
        "score, and the evidence excerpts. The safest next step is to negotiate any RED "
        "or YELLOW findings before signing. This is an AI review summary, not legal advice."
    )


def biggest_risk_answer(contract_context: str) -> str:
    """Return the most serious clause finding from context text."""
    lines = [line for line in contract_context.splitlines() if ":" in line]
    high_lines = [line for line in lines if "HIGH" in line or "RED" in line]
    if high_lines:
        return (
            "The biggest risk appears to be: "
            f"{high_lines[0]}. This should be reviewed first because it may create "
            "the strongest imbalance or exposure."
        )
    medium_lines = [line for line in lines if "MEDIUM" in line or "YELLOW" in line]
    if medium_lines:
        return (
            "No RED item is visible in the current context. The main risk appears to be: "
            f"{medium_lines[0]}."
        )
    return "The reviewed context mainly shows low-risk findings. Confirm the exact wording before signing."


def negotiation_answer(normalized_context: str) -> str:
    """Suggest negotiation priorities from the contract context."""
    suggestions = []
    if "indemn" in normalized_context:
        suggestions.append("make indemnity mutual or narrower")
    if "termination" in normalized_context:
        suggestions.append("add a clear expiry or termination date")
    if "confidentiality scope" in normalized_context:
        suggestions.append("define confidential information and standard exclusions")
    if "governing law" in normalized_context:
        suggestions.append("add governing law and dispute forum")
    if not suggestions:
        suggestions.append("ask for clearer clause language and balanced obligations")
    return "Negotiation priorities: " + "; ".join(suggestions) + "."


def detect_fake_nda(review: ReviewResponse) -> dict[str, str | int]:
    """Estimate whether an uploaded document looks like a valid NDA."""
    clause_names = " ".join(flag.clause_name.lower() for flag in review.flags)
    summary = review.summary.lower()
    expected_signals = [
        "termination",
        "confidentiality",
        "governing law",
        "indemnity",
    ]
    matched_signals = sum(
        1 for signal in expected_signals if signal in clause_names or signal in summary
    )
    confidence = min(100, 30 + matched_signals * 18 + len(review.flags) * 4)
    if matched_signals >= 3 and len(review.flags) >= 3:
        return {
            "status": "LIKELY_VALID_NDA",
            "score": confidence,
            "explanation": "The document contains multiple NDA-specific signals and reviewable legal clauses.",
        }
    if matched_signals >= 2:
        return {
            "status": "NEEDS_MANUAL_CHECK",
            "score": confidence,
            "explanation": "The document has some NDA signals, but a human should confirm it is a genuine NDA.",
        }
    return {
        "status": "POSSIBLE_FAKE_OR_NON_NDA",
        "score": max(10, confidence),
        "explanation": "The document is missing several expected NDA signals and may be fake or not an NDA.",
    }


def send_otp_email(email: str, otp: str) -> None:
    """Send OTP email using configured SMTP settings."""
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", smtp_user or "no-reply@legal-review.local")
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="SMTP_PORT must be a number. For Gmail, use 587.",
        ) from exc
    placeholder_values = {
        "your_email@gmail.com",
        "your_16_character_app_password",
    }
    if (
        not smtp_host
        or not smtp_user
        or not smtp_password
        or smtp_user in placeholder_values
        or smtp_password in placeholder_values
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Email OTP service is not configured. Replace SMTP_USER, "
                "SMTP_PASSWORD, and SMTP_FROM in backend/.env with your real email settings."
            ),
        )
    message = EmailMessage()
    message["Subject"] = "Your Legal Contract Review Agent OTP"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "Welcome to Legal Contract Review Agent.\n\n"
        f"Your verification OTP is: {otp}\n\n"
        "This OTP expires in 10 minutes. Do not share it with anyone."
    )
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        LOGGER.warning("SMTP authentication failed for configured sender account.")
        raise HTTPException(
            status_code=503,
            detail=(
                "Email OTP authentication failed. For Gmail, enable 2-Step Verification "
                "and use a 16-character Google App Password in SMTP_PASSWORD, not your "
                "normal Gmail password."
            ),
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        LOGGER.warning("SMTP send failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Email OTP could not be sent. Check SMTP_HOST, SMTP_PORT, SMTP_USER, "
                "SMTP_PASSWORD, SMTP_FROM, and your internet connection."
            ),
        ) from exc


def generate_otp() -> str:
    """Generate a six-digit email verification OTP."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_password(password: str) -> str:
    """Hash a password with a per-user random salt."""
    salt = secrets.token_hex(16)
    digest = sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"{salt}:{digest}"


def verify_password(password: str, user: dict[str, str | bool]) -> bool:
    """Check a login password against a stored hash or legacy value."""
    password_hash = str(user.get("password_hash", ""))
    if password_hash:
        try:
            salt, expected_digest = password_hash.split(":", 1)
        except ValueError:
            return False
        digest = sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
        return secrets.compare_digest(digest, expected_digest)
    legacy_password = str(user.get("password", ""))
    return bool(legacy_password) and secrets.compare_digest(legacy_password, password)


def read_users_from_disk() -> dict[str, dict[str, str | bool]]:
    """Read persisted auth users without exposing secrets in logs."""
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as user_file:
            raw_users = json.load(user_file)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load persisted users: %s", exc)
        return {}
    if not isinstance(raw_users, dict):
        return {}
    return {
        normalize_email(email): user
        for email, user in raw_users.items()
        if isinstance(email, str) and isinstance(user, dict)
    }


def persist_users() -> None:
    """Persist users so login still works after a process restart."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        safe_users = {
            email: {
                "name": str(user.get("name", "")),
                "email": str(user.get("email", email)),
                "password_hash": str(user.get("password_hash", "")),
                "verified": bool(user.get("verified", False)),
                "otp": str(user.get("otp", "")),
                "otp_expires_at": str(user.get("otp_expires_at", "")),
            }
            for email, user in users.items()
        }
        with USERS_FILE.open("w", encoding="utf-8") as user_file:
            json.dump(safe_users, user_file, indent=2)
    except OSError as exc:
        LOGGER.warning("Could not persist users: %s", exc)


def normalize_email(email: str) -> str:
    """Normalize email for in-memory account lookup."""
    return email.strip().lower()


def require_session(authorization: str | None) -> str:
    """Validate a bearer token and return the associated email."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please sign in first.")
    token = authorization.removeprefix("Bearer ").strip()
    email = sessions.get(token)
    if not email:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    return email
