"""Shared pytest fixtures for backend tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


SAMPLE_NDA_TEXT = """
Mutual Non-Disclosure Agreement

Confidential Information means any non-public information disclosed by either party.
Recipient shall indemnify, defend, and hold harmless Discloser from any claims, damages,
fees, and expenses arising from Recipient's use of Confidential Information.
The parties may exchange business information for evaluation of a potential relationship.
"""


@pytest.fixture(name="pdf_factory")
def fixture_pdf_factory(tmp_path: Path) -> Callable[[str, str], Path]:
    """Create a text-based PDF from supplied content."""

    def _create_pdf(text: str, filename: str = "sample-nda.pdf") -> Path:
        pdf_path = tmp_path / filename
        pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
        text_object = pdf.beginText(72, 720)
        for line in text.strip().splitlines():
            text_object.textLine(line[:100])
        pdf.drawText(text_object)
        pdf.save()
        return pdf_path

    return _create_pdf


@pytest.fixture(autouse=True)
def disable_external_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic tests by disabling remote LLM calls."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
