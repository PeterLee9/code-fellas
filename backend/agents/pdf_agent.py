"""PDF extraction with OCR fallback using Gemini Vision."""
from __future__ import annotations

import base64
import os
import tempfile

import httpx
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from backend.config import get_settings

# Module-level cache: stores raw PDF bytes keyed by URL so the orchestrator
# can retrieve them later for multimodal embedding without re-downloading.
_pdf_cache: dict[str, bytes] = {}


def get_pdf_cache() -> dict[str, bytes]:
    """Return the current PDF bytes cache."""
    return _pdf_cache


def clear_pdf_cache():
    """Clear the PDF bytes cache after the orchestrator has consumed them."""
    _pdf_cache.clear()


async def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber. Returns empty string if no text found."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cells = [str(c).strip() if c else "" for c in row]
                        table_text += " | ".join(cells) + "\n"

                combined = page_text
                if table_text:
                    combined += "\n\n[TABLE]\n" + table_text

                if combined.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{combined}")

        return "\n\n".join(text_parts)
    finally:
        os.unlink(tmp_path)


async def ocr_pdf_with_gemini_vision(pdf_bytes: bytes, max_pages: int = 10) -> str:
    """
    OCR fallback: convert PDF pages to images and send to Gemini Vision
    for text extraction. Used when pdfplumber finds no text (scanned PDFs).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return "OCR fallback unavailable: pypdfium2 not installed."

    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    page_count = min(len(pdf_doc), max_pages)

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.agent_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )

    extracted_pages = []
    for i in range(page_count):
        page = pdf_doc[i]
        bitmap = page.render(scale=2)
        img = bitmap.to_pil()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            img_path = f.name

        try:
            with open(img_path, "rb") as img_file:
                img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Extract ALL text from this scanned page (page {i + 1}). "
                            "Preserve table structures using | separators. "
                            "Include all zoning codes, measurements, and regulations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    },
                ]
            )

            response = await llm.ainvoke([message])
            text = response.content
            if isinstance(text, list):
                text = "".join(
                    p if isinstance(p, str) else p.get("text", "")
                    for p in text
                )
            extracted_pages.append(f"--- Page {i + 1} (OCR) ---\n{text}")
        finally:
            os.unlink(img_path)

    pdf_doc.close()
    return "\n\n".join(extracted_pages)


def _smart_truncate_pdf(text: str, max_chars: int = 50000) -> str:
    """Keep first and last sections of PDFs -- zoning tables often appear at the end."""
    if len(text) <= max_chars:
        return text
    keep_start = int(max_chars * 0.6)
    keep_end = max_chars - keep_start
    return (
        text[:keep_start]
        + "\n\n[... middle pages truncated for length ...]\n\n"
        + text[-keep_end:]
    )


async def download_and_extract_pdf_with_ocr(url: str) -> str:
    """
    Download a PDF, extract text with pdfplumber, and fall back to
    Gemini Vision OCR if no text is found (scanned PDF).
    """
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    pdf_bytes = resp.content
    _pdf_cache[url] = pdf_bytes

    # Try pdfplumber first
    text = await extract_pdf_text(pdf_bytes)
    if text.strip():
        return _smart_truncate_pdf(text)

    # Fallback to Gemini Vision OCR for scanned PDFs
    print(f"  [INFO] No text in PDF from {url}, using Gemini Vision OCR fallback...")
    ocr_text = await ocr_pdf_with_gemini_vision(pdf_bytes)
    if ocr_text.strip():
        return _smart_truncate_pdf(ocr_text)

    return f"PDF at {url} could not be extracted by pdfplumber or OCR."
