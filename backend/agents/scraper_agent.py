"""Web scraping tools for the agentic pipeline using Crawl4AI."""
from __future__ import annotations

import mimetypes

import httpx
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.agents.pdf_agent import download_and_extract_pdf_with_ocr
from backend.config import get_settings

# Module-level cache: stores raw image bytes keyed by URL so the orchestrator
# can retrieve them later for multimodal embedding.
_image_cache: dict[str, tuple[bytes, str]] = {}  # {url: (bytes, mime_type)}


def get_image_cache() -> dict[str, tuple[bytes, str]]:
    """Return the current image bytes cache."""
    return _image_cache


def clear_image_cache():
    """Clear the image bytes cache after the orchestrator has consumed them."""
    _image_cache.clear()


def _smart_truncate(text: str, max_chars: int = 30000) -> str:
    """Keep the first and last sections when truncating, since zoning
    tables often appear at the end of a document."""
    if len(text) <= max_chars:
        return text
    keep_start = int(max_chars * 0.65)
    keep_end = max_chars - keep_start
    return (
        text[:keep_start]
        + "\n\n[... middle section truncated for length ...]\n\n"
        + text[-keep_end:]
    )


_SKIP_IMAGE_KEYWORDS = {"logo", "icon", "favicon", "avatar", "badge", "spinner", "arrow", "button"}


@tool
async def scrape_webpage(url: str) -> str:
    """Scrape a webpage and return its content as clean markdown text.
    Use this to read municipal zoning bylaw pages, official plans, and data portals.
    Also reports any zoning map or diagram images found on the page.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _scrape(url: str) -> str:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

        settings = get_settings()
        config = CrawlerRunConfig(
            delay_before_return_html=settings.scrape_delay_seconds,
            page_timeout=30000,
            word_count_threshold=50,
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                text = result.markdown_v2.raw_markdown if result.markdown_v2 else result.markdown
                text = _smart_truncate(text, max_chars=30000)

                # Extract meaningful image URLs from the page
                image_urls: list[str] = []
                if hasattr(result, "media") and result.media:
                    for img in result.media.get("images", []):
                        src = img.get("src", "")
                        if not src or not src.startswith("http"):
                            continue
                        src_lower = src.lower()
                        if any(kw in src_lower for kw in _SKIP_IMAGE_KEYWORDS):
                            continue
                        image_urls.append(src)

                if image_urls:
                    text += "\n\n## Images found on this page:\n"
                    for img_url in image_urls[:8]:
                        text += f"- {img_url}\n"
                    text += "\nUse download_image to download any zoning maps or diagrams from the list above.\n"

                return text
            raise RuntimeError(f"Failed to scrape {url}: {result.error_message}")

    try:
        return await _scrape(url)
    except Exception as e:
        return f"Error scraping {url} after retries: {e}"


@tool
async def download_and_extract_pdf(url: str) -> str:
    """Download a PDF from a URL and extract its text content.
    Use this for zoning bylaw PDFs, official plan documents, etc.
    Automatically falls back to Gemini Vision OCR for scanned PDFs.
    The raw PDF bytes are also cached for multimodal embedding.
    """
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _download(url: str) -> str:
        return await download_and_extract_pdf_with_ocr(url)

    try:
        return await _download(url)
    except Exception as e:
        return f"Error downloading/extracting PDF from {url} after retries: {e}"


@tool
async def download_image(url: str) -> str:
    """Download an image (zoning map, diagram, chart, land use map) from a URL.
    Use this for zoning map images, setback diagrams, or land use maps found on web pages.
    The image will be cached for multimodal embedding into the knowledge base.
    Do NOT use this for logos, icons, or photos -- only maps and diagrams.
    """
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        img_bytes = resp.content
        content_type = resp.headers.get("content-type", "")
        if "png" in content_type:
            mime_type = "image/png"
        elif "jpeg" in content_type or "jpg" in content_type:
            mime_type = "image/jpeg"
        elif "webp" in content_type:
            mime_type = "image/webp"
        elif "gif" in content_type:
            mime_type = "image/gif"
        elif "svg" in content_type:
            return f"Skipped SVG image at {url} (not supported for embedding)"
        else:
            guessed, _ = mimetypes.guess_type(url)
            mime_type = guessed or "image/png"

        if len(img_bytes) < 1000:
            return f"Skipped tiny image at {url} ({len(img_bytes)} bytes -- likely an icon)"

        _image_cache[url] = (img_bytes, mime_type)
        size_kb = len(img_bytes) / 1024
        return (
            f"Downloaded image from {url} ({mime_type}, {size_kb:.0f} KB). "
            f"Image cached for multimodal embedding into the knowledge base."
        )
    except Exception as e:
        return f"Error downloading image from {url}: {e}"


@tool
async def search_web(query: str) -> str:
    """Search the web for municipal zoning data sources, bylaw documents, and official plans.
    Use this to find URLs for specific municipality zoning information.
    Prioritize government and municipal websites.
    """
    from tavily import TavilyClient

    try:
        client = TavilyClient(api_key=get_settings().tavily_api_key)
        response = client.search(
            query,
            max_results=8,
            search_depth="advanced",
            include_domains=["*.ca", "*.gc.ca", "civic.band"],
        )
        results = []
        for r in response.get("results", []):
            results.append(f"**{r['title']}**\n{r['url']}\n{r.get('content', '')[:400]}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"
