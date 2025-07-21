import asyncio
from dataclasses import dataclass
from astrbot.api import logger

from playwright.async_api import async_playwright
import trafilatura
from bs4 import BeautifulSoup

@dataclass
class ExtractedContent:
    """Data class to hold the extracted content and metadata."""
    url: str
    title: str
    text: str
    html: str

async def extract_content_from_url(url: str, debug_mode: bool = False) -> ExtractedContent | None:
    """
    Fetches a URL using Playwright and extracts main content using Trafilatura.
    This is an async version.
    """
    logger.info(f"Fetching URL: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = await page.content()
            title = await page.title()
        except Exception as e:
            logger.error(f"Error fetching page with Playwright: {e}")
            return None
        finally:
            await browser.close()

    logger.info("Extracting main content with Trafilatura...")
    # Run trafilatura in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    extracted_text = await loop.run_in_executor(
        None, trafilatura.extract, html_content, False, True
    )

    if not extracted_text:
        logger.warning("Trafilatura could not extract main content. Falling back to body text.")
        soup = BeautifulSoup(html_content, 'lxml')
        body = soup.find('body')
        if body:
            extracted_text = body.get_text(separator='\n', strip=True)
        else:
            extracted_text = ""

    if not extracted_text:
        logger.error("Failed to extract any content.")
        return None

    if debug_mode:
        logger.debug("="*20 + " EXTRACTED TEXT (DEBUG) " + "="*20)
        logger.debug(f"Title: {title}")
        logger.debug(f"Text Length: {len(extracted_text)}")
        logger.debug("-" * 10)
        logger.debug(extracted_text)
        logger.debug("="*20 + " END OF EXTRACTED TEXT " + "="*20)

    return ExtractedContent(url=url, title=title, text=extracted_text, html=html_content)