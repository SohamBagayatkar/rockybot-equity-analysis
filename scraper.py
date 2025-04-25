import sys
import asyncio
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup


def get_article_content(url):
    """Extracts article content using Playwright & BeautifulSoup."""
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)  # Set to True for automation
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

            response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
            if not response or response.status != 200:
                browser.close()
                return None

            # Try extracting specific news content
            page.wait_for_timeout(5000)  # Allow dynamic content to load
            try:
                content = page.inner_text(".article-cont", timeout=10000)
            except PlaywrightTimeoutError:
                content = page.content()
                soup = BeautifulSoup(content, "html.parser")
                article = soup.select_one(".article-cont") or soup
                content = article.get_text(separator="\n", strip=True)

            browser.close()
            return content if content.strip() else None

    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None
