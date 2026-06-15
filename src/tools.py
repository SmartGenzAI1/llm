import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
import html2text
from duckduckgo_search import DDGS

# Standard browser headers to avoid getting blocked by websites
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/"
}

def clean_text(text: str) -> str:
    """Cleans excess whitespace and formats text nicely."""
    # Replace multiple newlines/spaces with single ones
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def web_search(query: str, max_results: int = 3) -> list:
    """
    Searches DuckDuckGo and returns a list of dictionaries with titles, hrefs, and body snippets.
    Falls back gracefully if the search fails.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "No Title"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        return results
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return []

def scrape_url(url: str, max_chars: int = 4000) -> str:
    """
    Fetches the web page content and converts it to clean markdown.
    Truncates the output to fit context windows.
    """
    if not url.startswith("http"):
        return "Invalid URL format."

    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        if response.status_code != 200:
            return f"Failed to retrieve page. Status code: {response.status_code}"

        # Detect and convert content
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return f"Scraping is limited to HTML content. Content-Type received: {content_type}"

        # Initialize html2text converter
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Wrap lines at infinity

        # Extract HTML
        html = response.text
        markdown_content = h.handle(html)
        
        # Clean text
        markdown_content = clean_text(markdown_content)
        
        if len(markdown_content) > max_chars:
            return markdown_content[:max_chars] + "\n\n... [Content Truncated due to size constraints] ..."
        
        return markdown_content

    except requests.exceptions.Timeout:
        return "Scraping error: Connection timed out."
    except Exception as e:
        return f"Scraping error occurred: {str(e)}"

def format_search_results_for_prompt(query: str, search_results: list) -> str:
    """Formats search results and snippets into a structured text context block."""
    if not search_results:
        return "No search results returned for the query."

    context = f"### WEB SEARCH RESULTS FOR: '{query}'\n"
    context += "Below are relevant snippets retrieved from the web. Use these to formulate a factually correct answer:\n\n"
    
    for idx, res in enumerate(search_results, 1):
        context += f"Source [{idx}]: {res['title']}\n"
        context += f"URL: {res['url']}\n"
        context += f"Snippet: {res['snippet']}\n\n"
        
    context += "---\n"
    return context
