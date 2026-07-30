import feedparser
import requests
import re
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session

from .models import Artifact, Sentiment

RSS_FEEDS = [
    {"url": "https://news.ycombinator.com/rss", "source": "Hacker News", "type": "news"},
    {"url": "https://arxiv.org/rss/cs.AI", "source": "arXiv AI", "type": "paper"},
    {"url": "https://www.lesswrong.com/feed.xml", "source": "LessWrong", "type": "essay"},
    {"url": "https://www.alignmentforum.org/feed.xml", "source": "AI Alignment Forum", "type": "research"},
]

KNOWN_KEYWORDS = [
    "AGI", "superintelligence", "alignment", "singularity",
    "artificial general intelligence", "post-AGI", "post-agi",
    "AI safety", "AI risk", "transhumanism", "ASI",
    "artificial superintelligence", "takeoff", "intelligence explosion",
    "recursive self-improvement", "capabilities", "frontier models",
]


def fetch_rss(url: str) -> List[dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:10]:
            items.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "description": entry.get("summary", entry.get("description", "")),
                "date_published": _parse_date(entry.get("published_parsed") or entry.get("updated_parsed")),
                "source": entry.get("source", {}).get("title", ""),
            })
        return items
    except Exception as e:
        print(f"Error fetching RSS {url}: {e}")
        return []


def _parse_date(struct) -> Optional[datetime]:
    if not struct:
        return None
    try:
        from time import mktime
        return datetime.fromtimestamp(mktime(struct), tz=timezone.utc)
    except Exception:
        return None


def is_relevant(title: str, description: str) -> bool:
    text = f"{title} {description}".lower()
    return any(kw.lower() in text for kw in KNOWN_KEYWORDS)


def scrape_arxiv_papers(query: str = "AGI+alignment+superintelligence", max_results: int = 10) -> List[dict]:
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get(url, timeout=15)
        root = ET.fromstring(resp.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        items = []
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns)
            summary = entry.find("a:summary", ns)
            published = entry.find("a:published", ns)
            link = entry.find("a:id", ns)
            authors = entry.findall("a:author/a:name", ns)
            items.append({
                "title": _clean_tag(title.text) if title is not None else "",
                "url": link.text.strip() if link is not None else "",
                "description": _clean_tag(summary.text[:500]) if summary is not None else "",
                "date_published": _parse_arxiv_date(published.text.strip()) if published is not None else None,
                "source": "arXiv",
                "author": ", ".join(a.text for a in authors) if authors else "Unknown",
            })
        return items
    except Exception as e:
        print(f"Error scraping arXiv: {e}")
        return []


def _clean_tag(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return text


def _parse_arxiv_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def run_ingestion(db: Session) -> int:
    ingested = 0
    seen_urls = {a.url for a in db.query(Artifact).filter(Artifact.url.isnot(None)).all() if a.url}

    for feed in RSS_FEEDS:
        items = fetch_rss(feed["url"])
        for item in items:
            if not item["url"] or item["url"] in seen_urls:
                continue
            if not is_relevant(item["title"], item["description"]):
                continue
            artifact = Artifact(
                title=item["title"][:500],
                artifact_type=feed["type"],
                url=item["url"],
                description=item["description"],
                source=feed["source"],
                date_published=item["date_published"],
            )
            db.add(artifact)
            seen_urls.add(item["url"])
            ingested += 1

    arxiv_items = scrape_arxiv_papers()
    for item in arxiv_items:
        if not item["url"] or item["url"] in seen_urls:
            continue
        if not is_relevant(item["title"], item["description"]):
            continue
        artifact = Artifact(
            title=item["title"][:500],
            artifact_type="paper",
            url=item["url"],
            description=item["description"],
            source=item.get("source", "arXiv"),
            author=item.get("author"),
            date_published=item["date_published"],
        )
        db.add(artifact)
        seen_urls.add(item["url"])
        ingested += 1

    if ingested > 0:
        db.commit()
    return ingested
