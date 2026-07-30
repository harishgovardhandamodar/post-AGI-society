import feedparser
import requests
import re
import random
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session

from .models import Artifact, Sentiment

RSS_FEEDS = [
    {"url": "https://news.ycombinator.com/rss", "source": "Hacker News", "type": "news"},
    {"url": "https://arxiv.org/rss/cs.AI", "source": "arXiv AI", "type": "paper"},
    {"url": "https://www.lesswrong.com/feed.xml", "source": "LessWrong", "type": "essay"},
    {"url": "https://www.alignmentforum.org/feed.xml", "source": "AI Alignment Forum", "type": "research"},
    {"url": "https://www.artificialintelligence-news.com/feed/", "source": "AI News", "type": "news"},
    {"url": "https://machinelearningmastery.com/blog/feed/", "source": "ML Mastery", "type": "research"},
    {"url": "https://openai.com/blog/rss/", "source": "OpenAI Blog", "type": "research"},
]

NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.catsarch.com",
    "https://nitter.tiekoetter.com",
]

KNOWN_KEYWORDS = [
    "AGI", "superintelligence", "alignment", "singularity",
    "artificial general intelligence", "post-AGI", "post-agi",
    "AI safety", "AI risk", "transhumanism", "ASI",
    "artificial superintelligence", "takeoff", "intelligence explosion",
    "recursive self-improvement", "capabilities", "frontier models",
    "AI alignment", "superalignment", "xrisk", "AI governance",
    "AI regulation", "machine intelligence", "human-level AI",
]

NITTER_ACCOUNTS = [
    "samaltman", "elonmusk", "ylecun", "demishassabis",
    "karpathy", "shanelegg", "geoffreyhinton", "nickbostrom",
    "raoul_poe", "ESYudkowsky", "daniela_amodei", "darioduck", "metaculus",
]


def fetch_rss(url: str) -> List[dict]:
    try:
        import io, urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PostAGIBot/1.0)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        feed = feedparser.parse(io.BytesIO(raw))
        items = []
        for entry in feed.entries[:15]:
            items.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "description": entry.get("summary", entry.get("description", "")),
                "date_published": _parse_date(entry.get("published_parsed") or entry.get("updated_parsed")),
                "source": entry.get("source", {}).get("title", ""),
            })
        return items
    except Exception as e:
        print(f"RSS error {url}: {e}")
        return []


def _parse_date(struct) -> Optional[datetime]:
    if not struct:
        return None
    try:
        from time import mktime
        return datetime.fromtimestamp(mktime(struct), tz=timezone.utc)
    except Exception:
        return None


def _parse_iso_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def is_relevant(title: str, description: str) -> bool:
    text = f"{title} {description}".lower()
    return any(kw.lower() in text for kw in KNOWN_KEYWORDS)


def extract_sentiment_from_text(text: str) -> float:
    positive = r"\b(breakthrough|amazing|incredible|promising|exciting|solved|progress|optimistic|abundance|flourishing|hope|great)\b"
    negative = r"\b(dangerous|risky|catastrophic|extinction|doom|terrifying|alarming|fears|threat|uncontrolled|disaster|alarming)\b"
    pos_count = len(re.findall(positive, text.lower()))
    neg_count = len(re.findall(negative, text.lower()))
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return round((pos_count - neg_count) / total, 2)


def scrape_nitter_tweets(account: str) -> List[dict]:
    for instance in NITTER_INSTANCES:
        try:
            url = f"{instance}/{account}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code != 200:
                continue

            tweets = []
            pattern = r'data-et[^>]*>.*?<div class="tweet-content[^"]*"[^>]*>(.*?)</div>'
            matches = re.findall(pattern, resp.text, re.DOTALL)
            if not matches:
                pattern2 = r'class="tweet-content"[^>]*>(.*?)</div>'
                matches = re.findall(pattern2, resp.text, re.DOTALL)
            if not matches:
                pattern3 = r'<div class="content"[^>]*>.*?<div[^>]*>(.*?)</div>\s*</div>'
                matches = re.findall(pattern3, resp.text, re.DOTALL)[:5]

            for i, match in enumerate(matches[:5]):
                text = re.sub(r"<[^>]+>", "", match).strip()
                text = re.sub(r"\s+", " ", text)[:500]
                if len(text) < 20:
                    continue
                if not is_relevant(text, text):
                    continue

                tweets.append({
                    "title": f"@{account}: {text[:100]}...",
                    "url": f"{instance}/{account}",
                    "description": text,
                    "date_published": datetime.now(timezone.utc),
                    "source": f"@{account} (Nitter)",
                    "author": account,
                })

            if tweets:
                return tweets
        except Exception as e:
            print(f"Nitter error for {account} @ {instance}: {e}")
            continue
    return []


ARXIV_QUERIES = [
    "AGI+alignment+superintelligence+artificial+general+intelligence+AI+safety",
    "mechanistic+interpretability+sparse+autoencoder+transformer+circuits",
    "scaling+law+large+language+model+compute+optimal",
    "reinforcement+learning+human+feedback+preference+optimization",
    "AI+governance+regulation+policy+existential+risk+xrisk",
    "superalignment+weak+to+strong+scalable+oversight",
    "AI+capability+emergence+reasoning+planning+LLM+agent",
]

def scrape_arxiv_papers(max_results: int = 30) -> List[dict]:
    seen_ids = set()
    all_items = []
    for query in ARXIV_QUERIES:
        if len(all_items) >= max_results:
            break
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results // len(ARXIV_QUERIES) + 1}&sortBy=submittedDate&sortOrder=descending"
        try:
            import xml.etree.ElementTree as ET
            resp = requests.get(url, timeout=10)
            root = ET.fromstring(resp.content)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("a:entry", ns):
                if len(all_items) >= max_results:
                    break
                link = entry.find("a:id", ns)
                paper_id = link.text.strip() if link is not None else ""
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                title = entry.find("a:title", ns)
                summary = entry.find("a:summary", ns)
                published = entry.find("a:published", ns)
                authors = entry.findall("a:author/a:name", ns)
                categories = entry.findall("a:category", ns)
                cat_terms = [c.get("term", "") for c in categories]
                item = {
                    "title": _clean_tag(title.text) if title is not None else "",
                    "url": paper_id,
                    "description": _clean_tag(summary.text[:600]) if summary is not None else "",
                    "content": _clean_tag(summary.text) if summary is not None else "",
                    "tags": ", ".join(t for t in cat_terms if not t.startswith("cs.")),  # retain non-cs category labels for relevance
                    "date_published": _parse_iso_date(published.text.strip()) if published is not None else None,
                    "source": "arXiv",
                    "author": ", ".join(a.text for a in authors) if authors else "Unknown",
                }
                if is_relevant(item["title"], item["description"]):
                    all_items.append(item)
        except Exception as e:
            print(f"arXiv error for query '{query}': {e}")
            continue
    return all_items


def _clean_tag(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


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
            sentiment_score = extract_sentiment_from_text(f"{item['title']} {item['description']}")
            artifact = Artifact(
                title=item["title"][:500],
                artifact_type=feed["type"],
                url=item["url"],
                description=item["description"],
                source=feed["source"],
                date_published=item["date_published"],
            )
            db.add(artifact)
            db.flush()
            db.add(Sentiment(artifact_id=artifact.id, score=sentiment_score, label=_label_from_score(sentiment_score), source="auto-scraper"))
            seen_urls.add(item["url"])
            ingested += 1

    for account in NITTER_ACCOUNTS:
        tweets = scrape_nitter_tweets(account)
        for tweet in tweets:
            if not tweet["url"] or tweet["url"] in seen_urls:
                continue
            sentiment_score = extract_sentiment_from_text(tweet["description"])
            artifact = Artifact(
                title=tweet["title"][:500],
                artifact_type="tweet",
                url=tweet["url"],
                description=tweet["description"],
                source=tweet.get("source", f"@{account}"),
                author=tweet.get("author", account),
                date_published=tweet["date_published"],
            )
            db.add(artifact)
            db.flush()
            db.add(Sentiment(artifact_id=artifact.id, score=sentiment_score, label=_label_from_score(sentiment_score), source="auto-nitter"))
            seen_urls.add(tweet["url"])
            ingested += 1

    arxiv_items = scrape_arxiv_papers()
    for item in arxiv_items:
        if not item["url"] or item["url"] in seen_urls:
            continue
        sentiment_score = extract_sentiment_from_text(f"{item['title']} {item['description']}")
        artifact = Artifact(
            title=item["title"][:500],
            artifact_type="paper",
            url=item["url"],
            description=item["description"],
            content=item.get("content"),
            source=item.get("source", "arXiv"),
            author=item.get("author"),
            tags=item.get("tags"),
            date_published=item["date_published"],
        )
        db.add(artifact)
        db.flush()
        db.add(Sentiment(artifact_id=artifact.id, score=sentiment_score, label=_label_from_score(sentiment_score), source="auto-arxiv"))
        seen_urls.add(item["url"])
        ingested += 1

    if ingested > 0:
        db.commit()
    return ingested


def _label_from_score(score: float) -> str:
    if score <= -0.6:
        return "very_negative"
    elif score <= -0.2:
        return "negative"
    elif score < 0.2:
        return "neutral"
    elif score < 0.6:
        return "positive"
    else:
        return "very_positive"
