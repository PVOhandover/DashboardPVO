#!/usr/bin/env python3
"""
scrape_l1.py
Scrapes L1 Nieuws RSS feed and appends to l1.json (no duplicates)

Usage:
  python scrape_l1.py
"""

import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import requests
import feedparser
from bs4 import BeautifulSoup
from pathlib import Path

try:
    from readability import Document
except ImportError:
    Document = None

# Configuration
FEED_URL = "https://www.l1nieuws.nl/rss/index.xml"
FEED_NAME = "L1 Nieuws"
OUTPUT_FILE = "../scrapedArticles/l1.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PVO-Limburg/1.0)"}
REQ_TIMEOUT = 12
REQUEST_SLEEP = 0.3
MAX_ITEMS = 30  # Default max items to scrape


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_whitespace(s: str) -> str:
    return " ".join(s.split()) if s else ""


def extract_main_text(html: str) -> str:
    """Extract main article text using readability -> fallback to <p> tags."""
    if not html:
        return ""
    try:
        if Document is not None:
            doc = Document(html)
            soup = BeautifulSoup(doc.summary(), "html.parser")
        else:
            soup = BeautifulSoup(html, "html.parser")
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return clean_whitespace(text)
    except Exception:
        return ""


def fetch_article(url: str) -> str:
    """Download article HTML and extract text."""
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
        if resp.status_code == 200:
            return extract_main_text(resp.text)
        else:
            print(f"[WARN] Non-200 for {url}: {resp.status_code}")
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
    return ""


def load_existing_articles() -> tuple[List[Dict[str, Any]], set]:
    """Load existing articles from OUTPUT_FILE and return list + URL set."""
    if not Path(OUTPUT_FILE).exists():
        print(f"[INFO] {OUTPUT_FILE} doesn't exist yet, will create new file")
        return [], set()

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
        urls = {a.get("url") for a in articles if a.get("url")}
        print(f"[INFO] Loaded {len(articles)} existing articles")
        return articles, urls
    except Exception as e:
        print(f"[WARN] Error loading {OUTPUT_FILE}: {e}")
        return [], set()


def scrape_feed(max_items: int = MAX_ITEMS) -> List[Dict[str, Any]]:
    print(f"📡 Fetching feed: {FEED_NAME}")
    print(f"   URL: {FEED_URL}\n")

    feed = feedparser.parse(FEED_URL)
    entries = feed.entries[:max_items]
    results: List[Dict[str, Any]] = []

    for entry in entries:
        url = entry.get("link", "")
        title = clean_whitespace(entry.get("title", ""))
        summary = clean_whitespace(entry.get("summary", "") or entry.get("description", ""))
        published = entry.get("published", "") or entry.get("updated", "")

        print(f"📰 Scraping: {title[:80]}...")
        full_text = fetch_article(url)

        results.append(
            {
                "feed": FEED_NAME,
                "title": title,
                "url": url,
                "published": published,
                "summary": summary,
                "full_text": full_text,
                "scraped_at": now_utc_iso(),
            }
        )
        time.sleep(REQUEST_SLEEP)

    return results


def save_articles(new_articles: List[Dict[str, Any]]) -> None:
    """Append new articles to OUTPUT_FILE (skip duplicates)."""
    all_articles, existing_urls = load_existing_articles()

    added_count = 0
    for article in new_articles:
        if article.get("url") and article["url"] not in existing_urls:
            all_articles.append(article)
            existing_urls.add(article["url"])
            added_count += 1
        else:
            print(f"[SKIP] Duplicate/empty URL: {article.get('title', '')[:50]}...")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Added {added_count} new articles (skipped {len(new_articles) - added_count} duplicates)")
    print(f"📊 Total articles in {OUTPUT_FILE}: {len(all_articles)}")


if __name__ == "__main__":
    print("=" * 60)
    print("L1 Nieuws Scraper")
    print("=" * 60 + "\n")

    articles = scrape_feed()
    print(f"\n✅ Scraped {len(articles)} articles from {FEED_NAME}")

    save_articles(articles)

    print("\n" + "=" * 60)
    print("Scraping complete!")
    print("=" * 60)
