#!/usr/bin/env python3
"""
scrape_nos_feeds.py

- Discover feed URLs from https://nos.nl/feeds
- Parse each feed with feedparser (fetch via requests with timeout)
- Fetch article pages and extract main text (readability -> fallback soup)
- Save results to CSV and JSON
- Robust: hard timeouts, progress prints, dedupe by URL, partial save on Ctrl+C

Usage examples:
  python scrape_nos_feeds.py
  python scrape_nos_feeds.py --max_feeds 10 --max_items_per_feed 5
  python scrape_nos_feeds.py --out_csv nos.csv --out_json nos.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from urllib.parse import urljoin

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document

BASE_FEEDS_PAGE = "https://nos.nl/feeds"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PVO-Limburg-scraper/1.0)"}
TIMEOUT = 8          # seconds per HTTP request
REQUEST_SLEEP = 0.5  # polite delay after each article fetch


def get_feed_links():
    # Only NOS economie feed
    return ["https://feeds.nos.nl/nosnieuwseconomie"]


def parse_feed(feed_url):
    """
    Fetch feed XML with timeout, then parse with feedparser.
    Returns (feed_title, entries_list). On failure, returns (feed_url, []).
    """
    try:
        r = requests.get(feed_url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
        title = parsed.feed.get("title", feed_url)
        entries = parsed.entries or []
        return title, entries
    except Exception as e:
        print(f"[FEED SKIPPED] {feed_url} -> {e}")
        return feed_url, []


# --- NEW: helpers to handle podcast feeds safely ---

def classify_links(entry):
    """
    Prefer an HTML page if available; otherwise return any enclosure/audio.
    Returns the best target URL for our purposes.
    """
    # Check links array for an HTML 'alternate'
    for l in entry.get("links", []) or []:
        href = l.get("href")
        rel = (l.get("rel") or "").lower()
        typ = (l.get("type") or "").lower()
        if href and (rel in (None, "", "alternate")) and (typ.startswith("text/html") or typ == ""):
            return href

    # Fallback to entry.link / id if it doesn't look like audio
    url = entry.get("link") or entry.get("id") or ""
    audio_exts = (".mp3", ".m4a", ".aac", ".ogg", ".wav")
    if url and not url.lower().endswith(audio_exts):
        return url

    # Last resort: any non-audio link found
    for l in entry.get("links", []) or []:
        href = l.get("href") or ""
        if href and not href.lower().endswith(audio_exts):
            return href

    return url  # may still be audio; we'll filter by content-type later


def is_html_url(url: str) -> bool:
    """
    Quick guard against audio/binary targets: check extension and Content-Type.
    """
    if not url:
        return False
    lower = url.lower()
    if lower.endswith((".mp3", ".m4a", ".aac", ".ogg", ".wav", ".zip", ".pdf")):
        return False
    try:
        h = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        ctype = (h.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            return True
        if h.status_code >= 400:
            return False
        # Fallback lightweight GET (headers only)
        g = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        ctype = (g.headers.get("Content-Type") or "").lower()
        g.close()
        return "text/html" in ctype
    except Exception:
        return False


# --- end new helpers ---


def extract_full_text(url):
    """
    Fetch article HTML (timeout), try readability first, then fallback to <article>/<main>.
    Returns clean text or "".
    """
    # NEW: only attempt extraction for HTML pages
    if not is_html_url(url):
        return ""

    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"    [SKIP] fetch failed: {e}")
        return ""

    # Readability extraction
    try:
        doc = Document(r.text)
        html = doc.summary(html_partial=True)
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # Fallback: take paragraphs from <article> or <main>
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        container = soup.find("article") or soup.find("main") or soup
        text = " ".join(p.get_text(" ", strip=True) for p in container.find_all("p"))
        return text.strip()
    except Exception:
        return ""


def entry_to_row(feed_name, entry):
    # NEW: choose best target URL (HTML if possible; otherwise enclosure-safe)
    url = classify_links(entry)

    title = (entry.get("title") or "").strip()
    published = entry.get("published") or entry.get("updated") or ""
    summary = (entry.get("summary") or entry.get("description") or "").strip()

    print(f"  - fetching: {title[:80]} | {url}")
    # NEW: only extract text for HTML pages
    full_text = extract_full_text(url) if url else ""

    return {
        "feed": feed_name,
        "title": title,
        "url": url,
        "published": published,
        "summary": summary,
        "full_text": full_text,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_feeds", type=int, default=None,
                   help="Limit number of feeds processed.")
    p.add_argument("--max_items_per_feed", type=int, default=None,
                   help="Limit number of items per feed.")
    p.add_argument("--out_json", default="scrapedArticles\\nos_articles.json",
                   help="Output JSON filename.")
    return p.parse_args()


def main(out_json="nos_articles.json",
         max_feeds=None, max_items_per_feed=None):
    feed_links = get_feed_links()
    rows = []
    seen_urls = set()

    try:
        for i, feed_url in enumerate(feed_links, start=1):
            if max_feeds and i > max_feeds:
                break

            print(f"\n[FEED {i}/{len(feed_links)}] {feed_url}")
            feed_name, entries = parse_feed(feed_url)
            if not entries:
                print("  (no entries)")
                continue

            subset = entries[:max_items_per_feed] if max_items_per_feed else entries
            for j, entry in enumerate(subset, start=1):
                # NEW: dedupe on the chosen target URL
                url = classify_links(entry)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                try:
                    row = entry_to_row(feed_name, entry)
                    rows.append(row)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"    [ITEM SKIPPED] {e}")

                time.sleep(REQUEST_SLEEP)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Writing partial results...")

    finally:
        if rows:
            try:
                df = pd.DataFrame(rows)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                print(f"[DONE] Wrote {len(rows)} rows to {out_json}")
            except Exception as e:
                print(f"[ERROR] Failed to write outputs: {e}")
        else:
            print("[INFO] No rows collected.")


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(
                  out_json=args.out_json,
                  max_feeds=args.max_feeds,
                  max_items_per_feed=args.max_items_per_feed))
