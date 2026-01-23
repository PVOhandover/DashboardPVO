import os
import json
import glob
import hashlib
import pandas as pd
from datetime import datetime, timezone, timedelta

# -------- CONFIGURATION --------
INPUT_DIR = "scrapedArticles"       # folder where per-source JSON files are stored
OUTPUT_FILE = "all_articles.json"   # merged master JSON file

# Refresh / dedup (Option A)
REFRESH_STATE_FILE = os.path.join("data", "refresh", "rss_state.json")
SEEN_IDS_FILE = os.path.join("data", "refresh", "seen_article_ids.json")
OVERLAP_HOURS = 24  # safety window to catch reordering / delayed items

# Security.nl CSV location (input) + generated JSON location (output)
SECURITY_CSV_PATH = os.path.join("articles", "security_nl_articles.csv")
SECURITY_JSON_PATH = os.path.join(INPUT_DIR, "security_nl_articles.json")


# -------- HELPERS --------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json_safe(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def canonicalize_url(url: str) -> str:
    """
    Minimal canonicalization:
    - remove fragment (#...)
    - drop tracking-heavy query strings in a conservative way
    - strip trailing slash
    """
    if not url:
        return ""

    # Remove fragment
    url = url.split("#")[0]

    # Conservative query stripping: if common trackers appear, drop query entirely
    if "?" in url:
        base, query = url.split("?", 1)
        q = query.lower()
        if "utm_" in q or "fbclid=" in q or "gclid=" in q:
            url = base
        else:
            url = base + "?" + query

    return url.rstrip("/")


def make_article_id(url: str) -> str:
    canon = canonicalize_url(url)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def parse_rss_published(published: str):
    """
    Parse typical RSS date strings like:
    'Wed, 7 Jan 2026 20:53:31 +0100'
    Also accepts strings without timezone:
    'Wed, 7 Jan 2026 20:53:31' (assumed UTC)
    Returns timezone-aware datetime in UTC, or None if parsing fails.
    """
    if not published:
        return None

    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S"):
        try:
            dt = datetime.strptime(published, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def feed_key_from_file(file_path: str) -> str:
    """
    Use filename (without extension) as a stable key in state.
    Example: scrapedArticles/bd.json -> 'bd'
    """
    return os.path.splitext(os.path.basename(file_path))[0]


# -------- SECURITY.NL CSV -> JSON --------
def csv_to_json_security_nl(csv_file_path: str, json_file_path: str) -> bool:
    """
    Converts articles/security_nl_articles.csv into scrapedArticles/security_nl_articles.json

    Input columns:
      date,time,title,url,full_text
      date format: DD-MM-YYYY
      time format: HH:MM

    Output JSON matches the other sources:
      published (RSS-like with timezone), title, url, full_text, feed
    """
    if not os.path.exists(csv_file_path):
        print(f"ℹ️ Security.nl CSV not found at {csv_file_path} — skipping CSV import")
        return False

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"⚠️ Failed to read Security.nl CSV ({csv_file_path}): {e}")
        return False

    required = {"date", "time", "title", "url", "full_text"}
    if not required.issubset(set(df.columns)):
        print(f"⚠️ Security.nl CSV missing columns. Found: {list(df.columns)}")
        return False

    def to_rss_with_tz(row):
        dt = datetime.strptime(f"{row['date']} {row['time']}", "%d-%m-%Y %H:%M")
        dt = dt.replace(tzinfo=timezone.utc)  # consistent convention
        return dt.strftime("%a, %d %b %Y %H:%M:%S %z")  # includes +0000

    df["published"] = df.apply(to_rss_with_tz, axis=1)
    df["feed"] = "security.nl"

    # keep only the needed columns (same as other source JSONs)
    out_df = df[["published", "title", "url", "full_text", "feed"]].copy()

    ensure_parent_dir(json_file_path)
    out_df.to_json(json_file_path, orient="records", indent=4, force_ascii=False)

    print(f"✅ Successfully converted {csv_file_path} → {json_file_path} ({len(out_df)} rows)")
    return True


# -------- CORE: MERGE + INCREMENTAL REFRESH --------
def merge_json_files(input_dir: str, output_file: str):
    # Load master articles (append-only behavior)
    all_articles = load_json_safe(output_file, [])
    if not isinstance(all_articles, list):
        all_articles = []

    # Load refresh state and seen IDs
    state = load_json_safe(REFRESH_STATE_FILE, {"feeds": {}})
    if "feeds" not in state or not isinstance(state["feeds"], dict):
        state = {"feeds": {}}

    seen_data = load_json_safe(SEEN_IDS_FILE, {"ids": []})
    ids_list = seen_data.get("ids", [])
    if not isinstance(ids_list, list):
        ids_list = []
    seen_ids = set(ids_list)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in '{input_dir}'")

    newly_added = []

    for file_path in json_files:
        key = feed_key_from_file(file_path)

        # cursor per feed key
        feed_state = state["feeds"].get(key, {})
        last_ts_str = feed_state.get("last_published_ts", "1970-01-01T00:00:00Z")
        try:
            last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        except Exception:
            last_ts = datetime(1970, 1, 1, tzinfo=timezone.utc)

        cutoff = last_ts - timedelta(hours=OVERLAP_HOURS)

        print(f"\nProcessing {file_path} (state key='{key}')")
        print(f"  cursor={last_ts.isoformat()}  cutoff={cutoff.isoformat()}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
            continue

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print(f"⚠️ Skipping {file_path}: not a list/dict JSON")
            continue

        max_ingested_ts = None
        added_for_file = 0
        skipped_dup = 0
        skipped_old = 0

        for item in data:
            if not isinstance(item, dict):
                continue

            url = item.get("url")
            if not url:
                continue

            article_id = make_article_id(url)
            if article_id in seen_ids:
                skipped_dup += 1
                continue

            pub_dt = parse_rss_published(item.get("published", ""))

            # If we have a published date, apply time cutoff.
            # If missing published, we still allow ingestion (but it won't advance cursor).
            if pub_dt is not None and pub_dt < cutoff:
                skipped_old += 1
                continue

            # Enrich record (minimal additions)
            item["id"] = article_id
            item["canonical_url"] = canonicalize_url(url)
            item["published_ts"] = pub_dt.isoformat().replace("+00:00", "Z") if pub_dt else None
            item.setdefault("source_type", "rss")

            feed = (item.get("feed") or "").strip()
            norm_map = {
                "NOS Nieuws": "NOS.nl (Economie)",
                "NOS Economie": "NOS.nl (Economie)",
                "NOS.nl": "NOS.nl (Economie)",
                "NOS.nl (Economie)": "NOS.nl (Economie)",
                "Brabants Dagblad - Economie": "Brabants Dagblad - Economie",
                "De Gelderlander - Economie": "De Gelderlander - Economie",
                "Omroep West - Economie": "Omroep West - Economie",
                "L1 Nieuws": "L1 Nieuws",
                "RTV Noord": "RTV Noord",
                "Politie": "Politie.nl",
                "Politie.nl": "Politie.nl",
                "security.nl": "Security.nl",
                "Security.nl": "Security.nl",
            }
            if feed in norm_map:
                item["feed"] = norm_map[feed]
            # ---------------------------------------

            newly_added.append(item)
            seen_ids.add(article_id)
            added_for_file += 1

            if pub_dt is not None:
                max_ingested_ts = pub_dt if max_ingested_ts is None else max(max_ingested_ts, pub_dt)

        # Update state for this feed key
        state["feeds"][key] = {
            "last_published_ts": (
                max_ingested_ts.isoformat().replace("+00:00", "Z")
                if max_ingested_ts is not None
                else last_ts_str
            ),
            "last_run_ts": now_utc_iso(),
            "last_file": os.path.basename(file_path),
            "added_last_run": added_for_file
        }

        print(f"  Added: {added_for_file} | Skipped dup: {skipped_dup} | Skipped old: {skipped_old}")
        if max_ingested_ts is not None:
            print(f"  New cursor: {state['feeds'][key]['last_published_ts']}")

    # Append new items to master output
    if newly_added:
        all_articles.extend(newly_added)
        save_json(output_file, all_articles)

    # Save dedup + state
    save_json(SEEN_IDS_FILE, {"ids": sorted(seen_ids)})
    save_json(REFRESH_STATE_FILE, state)

    print("\n" + "=" * 60)
    print(f"Added {len(newly_added)} new articles")
    print(f"Total articles in {output_file}: {len(all_articles)}")
    print(f"Saved seen IDs to: {SEEN_IDS_FILE}")
    print(f"Saved RSS refresh state to: {REFRESH_STATE_FILE}")
    print("=" * 60 + "\n")


# -------- MAIN --------
if __name__ == "__main__":
    csv_to_json_security_nl(SECURITY_CSV_PATH, SECURITY_JSON_PATH)

    merge_json_files(INPUT_DIR, OUTPUT_FILE)


