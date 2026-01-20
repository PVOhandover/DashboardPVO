import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://api.politie.nl/v4/nieuws"

# Make OUTPUT_FILE independent of where you run the script from.
# This file is in: <project_root>/webScrapers/scrape_police.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "scrapedArticles" / "politie.json"


def convert_article(old):
    # --- extract and clean full text ---
    paragraphs = []
    try:
        for alinea in old.get("alineas", []):
            html = alinea.get("opgemaaktetekst", "")
            text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)
    except Exception:
        paragraphs.append("")

    full_text = " ".join(paragraphs)

    # --- convert publish date to RFC1123 ---
    try:
        dt = datetime.strptime(old["publicatiedatum"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        published_rfc = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    except Exception:
        published_rfc = old.get("publicatiedatum", "")

    return {
        "feed": "Politie",
        "title": old.get("titel", ""),
        "url": old.get("url", ""),
        "published": published_rfc,
        "summary": old.get("introductie", ""),
        "full_text": full_text,
        "scraped_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def fetch_news(from_date, to_date):
    all_items = []

    start_date = datetime.strptime(from_date, "%Y%m%d")
    end_date = datetime.strptime(to_date, "%Y%m%d")
    increment = timedelta(days=15)

    current_end = end_date

    while current_end >= start_date:
        current_start = max(current_end - increment + timedelta(days=1), start_date)
        offset = 0
        max_items = 25

        while True:
            params = {
                "fromdate": current_start.strftime("%Y%m%d"),
                "todate": current_end.strftime("%Y%m%d"),
                "language": "nl",
                "maxnumberofitems": max_items,
                "offset": offset,
            }

            print(f"Requesting {params['fromdate']} to {params['todate']}, offset {offset}...")
            response = requests.get(BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "nieuwsberichten" not in data or not data["nieuwsberichten"]:
                break

            all_items.extend(data["nieuwsberichten"])

            iterator = data.get("iterator", {})
            if iterator.get("last", True):
                break

            offset += max_items

        time.sleep(1)
        current_end = current_start - timedelta(days=1)

    return [convert_article(item) for item in all_items]


def scrape_1yr():
    today = datetime.today().strftime("%Y%m%d")
    one_year_before = datetime.today().replace(year=datetime.today().year - 1).strftime("%Y%m%d")

    result = fetch_news(one_year_before, today)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(result)} police articles to {OUTPUT_FILE}")


def merge_and_dedupe(items, key="url"):
    seen = set()
    merged = []
    for item in items:
        identifier = item.get(key)
        if identifier and identifier not in seen:
            seen.add(identifier)
            merged.append(item)
    return merged


def update_csvs():
    if not OUTPUT_FILE.exists():
        print(f"⚠️ {OUTPUT_FILE} not found. Run --scrape-year once to initialize.")
        return 2  # non-zero-ish, but controlled

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"⚠️ {OUTPUT_FILE} is empty. Run --scrape-year once to initialize.")
        return 2

    # Your file is sorted newest-first in practice; keep your logic:
    # take the first item as the newest "published".
    published_str = data[0].get("published", "")
    try:
        dt = datetime.strptime(published_str, "%a, %d %b %Y %H:%M:%S %Z")
    except Exception:
        print(f"⚠️ Could not parse published date: {published_str}. Falling back to 7-day update.")
        dt = datetime.today() - timedelta(days=7)

    update_from = dt.strftime("%Y%m%d")
    today = datetime.today().strftime("%Y%m%d")

    result = fetch_news(update_from, today)
    merged = merge_and_dedupe(result + data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Update complete. Added {len(result)} fetched items. Total now {len(merged)}. Saved to {OUTPUT_FILE}")
    return 0


def interactive_menu():
    while True:
        print("\n=== Main Menu ===")
        print("1) Scrape 1 year")
        print("2) Update")
        print("3) Exit")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            scrape_1yr()
        elif choice == "2":
            code = update_csvs()
            if code != 0:
                print("⚠️ Update did not run cleanly. Consider running option 1 once.")
        elif choice == "3":
            print("Bye.")
            raise SystemExit(0)
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape-year", action="store_true", help="Scrape 1 year once and exit")
    parser.add_argument("--update", action="store_true", help="Update once and exit")
    args = parser.parse_args()

    if args.scrape_year:
        scrape_1yr()
        return

    if args.update:
        code = update_csvs()
        raise SystemExit(code)

    interactive_menu()


if __name__ == "__main__":
    main()
