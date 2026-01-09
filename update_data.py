import os, sys, json, shutil, subprocess

OUT_DIR = os.path.join("cache", "public")
OUT_FILE = os.path.join(OUT_DIR, "latest.json")

DASHBOARD_INPUT = os.path.join("keywords", "all_articles_keywords.json")

def run(cmd):
    print("\n+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def ensure_json_ok(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of records for dashboard")
    return data

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # (Optional) scrape fresh RSS first — enable if you want nightly updates
    # run([sys.executable, os.path.join("webScrapers", "scrape_all_rss_feeds.py")])

    # Merge scrapedArticles/*.json -> all_articles.json
    run([sys.executable, "merge_jsons.py"])

    # Preprocess -> keywords/all_articles_keywords.json
    run([sys.executable, "pre_process.py"])

    if not os.path.exists(DASHBOARD_INPUT):
        raise FileNotFoundError(f"Expected {DASHBOARD_INPUT} after pre_process.py")

    shutil.copyfile(DASHBOARD_INPUT, OUT_FILE)
    data = ensure_json_ok(OUT_FILE)
    print(f"\n✅ Published {OUT_FILE} with {len(data)} records\n")

if __name__ == "__main__":
    main()
