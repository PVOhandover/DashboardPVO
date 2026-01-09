import os
import json
import glob
import pandas as pd
from datetime import datetime

# -------- CONFIGURATION --------
INPUT_DIR = "scrapedArticles"       # folder wherescraped JSON files are stored
OUTPUT_FILE = "all_articles.json"  # name of the merged output file
# --------------------------------

def merge_json_files(input_dir: str, output_file: str):
    all_articles = []
    seen_urls = set()

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"🔍 Found {len(json_files)} JSON files in '{input_dir}'")

    for file_path in json_files:
        print (file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                elif not isinstance(data, list):
                    print("Skipped for some reason")
                    continue  # skip malformed entries

                for item in data:
                    url = item.get("url")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(item)
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")

    print(f"✅ Merged {len(all_articles)} unique articles from {len(json_files)} files.")

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(all_articles, out, ensure_ascii=False, indent=2)

    print(f"💾 Saved merged file to: {output_file}")

def csv_to_json(csv_file_path, json_file_path, feed):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Combine date and time into a datetime object
    df['published'] = df.apply(
        lambda row: datetime.strptime(
            f"{row['date']} {row['time']}", "%d-%m-%Y %H:%M"
        ).strftime("%a, %d %b %Y %H:%M:%S"),
        axis=1
    )

    # Keep only the needed columns, in desired order
    df = df[['published', 'title', 'url', 'full_text']]
    df['feed'] = feed
    
    # Convert to JSON and save
    df.to_json(json_file_path, orient='records', indent=4, force_ascii=False)
    
    print(f"✅ Successfully converted {csv_file_path} → {json_file_path}")

if __name__ == "__main__":
    csv_path = os.path.join("articles", "security_nl_articles.csv")
    json_path = os.path.join(INPUT_DIR, "security_nl_articles.json")

    if os.path.exists(csv_path):
        csv_to_json(csv_path, json_path, "security.nl")
    else:
        print(f"ℹ️ CSV not found at {csv_path} — skipping CSV import.")

    merge_json_files(INPUT_DIR, OUTPUT_FILE)
