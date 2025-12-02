import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests

# ---------------- CONFIG SECTION ----------------

# Base endpoint for the CBS Open Data API
# Every dataset we request will be appended onto this URL
CBS_BASE_URL = "https://opendata.cbs.nl/ODataApi/odata"

# CBS dataset IDs we want to scrape.
# 83625NED: Regional SME/economy-related data
# 83648NED: crime statistics
DATASETS = ["83625NED", "83648NED"]

# Mapping so we can label each dataset as "sme" or "crime"
# naming output files and tagging articles
CRIME_DATASETS = {"83648NED"}
SME_DATASETS = {"83625NED"}


def topic_for(dataset_id: str) -> str:
    """Return a topic name (e.g., 'crime' or 'sme') based on the dataset ID."""
    if dataset_id in CRIME_DATASETS:
        return "crime"
    if dataset_id in SME_DATASETS:
        return "sme"
    return "other"


# How many rows to fetch per API request
# CBS returns data in "pages", so we keep requesting until the dataset is fully downloaded
PAGE_SIZE = 1000

# Folder where scraped JSON files will be saved
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPED_DIR = PROJECT_ROOT / "scrapedArticles"

# ---------------------------------------------------------------------------


def _now_published_rfc822() -> str:
    """
    Return the current time formatted the same way the Limburger scraper does.
    This keeps the JSON output consistent across all scrapers.
    """
    return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


def _now_iso_z() -> str:
    """
    Return the current time in ISO format used for 'scraped_at'.
    Example: 2025-10-21T14:13:47.786257Z
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_cbs_table(
    dataset_id: str,
    table_name: str = "TypedDataSet",
    extra_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Download ALL rows from a CBS dataset by automatically paging through results.

    CBS does not return entire datasets in one go — instead, it returns them in
    chunks of (up to) PAGE_SIZE rows. We keep requesting more pages using $skip
    until there are no more rows left.
    """
    all_rows: List[Dict[str, Any]] = []
    skip = 0

    params: Dict[str, Any] = dict(extra_params or {})
    params["$top"] = PAGE_SIZE # how many rows per request

    while True:
        params["$skip"] = skip # which "page" to request next
        url = f"{CBS_BASE_URL}/{dataset_id}/{table_name}"

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status() # stops the script if CBS returns an error

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from CBS for dataset {dataset_id}: {e}")

        rows = data.get("value", [])
        if not rows:
            # No more data — we reached the end of the dataset
            break

        all_rows.extend(rows)

        if len(rows) < PAGE_SIZE:
            # CBS returned fewer rows than requested, last pg
            break

        # Move to the next block of rows
        skip += PAGE_SIZE

    return all_rows


# JSON format conversion
def normalize_cbs_row(raw: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:

    # Extract region and year, if they exist
    region = (
        raw.get("RegioS")
        or raw.get("RegioNaam")
        or raw.get("RegioCode")
        or ""
    )
    period = raw.get("Perioden") or raw.get("Periods") or ""

    # Creating a title
    title_parts = [f"CBS dataset {dataset_id}"]
    if region:
        title_parts.append(f"regio: {region}")
    if period:
        title_parts.append(f"periode: {period}")
    title = " – ".join(title_parts)

    # Short description that describes what the row represents
    summary = (
        f"Rij uit CBS-dataset {dataset_id} voor "
        f"{region or 'onbekende regio'} in periode {period or 'onbekend'}."
    )

    # Convert raw CBS row into our final consistent JSON format
    return {
        "feed": "CBS",
        "title": title,
        "url": f"https://opendata.cbs.nl/statline/#/CBS/nl/dataset/{dataset_id}/table",
        "published": _now_published_rfc822(),
        "summary": summary,
        "full_text": json.dumps(raw, ensure_ascii=False), # raw CBS data as JSON
        "scraped_at": _now_iso_z(),
        "topic": topic_for(dataset_id), # label as crime/sme
    }


def ensure_output_dir() -> None:
    """Create the scrapedArticles folder if it doesn't exist yet."""
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)


def save_articles(dataset_id: str, articles: List[Dict[str, Any]]) -> Path:
    """
    Save the scraped and normalized CBS rows into a JSON file.
    The file is named based on the topic (e.g. cbs_sme.json or cbs_crime.json).
    """
    ensure_output_dir()

    topic = topic_for(dataset_id)
    out_path = SCRAPED_DIR / f"cbs_{topic}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    return out_path


def scrape_cbs_dataset(dataset_id: str) -> Path:
    """
    Download one CBS dataset, convert every row to the project format,
    and store it as a JSON file inside scrapedArticles/.
    """
    # Filters can be added here - we'll see about this part later
    extra_params: Dict[str, Any] = {}

    raw_rows = fetch_cbs_table(dataset_id, extra_params=extra_params)
    articles = [normalize_cbs_row(row, dataset_id) for row in raw_rows]
    return save_articles(dataset_id, articles)


def main() -> None:
    """Loop through all configured CBS datasets and scrape each one."""
    if not DATASETS:
        print("No CBS dataset IDs configured. Please add at least one.")
        return

    for dataset_id in DATASETS:
        print(f"Scraping CBS dataset {dataset_id}...")
        out_path = scrape_cbs_dataset(dataset_id)
        print(f" -> Saved to {out_path}")

if __name__ == "__main__":
    main()
