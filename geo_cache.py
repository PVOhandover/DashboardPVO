from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set

import requests

DEFAULT_CACHE = "cache/geocode_cache.json"

# important for Nominatim
DEFAULT_UA = "PVO-Limburg-GeoCacheUpdater/1.0 (internal use)"

_WS_RE = re.compile(r"\s+")

DEFAULT_SKIP = {
    "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag",
    "vandaag", "gisteren", "morgen",
}

_GEO_BBOXES = [
    (50.5, 3.2, 53.7, 7.3),   # Netherlands
    (49.4, 2.5, 51.7, 6.4),   # Belgium
    (50.3, 5.5, 52.0, 7.8),   # NRW
]

def _in_any_bbox(lat: float, lon: float) -> bool:
    for south, west, north, east in _GEO_BBOXES:
        if south <= lat <= north and west <= lon <= east:
            return True
    return False


def norm_loc(s: str) -> str:
    s = (s or "").strip()
    s = _WS_RE.sub(" ", s)
    return s


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_cache(cache_file: str) -> Dict[str, Dict[str, float]]:
    if os.path.exists(cache_file):
        try:
            data = load_json(cache_file)
            if isinstance(data, dict):
                out: Dict[str, Dict[str, float]] = {}
                for k, v in data.items():
                    if isinstance(v, dict) and "lat" in v and "lon" in v:
                        out[str(k)] = {"lat": float(v["lat"]), "lon": float(v["lon"])}
                return out
        except Exception:
            pass
    return {}


def extract_locations_from_articles(articles: List[Dict[str, Any]]) -> Set[str]:
    found: Set[str] = set()
    for a in articles:
        locs = a.get("locations", [])
        if isinstance(locs, list):
            for loc in locs:
                if isinstance(loc, str):
                    loc = norm_loc(loc)
                    if loc:
                        found.add(loc)
    return found


def geocode_nominatim(
    query: str,
    user_agent: str,
    timeout_s: int = 25,
    retries: int = 6,
) -> Optional[Tuple[float, float]]:
    """
    Robust Nominatim geocode with retries + exponential backoff + jitter.
    Returns (lat, lon) or None.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": user_agent}

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
            if r.status_code in (429, 502, 503, 504):
                raise requests.RequestException(f"HTTP {r.status_code}")

            if r.status_code != 200:
                return None

            data = r.json()
            if not data:
                return None

            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                requests.RequestException) as e:
            last_err = e
            sleep_s = min(45.0, (2 ** attempt) + random.random())
            time.sleep(sleep_s)

    # If debugging needed:
    # print(f"[WARN] Geocode failed for '{query}': {last_err}")
    return None


def update_cache(
    articles_paths: List[str],
    cache_file: str = DEFAULT_CACHE,
    user_agent: str = DEFAULT_UA,
    sleep_s: float = 1.2,
    max_new: int = 400,
    save_every: int = 25,
    only_bbox: bool = True,
    skip_words: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    cache = load_cache(cache_file)

    # Load all articles
    all_articles: List[Dict[str, Any]] = []
    for p in articles_paths:
        if not os.path.exists(p):
            continue
        data = load_json(p)
        if isinstance(data, list):
            all_articles.extend([x for x in data if isinstance(x, dict)])

    all_locs = extract_locations_from_articles(all_articles)

    skip_words = skip_words or set()
    missing = []
    for loc in sorted(all_locs):
        if loc in cache:
            continue
        if loc.strip().lower() in skip_words:
            continue
        if len(loc) < 3:
            continue
        missing.append(loc)

    added = 0
    skipped = 0
    filtered_outside_bbox = 0

    for loc in missing:
        if added >= max_new:
            break

        coords = geocode_nominatim(loc, user_agent=user_agent)
        if coords is None:
            skipped += 1
        else:
            lat, lon = coords

            if only_bbox and not _in_any_bbox(lat, lon):
                filtered_outside_bbox += 1
            else:
                cache[loc] = {"lat": lat, "lon": lon}
                added += 1

                if added % save_every == 0:
                    save_json(cache_file, cache)

        time.sleep(sleep_s)

    save_json(cache_file, cache)

    return {
        "cache_file": cache_file,
        "article_files": articles_paths,
        "locations_total": len(all_locs),
        "missing_before": len(missing),
        "added": added,
        "skipped": skipped,
        "filtered_outside_bbox": filtered_outside_bbox,
        "cache_size_now": len(cache),
        "only_bbox": only_bbox,
        "sleep_s": sleep_s,
        "max_new": max_new,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", nargs="+", required=True)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--user-agent", default=DEFAULT_UA)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--no-bbox-filter", action="store_true", help="If set, cache everything (not just NL/BE/NRW).")
    args = ap.parse_args()

    summary = update_cache(
        articles_paths=args.articles,
        cache_file=args.cache,
        user_agent=args.user_agent,
        sleep_s=args.sleep,
        max_new=args.max_new,
        save_every=args.save_every,
        only_bbox=not args.no_bbox_filter,
        skip_words=DEFAULT_SKIP,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
