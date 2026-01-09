## This file is for pre-processing the scrapped all_articles.json file. ##

## -------------------------------------------------------------- ##

## First I will load the all_articles JSON file into a DataFrame. ##
import pandas as pd
import json
from geo_filter import build_geo_df
from sme_filter import run_snorkel
from narrow_locations import apply_location_narrowing
from sector_classifier import add_sector_classification
from narrow_locations import apply_location_narrowing

### >>> CACHING ADDED >>>
import os
import hashlib

CACHE_FILE = "preprocess_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def make_article_id(article):
    url = (article.get("url") or "").strip()
    if url:
        key = url
    else:
        key = (
                (article.get("title", "") or "") +
                (article.get("published", "") or "") +
                (article.get("date", "") or "")
        )
    return hashlib.md5(key.encode("utf-8")).hexdigest()
### <<< END CACHING <<<


with open("all_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

### >>> CACHING ADDED — WRAP EXPENSIVE PART >>>
cache = load_cache()
processed_rows = []
new_cache_entries = {}

for art in data:
    aid = make_article_id(art)

    # Already cached → reuse
    if aid in cache:
        processed_rows.append(cache[aid])
        continue

    # Save single article to temp JSON (build_geo_df requires a file path)
    os.makedirs("cache", exist_ok=True)
    temp_path = os.path.join("cache", "_cache_single.json")
    with open(temp_path, "w", encoding="utf-8") as tf:
        json.dump([art], tf, ensure_ascii=False, indent=2)

    # GEO FILTER
    single_df = build_geo_df(temp_path, min_conf=0.6)
    if len(single_df) == 0:
        continue  # No geo info → skip article

    # SME FILTER
    single_df, _ = run_snorkel(single_df, min_conf=0.5)
    single_df = single_df[single_df["sme_probability"] > 0.6]
    if len(single_df) == 0:
        continue  # Not an SME article → skip

    # LOCATION NARROWING
    if len(single_df) > 0:
        single_df = apply_location_narrowing(single_df)

    # SECTOR CLASSIFICATION
    if len(single_df) > 0:
        single_df = add_sector_classification(single_df)

    # Keep result
    if len(single_df) > 0:
        result = single_df.to_dict(orient="records")[0]
        new_cache_entries[aid] = result
        processed_rows.append(result)

# Save cache if new entries were added
if new_cache_entries:
    cache.update(new_cache_entries)
    save_cache(cache)

# Final filtered DataFrame (replaces original sme_filtered)
sme_filtered = pd.DataFrame(processed_rows)
### <<< END CACHING <<<


print(sme_filtered)


#print(df[["title", "sme_probability", "sme_label"]].head()) # peek

## -------------------------------------------------------------- ##

## This is the method to clean the cell contents in each row of the chosen column. ##
import re
import wordninja
import unicodedata
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1) Stripping HTML to plain text:
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # 2) Normalizing whitespace/dashes/quotes:
    text = unicodedata.normalize("NFKC", text)

    # 3) Removing leading podcast/article number headers like "#202 - ":
    text = re.sub(r"^\s*#?\d+\s*[-–—:]\s*", "", text)

    # 4) Splitting #Hashtag/@Mentions into words (keeping the content, but not the symbol):
    def tag_handler(m):
        return " ".join(wordninja.split(m.group(1)))
    text = re.sub(r"[@#](\w+)", tag_handler, text)
    text = re.sub(r"[@#]", " ", text)  # leftover symbols

    # 5) Replacing punctuation with space (keeping letters incl. accents + digits):
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ\s]", " ", text)

    # 6) Collapsing multiple spaces and lowercase:
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text
## -------------------------------------------------------------- ##

## This is a method that returns vocab (a collections.Counter object, which is a special kind of dictionary). ##
## Keys = words (tokens from the dataset) and Values = counts (how many times each word appeared) ##
from collections import Counter

def build_vocabulary(dataset):
    vocab = Counter()

    for example in dataset:
        text = example['clean']
        words = text.split()
        vocab.update(words)

    return vocab
## -------------------------------------------------------------- ##

## This method returns a row with an added list of tokens (words or <unk>) ##
def word_tokenizer(example, vocab, unknown_token='<unk>'):
    text = example['clean']
    tokens = None

    words = text.split()

    tokens = [word if word in vocab else unknown_token for word in words]

    example['tokens'] = tokens
    return example
## -------------------------------------------------------------- ##

# -------------------------
# Dashboard keyword output
# -------------------------
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# If no SME articles, still write an empty file so Actions/Streamlit don't break
os.makedirs("keywords", exist_ok=True)
OUT_JSON = os.path.join("keywords", "all_articles_keywords.json")

if sme_filtered.empty:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print(f"⚠️ No SME articles found. Wrote empty {OUT_JSON}.")
    raise SystemExit(0)

def get_raw_text(row):
    if "full_text" in row and isinstance(row["full_text"], str) and row["full_text"].strip():
        return row["full_text"]
    title = row.get("title", "") or ""
    summary = row.get("summary", "") or ""
    return f"{title} {summary}".strip()

# Ensure each row has 'clean'
sme_filtered = sme_filtered.copy()
sme_filtered["clean"] = sme_filtered.apply(lambda r: clean_text(get_raw_text(r)), axis=1)

# Dutch stopwords
nltk.download("stopwords", quiet=True)
stopword_list = stopwords.words("dutch")

corpus = sme_filtered["clean"].fillna("").tolist()

vectorizer = TfidfVectorizer(max_features=10000, stop_words=stopword_list)
tfidf = vectorizer.fit_transform(corpus)
vocab = np.array(vectorizer.get_feature_names_out())

top_k = 10
keywords_col = []
for i in range(tfidf.shape[0]):
    row = tfidf[i].toarray().ravel()
    if row.sum() == 0:
        keywords_col.append([])
        continue
    top_idx = row.argsort()[-top_k:][::-1]
    kws = [{"word": vocab[j], "score": float(row[j])} for j in top_idx if row[j] > 0]
    keywords_col.append(kws)

sme_filtered["keywords"] = keywords_col

sme_filtered.to_json(
    OUT_JSON,
    orient="records",
    indent=2,
    force_ascii=False
)

print(f"✅ Wrote {OUT_JSON} with {len(sme_filtered)} SME articles.")
