## This file is for pre-processing the scrapped all_articles.json file. ##

## -------------------------------------------------------------- ##

import pandas as pd
import json
from geo_filter import build_geo_df
from layered_filter import run_crime_snorkel, run_sme_snorkel
from narrow_locations import apply_location_narrowing
from sector_classifier import add_sector_classification
import sys

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
    key = (article.get("title", "") + article.get("date", "")).encode("utf-8")
    return hashlib.md5(key).hexdigest()
### <<< END CACHING <<<


with open("all_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

### >>> CACHING ADDED >>>
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
    temp_path = "_cache_single.json"
    with open(temp_path, "w", encoding="utf-8") as tf:
        json.dump([art], tf, ensure_ascii=False, indent=2)

    # GEO FILTER
    single_df = build_geo_df(temp_path, min_conf=0.6)
    if len(single_df) == 0:
        continue  # No geo info → skip article

    # CRIME FILTER
    single_df = run_crime_snorkel(single_df)
    if len(single_df) == 0:
        continue

    # SME FILTER
    single_df = run_sme_snorkel(single_df)
    if len(single_df) == 0:
        continue  # Not an SME article → skip

    # LOCATION NARROWING
    if len(single_df) > 0:
        single_df = apply_location_narrowing(single_df)
    if len(single_df) == 0:
        continue


    # SECTOR CLASSIFICATION
    if len(single_df) > 0:
        single_df = add_sector_classification(single_df)
    if len(single_df) == 0:
        continue


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

# CLUSTERING 
from article_clustering import cluster_articles
if len(sme_filtered) > 0:
    
    sme_filtered = cluster_articles(
        sme_filtered,
        threshold=0.75,
        max_hours=72,
        verbose=True
    )

    
    # Get detailed statistics
    from article_clustering import get_clustering_stats
    stats = get_clustering_stats(sme_filtered)
    for key, value in stats.items():
        print(f"  {key}: {value}")


print(sme_filtered)



#print(df[["title", "sme_probability", "sme_label"]].head()) # peek

## -------------------------------------------------------------- ##

## clean cell contents in each row of chosen column. ##
import re
import wordninja
import unicodedata
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Stripping HTML to plain text:
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Normalizing whitespace/dashes/quotes:
    text = unicodedata.normalize("NFKC", text)

    # Removing leading podcast/article number headers like "#202 - ":
    text = re.sub(r"^\s*#?\d+\s*[-–—:]\s*", "", text)

    # Splitting #Hashtag/@Mentions into words (keeping the content, but not the symbol):
    def tag_handler(m):
        return " ".join(wordninja.split(m.group(1)))
    text = re.sub(r"[@#](\w+)", tag_handler, text)
    text = re.sub(r"[@#]", " ", text)  # leftover symbols

    # Replacing punctuation with space (keeping letters incl. accents + digits):
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ\s]", " ", text)

    # Collapsing multiple spaces and lowercase:
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text
## -------------------------------------------------------------- ##

## returns vocab (a collections.Counter object, which is a special kind of dictionary). ##
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

## returns a row with an added list of tokens (words or <unk>) ##
def word_tokenizer(example, vocab, unknown_token='<unk>'):
    text = example['clean']
    tokens = None

    words = text.split()

    tokens = [word if word in vocab else unknown_token for word in words]

    example['tokens'] = tokens
    return example
## -------------------------------------------------------------- ##

## TESTING AREA ##
## This is the where Train/Test split + vocab + word tokenization for NOS articles is being done. ##
from sklearn.model_selection import train_test_split
import sys

# --- after building sme_filtered ---
if len(sme_filtered) == 0:
    print("⚠️ No SME articles found — skipping split.")
    sys.exit(0)
elif len(sme_filtered) < 3:
    print(f"⚠️ Only {len(sme_filtered)} SME article(s) found — skipping train/test split.")
    sys.exit(0)


# split the DataFrame into train/test.
train_df, test_df = train_test_split(sme_filtered, test_size=0.2, random_state=42)

news_ds = {
    "train": train_df.to_dict(orient="records"),
    "test": test_df.to_dict(orient="records"),
}

def get_raw_text(row):
    if "full_text" in row and isinstance(row["full_text"], str) and row["full_text"].strip():
        return row["full_text"]
    # If full_text is missing, then title + summary:
    title = row.get("title", "") or ""
    summary = row.get("summary", "") or ""
    return f"{title} {summary}".strip()

for split in ["train", "test"]:
    for row in news_ds[split]:
        row["clean"] = clean_text(get_raw_text(row))

# Building vocabulary from TRAIN only
vocab_counter = build_vocabulary(news_ds["train"])
#print("Size of the vocabulary:", len(vocab_counter))

# Limiting vocab to top-10000 most frequent terms
max_vocab_size = 10000
vocab = vocab_counter.most_common(max_vocab_size)

# Casting to a plain list of words
vocab = [word for word, _ in vocab]
#print("Final vocab size (after cutoff):", len(vocab))

# Tokenizing TRAIN set
for i in range(len(news_ds["train"])):
    news_ds["train"][i] = word_tokenizer(news_ds["train"][i], vocab)

# Checking the OOV rate for the TRAIN set
total = 0
oov = 0
for row in news_ds["train"]:
    toks = row.get("tokens", [])
    total += len(toks)
    oov += sum(1 for t in toks if t == "<unk>")
#print(f"OOV rate: {oov}/{total} = {oov/total:.2%}")


# Shows first 10 examples from TRAIN set
#for i in range(min(10, len(news_ds["train"]))):
    #print("Original article:")
    #print(get_raw_text(news_ds["train"][i]))
    #print("Tokenized article:")
    #print(news_ds["train"][i]["tokens"])
    #print("-" * 40)
## -------------------------------------------------------------- ##

## -------------------------------------------------------------- ##
## TF-IDF Keyword Extraction for SME-filtered articles ##
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk


print("🔧 Building TF-IDF keyword lists for SME-filtered articles...")

# Download and load Dutch stopwords
nltk.download('stopwords', quiet=True)
stopword_list = stopwords.words('dutch')

# Build corpus from 'clean' column in the SME-filtered dataset
# (the split ensures each row already has a 'clean' field)
corpus = [row["clean"] for row in news_ds["train"] if row.get("clean", "").strip()]

# Build Bag-of-Words with stopword filtering
vectorizer = CountVectorizer(
    max_features=10000,
    stop_words=stopword_list,
)
bows = vectorizer.fit_transform(corpus).toarray()
vocab = np.array(vectorizer.get_feature_names_out())

# Calculate IDF
def calculate_idf(bows):
    """
    Calculates the IDF for each word in the vocabulary.
    Args:
        bows: numpy array of shape (N x D)
    Returns: numpy array of size D with IDF values for each token.
    """
    N = bows.shape[0]
    df = np.count_nonzero(bows, axis=0)
    df = np.where(df == 0, 1, df)  # avoid division by zero
    idf = np.log10(N / df)
    return idf

idf = calculate_idf(bows)

# Compute TF-IDF for each document
def compute_tfidf_matrix(bows, idf):
    tfidf_matrix = bows * idf
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return tfidf_matrix / norms

tfidf_matrix = compute_tfidf_matrix(bows, idf)

# Extract top-k keywords per document
def extract_keywords(tfidf_matrix, vocab, top_k=10):
    keywords = []
    for row in tfidf_matrix:
        top_indices = row.argsort()[-top_k:][::-1]
        top_words = vocab[top_indices]
        top_scores = row[top_indices]
        keywords.append(list(zip(top_words, top_scores)))
    return keywords

article_keywords = extract_keywords(tfidf_matrix, vocab, top_k=10)

# Attach top keywords back to the training DataFrame
train_df = pd.DataFrame(news_ds["train"]).copy()
train_df["keywords"] = [
    [{"word": w, "score": float(s)} for w, s in kws] for kws in article_keywords
]

# Save results to a new JSON file
train_df.to_json(
    "keywords/all_articles_keywords.json",
    orient="records",
    indent=2,
    force_ascii=False,
)

print("TF-IDF keyword extraction completed.")
print(train_df[["title", "keywords"]].head())
## -------------------------------------------------------------- ##

# TEST to see the top 20 keywords from news sources
from collections import Counter

# aggregate scores
global_keywords = Counter()
for kws in article_keywords:
    for w, s in kws:
        global_keywords[w] += float(s)

# deduplicate (Counter already merges same words)
# sort descending
top_keywords = global_keywords.most_common(20)

print("\n Top 20 overall SME keywords (deduplicated & sorted):")
for word, score in top_keywords:
    print(f"{word:<20} {score:.3f}")

# Save keywords to JSON file for visualization
output_file = "keywords/all_articles_top_keywords.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dict(top_keywords), f, ensure_ascii=False, indent=2)

print(f"\n Saved top 20 keywords to {output_file}")
