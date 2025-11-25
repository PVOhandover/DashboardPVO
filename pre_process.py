## This file is for pre-processing the scrapped all_articles.json file. ##

## -------------------------------------------------------------- ##

## First I will load the all_articles JSON file into a DataFrame. ##
import pandas as pd
import json
from geo_filter import build_geo_df
from sme_filter import run_snorkel
from narrow_locations import apply_location_narrowing
from sector_classifier import add_sector_classification


with open("all_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# We filter out the articles that are not from the region before pre-processing.
df = build_geo_df("all_articles.json", min_conf=0.6)

# We filter out the articles that are not about SMEs before pre-processing.
df, label_model = run_snorkel(df, min_conf=0.5)

sme_filtered = df[df["sme_probability"] > 0.6]

# We narrow down the locations
sme_filtered = apply_location_narrowing(sme_filtered)

# SECTOR CLASSIFICATION
sme_filtered = add_sector_classification(sme_filtered)

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


# 1) I split the DataFrame into train/test.
train_df, test_df = train_test_split(sme_filtered, test_size=0.2, random_state=42)

news_ds = {
    "train": train_df.to_dict(orient="records"),
    "test": test_df.to_dict(orient="records"),
}

# 2) Making sure that each row has a 'clean' field (uses 'full_text' if present, otherwise title+summary).
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

# 3) Building vocabulary from TRAIN only:
vocab_counter = build_vocabulary(news_ds["train"])
#print("Size of the vocabulary:", len(vocab_counter))

# 4) Limiting vocab to top-10000 most frequent terms:
max_vocab_size = 10000
vocab = vocab_counter.most_common(max_vocab_size)

# 5) Casting to a plain list of words (droping their counts):
vocab = [word for word, _ in vocab]
#print("Final vocab size (after cutoff):", len(vocab))

# 6) Tokenizing TRAIN set:
for i in range(len(news_ds["train"])):
    news_ds["train"][i] = word_tokenizer(news_ds["train"][i], vocab)

# 6) Checking the OOV rate for the TRAIN set:
total = 0
oov = 0
for row in news_ds["train"]:
    toks = row.get("tokens", [])
    total += len(toks)
    oov += sum(1 for t in toks if t == "<unk>")
#print(f"OOV rate: {oov}/{total} = {oov/total:.2%}")


# 7) Shows first 10 examples from TRAIN set:
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

# Build Bag-of-Words (BoW) representation
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
