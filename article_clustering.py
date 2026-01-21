import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import to_datetime, isna


# -------------------------
# Simple similarity helpers
# -------------------------

def calculate_title_similarity(t1, t2):
    if not t1 or not t2:
        return 0.0
    
    w1 = set(t1.lower().split())
    w2 = set(t2.lower().split())
    
    return len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.0


def extract_entities(text):
    if not isinstance(text, str):
        return set()
    
    return {w.lower() for w in text.split() if w[0].isupper()}


def calculate_entity_overlap(t1, t2):
    e1 = extract_entities(t1)
    e2 = extract_entities(t2)
    
    return len(e1 & e2) / len(e1 | e2) if (e1 | e2) else 0.0


# -------------------------
# Text for TF-IDF
# -------------------------

def extract_article_text(row):
    title = row.get("title", "")
    summary = row.get("summary", "")
    
    # Give title more weight
    return f"{title} {title} {title} {summary}".lower()


# -------------------------
# Similarity matrix
# -------------------------

def calculate_similarity_matrix(df, max_features=1000):
    texts = df.apply(extract_article_text, axis=1).tolist()
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        stop_words=None
    )
    
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


# -------------------------
# Time filtering
# -------------------------
from dateutil.parser import parse

def robust_parse_date(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return pd.NaT
    try:
        dt_obj = parse(str(s), fuzzy=True)  # flexible parsing
        # Remove timezone info for consistency
        if dt_obj.tzinfo is not None:
            dt_obj = dt_obj.replace(tzinfo=None)
        return dt_obj
    except Exception:
        return pd.NaT


def are_temporally_close(d1, d2, max_hours=72):
    d1, d2 = robust_parse_date(d1), robust_parse_date(d2)

    if isna(d1) or isna(d2):
        return True
    # Drop tz info if present
    d1, d2 = getattr(d1, 'tz_localize', lambda x=None: d1)(None), getattr(d2, 'tz_localize', lambda x=None: d2)(None)
    return abs((d1 - d2).total_seconds()) / 3600 <= max_hours


# -------------------------
# Clustering
# -------------------------

def find_clusters(df, sim_matrix, threshold=0.75, max_hours=72):
    n = len(df)
    assigned = set()
    clusters = []
    
    for i in range(n):
        print(f"{i} of {n}")
        if i in assigned:
            continue
        
        cluster = [i]
        assigned.add(i)
        
        for j in range(i + 1, n):
            if j in assigned:
                continue
            
            if not are_temporally_close(df.loc[i, "published"], df.loc[j, "published"], max_hours):
                continue
            
            tfidf_sim = sim_matrix[i][j]
            title_sim = calculate_title_similarity(df.loc[i, "title"], df.loc[j, "title"])
            title_i = str(df.loc[i, "title"] or "")
            summary_i = str(df.loc[i, "summary"] or "")  # or df.loc[i, "clean_geo"]
            title_j = str(df.loc[j, "title"] or "")
            summary_j = str(df.loc[j, "summary"] or "")

            entity_sim = calculate_entity_overlap(
                f"{title_i} {summary_i}",
                f"{title_j} {summary_j}"
            )
            
            is_duplicate = (
                tfidf_sim >= threshold or
                title_sim >= 0.6 or
                (entity_sim >= 0.5 and tfidf_sim >= threshold - 0.1) or
                (tfidf_sim >= 0.6 and title_sim >= 0.4 and entity_sim >= 0.3)
            )
            
            if is_duplicate:
                cluster.append(j)
                assigned.add(j)
        
        clusters.append(cluster)
    
    return clusters


# -------------------------
# Merge duplicates
# -------------------------

def merge_cluster(df, indices):
    group = df.iloc[indices].sort_values("published")
    base = group.iloc[0].to_dict()
    
    base["cluster_size"] = len(indices)
    base["sources"] = group["feed"].unique().tolist()
    base["all_urls"] = group["url"].dropna().unique().tolist()
    base["combined_summary"] = " | ".join(group["summary"].dropna().unique())
    base["is_clustered"] = True
    
    return base


# -------------------------
# Main function
# -------------------------

def cluster_articles(df, threshold=0.75, max_hours=72, verbose=True):
    df = df.reset_index(drop=True)
    
    if verbose:
        print(f"Analyzing {len(df)} articles...")
    
    sim_matrix = calculate_similarity_matrix(df)
    clusters = find_clusters(df, sim_matrix, threshold, max_hours)
    
    merged = []
    
    for c in clusters:
        if len(c) == 1:
            article = df.iloc[c[0]].to_dict()
            article["cluster_size"] = 1
            article["is_clustered"] = False
            merged.append(article)
        else:
            merged.append(merge_cluster(df, c))
    
    result = pd.DataFrame(merged)
    
    if verbose:
        removed = len(df) - len(result)
        print(f"Removed {removed} duplicates → {len(result)} unique articles")
    
    return result


# -------------------------
# Stats
# -------------------------

def get_clustering_stats(df):
    return {
        "total_articles": len(df),
        "merged_groups": len(df[df["cluster_size"] > 1]),
        "largest_cluster": df["cluster_size"].max(),
        "avg_cluster_size": df[df["cluster_size"] > 1]["cluster_size"].mean()
    }
