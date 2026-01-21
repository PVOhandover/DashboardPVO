## After some attempts of applying general NLP methods to the articles that we got from
#  RSS-feed of the NOS(I believe any generic news website would give similar results)
#  it is visible that there is some issues when it comes to filtering the location.
#  To solve this issue, we used a pre-trained NER(Named Entity Recognition) model to extract
#  locations from the articles.
#  We sum up the votes per country, then return country, country_score,
#  and evidence (matched city, postal code, etc.) and finally produce a country probability.
#  If the probability score is less than the threshold, then it gets marked “uncertain”.
#  
# The methods on this file(filtering) are used BEFORE the word-tokenization step at the pre-processing file. ##

## --------------------------------------------------------------##

## This is the method to clean the cell contents in each row of the chosen column.
#  It is slightly different than the clean_text method in pre_processing.py file,
#  Because it keeps original casing and punctuation, so we don't break spaCy’s ability to detect place names. ##
import unicodedata
from bs4 import BeautifulSoup

def clean_text_geo(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # HTML to plain text
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Normalize
    text = unicodedata.normalize("NFKC", text)

    return text

# uses 'full_text' if present, otherwise title+summary
def get_raw_text_geo(row):
    if row.get("full_text") and isinstance(row["full_text"], str) and row["full_text"].strip():
        return row["full_text"]
    # If full_text is missing, then title + summary:
    title = row.get("title", "") or ""
    summary = row.get("summary", "") or ""
    return f"{title} {summary}".strip()

## -------------------------------------------------------------- ##

## Detect candidate location names  ##
import spacy

nlp = spacy.load('nl_core_news_sm', disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])

def detect_candidate_locations(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in {"LOC","GPE"}]

## -------------------------------------------------------------- ##

## Create a dictionary of place names ##
from geoNames.gazetteer_parser import load_geonames_file

# Load dictionaries
nl_gaz = load_geonames_file("geoNames/NL.txt", keep_countries={"NL"})
be_gaz = load_geonames_file("geoNames/BE.txt", keep_countries={"BE"})
de_gaz = load_geonames_file("geoNames/DE.txt", keep_countries={"DE"})

# Merge into one
gazetteer = {}
for g in (nl_gaz, be_gaz, de_gaz):
    gazetteer.update(g)

gazetteer = {k.lower(): v for k, v in gazetteer.items()}

# print("Total entries in gazetteer:", len(gazetteer))
# print(list(gazetteer.items())[:20]) # peek
## -------------------------------------------------------------- ##

# Load trained logistic regression location classifier
import joblib
import pandas as pd
from location_classifier import extract_features
from location_model_loader import is_likely_location_from_model
try:
    bundle = joblib.load("models/location_classifier_latest.pkl")
    clf = bundle["model"]
    feature_cols = bundle["feature_columns"]
    print("Loaded location classifier model.")
except Exception as e:
    clf = None
    feature_cols = []
    print("Warning: location classifier not found or failed to load:", e)


def is_likely_location(word, threshold=0.5):
    """Check if a word is likely a real location."""
    if not clf:
        return True  # fallback if model missing
    feat = pd.DataFrame([extract_features(word)])[feature_cols]
    prob = clf.predict_proba(feat)[0, 1]
    return prob > threshold
## -------------------------------------------------------------- ##

## This is the voting method to know if the places mentioned in the articles are around the NL/BE/DE or not. ##
def voting_country_from_locations(locations, gazetteer, threshold=0.6):
    # Initializing a vote counter
    votes = {"NL": 0, "BE": 0, "DE": 0}
    evidence = [] # Evidence will store which place name matched which country.

    # It loops through all places spaCy found, and if the place is in the gazetteer:
    # it gets its country code
    # adds +1 to that country’s votes
    # and saves evidence.
    for loc in locations:
        cc = gazetteer.get(loc.lower())
        if cc in votes:
            votes[cc] += 1
            evidence.append((loc, cc))

    total = sum(votes.values())
    if total == 0:
        return "uncertain", 0.0, [] # If no place matched NL/BE/DE, then returns "uncertain".

    # Picks the country with the highest number of votes.
    # Confidence = proportion of votes for that winner.
    best_cc, best_val = max(votes.items(), key=lambda kv: kv[1])
    confidence = best_val / total

    if confidence < threshold:
        return "uncertain", confidence, evidence
    
    return best_cc, confidence, evidence

## -------------------------------------------------------------- ##

## This is the method to filter out articles that aren't from the region ##
TARGET_COUNTRIES = {"NL", "BE", "DE"}

def filtering_articles_by_country(df, min_conf: float = 0.6):
    """
    Keep only rows confidently resolved to NL/BE/DE.
    Drops 'uncertain' and low-confidence rows.
    Returns a *copy* so the original df stays intact.
    """
    if "country" not in df.columns or "country_score" not in df.columns:
        raise ValueError("Missing required columns: 'country' and 'country_score'.")

    mask = (df["country"].isin(TARGET_COUNTRIES)) & (df["country_score"] >= min_conf)
    df_filtered_by_country = df.loc[mask].copy()

    kept = len(df_filtered_by_country)
    total = len(df)
    print(f"[geo_resolution] kept {kept}/{total} rows ({kept/total:.1%}) with country in {TARGET_COUNTRIES} and score ≥ {min_conf}") # peek

    return df_filtered_by_country

## -------------------------------------------------------------- ##

## This is the method that needs to be called from pre_processing.py file ##
def build_geo_df(json_path="nos_articles.json", min_conf=0.6):
    """
    Load NOS articles, enrich them with geo info, 
    and return only rows confidently resolved to NL/BE/DE.
    """
    import pandas as pd, json
    # I will load the nos_articles JSON file into a DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Clean the JSON file:
    df["clean_geo"] = df.apply(lambda r: clean_text_geo(get_raw_text_geo(r)), axis=1)

    # 3Detect candidate locations for all articles:
    texts = df["clean_geo"].tolist()
    docs = list(nlp.pipe(texts, batch_size=50))
    df["locations"] = [[ent.text for ent in doc.ents if ent.label_ in {"LOC","GPE"}] for doc in docs]

    # Filter out non-location words from detected locations using the logistic-regression model:
    all_candidates = {loc for locs in df["locations"] for loc in locs}
    if clf and len(all_candidates) > 0:
        feat_df = pd.DataFrame([extract_features(w) for w in all_candidates])
        feat_df["word"] = list(all_candidates)
        feat_df = feat_df.set_index("word")
        feat_df["is_loc"] = clf.predict_proba(feat_df[feature_cols])[:, 1] > 0.5
        loc_dict = feat_df["is_loc"].to_dict()
        df["locations"] = df["locations"].apply(lambda locs: [loc for loc in locs if loc_dict.get(loc, False)])
    print("[geo_filter] Filtered non-location terms from locations column.")

    # Get the voting per article:
    results = df["locations"].apply(lambda locs: voting_country_from_locations(locs, gazetteer))
    df[["country", "country_score", "country_evidence"]] = pd.DataFrame(results.tolist(), index=df.index)

    # Filter only the rows around the region:
    return filtering_articles_by_country(df, min_conf=min_conf)
## -------------------------------------------------------------- ##

