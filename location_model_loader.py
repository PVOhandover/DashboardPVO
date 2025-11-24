# location_model_loader.py
import joblib
import pandas as pd

# This is implemented to load the logistic regression model only once and 
# not everytime the pre-process runs to not slow down the pipeline.
try:
    bundle = joblib.load("models/location_classifier_latest.pkl")
    clf = bundle["model"]
    feature_cols = bundle["feature_columns"]
    print("✅ Location classifier loaded.")
except Exception as e:
    clf = None
    feature_cols = []
    print("⚠️ Could not load location classifier:", e)

def is_likely_location_from_model(word, extract_features_fn, threshold=0.5):
    """Use pretrained model to test if word is likely a location."""
    if clf is None:
        return True  # fallback: treat all as valid
    feat = pd.DataFrame([extract_features_fn(word)])[feature_cols]
    prob = clf.predict_proba(feat)[0, 1]
    return prob > threshold
