
#
# header comment
#

# ------ imports & constants ------
import re
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel

ABSTAIN = -1 # Meaning this LF(labeling function) cannot decide for this example.
YES = 1 # Meaning this LF(labeling function) believes the article is about an SME.
NOT = 0 # Meaning this LF(labeling function) believes the article is not about an SME.

# ------ labeling functions ------

## layer 1 crime

@labeling_function()
def lf_financial_crime(x):
    text = (x.get("clean_geo") or "").lower()
    return YES if re.search(
        r"\b(fraude|oplichting|witwassen|corruptie|verduistering|"
        r"belastingfraude|subsidiefraude|verzekeringsfraude|"
        r"beleggingsfraude|nepfactuur|valsheid in geschrifte)\b",
        text
    ) else ABSTAIN

@labeling_function()
def lf_cyber_crime(x):
    text = (x.get("clean_geo") or "").lower()
    return YES if re.search(
        r"\b(hack|gehackt|hacker|cyberaanval|ransomware|phishing|"
        r"smishing|spoofing|datalek|datadiefstal|malware|ddos)\b",
        text
    ) else ABSTAIN

@labeling_function()
def lf_police_justice(x):
    text = (x.get("clean_geo") or "").lower()
    return YES if re.search(
        r"\b(politie|recherche|aangehouden|arrestatie|"
        r"verdachte|onderzoek|officier van justitie|"
        r"rechtbank|veroordeeld|vonnis|celstraf|taakstraf)\b",
        text
    ) else ABSTAIN

@labeling_function()
def lf_general_crime(x):
    text = (x.get("clean_geo") or "").lower()
    return YES if re.search(
        r"\b(moord|doodslag|mishandeling|bedreiging|"
        r"inbraak|diefstal|beroving|roof|ontvoering|afpersing)\b",
        text
    ) else ABSTAIN


@labeling_function()
def lf_organizational_crime(x):
    text = (x.get("clean_geo") or "").lower()
    return YES if re.search(
        r"\b(bedrijf|organisatie|onderneming|bank|instelling|"
        r"directeur|bestuurder)\b", text
    ) and re.search(
        r"\b(fraude|witwassen|corruptie|oplichting|"
        r"onderzoek|aangifte)\b", text
    ) else ABSTAIN

@labeling_function()
def lf_not_sports_entertainment(x):
    text = (x.get("clean_geo") or "").lower()
    return NOT if re.search(
        r"\b(honkbal|voetbal|sport|theater|film|serie|muziek|concert|festival|wedstrijden)\b", 
        text
    ) else ABSTAIN

@labeling_function()
def lf_not_politics(x):
    text = (x.get("clean_geo") or "").lower()
    return NOT if re.search(
        r"\b(verkiezing|beleid|debat|parlement|kamer|motie|wetgeving)\b",
        text
    ) else ABSTAIN

@labeling_function()
def lf_not_weather(x):
    text = (x.get("clean_geo") or "").lower()
    return NOT if re.search(
        r"\b(storm|regen|overstroming|hittegolf|weer|natuurbrand)\b",
        text
    ) else ABSTAIN

@labeling_function()
def lf_not_lifestyle(x):
    text = (x.get("clean_geo") or "").lower()
    return NOT if re.search(
        r"\b(reizen|wonen|eten|familie|relatie|vrijetijd|hobby)\b",
        text
    ) else ABSTAIN

@labeling_function()
def lf_not_transport(x):
    text = (x.get("clean_geo") or "").lower()
    return NOT if re.search(
        r"\b(trein|bus|verkeer|spoor|station|infrastructuur|ov)\b",
        text
    ) else ABSTAIN

## layer 2 sme

# Generic entrepreneurship terms → ondernemer, zelfstandige, start-up, zzp.

SME_TERMS = [
    "mkb", "midden- en kleinbedrijf", "kleinbedrijf", "kleine onderneming",
    "zzp", "zzp'er", "ondernemer", "onderneming", "bedrijf", "winkel", "zaak",
    "restaurant", "horeca", "café", "bakkerij", "slagerij", "kapsalon",
    "garage", "autobedrijf", "webshop"
]

@labeling_function()
def lf_sme_terms(x):
    text = (x.get("clean_geo") or "").lower()
    if any(term in text for term in SME_TERMS):
        return YES
    return ABSTAIN

CRIME_TERMS = [
    "overval", "inbraak", "diefstal", "afpersing", "beroving",
    "bedreiging", "oplichting", "fraude", "vandalisme",
    "brandstichting", "ramkraak"
]

@labeling_function()
def lf_crime_against_business(x):
    text = (x.get("clean_geo") or "").lower()
    if any(c in text for c in CRIME_TERMS) and any(b in text for b in SME_TERMS):
        return YES
    return ABSTAIN

OWNER_TERMS = ["eigenaar", "ondernemer", "zaakvoerder", "bedrijfsleider"]

@labeling_function()
def lf_owner_victim(x):
    text = (x.get("clean_geo") or "").lower()
    if any(o in text for o in OWNER_TERMS) and any(c in text for c in CRIME_TERMS):
        return YES
    return ABSTAIN


@labeling_function()
def lf_private_home(x):
    text = (x.get("clean_geo") or "").lower()
    if "woning" in text or "thuis" in text or "bewoner" in text:
        return NOT
    return ABSTAIN


@labeling_function()
def lf_public_institution(x):
    text = (x.get("clean_geo") or "").lower()
    if any(word in text for word in ["gemeente", "overheid", "school", "ziekenhuis", "politie", "brandweer"]):
        return YES
    return NOT

TECH_TERMS = [
    "software", "app", "website", "webshop", "cloud", "it-bedrijf", 
    "it-dienst", "it-consultant", "ict", "server", "netwerk", "database",
    "cybersecurity"
]

@labeling_function()
def lf_tech_terms(x):
    text = (x.get("clean_geo") or "").lower()
    if any(term in text for term in TECH_TERMS):
        return YES
    return ABSTAIN

@labeling_function()
def lf_general_crime(x):
    text = (x.get("clean_geo") or "").lower()
    if any(word in text for word in ["straat", "park", "wijk", "buurt", "woninginbraak", "diefstal uit huis"]):
        return NOT
    return ABSTAIN

CYBER_CRIME_TERMS = [
    "hack", "hacken", "datadiefstal", "ransomware", "phishing",
    "ddos", "cyberaanval", "malware", "virusscanner", "identiteitsfraude"
]

@labeling_function()
def lf_cybercrime_terms(x):
    text = (x.get("clean_geo") or "").lower()
    if any(term in text for term in CYBER_CRIME_TERMS):
        return YES
    return ABSTAIN

ONLINE_TERMS = ["webshop", "online platform", "e-commerce", "internetbedrijf", "digitaal bedrijf"]

@labeling_function()
def lf_online_business(x):
    text = (x.get("clean_geo") or "").lower()
    if any(term in text for term in ONLINE_TERMS) and any(term in text for term in CYBER_CRIME_TERMS):
        return YES
    return ABSTAIN


# ----------- snorkel ------------

crime_layer = [
    lf_cyber_crime,
    lf_financial_crime,
    lf_general_crime,
    lf_not_lifestyle,
    lf_not_transport,
    lf_not_weather,
    lf_not_politics,
    lf_police_justice,
    lf_organizational_crime,
    lf_not_sports_entertainment
]

sme_layer = [
    lf_sme_terms,
    lf_crime_against_business,
    lf_owner_victim,
    lf_private_home,
    lf_public_institution,
    lf_tech_terms,
    lf_online_business
]

def apply_lfs(df, lfs):
    """Apply labeling functions to a DataFrame and return the label matrix."""
    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(df)
    return L

def train_label_model(L, cardinality=2, n_epochs=500, seed=42):
    """Train a Snorkel LabelModel and return the model and predictions."""
    label_model = LabelModel(cardinality=cardinality, verbose=True)
    label_model.fit(L_train=L, n_epochs=n_epochs, log_freq=100, seed=seed)
    preds = label_model.predict(L)
    probs = label_model.predict_proba(L)
    return label_model, preds, probs

def filter_df(df, preds, probs, min_confidence=0.7):
    """Filter DataFrame rows based on minimum confidence."""
    df = df.copy()
    df["preds"] = preds

    # Get probability of the predicted class
    df["pred_prob"] = [
        probs[i][preds[i]] for i in range(len(preds))
    ]

    filtered_df = df[df["pred_prob"] >= min_confidence]
    return filtered_df

def run_crime_snorkel(df, lfs=crime_layer, min_confidence=0.6):
    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(df=df)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L, n_epochs=200, log_freq=50, seed=123)
    
    probs = label_model.predict_proba(L=L)  # shape (n,2)
    df["crime_prob"] = probs[:, 1]

    df = df[df["crime_prob"] >= min_confidence].copy()
    
    return df

def run_sme_snorkel(df, lfs=sme_layer, min_confidence=0.6):
    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(df=df)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L, n_epochs=200, log_freq=50, seed=123)
    
    probs = label_model.predict_proba(L=L)  # shape (n,2)
    df["sme_probability"] = probs[:, 1]

    df = df[df["sme_probability"] >= min_confidence].copy()
    
    return df

def old_run_snorkel(df, layer1_lfs=crime_layer, layer2_lfs=sme_layer, min_confidence=0.6):
    # ---- Layer 1 ----
    L1 = apply_lfs(df, layer1_lfs)
    lm1, preds1, probs1 = train_label_model(L1)
    
    df_layer2 = filter_df(df, preds1, probs1, min_confidence)
    print(f"Layer 1 filtered {len(df_layer2)} articles for Layer 2")
    
    if len(df_layer2) == 0:
        return pd.DataFrame(), None
    # ---- Layer 2 ----
    L2 = apply_lfs(df_layer2, layer2_lfs)
    lm2, preds2, probs2 = train_label_model(L2)

    df_layer2 = filter_df(df_layer2, preds2, probs2, min_confidence)
    df_layer2["preds_layer2"] = preds2

    return df_layer2, lm2
