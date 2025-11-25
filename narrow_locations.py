# Currently, after geo-filtering using Named Entity Recognition (NER) and voting-based comparison 
# between detected locations and GeoNames gazetteers, we often end up with multiple location 
# mentions per article. To accurately find the true location of the crime and SMEs we implement
# a “contextual proximity heuristic”, which is commonly used in entity disambiguation and relation extraction. 
# Basically, we measure the distance between the location and business or crime mentions (context words),
# and select the final location based on the closest association.

# We first tried to do this in geo_filter.py file, but as the number of articles increased, it became
# computationally heavy, so we are doing this step after all ofthe filtering 
# since this step is done mostly for visualization.

# The methods on this file are used AFTER geo_filtering and sme_filter
# but BEFORE the word-tokenization step at the pre-processing file. ##

## --------------------------------------------------------------##

# This method is used to identify business or crime mentions (context words) within the same sentence.
# It returns returns a list of integer positions, where each number is the character index in the text
# where a business or crime keyword starts.
# To identify these business-related and crime-related we use the same keywords from the LFs in sme_filter.py file.
# After implementing a named entity recognition file for detecting Dutch SMEs, than we should add that to the business
# and crime terms here.

import re

def find_context_word_positions(text):

    if not text:
        return []

    lower_text = text.lower()
    positions = []

    business_terms = [ "mkb", "midden- en kleinbedrijf", "kmo", "kleine onderneming", "kleine bedrijven", "small and medium enterprise", "mkb-ondernemer", "mkb-ondernemers", "mkb-bedrijf", "mkb-bedrijven", "mkb-sector", "mkb'er", "mkb'ers", "ondernemersvereniging", "ondernemersloket", "bedrijf", "bedrijven", "onderneming", "ondernemingen", "zaak", "zaken", "ondernemingshuis", "bedrijfsleven", "organisatie", "organisaties", "bedrijfstak", "bedrijfssector", "ondernemer", "ondernemers", "zelfstandige", "zelfstandigen", "zzp", "start-up", "startup", "startups", "ondernemerschap", "freelancer", "freelancers", "bedrijf starten", "bedrijf oprichten", "winkel", "horeca", "restaurant", "café", "bar", "hotel", "bouwbedrijf", "installatiebedrijf", "aannemer", "logistiek", "transportbedrijf", "koerier", "energiebedrijf", "ict", "softwarebedrijf", "adviesbureau", "consultancy", "marketingbureau", "juridisch advies", "accountantskantoor", "verzekeringskantoor", "boekhoudkantoor", "bank", "verzekeraar", "makelaar", "vastgoed", "onderwijsinstelling", "praktijk", "kliniek", "zorginstelling", "sportschool", "fitnesscentrum", "sportvereniging" ]
    crime_terms = [ "fraude", "oplichting", "witwassen", "corruptie", "diefstal", "verduistering", "afpersing", "valsheid in geschrifte", "onderzoek naar", "aangifte", "cyber", "cybercrime", "digitale", "phishing", "ransomware", "hack", "hacker", "veiligheid", "weerbaarheid", "inbraak", "aanval", "incident", "drugs", "ongeluk", "ramp", "brand", "moord", "criminaliteit", "aanrijding", "explosie", "rellen" ]

    for kw in business_terms + crime_terms:
        for match in re.finditer(r"\b" + re.escape(kw) + r"\b", lower_text):
            positions.append(match.start())

    return sorted(positions)
# --------------------------------------------------------------

# This method is used to reduce multiple locations of an article to 1 true location of the SME or crime-scene.
# If context words exist, then the true location returned is the one location that is closest to the context words.
# If no context words are found, it keeps all locations as is.
# If multiple locations are equally close, it chooses the first one seen in text.
def narrow_down_locations(text, locations):

    if not locations:
        return []

    context_positions = find_context_word_positions(text)
    if not context_positions:
        print("There is no context word in the article, so locations cannot be narrowed down to 1.")
        return locations # If there is no match with the context words, then keep the locations as it is.

    distances = []
    for loc in locations:
        for match in re.finditer(re.escape(loc), text, re.IGNORECASE):
            loc_pos = match.start()
            nearest_distance = min(abs(loc_pos - cpos) for cpos in context_positions)
            distances.append((nearest_distance, loc_pos, loc))

    if not distances:
        print("Distance cannot be computed, so locations cannot be narrowed down to 1.")
        return locations

    distances.sort(key=lambda x: (x[0], x[1]))
    closest_loc = distances[0][2]
    return [closest_loc]
# --------------------------------------------------------------


# This is the method that needs to be called from pre_processing.py file. ##
def apply_location_narrowing(df):
    """
    Takes a DataFrame that already has:
        - 'clean_geo' (full text)
        - 'locations' (list of location strings)
    Returns the same DataFrame with narrowed 'locations'.
    """
    df = df.copy()
    print(f"Applying contextual location narrowing on {len(df)} SME articles...")

    before_counts = df["locations"].apply(len)
    df["locations"] = df.apply(lambda r: narrow_down_locations(r["clean_geo"], r["locations"]), axis=1)
    after_counts = df["locations"].apply(len)

    narrowed = sum((before_counts > 1) & (after_counts == 1))
    unchanged = sum(before_counts == after_counts)
    no_locs = sum(before_counts == 0)

    # TEST:
    print("\n[narrow_locations] --- Summary ---")
    print(f"Total articles: {len(df)}")
    print(f"Multiple → single location: {narrowed}")
    print(f"Unchanged: {unchanged}")
    print(f"No locations: {no_locs}")
    print(f"Reduction ratio: {narrowed / max(1, len(df)):.2%}")
    print("----------------------------------\n")

    return df
# --------------------------------------------------------------
