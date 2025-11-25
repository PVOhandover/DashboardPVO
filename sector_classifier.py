# sector_classifier.py
"""
Classifies SMEs mentioned in articles into CBS SBI sectors.
Uses NER (Named Entity Recognition) to extract company names for better accuracy.
"""

import re
import spacy
from typing import Dict, List, Optional

# Load Dutch language model for NER
try:
    nlp = spacy.load('nl_core_news_sm')
except OSError:
    print("⚠️ Warning: Dutch spaCy model not found. Company extraction disabled.")
    nlp = None

# CBS SBI sector mappings (expanded for better coverage)
# Format: sector_code -> (sector_name, [keywords])
SECTOR_KEYWORDS = {
    # Motor vehicles
    "45": ("Wholesale and retail trade and repair of motor vehicles", 
           ["garage", "autodealers", "autowerkplaats", "autobedrijf", "carwash", "autohandel", 
            "apk", "bandenservice", "autoschade", "occasion", "autodealer"]),
    
    # Food retail - supermarkets
    "47.11": ("Retail sale in non-specialised stores with food predominance",
              ["supermarkt", "supermarket", "albert heijn", "jumbo", "lidl", "aldi", "plus", 
               "coop", "dirk", "vomar", "deen", "nettorama", "boni"]),
    
    # General retail stores
    "47.19": ("Other retail sale in non-specialised stores",
              ["warenhuis", "department store", "winkelcentrum", "shopping center", "v&d", 
               "hema", "action", "kruidvat", "etos", "trekpleister"]),
    
    # Specialized food stores
    "47.2": ("Retail sale of food in specialised stores",
             ["bakkerij", "bakery", "slagerij", "butcher", "viswinkel", "groenteboer", 
              "banketbakkerij", "patisserie", "kaaswinkel", "delicatessen", "broodjeszaak",
              "poelier", "groentewinkel"]),
    
    # Other specialized retail
    "47.7": ("Retail sale of other goods in specialised stores",
             ["kledingwinkel", "fashion", "juwelier", "jewelry", "boekhandel", "drogist",
              "schoenwinkel", "modewinkel", "opticien", "brilwinkel", "sportwinkel",
              "speelgoedwinkel", "bloemenwinkel", "dierenwinkel", "fietsenwinkel"]),
    
    # Restaurants
    "56.10": ("Restaurants and mobile food service activities",
              ["restaurant", "bistro", "eetcafe", "cafetaria", "snackbar", "pizzeria",
               "chinees", "italiaans", "sushi", "kebab", "lunchroom", "brasserie",
               "afhaalrestaurant", "grillroom", "steakhouse", "friettent", "döner"]),
    
    # Bars and cafés
    "56.3": ("Beverage serving activities",
            ["café", "bar", "pub", "kroeg", "discotheek", "nightclub", "terras", 
             "tapperij", "proeflokaal", "loungebar", "cocktailbar", "eetcafé"]),
    
    # Hotels and accommodation
    "55": ("Accommodation",
           ["hotel", "motel", "pension", "bed and breakfast", "vakantiepark", "hostel",
            "resort", "appartement", "vakantiewoning", "camping", "b&b"]),
    
    # Real estate
    "68": ("Real estate activities",
           ["makelaar", "vastgoed", "real estate", "makelaardij", "hypotheek",
            "woningbemiddeling", "vastgoedmakelaar"]),
    
    # IT and technology services
    "62": ("Computer programming and consultancy",
           ["it-bedrijf", "software", "ict", "programmeur", "developer", "webdesign", 
            "hosting", "webshop", "online", "digitaal", "cybersecurity", "netwerk",
            "cloud", "app", "website", "platform", "tech", "data"]),
    
    # Business services
    "69-82": ("Business services",
              ["accountant", "advocaat", "lawyer", "notaris", "consultant", "adviesbureau",
               "administratiekantoor", "boekhouder", "juridisch", "belasting"]),
    
    # Healthcare
    "86": ("Human health activities",
           ["huisarts", "tandarts", "dentist", "apotheek", "pharmacy", "kliniek",
            "ziekenhuis", "arts", "fysio", "psycholoog", "therapeut", "praktijk"]),
    
    # Personal services
    "96": ("Other personal service activities",
           ["kapper", "hairdresser", "schoonheidssalon", "fitness", "sportschool",
            "nagelstudio", "wellness", "spa", "massage", "beautysalon", "gym"]),
    
    # Construction
    "41-43": ("Construction",
              ["bouw", "aannemer", "schilder", "loodgieter", "elektricien", "timmerman", 
               "stukadoor", "dakdekker", "installateur", "renovatie", "verbouwing"]),
    
    # Transportation
    "49-53": ("Transportation and storage",
              ["transport", "vervoer", "logistiek", "taxi", "vrachtwagen", "distributie", 
               "opslag", "koerier", "bezorging", "expeditie", "chauffeur"]),
    
    # Architecture and engineering
    "71": ("Architectural and engineering activities",
           ["architect", "ingenieur", "bouwkundig", "constructie", "adviesbureau",
            "technisch", "ontwerp", "engineering"]),
    
    # Retail - general (catch-all for retail mentions)
    "47": ("Retail trade",
           ["winkel", "detailhandel", "retail", "webwinkel", "shop", "verkoop",
            "handelaar", "winkelier"]),
    
    # General business (catch-all)
    "00": ("General business",
           ["bedrijf", "onderneming", "zaak", "organisatie", "firma", "concern",
            "ondernemer", "eigenaar", "mkb", "kmo", "handel", "commercieel"])
}


def extract_companies(text: str) -> List[str]:
    """
    Extract company/organization names from text using NER.
    
    Args:
        text: Article text to analyze
        
    Returns:
        List of company names found
    """
    if not nlp or not text:
        return []
    
    try:
        doc = nlp(text[:5000])  # Limit to first 5000 chars for performance
        companies = []
        
        for ent in doc.ents:
            # Extract organizations (ORG) and some persons (might be business names)
            if ent.label_ in ['ORG', 'PERSON']:
                company_name = ent.text.strip()
                # Filter out very short names (likely not real companies)
                if len(company_name) > 2:
                    companies.append(company_name)
        
        return companies
    except Exception as e:
        print(f"⚠️ Error extracting companies: {e}")
        return []


def is_company_victim(company_name: str, text: str) -> bool:
    """
    Check if a company is actually a victim/target in the article.
    Looks for victim indicators near the company name.
    
    Args:
        company_name: Name of the company
        text: Full article text
        
    Returns:
        True if company appears to be a victim, False otherwise
    """
    text_lower = text.lower()
    company_lower = company_name.lower()
    
    # Find where company is mentioned in text
    if company_lower not in text_lower:
        return True  # Default to true if can't find (safe assumption)
    
    # Get text window around company name (50 chars before and after)
    company_pos = text_lower.find(company_lower)
    start = max(0, company_pos - 50)
    end = min(len(text_lower), company_pos + len(company_lower) + 50)
    context = text_lower[start:end]
    
    # Victim indicators (words that suggest the company was affected)
    victim_indicators = [
        'getroffen', 'hit', 'slachtoffer', 'victim', 'gehacked', 'hacked',
        'aangevallen', 'attacked', 'overvallen', 'robbed', 'inbraak', 'burglary',
        'gestolen', 'stolen', 'verloor', 'lost', 'schade', 'damage',
        'ransomware', 'malware', 'lek', 'leak', 'datalek', 'data breach',
        'bij', 'at', 'in', 'van', 'of'  # Prepositions indicating location/possession
    ]
    
    # Non-victim indicators (words suggesting NOT a victim)
    non_victim_indicators = [
        'niet getroffen', 'not affected', 'niet geraakt', 'not hit',
        'geen schade', 'no damage', 'niet bij betrokken', 'not involved',
        'ontsnapte', 'escaped', 'vermeed', 'avoided', 'beschermd', 'protected',
        'helpt', 'helps', 'onderzoekt', 'investigates', 'politie', 'police'
    ]
    
    # Check for non-victim indicators first (stronger signal)
    for indicator in non_victim_indicators:
        if indicator in context:
            return False  # Explicitly mentioned as NOT affected
    
    # Check for victim indicators
    for indicator in victim_indicators:
        if indicator in context:
            return True  # Company appears to be victim
    
    # Default: if no clear signal, assume mentioned = relevant
    return True


def classify_single_company(company_name: str, article_text: str, check_victim: bool = True) -> Dict[str, str]:
    """
    Classify a single company into a CBS SBI sector.
    Now checks if company is actually a victim!
    
    Args:
        company_name: Name of the company to classify
        article_text: Full article text for context
        check_victim: Whether to check if company is a victim (default: True)
        
    Returns:
        Dictionary with 'company_name', 'sector_code', 'sector_name', and 'is_victim'
    """
    # Check if this company is actually a victim
    is_victim = True
    if check_victim:
        is_victim = is_company_victim(company_name, article_text)
    
    # If not a victim, mark as irrelevant
    if not is_victim:
        return {
            "company_name": company_name,
            "sector_code": "not_affected",
            "sector_name": "Not affected by incident",
            "is_victim": False
        }
    
    # Prepare search text - prioritize company name with context
    search_text = f"{company_name.lower()} {company_name.lower()} {article_text.lower()}"
    
    # Score each sector based on keyword matches
    sector_scores = {}
    for code, (name, keywords) in SECTOR_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in search_text)
        if score > 0:
            sector_scores[code] = (score, name)
    
    # Return the sector with highest score
    if sector_scores:
        best_code = max(sector_scores, key=lambda x: sector_scores[x][0])
        return {
            "company_name": company_name,
            "sector_code": best_code,
            "sector_name": sector_scores[best_code][1],
            "is_victim": True
        }
    
    return {
        "company_name": company_name,
        "sector_code": "unknown",
        "sector_name": "Unclassified",
        "is_victim": True
    }

def classify_sector_from_text(text: str, company_name: Optional[str] = None) -> Dict:
    """
    Classify all companies mentioned in text into CBS SBI sectors.
    Now classifies EACH company separately!
    
    Args:
        text: Full article text to analyze
        company_name: Optional specific company name (for backward compatibility)
        
    Returns:
        Dictionary with:
        - 'primary_sector': Main sector (most common or first company's sector)
        - 'companies_classified': List of all companies with their sectors
        - 'sectors_mentioned': List of unique sectors in article
    """
    if not text:
        return {
            "primary_sector": {"sector_code": "unknown", "sector_name": "Unclassified"},
            "companies_classified": [],
            "sectors_mentioned": []
        }
    
    # Extract company names
    companies_found = []
    if nlp:
        companies_found = extract_companies(text)
    
    # If specific company provided, add it
    if company_name and company_name not in companies_found:
        companies_found.insert(0, company_name)
    
    # Classify each company separately
    companies_classified = []
    if companies_found:
        for company in companies_found[:10]:  # Limit to first 10 to avoid processing too many
            classification = classify_single_company(company, text)
            companies_classified.append(classification)
    
    # Determine primary sector (most common, or first company's sector)
    if companies_classified:
        # Count sector occurrences
        sector_counts = {}
        for comp in companies_classified:
            code = comp['sector_code']
            if code != 'unknown':
                sector_counts[code] = sector_counts.get(code, 0) + 1
        
        if sector_counts:
            # Most common sector becomes primary
            primary_code = max(sector_counts, key=sector_counts.get)
            primary_name = next(c['sector_name'] for c in companies_classified if c['sector_code'] == primary_code)
            primary_sector = {"sector_code": primary_code, "sector_name": primary_name}
        else:
            # All unknown, use first company
            primary_sector = {"sector_code": companies_classified[0]['sector_code'], 
                            "sector_name": companies_classified[0]['sector_name']}
        
        # Get unique sectors mentioned
        sectors_mentioned = list(set(
            c['sector_code'] for c in companies_classified if c['sector_code'] != 'unknown'
        ))
    else:
        # No companies found, classify article as a whole (fallback)
        sector_scores = {}
        search_text = text.lower()
        for code, (name, keywords) in SECTOR_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in search_text)
            if score > 0:
                sector_scores[code] = (score, name)
        
        if sector_scores:
            best_code = max(sector_scores, key=lambda x: sector_scores[x][0])
            primary_sector = {"sector_code": best_code, "sector_name": sector_scores[best_code][1]}
            sectors_mentioned = [best_code]
        else:
            primary_sector = {"sector_code": "unknown", "sector_name": "Unclassified"}
            sectors_mentioned = []
    
    return {
        "primary_sector": primary_sector,
        "companies_classified": companies_classified,
        "sectors_mentioned": sectors_mentioned
    }


def add_sector_classification(df):
    """
    Add sector classification to each article in the DataFrame.
    Now classifies EACH company separately and tracks multiple sectors!
    
    Args:
        df: DataFrame with SME-filtered articles
        
    Returns:
        DataFrame with added sector classification columns
    """
    print("🏢 SECTOR CLASSIFICATION STARTING...")
    print(f"DataFrame shape: {df.shape}")
    print("🏢 Classifying EACH company by sector with NER extraction...")
    
    primary_sectors = []
    companies_classified_list = []
    sectors_mentioned_list = []
    
    for _, row in df.iterrows():
        # Get the full text
        text = row.get('full_text', '') or f"{row.get('title', '')} {row.get('summary', '')}"
        
        # Classify all companies in article
        result = classify_sector_from_text(text)
        
        primary_sectors.append(result['primary_sector'])
        companies_classified_list.append(result['companies_classified'])
        sectors_mentioned_list.append(result['sectors_mentioned'])
    
    # Add new columns
    df['sector_info'] = primary_sectors
    df['companies_with_sectors'] = companies_classified_list
    df['all_sectors_in_article'] = sectors_mentioned_list
    
    # Print statistics
    total_articles = len(df)
    
    # Primary sector stats
    sector_counts = {}
    for sector in primary_sectors:
        code = sector['sector_code']
        sector_counts[code] = sector_counts.get(code, 0) + 1
    
    classified = sum(1 for s in primary_sectors if s['sector_code'] != 'unknown')
    
    # Company classification stats
    articles_with_companies = sum(1 for companies in companies_classified_list if len(companies) > 0)
    total_companies = sum(len(companies) for companies in companies_classified_list)
    total_companies_classified = sum(
        sum(1 for c in companies if c['sector_code'] != 'unknown') 
        for companies in companies_classified_list
    )
    
    # Multi-sector articles
    multi_sector_articles = sum(1 for sectors in sectors_mentioned_list if len(sectors) > 1)
    
    print(f"\n📊 Sector classification results:")
    print(f"  Total articles: {total_articles}")
    print(f"  Articles classified: {classified} ({classified/total_articles*100:.1f}%)")
    print(f"  Unclassified: {total_articles - classified} ({(total_articles-classified)/total_articles*100:.1f}%)")
    
    if nlp:
        print(f"\n🏢 Company-level classification:")
        print(f"  Articles with companies: {articles_with_companies} ({articles_with_companies/total_articles*100:.1f}%)")
        print(f"  Total companies found: {total_companies}")
        print(f"  Companies classified: {total_companies_classified} ({total_companies_classified/total_companies*100:.1f}% of companies)")
        print(f"  Articles mentioning multiple sectors: {multi_sector_articles} ({multi_sector_articles/total_articles*100:.1f}%)")
        print(f"  Average companies per article: {total_companies/total_articles:.1f}")
    
    print(f"\n  Primary sector breakdown:")
    sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
    for code, count in sorted_sectors:
        # Find sector name
        sector_name = next((s['sector_name'] for s in primary_sectors if s['sector_code'] == code), "Unknown")
        percentage = (count / total_articles) * 100
        print(f"    {code}: {sector_name} - {count} articles ({percentage:.1f}%)")
    
    # Show examples of multi-sector articles
    if multi_sector_articles > 0:
        print(f"\n  Sample multi-sector articles:")
        shown = 0
        for idx, row in df.iterrows():
            if len(row['all_sectors_in_article']) > 1 and shown < 3:
                companies = row['companies_with_sectors']
                print(f"    • Article: {row['title'][:60]}...")
                print(f"      Sectors: {', '.join(row['all_sectors_in_article'])}")
                print(f"      Companies: {', '.join([c['company_name'] for c in companies[:3]])}...")
                shown += 1
    
    # Show sample company classifications
    if total_companies > 0:
        print(f"\n  Sample company classifications:")
        shown = 0
        for idx, row in df.iterrows():
            if len(row['companies_with_sectors']) > 0 and shown < 5:
                for company in row['companies_with_sectors'][:2]:  # Show first 2 companies
                    if company['sector_code'] != 'unknown':
                        print(f"    • {company['company_name']} → {company['sector_code']} ({company['sector_name']})")
                        shown += 1
                        if shown >= 5:
                            break
    
    return df