# PVO_Limburg

# Project Prototype Overview
>> This repository contains a prototype system for PVO Limburg. It consists of a data pipeline and a Streamlit based dashboard for collecting, filtering, and visualizing public information sources (news and reports), with a focus on items relevant to SMEs

>> The current prototype can:
>> collect new articles from multiple sources (RSS feeds and scraping)
>> merge and deduplicate collected articles into one master dataset
>> filter articles by geographic relevance and SME/crime relevance
>> enrich articles with locations, sector labels, and keywords
>> present the results in an interactive dashboard (search, filters, map, spotlight, keyword summaries and trends)

# Main folders and files
>> webScrapers/
>> Contains the source scrapers (mostly RSS-based, some web scraping). Each script writes JSON output into scrapedArticles/

>> scrapedArticles/
>> Contains per-source JSON files produced by the scrapers. These are the raw inputs for the merge step

>> merge_jsons.py
>> Merges all JSON files in scrapedArticles/ into all_articles.json and removes duplicates based on URL/canonical URL logic
>> It also maintains refresh state under data/refresh/ for incremental updates

>> all_articles.json
>> The merged master dataset used as input for the preprocessing step

>> geo_filter.py
>> Geographic filtering and location extraction. Uses spaCy NER plus GeoNames gazetteers (NL/BE/DE) and a location validity model

>> layered_filter.py and sme_filter.py
>> Weak supervision filters (Snorkel) for content relevance. These assign probability scores (crime_prob and sme_probability) and keep only items above thresholds

>> narrow_locations.py
>> Reduces multiple detected location mentions to a single most plausible location using a simple context proximity heuristic

>> sector_classifier.py
>> Adds coarse sector classification using spaCy-based entity extraction and keyword matching

>> pre_process.py
>> Main preprocessing script. Reads all_articles.json, runs filtering/enrichment stages, and writes dashboard-ready output to keywords/all_articles_keywords.json
>> Also uses preprocess_cache.json to avoid reprocessing unchanged articles

>> update_data.py
>> One command runner that executes ingestion + preprocessing and then publishes the final dataset to cache/public/latest.json
>> This is the easiest way to run everything locally

>> cache/public/latest.json
>> Published dataset used for deployments. Streamlit Cloud can read this through DATA_URL

>> dashboard.py
>> Streamlit dashboard UI. Reads data either from DATA_URL (remote JSON) or from keywords/all_articles_keywords.json (local mode)




# How to run the project locally

>> 1) Setup a virtual environment and install dependencies 

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
python -m spacy download nl_core_news_sm
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
python -m spacy download nl_core_news_sm
```


>> 2) Run the full pipeline (scrape + process + publish)
```bash
python update_data.py
```

>> After a successful run, these files are updated/created:
>> all_articles.json
>> keywords/all_articles_keywords.json
>> cache/public/latest.json
>> preprocess_cache.json

>> 3) Run the dashboard
```bash
streamlit run dashboard.py
```

If needed:
```bash
python -m streamlit run dashboard.py
```

To stop the dashboard:
>> press Ctrl + C in the terminal



# Streamlit Cloud / Deployment notes
>> Streamlit Cloud runs only dashboard.py. It does not scrape by itself.

>> The intended setup is:
>> GitHub Actions runs update_data.py on a schedule and commits cache/public/latest.json
>> Streamlit Cloud loads the dataset from DATA_URL

>> DATA_URL should point to the raw GitHub file for cache/public/latest.json, for example:
>> https://raw.githubusercontent.com/PVOhandover/DashboardPVO/main/cache/public/latest.json

>> If the repository or branch changes, DATA_URL must be updated accordingly.



# Common issues
>> spaCy model error: Can't find model 'nl_core_news_sm'
Run:
```bash
python -m spacy download nl_core_news_sm
```


>> Use the sidebar Reset filters button to return to the full range.

>> Map shows fewer points than the number of matching articles
>> Only articles that have cached coordinates (cache/geocode_cache.json) and fall within the configured region bounding boxes are plotted.


## Live Dashboard

The deployed dashboard is available here:

https://dashboardpvo.streamlit.app/

Open the link in a browser to use the application

# What We Have Learned
Through this prototype, we learned how to transform unstructured information into a usable overview that can support decision making for SMEs and organizations such as PVO Limburg. We gained hands-on experience with a full pipeline from data collection and integration (RSS and scraping, merging and deduplication) to text preprocessing and filtering for relevance.

We explored NLP techniques such as named entity recognition, weak supervision with labeling functions, and keyword extraction, and saw how these methods can be combined to reduce information overload while preserving transparency. In addition, we learned to design an interactive dashboard in Streamlit that supports exploration through filters, map based views, and trend summaries, which helped connect the technical pipeline to a practical interface for end users.
