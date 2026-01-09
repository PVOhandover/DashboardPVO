import streamlit as st
import pandas as pd
import json
import os
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime, date, timedelta
import altair as alt
from sector_classifier import add_sector_classification
from html import escape as _e

# streamlit run dashboard.py


#streamlit help
def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def _snippet(txt, n=220):
    if not txt: return ""
    s = " ".join(str(txt).split())
    return (s[:n] + "…") if len(s) > n else s


def _parse_to_date(x):
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x).date()
        except Exception:
            pass
    return datetime.today().date()

def _clamp_date_range(min_d: date, max_d: date, value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return (min_d, max_d)
    sd, ed = _parse_to_date(value[0]), _parse_to_date(value[1])
    if sd < min_d: sd = min_d
    if sd > max_d: sd = max_d
    if ed < min_d: ed = min_d
    if ed > max_d: ed = max_d
    if sd > ed:
        sd, ed = min_d, max_d
    return (sd, ed)



def _time_ago(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    now = pd.Timestamp.utcnow().tz_localize(None)
    delta = now - ts
    d = delta.days
    h = delta.seconds // 3600
    m = (delta.seconds % 3600) // 60
    if d > 0:
        return f"{d}d {h}h"
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


#preset storage
PRESETS_FILE = os.path.join("cache", "filter_presets.json")

def _ensure_cache_dir():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("digests", exist_ok=True)

def _load_presets():
    _ensure_cache_dir()
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_presets(presets: dict):
    _ensure_cache_dir()
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2, default=str)


def _reset_all_filters(feed_options_all, _min_date, _max_date):
    st.session_state["text_filter"] = ""
    st.session_state["selected_feeds"] = list(feed_options_all)
    st.session_state["location_search"] = ""
    dr = (_min_date, _max_date)
    st.session_state["date_range"] = dr
    st.session_state["date_range_widget"] = dr
    st.session_state["min_sme_probability"] = 0.0


    #future
    st.session_state["selected_sectors"] = []
    st.session_state["show_codes"] = False
    st.session_state["only_limburg"] = False
    st.session_state["highlight_dups"] = False


#caching the sector of each article sector
@st.cache_data(show_spinner=True)
def _classify_all_articles(df_records):
    df_local = pd.DataFrame(df_records)
    out = add_sector_classification(df_local.copy())
    return out.to_dict(orient="records")


# all geoNames in Limburg
@st.cache_data
def limburg_box():
    geo_df = pd.read_csv(
        "geoNames\\NL.txt", 
        sep="\t", 
        header=None,
        dtype={4: float, 5: float},  # lat/lon columns
        names=[
            "geonameid","name","ascii_name","alternate_names",
            "latitude","longitude","feature_class","feature_code","country_code",
            "cc2","admin1_code","admin2_code","admin3_code","admin4_code",
            "population","elevation","dem","timezone","modification_date"
        ]
    )

    geo_df = geo_df[["name", "latitude", "longitude", "admin1_code"]]
    geo_in_box = geo_df[geo_df["admin1_code"] == 5]



    locations_in_box = set(geo_in_box['name'].str.lower())
    return locations_in_box
limburg = limburg_box()

#--------------------------------

FILE_PATH = "keywords\\all_articles_keywords.json"
st.title("PVO Dashboard")


st.markdown("""
<style>
.divider{
  height: 10px;
  background: #3b3b3b;
  border-radius: 6px;
  margin: 14px 0 20px;
}
</style>
""", unsafe_allow_html=True)

def divider():
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


try:

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    try:
        df = pd.DataFrame(_classify_all_articles(df.to_dict(orient="records")))
    except Exception as e:
        st.warning(f"Sector classification skipped: {e}")
    # st.subheader(f"Data loaded from: `{FILE_PATH}`")
    # st.write(f"{len(df)} Articles")

    # -------------------------
    # Filtering UI
    # -------------------------
    st.sidebar.header("Filter options")


    # Column selector
    # cols_to_show = st.sidebar.multiselect(
    #     "Select columns to display",
    #     options=df.columns.tolist(),
    #     default=df.columns.tolist()
    # )





    #session-state defaults, global options for reset/presets
    feed_options_all = df["feed"].dropna().unique().tolist() if "feed" in df.columns else []
    if "published" in df.columns:
        _tmp_dates = pd.to_datetime(df["published"], errors="coerce")
        _min_date = _tmp_dates.min().date() if _tmp_dates.notna().any() else datetime.today().date()
        _max_date = _tmp_dates.max().date() if _tmp_dates.notna().any() else datetime.today().date()
    else:
        _min_date = _max_date = datetime.today().date()

    st.session_state.setdefault("min_sme_probability", 0.0)
    st.session_state.setdefault("selected_sectors", [])
    st.session_state.setdefault("show_codes", False)
    st.session_state.setdefault("highlight_dups", False)


    if "text_filter" not in st.session_state: st.session_state.text_filter = ""
    if "selected_feeds" not in st.session_state: st.session_state.selected_feeds = feed_options_all
    if "location_search" not in st.session_state: st.session_state.location_search = ""
    if "date_range" not in st.session_state: st.session_state.date_range = (_min_date, _max_date)

    p = st.session_state.pop("_pending_preset", None)
    if p:
        st.session_state.text_filter     = p.get("text_filter",     st.session_state.get("text_filter", ""))
        st.session_state.selected_feeds  = p.get("selected_feeds",  st.session_state.get("selected_feeds", feed_options_all))
        st.session_state.location_search = p.get("location_search", st.session_state.get("location_search", ""))

        try:
            st.session_state.min_sme_probability = float(
                p.get("min_sme_probability", st.session_state.get("min_sme_probability", 0.0))
            )
        except Exception:
            pass
        st.session_state.show_codes     = bool(p.get("show_codes",     st.session_state.get("show_codes", False)))
        st.session_state.highlight_dups = bool(p.get("highlight_dups", st.session_state.get("highlight_dups", False)))

        #lists
        sel_secs = p.get("selected_sectors", st.session_state.get("selected_sectors", []))
        st.session_state.selected_sectors = sel_secs if isinstance(sel_secs, list) else []

        # date range
        dr = p.get("date_range", (_min_date, _max_date))
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            dr = (_parse_to_date(dr[0]), _parse_to_date(dr[1]))
        else:
            dr = (_min_date, _max_date)
        st.session_state["date_range"] = dr
        st.session_state["date_range_widget"] = dr





    #refresh + reset
    c_ref, c_reset = st.sidebar.columns(2)
    with c_ref:
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            _rerun()
    with c_reset:
        if st.button("Reset filters", use_container_width=True):
            _reset_all_filters(feed_options_all, _min_date, _max_date)
            _rerun()




    #text search filter
    # query search across all string-like columns (AND / OR / NOT with quotes)
    st.sidebar.markdown("""
    **Query search syntax**
    - Words are default **ANDed**: `ransomware privacy`
    - Put **OR** to search for either: `privacy OR salesforce`
    - Exclude with minus: `-microsoft`
    - Exact phrase in quotes: `"data leak"`
    - Leave empty to match all
    """)

    query = st.sidebar.text_input(
        "Search query",
        key="text_filter",
        label_visibility="collapsed",
        placeholder=''
    )

    filtered_df = df.copy()
    if query.strip():
        import shlex
        tokens = shlex.split(query)

        include_terms, any_terms, exclude_terms = [], [], []
        for i, t in enumerate(tokens):
            if t.upper() in ("OR", "|"):
                continue
            if t.startswith("-"):
                exclude_terms.append(t[1:])
                continue
            prev_is_or = i > 0 and tokens[i-1].upper() in ("OR", "|")
            next_is_or = i < len(tokens)-1 and tokens[i+1].upper() in ("OR", "|")
            (any_terms if (prev_is_or or next_is_or) else include_terms).append(t)

        str_cols = filtered_df.select_dtypes(include=["object", "string"]).columns


        def contains_any(series, term):
            #case-insensitive search (lists get stringified)
            return series.astype(str).str.contains(term, case=False, na=False, regex=False)

        #all rows
        mask = pd.Series(True, index=filtered_df.index)

        # AND
        for term in include_terms:
            term_mask = pd.Series(False, index=filtered_df.index)
            for c in str_cols:
                term_mask |= contains_any(filtered_df[c], term)
            mask &= term_mask

        # OR
        if any_terms:
            any_mask = pd.Series(False, index=filtered_df.index)
            for term in any_terms:
                term_mask = pd.Series(False, index=filtered_df.index)
                for c in str_cols:
                    term_mask |= contains_any(filtered_df[c], term)
                any_mask |= term_mask
            mask &= any_mask

        #NOT
        if exclude_terms:
            ex_mask = pd.Series(False, index=filtered_df.index)
            for term in exclude_terms:
                term_mask = pd.Series(False, index=filtered_df.index)
                for c in str_cols:
                    term_mask |= contains_any(filtered_df[c], term)
                ex_mask |= term_mask
            mask &= ~ex_mask

        filtered_df = filtered_df[mask]



    # Feed filter
    if "feed" in df.columns:
        feed_options = sorted(df["feed"].dropna().unique().tolist())
    else:
        feed_options = []

    if "selected_feeds" not in st.session_state or not isinstance(st.session_state.selected_feeds, list):
        st.session_state.selected_feeds = feed_options_all

    selected_feeds = st.sidebar.multiselect(
        "Filter by feed",
        options=feed_options,
        default=st.session_state.selected_feeds,
        key="selected_feeds",
    )

    if selected_feeds:
        filtered_df = filtered_df[filtered_df["feed"].isin(selected_feeds)]



    #SME probability threshold
    if "sme_probability" in filtered_df.columns:
        min_sme_prob = st.sidebar.slider(
            "Min SME probability",
            0.0, 1.0, value=0.0, step=0.05, key="min_sme_probability",
            help="Hide articles with SME probability below this threshold"
        )
        filtered_df = filtered_df[filtered_df["sme_probability"].fillna(0) >= min_sme_prob]



    # Sector filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter by sector")

    #available sector codes
    sector_pairs = []
    if "sector_info" in df.columns:
        for si in df["sector_info"].dropna():
            if isinstance(si, dict):
                code = si.get("sector_code", "unknown")
                name = si.get("sector_name", "Unclassified")
                sector_pairs.append((code, name))

    seen = {}
    for code, name in sector_pairs:
        if code not in seen:
            seen[code] = name

    #check mark
    st.sidebar.checkbox("Show sector codes", value=st.session_state.get("show_codes", False), key="show_codes")


    def _sector_label(code: str) -> str:
        name = seen.get(code, "Unclassified")
        return f"{name} ({code})" if st.session_state.get("show_codes") else name

    sector_options = list(seen.keys())

    if "selected_sectors" not in st.session_state:
        st.session_state.selected_sectors = []

    selected_sectors = st.sidebar.multiselect(
        "Choose sector(s)",
        options=sector_options,
        default=st.session_state.selected_sectors,
        format_func=_sector_label,
        key="selected_sectors",
    )



    if selected_sectors:
        def _row_in_selected(row):
            #primary sector match
            primary_ok = False
            si = row.get("sector_info", {})
            if isinstance(si, dict):
                primary_ok = si.get("sector_code") in selected_sectors

            #any match
            any_ok = False
            codes = row.get("all_sectors_in_article", [])
            if isinstance(codes, list):
                any_ok = any(c in selected_sectors for c in codes)

            return primary_ok or any_ok

        filtered_df = filtered_df[filtered_df.apply(_row_in_selected, axis=1)]




    # -------------------------
    # Location filter
    # -------------------------
    location_search = st.sidebar.text_input(
        "Filter by locations (type one or more tags, separated by commas)",
        value=st.session_state.location_search, key="location_search"
    )

    if location_search.strip():
        search_tags = [tag.strip() for tag in location_search.split(",") if tag.strip()]
        filtered_df = filtered_df[
            filtered_df["locations"].apply(
                lambda loc_list: any(
                    any(search_tag.lower() in loc.lower() for loc in loc_list)
                    for search_tag in search_tags
                )
                if isinstance(loc_list, list) else False
            )
        ]

    # -------------------------
    # Date filter new
    # -------------------------
    date_col = "published"
    if date_col in df.columns:
        overall_min = _min_date
        overall_max = _max_date

        cur_start, cur_end = st.session_state.get("date_range", (overall_min, overall_max))
        cur_start, cur_end = _clamp_date_range(overall_min, overall_max, (cur_start, cur_end))

        st.sidebar.markdown("**Quick date presets**")
        c1, c2, c3, c4 = st.sidebar.columns(4)

        # 30d
        if c1.button("30d"):
            new_start = overall_max - timedelta(days=30)
            if new_start < overall_min:
                new_start = overall_min
            st.session_state.date_range = (new_start, overall_max)
            st.session_state["date_range_widget"] = st.session_state.date_range


        #90d
        if c2.button("90d"):
            new_start = overall_max - timedelta(days=90)
            if new_start < overall_min:
                new_start = overall_min
            st.session_state.date_range = (new_start, overall_max)
            st.session_state["date_range_widget"] = st.session_state.date_range

        # 1y
        if c3.button("1y"):
            new_start = overall_max - timedelta(days=365)
            if new_start < overall_min:
                new_start = overall_min
            st.session_state.date_range = (new_start, overall_max)
            st.session_state["date_range_widget"] = st.session_state.date_range

        #all
        if c4.button("All"):
            st.session_state.date_range = (overall_min, overall_max)
            st.session_state["date_range_widget"] = st.session_state.date_range

        cur_start, cur_end = st.session_state.date_range

        picked = st.sidebar.date_input(
            "Filter by date",
            value=(cur_start, cur_end),
            min_value=overall_min,
            max_value=overall_max,
            key="date_range_widget"
        )

        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            start_date, end_date = picked
        else:
            start_date = end_date = picked

        start_date, end_date = _clamp_date_range(overall_min, overall_max, (start_date, end_date))
        st.session_state.date_range = (start_date, end_date)

        #filter
        tmp_dates = pd.to_datetime(filtered_df[date_col], errors="coerce")
        filtered_df = filtered_df.loc[tmp_dates.notna()].copy()
        tmp_dates = tmp_dates.loc[tmp_dates.notna()]
        filtered_df = filtered_df[
            (tmp_dates.dt.date >= start_date) & (tmp_dates.dt.date <= end_date)
            ]

    # -------------------------
    # Display filtered DataFrame
    # -------------------------
    # st.subheader("📈 Filtered DataFrame")
    # st.dataframe(filtered_df[cols_to_show])


    #filter summary
    _sd, _ed = st.session_state.get("date_range", (_min_date, _max_date))
    _sector_ct = len(st.session_state.get("selected_sectors", []))
    st.markdown(
        f"**Current filters:** {len(st.session_state.get('selected_feeds', []))} feed(s) • "
        f"{_sd} → {_ed} • search: “{st.session_state.get('text_filter','')}” • "
        f"locations: “{st.session_state.get('location_search','')}” • "
        f"sectors: {_sector_ct or 'all'}"
    )
    st.write(f"**{len(filtered_df)}** articles match.")
    divider()

    # Downloads for filtered data
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered (CSV)", csv_bytes,
                           "filtered_articles.csv", "text/csv", use_container_width=True)
    with c_dl2:
        json_text = filtered_df.to_json(orient="records", force_ascii=False, indent=2)
        st.download_button("Download filtered (JSON)", json_text,
                           "filtered_articles.json", "application/json", use_container_width=True)


    #presets
    st.sidebar.markdown("---")
    st.sidebar.subheader("Presets")
    _presets = _load_presets()
    _names = ["—"] + list(_presets.keys())
    c_load, c_btn = st.sidebar.columns([3,1])
    with c_load:
        _sel = c_load.selectbox("Load preset", options=_names)
    with c_btn:
        if st.button("Load"):
            if _sel != "—":
                st.session_state["_pending_preset"] = _presets[_sel]
                _rerun()

    _new_preset = st.sidebar.text_input("Save current as…", placeholder="Limburg 90d")
    if st.sidebar.button("Save preset") and _new_preset:
        _presets[_new_preset] = {
            "text_filter":        st.session_state.get("text_filter", ""),
            "selected_feeds":     st.session_state.get("selected_feeds", []),
            "location_search":    st.session_state.get("location_search", ""),
            "date_range":         list(st.session_state.get("date_range", (_min_date, _max_date))),
            "min_sme_probability": float(st.session_state.get("min_sme_probability", 0.0)),
            "selected_sectors":   list(st.session_state.get("selected_sectors", [])),
            "show_codes":         bool(st.session_state.get("show_codes", False)),
            "highlight_dups":     bool(st.session_state.get("highlight_dups", False)),
        }
        _save_presets(_presets)
        st.sidebar.success(f"Saved preset “{_new_preset}”.")


    #confirm delete
    st.sidebar.markdown("**Delete a preset**")

    _presets = _load_presets()
    preset_names = list(_presets.keys())

    if preset_names:
        del_sel = st.sidebar.selectbox("Choose preset to delete", options=preset_names, key="del_preset_name")
        if st.sidebar.button("Delete selected", key="delete_selected_preset_btn"):
            # store the choice and show a confirmation step
            st.session_state["__confirm_del_name"] = del_sel
            _rerun()

        # confirmation step
        name_to_confirm = st.session_state.get("__confirm_del_name")
        if name_to_confirm:
            st.sidebar.warning(f"Delete preset “{name_to_confirm}”? This cannot be undone")
            c_del, c_cancel = st.sidebar.columns(2)
            with c_del:
                if st.button("Yes, delete", key="do_delete_preset"):
                    _presets.pop(name_to_confirm, None)
                    _save_presets(_presets)
                    st.sidebar.success(f"Deleted “{name_to_confirm}”.")
                    st.session_state["__confirm_del_name"] = None
                    _rerun()
            with c_cancel:
                if st.button("Cancel", key="cancel_delete_preset"):
                    st.session_state["__confirm_del_name"] = None
                    _rerun()
    else:
        st.sidebar.caption("no presets to delete yet")




    # -------------------------
    # Map Section — Using Cached Geocoded Data
    # -------------------------
    st.subheader("Interactive map")

    def geocode_locations_with_cache(rows, cache_file="cache/geocode_cache.json"):
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)

        recs = []
        for _, row in rows.iterrows():
            locs = row.get("locations", [])
            if isinstance(locs, list) and locs:
                chosen = None
                for loc in locs:
                    if loc in cache:
                        chosen = (loc, cache[loc]["lat"], cache[loc]["lon"])
                        break
                if chosen:
                    si = row.get("sector_info", {}) if isinstance(row.get("sector_info", {}), dict) else {}
                    recs.append({
                        "title": row.get("title", "Untitled"),
                        "url": row.get("url", ""),
                        "location": chosen[0],
                        "lat": chosen[1],
                        "lon": chosen[2],
                        "sector_code": si.get("sector_code", "unknown"),
                        "sector_name": si.get("sector_name", "Unclassified"),
                        "source": row.get("feed", ""),
                        #"summary": row.get("summary") or row.get("full_text") or "",
                    })
        return recs



    geo_article_records = geocode_locations_with_cache(filtered_df)

    if geo_article_records:
        #koloren

        MARKER_COLOR = "#ff6b6b"



        def _text_on(bg_hex: str) -> str:
            # WCAG relative luminance helper 3000
            bg = bg_hex.lstrip("#")
            r, g, b = [int(bg[i:i+2], 16)/255.0 for i in (0, 2, 4)]
            def lin(u): return u/12.92 if u <= 0.03928 else ((u+0.055)/1.055)**2.4
            L = 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)
            return "#000" if L > 0.55 else "#fff"


        # palettete


        #Map sector_code is color name


        show_codes = bool(st.session_state.get("show_codes", False))




        #build map
        m = folium.Map(location=[52.1, 5.3], zoom_start=7)

        map_mode = st.radio("Map mode", ["Heatmap", "Sector markers"], index=1, horizontal=True)
        if map_mode == "Heatmap":
            from collections import defaultdict
            by_latlon = defaultdict(int)
            for r in geo_article_records:
                by_latlon[(r["lat"], r["lon"])] += 1
            heat_data = [[lat, lon, wt] for (lat, lon), wt in by_latlon.items()]
            HeatMap(heat_data, radius=18, blur=15, max_zoom=6).add_to(m)
        else:
            cluster = MarkerCluster(
                #spidey
                options={
                    "spiderfyOnMaxZoom": True,
                    "disableClusteringAtZoom": 20,
                    "showCoverageOnHover": False,
                    "spiderfyDistanceMultiplier": 1.8,
                }
            ).add_to(m)
            for r in geo_article_records:
                bg = MARKER_COLOR
                fg = _text_on(bg)
                label = f"{r['sector_name']} ({r['sector_code']})" if show_codes else r["sector_name"]

                chip_html = (
                    f"<span style='display:inline-block;padding:2px 6px;border-radius:6px;"
                    f"background:{bg};color:{fg};border:1px solid rgba(0,0,0,.25);margin-right:6px;'>"
                    f"{_e(label)}</span>"
                )
                popup_html = f"""
                <div style="max-width:280px;font:13px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
                  <div style="font-weight:700;font-size:15px;margin-bottom:6px;">{_e(r['title'])}</div>
                  <div style="color:#666;margin-bottom:6px;">{chip_html}<span>Source: {_e(r.get('source',''))}</span></div>
                  {f"<a href='{_e(r['url'])}' target='_blank' style='display:inline-block;padding:6px 10px;border:1px solid #ccc;border-radius:6px;text-decoration:none;'>Open</a>" if r.get("url") else ""}
                </div>
                """

                folium.CircleMarker(
                    [r["lat"], r["lon"]],
                    radius=8,
                    color=bg,
                    fill=True,
                    fill_color=bg,
                    fill_opacity=0.9,
                    tooltip=f"{r['location']}: {label}",
                    popup=popup_html
                ).add_to(cluster)

        st_folium(m, width=1000, height=600)
        st.write(f"Showing {len(geo_article_records)} articles with cached coordinates")
        divider()
    else:
        st.info("No cached geocoded locations found for current filter")


    # -----------------------------------------
    # ARTICLE SPOTLIGHT
    # -----------------------------------------

    # heuristic 1: in Limburg

    in_limburg_df = filtered_df[
        filtered_df['locations'].apply(
            lambda tags: any(tag.lower() in limburg for tag in tags)
        )
    ].copy()

    # heuristic 2: compare to top keywords
    #???

    # heuristic 3: sme probabilty > 0.9 or head k?
    k = 5
    sme_df = filtered_df.sort_values(by='sme_probability', ascending=False).head(k)



    spotlight_df = pd.concat([in_limburg_df, sme_df])
    spotlight_df = spotlight_df[~spotlight_df.index.duplicated(keep='first')]



    # spotliht

    st.subheader("Spotlight")
    pretty = spotlight_df.copy()

    pretty["published_dt"] = pd.to_datetime(pretty.get("published"), errors="coerce", utc=True).dt.tz_convert(None)
    pretty = pretty.rename(columns={"feed": "source"})


    if "sector_info" in pretty.columns:
        pretty["sector"] = pretty["sector_info"].apply(
            lambda d: (d or {}).get("sector_name", "") if isinstance(d, dict) else ""
        )
    else:
        pretty["sector"] = ""


    pretty["date"] = pretty["published_dt"].dt.strftime("%Y-%m-%d %H:%M")
    pretty["age"]  = pretty["published_dt"].apply(_time_ago)

    if "url" not in pretty.columns:
        pretty["url"] = ""




    display_cols = ["source", "title", "sector", "date", "age", "url"]
    pretty = pretty.reindex(columns=display_cols).sort_values("date", ascending=False)


    st.dataframe(
        pretty,
        hide_index=True,
        width='stretch',
        column_config={
            "source": st.column_config.TextColumn("Source", width="small"),
            "title":  st.column_config.TextColumn("Title", width="medium"),
            "sector": st.column_config.TextColumn("Sector", width="small"),
            "date":   st.column_config.TextColumn("Published", width="small"),
            "age":    st.column_config.TextColumn("Age", help="Time since publication"),
            "url":    st.column_config.LinkColumn("Link", display_text="Open")
        }
    )


    st.caption(
        "Why these articles are in spotlight: "
        " located in **Limburg** and/or among the **top 5** by **SME probability** "
        "Duplicates are removed"
    )


    divider()



    # -------------------------
    # Top Keywords from Filtered Articles
    # -------------------------
    st.subheader("Top keywords")

    def extract_keywords(df):
        all_keywords = []
        for _, row in df.iterrows():
            kw_list = row.get("keywords", [])
            if isinstance(kw_list, list):
                for kw in kw_list:
                    if isinstance(kw, dict) and "word" in kw and "score" in kw:
                        all_keywords.append(kw)
        return all_keywords

    keywords = extract_keywords(filtered_df)



    if keywords:
        kw_df = pd.DataFrame(keywords)

        #most important keywords overall by total score across the filtered set
        kw_stats = (
            kw_df.groupby("word")
            .agg(mentions=("word", "size"), score=("score", "sum"))
            .reset_index()
            .sort_values(["score", "mentions"], ascending=False)
            .head(10)# cap
        )

        #top 20 keywords for charts
        kw_select = (
            kw_df.groupby("word")
            .agg(mentions=("word", "size"), score=("score", "sum"))
            .reset_index()
            .sort_values(["score", "mentions"], ascending=False)
            .head(20)
        )

        #aplhabet sort
        def _alphanum_sort(words):
            return sorted(words, key=lambda w: (not w[0].isdigit(), w.lower()))

        _kw_candidates = _alphanum_sort(kw_select["word"].tolist())

        if "focus_keywords" not in st.session_state:
            st.session_state["focus_keywords"] = _alphanum_sort(_kw_candidates)[:5]
        st.session_state["kw_candidates"] = _kw_candidates

        #chart
        kw_chart = (
            alt.Chart(kw_stats)
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title="Score"),
                y=alt.Y("word:N", sort='-x', title="Keyword"),
                tooltip=["word", "score", "mentions"]
            )
            .properties(height=400)
        )
        st.altair_chart(kw_chart, use_container_width=True)

        with st.expander("Statistics table", expanded=False):
            #table
            st.dataframe(
                kw_stats.rename(columns={"word": "Keyword", "score": "Score", "mentions": "Mentions"}),
                hide_index=True,
                width='stretch',
                column_config={
                    "Keyword":  st.column_config.TextColumn("Keyword"),
                    "Mentions": st.column_config.NumberColumn("Mentions"),
                    "Score":    st.column_config.NumberColumn("Score"),
                }
            )


        st.caption(
            "Score = sum of each keyword’s per article score across the currently filtered articles. "
            "Higher means the keyword appears more (and/or in articles where it was ranked highly)"
        )
    else:
        st.info("No keywords found for the current filter")


    divider()



    #Keyword trends over time
    st.subheader("Keyword trends over time")

    if "kw_trend_mode" not in st.session_state:
        st.session_state.kw_trend_mode = "normalized"
    if "kw_trend_roll" not in st.session_state:
        st.session_state.kw_trend_roll = 6

    _kw_opts = st.session_state.get("kw_candidates", [])
    _focus_sel = st.multiselect(
        "Focus keywords (pick 2–5)",
        options=_kw_opts,
        default=st.session_state.get("focus_keywords", _kw_opts[:5]),
        key="focus_keywords",
        help="These keywords will be shown in the line and heatmap below"
    )

    # Enforce at most 5 in charts
    focus_effective = (_focus_sel or _kw_opts[:5])[:5]
    if len(_focus_sel) > 5:
        st.warning("Showing the first 5 selected keywords")
    if len(focus_effective) < 2:
        st.info("Pick at least 2 for a useful comparison")
        st.session_state.setdefault("kw_trend_mode", "normalized")
        st.session_state.setdefault("kw_trend_roll", 6)



    #quick presets
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Long term trends", use_container_width=True):
            st.session_state.kw_trend_mode = "normalized"
            st.session_state.kw_trend_roll = 6
            _rerun()
    with p2:
        if st.button("Short term trends", use_container_width=True):
            st.session_state.kw_trend_mode = "normalized"
            st.session_state.kw_trend_roll = 3
            _rerun()
    with p3:
        if st.button("Raw counts", use_container_width=True):
            st.session_state.kw_trend_mode = "raw"
            st.session_state.kw_trend_roll = 0
            _rerun()



    #sliders
    trend_roll = st.slider(
        "Rolling window (months)",
        min_value=0, max_value=12,
        key="kw_trend_roll",
        help="0 = no smoothing",
    )

    trend_roll = int(st.session_state.kw_trend_roll)



    #mode switch
    mode_now = st.session_state.kw_trend_mode
    mode_next = "raw" if mode_now == "normalized" else "normalized"
    c_mode, c_btn = st.columns([3, 1])
    with c_mode:
        st.markdown(f"**Mode:** {'Normalized share (%)' if mode_now == 'normalized' else 'Raw counts'}")
    with c_btn:
        if st.button(
                f"Switch to {('Raw counts' if mode_now=='normalized' else 'Normalized share')}",
                use_container_width=True, key="kw_trend_switch"
        ):
            st.session_state.kw_trend_mode = mode_next
            _rerun()


    if not st.session_state.get("focus_keywords"):
        st.info("No focus keywords available for the current filters.")
        st.stop()

    #building
    if filtered_df.empty or "published" not in filtered_df.columns:
        st.info("No keyword trend data for the current filters/date range")
    else:
        trend_rows = []
        for _, row in filtered_df.iterrows():
            pub = row.get("published")
            if not pub:
                continue
            pub_date = pd.to_datetime(pub, errors="coerce")
            if pd.isna(pub_date):
                continue
            month_str = pub_date.strftime("%Y-%m")
            kw_list = row.get("keywords", [])
            if isinstance(kw_list, list):
                for kw in kw_list:
                    if isinstance(kw, dict) and "word" in kw:
                        trend_rows.append({"keyword": kw["word"], "month": month_str})

        if not trend_rows:
            st.info("No keyword trend data for the current filters/date range")
        else:
            base = pd.DataFrame(trend_rows)


            #mentions per (keyword, month)
            counts = base.groupby(["keyword", "month"]).size().reset_index(name="count")

            #top-K keywords in current date range
            focus_kw = focus_effective
            counts = counts[counts["keyword"].isin(focus_kw)]

            custom_palette = ['#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#D55E00']

            color_enc = alt.Color(
                "keyword:N",
                title="Keyword",
                scale=alt.Scale(
                    domain=focus_kw,
                    range=custom_palette[:len(focus_kw)]
                )
            )

            months_sorted = sorted(counts["month"].unique())
            grid = pd.MultiIndex.from_product([focus_kw, months_sorted], names=["keyword", "month"]).to_frame(index=False)
            raw = grid.merge(counts, on=["keyword", "month"], how="left").fillna({"count": 0})

            #smoothing
            use_smooth = (trend_roll > 0)




            #raw
            if use_smooth:
                raw["y_val"] = (
                    raw.sort_values(["keyword", "month"])
                    .groupby("keyword")["count"]
                    .transform(lambda s: s.rolling(trend_roll, min_periods=1).mean())
                )
            else:
                raw["y_val"] = raw["count"]



            #NORMALIZED
            month_totals = raw.groupby("month")["count"].sum().rename("month_total")
            norm = raw.merge(month_totals, on="month", how="left")
            norm["share_pct"] = (norm["count"] / norm["month_total"].replace(0, 1)) * 100
            if use_smooth:
                norm["y_val"] = (
                    norm.sort_values(["keyword", "month"])
                    .groupby("keyword")["share_pct"]
                    .transform(lambda s: s.rolling(trend_roll, min_periods=1).mean())
                )
            else:
                norm["y_val"] = norm["share_pct"]




            #charts
            if st.session_state.kw_trend_mode == "normalized":
                y_title = "Share of monthly mentions (%)" + (" (smoothed)" if use_smooth else "")
                line = (
                    alt.Chart(norm).mark_line().encode(
                        x=alt.X("month:N", title="Month", sort="ascending"),
                        y=alt.Y("y_val:Q", title=y_title),
                        color=color_enc,
                        tooltip=["keyword", "month", alt.Tooltip("y_val:Q", title="share%", format=".2f")]
                    ).properties(height=420)
                )
                st.altair_chart(line, use_container_width=True)

                heat = (
                    alt.Chart(norm).mark_rect().encode(
                        x=alt.X("month:N", title="Month", sort="ascending"),
                        y=alt.Y("keyword:N", title="Keyword"),
                        color=alt.Color("y_val:Q", title="share%" + (" (smoothed)" if use_smooth else ""),
                                        scale=alt.Scale(scheme="reds")),
                        opacity=alt.condition(alt.datum.y_val > 0, alt.value(1), alt.value(0)),
                        tooltip=["keyword", "month", alt.Tooltip("y_val:Q", title="share%", format=".2f")]
                    ).properties(height=420)
                )
                st.altair_chart(heat, use_container_width=True)

                st.caption(
                    ("Showing your selected keywords - "
                     + ("**normalized share per month (smoothed)** with a "
                        f"**{trend_roll}-month** rolling average" if use_smooth
                        else "**normalized share per month (no smoothing)**"))
                )
            else:
                y_title = "Mentions" + (" (smoothed)" if use_smooth else "")
                line2 = (
                    alt.Chart(raw).mark_line().encode(
                        x=alt.X("month:N", title="Month", sort="ascending"),
                        y=alt.Y("y_val:Q", title=y_title),
                        color=color_enc,
                        tooltip=["keyword", "month", alt.Tooltip("y_val:Q", title="mentions", format=".2f")]
                    ).properties(height=420)
                )
                st.altair_chart(line2, use_container_width=True)

                heat2 = (
                    alt.Chart(raw).mark_rect().encode(
                        x=alt.X("month:N", title="Month", sort="ascending"),
                        y=alt.Y("keyword:N", title="Keyword"),
                        color=alt.Color(("y_val:Q" if use_smooth else "count:Q"),
                                        title=("Mentions (smoothed)" if use_smooth else "Mentions"),
                                        scale=alt.Scale(scheme="blues")),
                        opacity=alt.condition((alt.datum.y_val if use_smooth else alt.datum.count) > 0,
                                              alt.value(1), alt.value(0)),
                        tooltip=["keyword", "month",
                                 (alt.Tooltip("y_val:Q", title="mentions", format=".2f") if use_smooth else "count")]
                    ).properties(height=420)
                )
                st.altair_chart(heat2, use_container_width=True)

                st.caption(
                    ("Showing your selected keywords - "
                     + ("**raw mentions (smoothed)** with a "
                        f"**{trend_roll}-month** rolling average" if use_smooth
                        else "**raw mentions (no smoothing)**"))
                )




    # with st.expander("Show raw JSON data"):
    #     st.json(data)

except FileNotFoundError:
    st.error(f"File not found at path: `{FILE_PATH}`")
except json.JSONDecodeError:
    st.error("The file is not valid JSON.")
except Exception as e:
    st.error(f"Unexpected error: {e}")
