#!/usr/bin/env python3
"""
Historical Disaster Dataset Downloader  v4  (reliability patch)
===============================================================
Downloads ~10 representative images per event from Wikimedia Commons.

Target  : 50 events x 10 images = 500 images
Categories : flood  wildfire  earthquake  landslide  cyclone

Changes from v3
---------------
* Thumbnail URLs  -- uses iiurlwidth=1200 (/thumb/ path, lighter rate-limiting)
* Exp backoff     -- 429 / 503 / connection reset: 5→10→20→40→80s cap 120s
* 5 retries       -- per failed image download (was 3)
* Event delay     -- 5 s between events
* Jitter          -- 1-3 s random sleep between image downloads
* Resume          -- skips existing files; URL→file map in checkpoint.json
* Checkpoint log  -- saves progress after every successful download
* Stats           -- downloaded / failed / skipped per event and overall
* Default scope   -- flood + cyclone + earthquake (pass --all-categories for all 5)

Outputs (all written next to this script):
  images/{category}/{event_id}_{nn}.jpg   -- downloaded images
  metadata.csv                            -- one row per image
  historical_events.json                  -- flat (1 entry per image)
  dataset_summary.xlsx                    -- overview / per-event / per-category
  checkpoint.json                         -- resume map (url → filepath)
  query_audit.csv                         -- one row per event; query + candidate count
  candidate_counts.xlsx                   -- audit detail: per-event and per-category

Usage
-----
  # Download flood + cyclone + earthquake (default)
  python datasets/historical/download_historical_dataset.py

  # All 5 categories
  python datasets/historical/download_historical_dataset.py --all-categories

  # Single category
  python datasets/historical/download_historical_dataset.py --category flood

  # Single event
  python datasets/historical/download_historical_dataset.py --event flood_kerala_2018

  # Audit coverage without downloading
  python datasets/historical/download_historical_dataset.py --audit
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import csv
import json
import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR          = Path(__file__).resolve().parent
IMAGES_DIR        = BASE_DIR / "images"
METADATA_CSV      = BASE_DIR / "metadata.csv"
EVENTS_JSON       = BASE_DIR / "historical_events.json"
SUMMARY_XLSX      = BASE_DIR / "dataset_summary.xlsx"
AUDIT_CSV         = BASE_DIR / "query_audit.csv"
CANDIDATE_XLSX    = BASE_DIR / "candidate_counts.xlsx"
CHECKPOINT_JSON   = BASE_DIR / "checkpoint.json"

CATEGORIES = ["flood", "wildfire", "earthquake", "landslide", "cyclone"]
DEFAULT_CATEGORIES = ["flood", "cyclone", "earthquake"]   # scope when --category not given

# ---------------------------------------------------------------------------
# Download / search parameters
# ---------------------------------------------------------------------------
TARGET_PER_EVENT  = 10
MIN_TARGET        = 5            # minimum acceptable per event
THUMB_WIDTH       = 1200         # px — thumbnail width requested (lighter rate-limiting)
MIN_SIDE          = 600          # minimum original width or height (px)
MAX_CANDIDATES    = 30           # API results fetched per query
API_DELAY         = 3.0          # seconds between Wikimedia API calls
INTERNAL_DELAY    = 1.5          # seconds between step-1 and step-2 in _wikimedia_search
MAX_RETRIES_API   = 5            # retry attempts for API calls
MAX_RETRIES_DL    = 5            # retry attempts per image download
BASE_BACKOFF      = 5            # seconds — starting wait for exp backoff
MAX_BACKOFF       = 120          # seconds — cap for exp backoff
EVENT_DELAY       = 5.0          # seconds to wait between events
JITTER_MIN        = 1.0          # seconds — min random delay between downloads
JITTER_MAX        = 3.0          # seconds — max random delay between downloads
GATE_THRESHOLD    = 0.80         # fraction of events needing 10+ candidates
WIKIMEDIA_API     = "https://commons.wikimedia.org/w/api.php"
USER_AGENT        = "VLM-Disaster-Analyzer/4.0 (research; ujjeshanityarouthu1312@gmail.com)"

_SKIP_RE = re.compile(
    r"(map|logo|icon|flag|chart|graph|infograph|diagram|locator|route|track"
    r"|symbol|badge|seal|coat|emblem|avatar|blank|template|vector"
    r"|portrait|headshot|mugshot|person|politician|minister|official)",
    re.IGNORECASE,
)
_ALLOWED_MIME = {"image/jpeg", "image/png"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Events database  (50 events; search queries generated at runtime)
# wikimedia_name  -- optional override used instead of event_name when
#                    building Wikimedia search strings; useful for events
#                    whose display names are too long or have uncommon terms
# ---------------------------------------------------------------------------
EVENTS: list[dict] = [

    # ======== FLOODS (10) ===================================================
    {
        "id": "flood_kerala_2018",
        "event_name": "Kerala Floods",
        "year": 2018, "category": "flood",
        "country": "India", "state_or_region": "Kerala",
        "fatalities": 483, "affected_population": "5,400,000",
        "economic_damage_usd": "3,500,000,000",
        "short_description": (
            "Worst flooding in Kerala in nearly a century triggered by unusually "
            "high monsoon rainfall. All 14 districts were placed on red alert."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2018_Kerala_floods",
    },
    {
        "id": "flood_assam_2022",
        "event_name": "Assam Floods",
        "year": 2022, "category": "flood",
        "country": "India", "state_or_region": "Assam",
        "fatalities": 192, "affected_population": "5,500,000",
        "economic_damage_usd": "800,000,000",
        "short_description": (
            "Annual Brahmaputra river flooding affecting 32 of 35 districts, "
            "displacing millions and inundating Kaziranga National Park."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2022_Assam_floods",
    },
    {
        "id": "flood_chennai_2015",
        "event_name": "Chennai Floods",
        "year": 2015, "category": "flood",
        "country": "India", "state_or_region": "Tamil Nadu",
        "fatalities": 500, "affected_population": "4,000,000",
        "economic_damage_usd": "3,000,000,000",
        "short_description": (
            "Record rainfall submerged large parts of Chennai, suspending airport "
            "operations for over a week."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2015_South_India_floods",
        "wikimedia_name": "South India floods 2015",
    },
    {
        "id": "flood_uttarakhand_2013",
        "event_name": "Uttarakhand Flash Floods",
        "year": 2013, "category": "flood",
        "country": "India", "state_or_region": "Uttarakhand",
        "fatalities": 5748, "affected_population": "100,000",
        "economic_damage_usd": "600,000,000",
        "short_description": (
            "Catastrophic cloudbursts triggered flash floods in the Himalayan state, "
            "devastating the Kedarnath valley -- often called the Himalayan Tsunami."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2013_North_India_floods",
        "wikimedia_name": "Uttarakhand floods 2013",
    },
    {
        "id": "flood_pakistan_2022",
        "event_name": "Pakistan Floods",
        "year": 2022, "category": "flood",
        "country": "Pakistan", "state_or_region": "Sindh, Balochistan, KPK",
        "fatalities": 1739, "affected_population": "33,000,000",
        "economic_damage_usd": "30,000,000,000",
        "short_description": (
            "One-third of Pakistan submerged by catastrophic monsoon flooding -- "
            "a climate-amplified disaster affecting 33 million people."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2022_Pakistan_floods",
    },
    {
        "id": "flood_mumbai_2005",
        "event_name": "Mumbai Floods",
        "year": 2005, "category": "flood",
        "country": "India", "state_or_region": "Maharashtra",
        "fatalities": 1094, "affected_population": "20,000,000",
        "economic_damage_usd": "690,000,000",
        "short_description": (
            "Mumbai received 944 mm of rainfall in a single day, causing "
            "catastrophic urban flooding across the city."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2005_Maharashtra_floods",
        "wikimedia_name": "Maharashtra floods 2005",
    },
    {
        "id": "flood_bihar_2008",
        "event_name": "Bihar Floods",
        "year": 2008, "category": "flood",
        "country": "India", "state_or_region": "Bihar",
        "fatalities": 527, "affected_population": "3,000,000",
        "economic_damage_usd": "250,000,000",
        "short_description": (
            "The Kosi river breached an embankment in Nepal causing devastating "
            "floods in North Bihar, displacing over three million people."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2008_Bihar_flood",
        "wikimedia_name": "Bihar flood 2008",
    },
    {
        "id": "flood_kashmir_2014",
        "event_name": "Kashmir Floods",
        "year": 2014, "category": "flood",
        "country": "India", "state_or_region": "Jammu and Kashmir",
        "fatalities": 277, "affected_population": "1,300,000",
        "economic_damage_usd": "2,000,000,000",
        "short_description": (
            "Persistent heavy rainfall flooded the Kashmir Valley, submerging "
            "Srinagar under several feet of water."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2014_Kashmir_floods",
    },
    {
        "id": "flood_bangladesh_2020",
        "event_name": "Bangladesh Floods",
        "year": 2020, "category": "flood",
        "country": "Bangladesh", "state_or_region": "Sylhet, Dhaka",
        "fatalities": 140, "affected_population": "5,400,000",
        "economic_damage_usd": "300,000,000",
        "short_description": (
            "Extended monsoon flooding inundated a third of Bangladesh for "
            "over a month."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2020_Bangladesh_floods",
    },
    {
        "id": "flood_germany_2021",
        "event_name": "Germany Floods",
        "year": 2021, "category": "flood",
        "country": "Germany", "state_or_region": "Rhineland-Palatinate",
        "fatalities": 222, "affected_population": "180,000",
        "economic_damage_usd": "40,000,000,000",
        "short_description": (
            "Catastrophic flash flooding devastated the Ahr Valley -- the "
            "deadliest flooding in Germany's post-war history."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2021_European_floods",
        "wikimedia_name": "2021 European floods",
    },

    # ======== CYCLONES (10) =================================================
    {
        "id": "cyclone_fani_2019",
        "event_name": "Cyclone Fani",
        "year": 2019, "category": "cyclone",
        "country": "India", "state_or_region": "Odisha",
        "fatalities": 89, "affected_population": "28,000,000",
        "economic_damage_usd": "8,100,000,000",
        "short_description": (
            "Extremely severe cyclone striking Odisha near Puri at 175-185 km/h. "
            "Mass evacuation of 1.2 million people minimised casualties."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Fani",
    },
    {
        "id": "cyclone_amphan_2020",
        "event_name": "Cyclone Amphan",
        "year": 2020, "category": "cyclone",
        "country": "India", "state_or_region": "West Bengal, Odisha",
        "fatalities": 128, "affected_population": "13,000,000",
        "economic_damage_usd": "13,600,000,000",
        "short_description": (
            "Strongest cyclone to strike the Bay of Bengal since 1999, causing "
            "catastrophic destruction across coastal West Bengal."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Amphan",
    },
    {
        "id": "cyclone_yaas_2021",
        "event_name": "Cyclone Yaas",
        "year": 2021, "category": "cyclone",
        "country": "India", "state_or_region": "West Bengal, Odisha",
        "fatalities": 19, "affected_population": "15,000,000",
        "economic_damage_usd": "3,000,000,000",
        "short_description": (
            "Very severe cyclone making landfall near Balasore; storm surge "
            "coincided with high tide causing severe coastal flooding."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Yaas",
    },
    {
        "id": "cyclone_tauktae_2021",
        "event_name": "Cyclone Tauktae",
        "year": 2021, "category": "cyclone",
        "country": "India", "state_or_region": "Gujarat",
        "fatalities": 198, "affected_population": "16,000,000",
        "economic_damage_usd": "1,500,000,000",
        "short_description": (
            "Extremely severe cyclone with 185 km/h winds making landfall in "
            "Gujarat -- the strongest to affect the region since 1998."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Tauktae",
    },
    {
        "id": "cyclone_odisha_1999",
        "event_name": "Odisha Super Cyclone",
        "year": 1999, "category": "cyclone",
        "country": "India", "state_or_region": "Odisha",
        "fatalities": 10000, "affected_population": "15,000,000",
        "economic_damage_usd": "2,500,000,000",
        "short_description": (
            "One of the most powerful cyclones ever to strike India, with wind "
            "speeds exceeding 260 km/h and a massive storm surge."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/1999_Odisha_cyclone",
        "wikimedia_name": "1999 Odisha cyclone",
    },
    {
        "id": "cyclone_hudhud_2014",
        "event_name": "Cyclone Hudhud",
        "year": 2014, "category": "cyclone",
        "country": "India", "state_or_region": "Andhra Pradesh",
        "fatalities": 124, "affected_population": "12,000,000",
        "economic_damage_usd": "7,000,000,000",
        "short_description": (
            "Very severe cyclone making landfall near Visakhapatnam, causing "
            "widespread destruction to the city and coastal districts."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Hudhud",
    },
    {
        "id": "cyclone_phailin_2013",
        "event_name": "Cyclone Phailin",
        "year": 2013, "category": "cyclone",
        "country": "India", "state_or_region": "Odisha, Andhra Pradesh",
        "fatalities": 45, "affected_population": "12,000,000",
        "economic_damage_usd": "700,000,000",
        "short_description": (
            "Very severe cyclone making landfall near Gopalpur at over 200 km/h. "
            "Massive pre-disaster evacuation significantly reduced fatalities."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Phailin",
    },
    {
        "id": "cyclone_nargis_2008",
        "event_name": "Cyclone Nargis",
        "year": 2008, "category": "cyclone",
        "country": "Myanmar", "state_or_region": "Ayeyarwady Delta",
        "fatalities": 138366, "affected_population": "2,400,000",
        "economic_damage_usd": "10,000,000,000",
        "short_description": (
            "Catastrophic cyclone striking Myanmar's Irrawaddy Delta with a "
            "devastating storm surge -- one of the deadliest natural disasters ever."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Nargis",
    },
    {
        "id": "cyclone_mocha_2023",
        "event_name": "Cyclone Mocha",
        "year": 2023, "category": "cyclone",
        "country": "Myanmar", "state_or_region": "Rakhine State",
        "fatalities": 145, "affected_population": "5,400,000",
        "economic_damage_usd": "1,000,000,000",
        "short_description": (
            "Extremely severe cyclone striking Rakhine State at over 250 km/h, "
            "severely damaging Sittwe and surrounding Rohingya camps."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Mocha",
    },
    {
        "id": "cyclone_biparjoy_2023",
        "event_name": "Cyclone Biparjoy",
        "year": 2023, "category": "cyclone",
        "country": "India", "state_or_region": "Gujarat",
        "fatalities": 2, "affected_population": "180,000",
        "economic_damage_usd": "500,000,000",
        "short_description": (
            "Extremely severe cyclone making landfall near Jakhau Port, Gujarat, "
            "after nearly two weeks over the Arabian Sea."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Cyclone_Biparjoy",
    },

    # ======== EARTHQUAKES (10) ==============================================
    {
        "id": "earthquake_gujarat_2001",
        "event_name": "Bhuj Earthquake",
        "year": 2001, "category": "earthquake",
        "country": "India", "state_or_region": "Gujarat",
        "fatalities": 20085, "affected_population": "15,900,000",
        "economic_damage_usd": "5,000,000,000",
        "short_description": (
            "A 7.7 magnitude earthquake struck Bhuj on Republic Day morning, "
            "causing catastrophic destruction across Kutch district."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2001_Gujarat_earthquake",
        "wikimedia_name": "2001 Gujarat earthquake",
    },
    {
        "id": "earthquake_nepal_2015",
        "event_name": "Nepal Earthquake",
        "year": 2015, "category": "earthquake",
        "country": "Nepal", "state_or_region": "Gorkha, Kathmandu",
        "fatalities": 8964, "affected_population": "8,000,000",
        "economic_damage_usd": "7,000,000,000",
        "short_description": (
            "A 7.8 magnitude earthquake caused widespread destruction in "
            "Kathmandu and triggered deadly avalanches on Mount Everest."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2015_Nepal_earthquake",
        "wikimedia_name": "2015 Nepal earthquake",
    },
    {
        "id": "earthquake_turkey_2023",
        "event_name": "Turkey Earthquake",
        "year": 2023, "category": "earthquake",
        "country": "Turkey", "state_or_region": "Kahramanmaras",
        "fatalities": 55000, "affected_population": "13,500,000",
        "economic_damage_usd": "103,000,000,000",
        "short_description": (
            "Twin 7.8 and 7.7 magnitude earthquakes struck southeastern Turkey "
            "and northern Syria, causing catastrophic building collapses."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2023_Turkey-Syria_earthquake",
        "wikimedia_name": "2023 Turkey earthquake",
    },
    {
        "id": "earthquake_kashmir_2005",
        "event_name": "Kashmir Earthquake",
        "year": 2005, "category": "earthquake",
        "country": "Pakistan", "state_or_region": "Azad Kashmir",
        "fatalities": 87351, "affected_population": "5,000,000",
        "economic_damage_usd": "5,200,000,000",
        "short_description": (
            "A 7.6 magnitude earthquake levelled towns across "
            "Pakistan-administered Kashmir's Jhelum Valley."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2005_Kashmir_earthquake",
        "wikimedia_name": "2005 Kashmir earthquake",
    },
    {
        "id": "earthquake_sikkim_2011",
        "event_name": "Sikkim Earthquake",
        "year": 2011, "category": "earthquake",
        "country": "India", "state_or_region": "Sikkim",
        "fatalities": 111, "affected_population": "500,000",
        "economic_damage_usd": "200,000,000",
        "short_description": (
            "A 6.9 magnitude earthquake struck Sikkim and parts of Nepal, "
            "Tibet and Bhutan."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2011_Sikkim_earthquake",
        "wikimedia_name": "2011 Sikkim earthquake",
    },
    {
        "id": "earthquake_japan_2011",
        "event_name": "Tohoku Earthquake",
        "year": 2011, "category": "earthquake",
        "country": "Japan", "state_or_region": "Tohoku",
        "fatalities": 19759, "affected_population": "500,000",
        "economic_damage_usd": "235,000,000,000",
        "short_description": (
            "A 9.0 magnitude megathrust earthquake triggered a tsunami "
            "reaching 40.5 m and caused the Fukushima nuclear disaster."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2011_Tohoku_earthquake",
        "wikimedia_name": "2011 Tohoku earthquake",
    },
    {
        "id": "earthquake_haiti_2010",
        "event_name": "Haiti Earthquake",
        "year": 2010, "category": "earthquake",
        "country": "Haiti", "state_or_region": "Port-au-Prince",
        "fatalities": 316000, "affected_population": "3,000,000",
        "economic_damage_usd": "8,000,000,000",
        "short_description": (
            "A catastrophic 7.0 magnitude earthquake near Port-au-Prince caused "
            "unprecedented urban destruction."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2010_Haiti_earthquake",
        "wikimedia_name": "2010 Haiti earthquake",
    },
    {
        "id": "earthquake_morocco_2023",
        "event_name": "Morocco Earthquake",
        "year": 2023, "category": "earthquake",
        "country": "Morocco", "state_or_region": "Al Haouz, Marrakesh",
        "fatalities": 2946, "affected_population": "300,000",
        "economic_damage_usd": "2,000,000,000",
        "short_description": (
            "A 6.8 magnitude earthquake struck the High Atlas mountains, "
            "destroying entire villages and killing thousands."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2023_Marrakesh-Safi_earthquake",
        "wikimedia_name": "2023 Morocco earthquake",
    },
    {
        "id": "earthquake_indonesia_2018",
        "event_name": "Sulawesi Earthquake",
        "year": 2018, "category": "earthquake",
        "country": "Indonesia", "state_or_region": "Central Sulawesi",
        "fatalities": 4340, "affected_population": "1,500,000",
        "economic_damage_usd": "531,000,000",
        "short_description": (
            "A 7.5 magnitude earthquake triggered a tsunami and soil liquefaction "
            "that buried entire neighbourhoods in Palu."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2018_Sulawesi_earthquake",
        "wikimedia_name": "2018 Sulawesi earthquake",
    },
    {
        "id": "earthquake_afghanistan_2022",
        "event_name": "Afghanistan Earthquake",
        "year": 2022, "category": "earthquake",
        "country": "Afghanistan", "state_or_region": "Khost, Paktika",
        "fatalities": 1163, "affected_population": "270,000",
        "economic_damage_usd": "100,000,000",
        "short_description": (
            "A 5.9 magnitude shallow earthquake struck Khost and Paktika "
            "provinces -- one of Afghanistan's deadliest recent seismic events."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2022_Afghanistan_earthquakes",
        "wikimedia_name": "2022 Afghanistan earthquake",
    },

    # ======== LANDSLIDES (10) ===============================================
    {
        "id": "landslide_malin_2014",
        "event_name": "Malin Landslide",
        "year": 2014, "category": "landslide",
        "country": "India", "state_or_region": "Maharashtra",
        "fatalities": 151, "affected_population": "44 households",
        "economic_damage_usd": "5,000,000",
        "short_description": (
            "A sudden catastrophic landslide buried the village of Malin "
            "in Pune district following heavy monsoon rainfall."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2014_Malin_landslide",
    },
    {
        "id": "landslide_chamoli_2021",
        "event_name": "Chamoli Disaster",
        "year": 2021, "category": "landslide",
        "country": "India", "state_or_region": "Uttarakhand",
        "fatalities": 204, "affected_population": "4,000",
        "economic_damage_usd": "150,000,000",
        "short_description": (
            "A glacial lake outburst triggered a catastrophic flash flood "
            "destroying two hydropower projects in the Rishiganga valley."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2021_Chamoli_disaster",
        "wikimedia_name": "2021 Chamoli disaster",
    },
    {
        "id": "landslide_kerala_2020",
        "event_name": "Pettimudi Landslide",
        "year": 2020, "category": "landslide",
        "country": "India", "state_or_region": "Kerala",
        "fatalities": 57, "affected_population": "1,000",
        "economic_damage_usd": "20,000,000",
        "short_description": (
            "Landslides struck Pettimudi in Munnar, burying tea estate "
            "worker settlements under debris."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2020_Rajamala_landslides",
        "wikimedia_name": "Pettimudi landslide",
    },
    {
        "id": "landslide_sikkim_2023",
        "event_name": "Sikkim Flood",
        "year": 2023, "category": "landslide",
        "country": "India", "state_or_region": "Sikkim",
        "fatalities": 98, "affected_population": "100,000",
        "economic_damage_usd": "800,000,000",
        "short_description": (
            "A glacial lake outburst from South Lhonak Lake triggered a "
            "devastating flash flood in the Teesta River valley."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2023_Sikkim_glacial_lake_outburst_flood",
        "wikimedia_name": "2023 Sikkim flood",
    },
    {
        "id": "landslide_darjeeling_2017",
        "event_name": "Darjeeling Landslide",
        "year": 2017, "category": "landslide",
        "country": "India", "state_or_region": "West Bengal",
        "fatalities": 18, "affected_population": "50,000",
        "economic_damage_usd": "30,000,000",
        "short_description": (
            "Widespread landslides triggered by monsoon rainfall caused "
            "significant damage to roads in the Darjeeling hills."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2017_North_Bengal_floods",
        "wikimedia_name": "Darjeeling landslide",
    },
    {
        "id": "landslide_kedarnath_2013",
        "event_name": "Kedarnath Landslide",
        "year": 2013, "category": "landslide",
        "country": "India", "state_or_region": "Uttarakhand",
        "fatalities": 5748, "affected_population": "100,000",
        "economic_damage_usd": "400,000,000",
        "short_description": (
            "Massive landslides buried the Kedarnath pilgrim route under "
            "metres of debris and rock during the 2013 disaster."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2013_North_India_floods",
        "wikimedia_name": "Kedarnath flood 2013",
    },
    {
        "id": "landslide_oso_2014",
        "event_name": "Oso Mudslide",
        "year": 2014, "category": "landslide",
        "country": "USA", "state_or_region": "Washington",
        "fatalities": 43, "affected_population": "49",
        "economic_damage_usd": "60,000,000",
        "short_description": (
            "A massive landslide near Oso, Washington buried a neighbourhood "
            "and blocked the North Fork Stillaguamish River."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2014_Oso_mudslide",
        "wikimedia_name": "2014 Oso mudslide",
    },
    {
        "id": "landslide_atami_2021",
        "event_name": "Atami Mudslide",
        "year": 2021, "category": "landslide",
        "country": "Japan", "state_or_region": "Shizuoka",
        "fatalities": 20, "affected_population": "1,500",
        "economic_damage_usd": "100,000,000",
        "short_description": (
            "A powerful mudflow swept through the coastal resort city of Atami, "
            "destroying dozens of buildings."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2021_Atami_mudslide",
        "wikimedia_name": "2021 Atami mudslide",
    },
    {
        "id": "landslide_montecito_2018",
        "event_name": "Montecito Mudslide",
        "year": 2018, "category": "landslide",
        "country": "USA", "state_or_region": "California",
        "fatalities": 23, "affected_population": "30,000",
        "economic_damage_usd": "422,000,000",
        "short_description": (
            "Catastrophic debris flows swept through Montecito following "
            "heavy rainfall on post-fire slopes, burying dozens of homes."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2018_Montecito_mudflows",
        "wikimedia_name": "Montecito mudflows 2018",
    },
    {
        "id": "landslide_petropolis_2022",
        "event_name": "Petropolis Landslide",
        "year": 2022, "category": "landslide",
        "country": "Brazil", "state_or_region": "Rio de Janeiro state",
        "fatalities": 237, "affected_population": "30,000",
        "economic_damage_usd": "400,000,000",
        "short_description": (
            "Catastrophic landslides struck Petropolis after 260 mm of rain "
            "in three hours -- more than double the city's February average."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2022_Petropolis_landslides",
        "wikimedia_name": "2022 Petropolis landslides",
    },

    # ======== WILDFIRES (10) ================================================
    {
        "id": "wildfire_australia_2020",
        "event_name": "Australian Bushfires",
        "year": 2020, "category": "wildfire",
        "country": "Australia", "state_or_region": "New South Wales, Victoria",
        "fatalities": 33, "affected_population": "3,000,000",
        "economic_damage_usd": "4,400,000,000",
        "short_description": (
            "The most destructive bushfire season in Australian history, "
            "burning 18.6 million hectares and destroying over 3,000 homes."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2019-20_Australian_bushfire_season",
        "wikimedia_name": "Australia bushfire 2019",
    },
    {
        "id": "wildfire_maui_2023",
        "event_name": "Maui Wildfire",
        "year": 2023, "category": "wildfire",
        "country": "USA", "state_or_region": "Hawaii",
        "fatalities": 99, "affected_population": "11,000",
        "economic_damage_usd": "5,600,000,000",
        "short_description": (
            "Fast-moving wildfire destroyed historic Lahaina -- the deadliest "
            "US wildfire in over a century."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2023_Hawaii_wildfires",
        "wikimedia_name": "Lahaina fire 2023",
    },
    {
        "id": "wildfire_california_2023",
        "event_name": "California Wildfire",
        "year": 2023, "category": "wildfire",
        "country": "USA", "state_or_region": "California",
        "fatalities": 0, "affected_population": "100,000",
        "economic_damage_usd": "200,000,000",
        "short_description": (
            "The Park Fire became California's fourth-largest wildfire, "
            "burning over 429,000 acres."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/Park_Fire",
        "wikimedia_name": "California wildfire 2023",
    },
    {
        "id": "wildfire_canada_2023",
        "event_name": "Canada Wildfire",
        "year": 2023, "category": "wildfire",
        "country": "Canada", "state_or_region": "British Columbia, Alberta",
        "fatalities": 8, "affected_population": "235,000",
        "economic_damage_usd": "5,000,000,000",
        "short_description": (
            "Record-breaking season burning over 18 million hectares across "
            "Canada, forcing mass evacuations."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2023_Canadian_wildfires",
        "wikimedia_name": "2023 Canada wildfires",
    },
    {
        "id": "wildfire_uttarakhand_2016",
        "event_name": "Uttarakhand Forest Fire",
        "year": 2016, "category": "wildfire",
        "country": "India", "state_or_region": "Uttarakhand",
        "fatalities": 7, "affected_population": "multiple districts",
        "economic_damage_usd": "50,000,000",
        "short_description": (
            "Widespread forest fires ravaged thousands of hectares across "
            "Uttarakhand during extended dry months."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2016_Uttarakhand_forest_fires",
        "wikimedia_name": "Uttarakhand forest fire 2016",
    },
    {
        "id": "wildfire_amazon_2019",
        "event_name": "Amazon Fires",
        "year": 2019, "category": "wildfire",
        "country": "Brazil", "state_or_region": "Amazonas, Para",
        "fatalities": 0, "affected_population": "500,000",
        "economic_damage_usd": "1,000,000,000",
        "short_description": (
            "Record fires in the Amazon with over 72,000 fires recorded -- "
            "an 84 percent increase over the previous year."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2019_Amazon_rainforest_wildfires",
        "wikimedia_name": "Amazon fires 2019",
    },
    {
        "id": "wildfire_greece_2023",
        "event_name": "Greece Wildfire",
        "year": 2023, "category": "wildfire",
        "country": "Greece", "state_or_region": "Evros, Rhodes",
        "fatalities": 26, "affected_population": "200,000",
        "economic_damage_usd": "500,000,000",
        "short_description": (
            "The Dadia fire became one of the largest ever in the EU during "
            "a record heat wave; Rhodes forced mass tourist evacuation."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2023_Greek_wildfires",
        "wikimedia_name": "2023 Greek wildfires",
    },
    {
        "id": "wildfire_turkey_2021",
        "event_name": "Turkey Wildfire",
        "year": 2021, "category": "wildfire",
        "country": "Turkey", "state_or_region": "Antalya, Mugla",
        "fatalities": 8, "affected_population": "50,000",
        "economic_damage_usd": "300,000,000",
        "short_description": (
            "Devastating wildfires swept through Turkey's Aegean and "
            "Mediterranean coastal regions during an extreme heat wave."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2021_Turkey_wildfires",
        "wikimedia_name": "2021 Turkey wildfires",
    },
    {
        "id": "wildfire_siberia_2021",
        "event_name": "Siberia Wildfire",
        "year": 2021, "category": "wildfire",
        "country": "Russia", "state_or_region": "Yakutia",
        "fatalities": 0, "affected_population": "500,000",
        "economic_damage_usd": "400,000,000",
        "short_description": (
            "Record wildfires burned 18.8 million hectares in Siberia, "
            "releasing unprecedented carbon emissions."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2021_Siberian_wildfires",
        "wikimedia_name": "Siberia wildfire 2021",
    },
    {
        "id": "wildfire_chile_2024",
        "event_name": "Chile Wildfire",
        "year": 2024, "category": "wildfire",
        "country": "Chile", "state_or_region": "Valparaiso",
        "fatalities": 134, "affected_population": "16,000",
        "economic_damage_usd": "350,000,000",
        "short_description": (
            "Fast-moving wildfires destroyed over 7,000 homes in Valparaiso "
            "hills -- the deadliest wildfire disaster in Chilean history."
        ),
        "reference_url": "https://en.wikipedia.org/wiki/2024_Valparaiso_fires",
        "wikimedia_name": "2024 Valparaiso fires",
    },
]

# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def _clean_parens(s: str) -> str:
    """Strip parenthetical suffixes: 'Gujarat Earthquake (Bhuj)' -> 'Gujarat Earthquake'"""
    return re.sub(r"\s*\(.*?\)", "", s).strip()


def _make_queries(event: dict) -> list[str]:
    """
    Generate 3 short Wikimedia-compatible queries.
    Uses wikimedia_name override when present; falls back to event_name.
    Never appends the year when it already appears in the name string.

    wikimedia_name already has year  ->  ["1999 Odisha cyclone",
                                          "Odisha Super Cyclone",
                                          "Odisha cyclone 1999"]
    wikimedia_name has no year       ->  ["Kerala Floods 2018",
                                          "Kerala Floods",
                                          "2018 Kerala Floods"]
    """
    wiki  = event.get("wikimedia_name") or event["event_name"]
    year  = str(event["year"])
    ename = _clean_parens(event["event_name"])

    if year in wiki:
        # Name already embeds the year — build variants that avoid duplication
        bare = wiki.replace(year, "").strip()   # "Odisha cyclone" from "1999 Odisha cyclone"
        candidates = [wiki, ename, f"{bare} {year}"]
    else:
        candidates = [f"{wiki} {year}", wiki, f"{year} {wiki}"]

    # Deduplicate while preserving order
    seen: set[str] = set()
    out:  list[str] = []
    for q in candidates:
        q = " ".join(q.split())   # normalise whitespace
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _make_alt_queries(event: dict) -> list[str]:
    """
    Alternative queries used when primary candidates < TARGET_PER_EVENT.
    Based on location + category + year.
    """
    cat     = event["category"]
    year    = str(event["year"])
    region  = event.get("state_or_region", "").split(",")[0].strip()
    country = event.get("country", "")

    type_word = {
        "flood":      "flood",
        "wildfire":   "wildfire",
        "earthquake": "earthquake",
        "landslide":  "landslide",
        "cyclone":    "cyclone",
    }.get(cat, cat)

    alts: list[str] = []
    if region:
        alts.append(f"{region} {type_word} {year}")
    if country and country != region:
        alts.append(f"{country} {type_word} {year}")
    alts.append(f"{type_word} {year}")
    return alts[:3]


# ---------------------------------------------------------------------------
# Wikimedia Commons API
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: BASE * 2^(attempt-1) capped at MAX_BACKOFF."""
    return min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF) + random.uniform(0, 2)


def _wikimedia_get(params: dict) -> requests.Response | None:
    """GET Wikimedia API with exponential backoff on 429 / 503 / connection errors."""
    for attempt in range(1, MAX_RETRIES_API + 1):
        wait = _backoff(attempt)
        try:
            r = _session.get(WIKIMEDIA_API, params=params, timeout=20)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as exc:
            log.warning("  [api conn error] %s (attempt %d/%d), retry in %.0fs",
                        type(exc).__name__, attempt, MAX_RETRIES_API, wait)
            if attempt < MAX_RETRIES_API:
                time.sleep(wait)
            continue
        except Exception as exc:
            log.warning("  [api error] %s", exc)
            return None

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", wait))
            log.warning("  [429] rate limited, waiting %ds (attempt %d/%d)",
                        retry_after, attempt, MAX_RETRIES_API)
            if attempt < MAX_RETRIES_API:
                time.sleep(retry_after)
            continue
        if r.status_code == 503:
            log.warning("  [503] service unavailable, waiting %.0fs (attempt %d/%d)",
                        wait, attempt, MAX_RETRIES_API)
            if attempt < MAX_RETRIES_API:
                time.sleep(wait)
            continue
        try:
            r.raise_for_status()
        except Exception as exc:
            log.warning("  [HTTP error] %s", exc)
            return None
        return r

    log.warning("  All %d API attempts exhausted for: %s",
                MAX_RETRIES_API, params.get("srsearch") or params.get("titles", "")[:60])
    return None


def _wikimedia_search(query: str, limit: int = MAX_CANDIDATES) -> list[dict]:
    """
    Search Wikimedia Commons File namespace for query.
    Two-step: list=search -> batch imageinfo.
    429s are retried automatically via _wikimedia_get.
    """
    # Step 1 -- get file titles
    r1 = _wikimedia_get({
        "action":      "query",
        "format":      "json",
        "list":        "search",
        "srnamespace": 6,
        "srsearch":    query,
        "srlimit":     limit,
    })
    if r1 is None:
        return []
    hits = r1.json().get("query", {}).get("search", [])
    if not hits:
        return []

    time.sleep(INTERNAL_DELAY)  # pause before step 2 to avoid burst

    # Step 2 -- batch-fetch imageinfo for those titles (include thumburl at THUMB_WIDTH)
    titles = "|".join(h["title"] for h in hits[:50])
    r2 = _wikimedia_get({
        "action":    "query",
        "format":    "json",
        "titles":    titles,
        "prop":      "imageinfo",
        "iiprop":    "url|thumburl|size|mime|mediatype",
        "iiurlwidth": THUMB_WIDTH,
    })
    if r2 is None:
        return []
    pages = r2.json().get("query", {}).get("pages", {})
    return list(pages.values())


def _is_valid_candidate(page: dict) -> tuple[bool, str, int, int]:
    """
    Returns (valid, download_url, orig_width, orig_height).
    download_url is the thumbnail URL when available (lighter rate-limiting),
    falling back to the full-resolution original URL.
    """
    info_list = page.get("imageinfo", [])
    if not info_list:
        return False, "", 0, 0
    info      = info_list[0]
    mime      = info.get("mime", "")
    mediatype = info.get("mediatype", "")
    width     = info.get("width", 0)
    height    = info.get("height", 0)
    # Prefer thumbnail URL (goes through /thumb/ — less aggressively rate-limited)
    url       = info.get("thumburl") or info.get("url", "")
    title     = page.get("title", "").lower()

    if mime not in _ALLOWED_MIME:               return False, "", 0, 0
    if mediatype != "BITMAP":                   return False, "", 0, 0
    if width < MIN_SIDE or height < MIN_SIDE:   return False, "", 0, 0
    if _SKIP_RE.search(title):                  return False, "", 0, 0
    return True, url, width, height


def _download_image_with_retry(url: str, dest: Path) -> tuple[bool, str]:
    """
    Download url → dest with up to MAX_RETRIES_DL attempts.
    Returns (success, reason) where reason is one of:
      "ok" | "too_small" | "error" | "max_retries"
    Retries on 429, 503, and connection/reset errors with exponential backoff.
    """
    for attempt in range(1, MAX_RETRIES_DL + 1):
        wait = _backoff(attempt)
        try:
            r = _session.get(url, timeout=45, stream=True)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout) as exc:
            log.warning("  [conn %d/%d] %s — retry in %.0fs",
                        attempt, MAX_RETRIES_DL, type(exc).__name__, wait)
            dest.unlink(missing_ok=True)
            if attempt < MAX_RETRIES_DL:
                time.sleep(wait)
            continue
        except Exception as exc:
            log.warning("  [dl error] %s", exc)
            dest.unlink(missing_ok=True)
            return False, "error"

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", wait))
            log.warning("  [429 %d/%d] rate limited, waiting %ds",
                        attempt, MAX_RETRIES_DL, retry_after)
            if attempt < MAX_RETRIES_DL:
                time.sleep(retry_after)
            continue
        if r.status_code == 503:
            log.warning("  [503 %d/%d] service unavailable, waiting %.0fs",
                        attempt, MAX_RETRIES_DL, wait)
            if attempt < MAX_RETRIES_DL:
                time.sleep(wait)
            continue
        try:
            r.raise_for_status()
        except Exception as exc:
            log.warning("  [HTTP %s] %s", r.status_code, exc)
            dest.unlink(missing_ok=True)
            return False, "error"

        try:
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(8192):
                    fh.write(chunk)
        except Exception as exc:
            log.warning("  [write error] %s", exc)
            dest.unlink(missing_ok=True)
            return False, "error"

        if dest.stat().st_size < 4096:
            dest.unlink(missing_ok=True)
            return False, "too_small"
        return True, "ok"

    log.warning("  [max_retries] %d attempts exhausted: %s", MAX_RETRIES_DL, url[:80])
    return False, "max_retries"


# ---------------------------------------------------------------------------
# Checkpoint (resume support)
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    """Load URL→filepath map from checkpoint.json (empty dict if missing/corrupt)."""
    if CHECKPOINT_JSON.exists():
        try:
            return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_checkpoint(checkpoint: dict) -> None:
    """Persist checkpoint to disk after each successful download."""
    CHECKPOINT_JSON.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# AUDIT phase
# ---------------------------------------------------------------------------

def run_audit(events: list[dict]) -> list[dict]:
    """
    For every event:
      1. Run 3 primary queries, count valid candidates each.
      2. If best < TARGET_PER_EVENT, run up to 3 alt queries.
      3. Record best query and candidate count.
    Returns list of audit result dicts.
    """
    results: list[dict] = []
    total = len(events)

    for i, event in enumerate(events, 1):
        log.info("[%d/%d]  %s (%d)  [%s]",
                 i, total, event["event_name"], event["year"], event["category"])

        primary_queries = _make_queries(event)
        query_log: list[dict] = []
        best_count = 0
        best_query = primary_queries[0]

        for q in primary_queries:
            pages  = _wikimedia_search(q)
            valid  = sum(1 for p in pages if _is_valid_candidate(p)[0])
            total_returned = len(pages)
            query_log.append({
                "query": q, "type": "primary",
                "total_returned": total_returned, "valid": valid,
            })
            if valid > best_count:
                best_count = valid
                best_query = q
            log.info("  primary  %-45s  total=%-3d  valid=%d", repr(q)[:45], total_returned, valid)
            time.sleep(API_DELAY)

        alt_used = False
        if best_count < TARGET_PER_EVENT:
            alt_queries = _make_alt_queries(event)
            log.info("  < %d candidates -- trying %d alt queries", TARGET_PER_EVENT, len(alt_queries))
            for q in alt_queries:
                pages = _wikimedia_search(q)
                valid = sum(1 for p in pages if _is_valid_candidate(p)[0])
                total_returned = len(pages)
                query_log.append({
                    "query": q, "type": "alt",
                    "total_returned": total_returned, "valid": valid,
                })
                if valid > best_count:
                    best_count = valid
                    best_query = q
                    alt_used   = True
                log.info("  alt      %-45s  total=%-3d  valid=%d", repr(q)[:45], total_returned, valid)
                time.sleep(API_DELAY)

        status = "READY" if best_count >= TARGET_PER_EVENT else (
                 "LOW"   if best_count >= 5 else "MISSING")
        log.info("  => best_query=%r  candidates=%d  status=%s",
                 best_query, best_count, status)

        results.append({
            "event_id":    event["id"],
            "event_name":  event["event_name"],
            "year":        event["year"],
            "category":    event["category"],
            "best_query":  best_query,
            "candidates":  best_count,
            "alt_used":    alt_used,
            "status":      status,
            "query_log":   query_log,
        })

    return results


def _write_audit_csv(results: list[dict]) -> None:
    fields = ["event_id", "event_name", "year", "category",
              "best_query", "candidates", "alt_used", "status"]
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Saved %s  (%d rows)", AUDIT_CSV, len(results))


def _write_candidate_xlsx(results: list[dict]) -> None:
    summary_rows = [{k: r[k] for k in
                     ["event_id", "event_name", "year", "category",
                      "best_query", "candidates", "alt_used", "status"]}
                    for r in results]
    df_summary = pd.DataFrame(summary_rows)

    # Per-category aggregation
    cat_rows = []
    for cat in CATEGORIES:
        sub = [r for r in results if r["category"] == cat]
        cat_rows.append({
            "category":       cat,
            "events":         len(sub),
            "total_candidates": sum(r["candidates"] for r in sub),
            "avg_candidates": round(sum(r["candidates"] for r in sub) / max(len(sub), 1), 1),
            "ready_events":   sum(1 for r in sub if r["status"] == "READY"),
            "low_events":     sum(1 for r in sub if r["status"] == "LOW"),
            "missing_events": sum(1 for r in sub if r["status"] == "MISSING"),
        })
    df_cat = pd.DataFrame(cat_rows)

    # All queries tried
    query_rows = []
    for r in results:
        for ql in r.get("query_log", []):
            query_rows.append({
                "event_id":      r["event_id"],
                "event_name":    r["event_name"],
                "category":      r["category"],
                "query":         ql["query"],
                "type":          ql["type"],
                "total_returned": ql["total_returned"],
                "valid":         ql["valid"],
            })
    df_queries = pd.DataFrame(query_rows)

    # Gate summary
    ready = sum(1 for r in results if r["status"] == "READY")
    gate_pct = ready / len(results)
    gate_pass = gate_pct >= GATE_THRESHOLD
    df_gate = pd.DataFrame([{
        "total_events":     len(results),
        "ready_events":     ready,
        "low_events":       sum(1 for r in results if r["status"] == "LOW"),
        "missing_events":   sum(1 for r in results if r["status"] == "MISSING"),
        "gate_threshold":   f"{GATE_THRESHOLD*100:.0f}%",
        "gate_pct":         f"{gate_pct*100:.1f}%",
        "gate_result":      "PASS" if gate_pass else "FAIL",
        "projected_images": sum(min(r["candidates"], TARGET_PER_EVENT) for r in results),
    }])

    with pd.ExcelWriter(CANDIDATE_XLSX, engine="openpyxl") as xl:
        df_gate.to_excel(xl,     sheet_name="Gate Check",      index=False)
        df_summary.to_excel(xl,  sheet_name="Per Event",       index=False)
        df_cat.to_excel(xl,      sheet_name="Per Category",    index=False)
        df_queries.to_excel(xl,  sheet_name="All Queries",     index=False)

    log.info("Saved %s", CANDIDATE_XLSX)


def _check_80pct_gate(results: list[dict]) -> bool:
    ready = sum(1 for r in results if r["status"] == "READY")
    return (ready / len(results)) >= GATE_THRESHOLD


def _print_audit_report(results: list[dict]) -> None:
    ready   = sum(1 for r in results if r["status"] == "READY")
    low     = sum(1 for r in results if r["status"] == "LOW")
    missing = sum(1 for r in results if r["status"] == "MISSING")
    gate_ok = _check_80pct_gate(results)

    proj_imgs = sum(min(r["candidates"], TARGET_PER_EVENT) for r in results)

    w = 70
    print()
    print("=" * w)
    print("  AUDIT REPORT -- WIKIMEDIA CANDIDATE COVERAGE")
    print("=" * w)
    print(f"  {'Event':<42}  {'Cat':<10}  {'Cands':>5}  {'Status'}")
    print("-" * w)
    for r in results:
        mark = "OK" if r["status"] == "READY" else ("~~" if r["status"] == "LOW" else "XX")
        print(f"  [{mark}] {r['event_name']:<40}  {r['category']:<10}  {r['candidates']:>5}  {r['status']}")

    print()
    print("-" * w)
    print(f"  Per-category summary:")
    for cat in CATEGORIES:
        sub   = [r for r in results if r["category"] == cat]
        total = sum(r["candidates"] for r in sub)
        rdy   = sum(1 for r in sub if r["status"] == "READY")
        print(f"    {cat:<14}  {rdy}/10 events ready  {total:>4} total candidates")

    print()
    print("-" * w)
    print(f"  Events READY    (>=10 candidates) : {ready:>3} / {len(results)}")
    print(f"  Events LOW      ( 5-9 candidates) : {low:>3}")
    print(f"  Events MISSING  (  0-4 candidates): {missing:>3}")
    print(f"  Projected images (capped at 10)   : {proj_imgs}")
    print()
    gate_label = "PASS" if gate_ok else "FAIL"
    print(f"  80% gate ({int(GATE_THRESHOLD*100)}% events need >=10 candidates) : {gate_label}")
    if not gate_ok:
        print()
        print("  Events needing attention:")
        for r in results:
            if r["status"] != "READY":
                print(f"    [{r['status']}] {r['event_name']} ({r['year']})  --  best_query={r['best_query']!r}")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Per-event download
# ---------------------------------------------------------------------------

def _location(event: dict) -> str:
    parts = [event.get("state_or_region", ""), event.get("country", "")]
    return ", ".join(p for p in parts if p)


def _row_for_file(path: Path, event: dict) -> dict:
    cat = event["category"]
    return {
        "image_file": f"{cat}/{path.name}",
        "event_name": event["event_name"],
        "year":       event["year"],
        "category":   cat,
        "location":   _location(event),
        "source_url": "",
    }


def _process_event(
    event:      dict,
    checkpoint: dict,
    dry_run:    bool = False,
    best_query: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Download up to TARGET_PER_EVENT images for the event.

    Returns:
      rows  -- metadata rows (one per saved image, existing + new)
      stats -- {"downloaded": int, "failed": int, "skipped": int}

    checkpoint is updated in-place and flushed to disk after each successful download.
    """
    cat      = event["category"]
    eid      = event["id"]
    dest_dir = IMAGES_DIR / cat

    existing = sorted(
        list(dest_dir.glob(f"{eid}_*.jpg")) +
        list(dest_dir.glob(f"{eid}_*.png"))
    )
    existing_rows = [_row_for_file(f, event) for f in existing]
    need = TARGET_PER_EVENT - len(existing)

    stats = {"downloaded": 0, "failed": 0, "skipped": 0}

    if need <= 0:
        log.info("  [SKIP] %s -- %d images already on disk (target met)",
                 event["event_name"], len(existing))
        return existing_rows, stats

    if dry_run:
        log.info("  [DRY-RUN] %s -- would download %d images", event["event_name"], need)
        return existing_rows, stats

    # Build query list: best_query first, then primary, then alt
    queries: list[str] = []
    if best_query:
        queries.append(best_query)
    for q in _make_queries(event):
        if q not in queries:
            queries.append(q)
    for q in _make_alt_queries(event):
        if q not in queries:
            queries.append(q)

    # seen_urls: already-checkpointed URLs + any seen this run
    seen_urls: set[str] = set(checkpoint.keys())
    new_rows:  list[dict] = []
    idx = len(existing) + 1

    for query in queries:
        if len(new_rows) >= need:
            break
        log.info("  Query: %r", query)
        candidates = _wikimedia_search(query)
        time.sleep(API_DELAY)

        for page in candidates:
            if len(new_rows) >= need:
                break
            valid, url, w, h = _is_valid_candidate(page)
            if not valid:
                continue
            if url in seen_urls:
                stats["skipped"] += 1
                continue
            seen_urls.add(url)

            ext   = ".png" if url.lower().endswith(".png") else ".jpg"
            fname = f"{eid}_{idx:02d}{ext}"
            dest  = dest_dir / fname

            # Resume: file already on disk from a previous run
            if dest.exists() and dest.stat().st_size > 4096:
                log.info("  [resume] %s already on disk, skipping", fname)
                stats["skipped"] += 1
                checkpoint[url] = f"{cat}/{fname}"
                idx += 1
                continue

            log.info("  Downloading [%02d] %s  (%dx%d)  attempt 1/%d",
                     idx, fname, w, h, MAX_RETRIES_DL)
            ok, reason = _download_image_with_retry(url, dest)
            if ok:
                rel_path = f"{cat}/{fname}"
                row = {
                    "image_file": rel_path,
                    "event_name": event["event_name"],
                    "year":       event["year"],
                    "category":   cat,
                    "location":   _location(event),
                    "source_url": url,
                }
                new_rows.append(row)
                checkpoint[url] = rel_path
                _save_checkpoint(checkpoint)
                stats["downloaded"] += 1
                idx += 1
                jitter = random.uniform(JITTER_MIN, JITTER_MAX)
                log.info("  [ok] %s saved — jitter %.1fs", fname, jitter)
                time.sleep(jitter)
            else:
                log.warning("  [fail:%s] %s", reason, url[:80])
                stats["failed"] += 1

    total = len(existing) + len(new_rows)
    log.info("  => %s: %d total  (new=%d  failed=%d  skipped=%d)",
             event["event_name"], total,
             stats["downloaded"], stats["failed"], stats["skipped"])
    return existing_rows + new_rows, stats


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------

def _usd_to_billion(raw: str) -> float:
    try:
        return round(float(str(raw).replace(",", "")) / 1e9, 3)
    except (ValueError, TypeError):
        return 0.0


def _write_events_json(rows: list[dict]) -> None:
    entries = []
    for row in rows:
        ev = next((e for e in EVENTS
                   if e["event_name"] == row["event_name"] and e["year"] == row["year"]), None)
        if ev is None:
            continue
        base = Path(row["image_file"]).stem
        entries.append({
            "id":                  base,
            "name":                ev["event_name"],
            "year":                ev["year"],
            "category":            ev["category"],
            "location":            _location(ev),
            "description":         ev["short_description"],
            "casualties":          ev.get("fatalities"),
            "affected_population": ev.get("affected_population", ""),
            "damage_usd_billion":  _usd_to_billion(ev.get("economic_damage_usd", "0")),
            "source":              ev.get("reference_url", ""),
            "image_filename":      row["image_file"],
            "event_id":            ev["id"],
            "country":             ev.get("country", ""),
            "state_or_region":     ev.get("state_or_region", ""),
        })
    out = {
        "_schema_version": "2.0",
        "_description":    (
            "Historical disaster event database for CLIP/FAISS similarity retrieval. "
            f"Flat format: one entry per image. {len(entries)} entries."
        ),
        "_generated":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events":          entries,
    }
    EVENTS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s  (%d entries)", EVENTS_JSON, len(entries))


def _write_metadata_csv(rows: list[dict]) -> None:
    fields = ["image_file", "event_name", "year", "category", "location", "source_url"]
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("Saved %s  (%d rows)", METADATA_CSV, len(rows))


def _write_summary_xlsx(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No rows -- skipping dataset_summary.xlsx")
        return

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as xl:
        cat_counts = df.groupby("category").size().reset_index(name="images")
        overview   = pd.DataFrame({
            "Metric": ["Total Images", "Total Events", "Categories", ""]
                      + [c.capitalize() for c in CATEGORIES],
            "Value":  [len(df), len(EVENTS), 5, ""]
                      + [cat_counts.set_index("category")["images"].get(c, 0)
                         for c in CATEGORIES],
        })
        overview.to_excel(xl, sheet_name="Overview",       index=False)
        (df.groupby(["category", "event_name", "year"])
           .size().reset_index(name="image_count")
           .sort_values(["category", "event_name"])
           .to_excel(xl, sheet_name="Per Event",           index=False))
        cat_counts.to_excel(xl, sheet_name="Per Category", index=False)
        df.to_excel(xl,         sheet_name="Full Metadata",index=False)

    log.info("Saved %s", SUMMARY_XLSX)


def _print_report(rows: list[dict],
                  run_stats: dict | None = None,
                  scoped_events: list[dict] | None = None) -> None:
    """
    Print a completion report.
    run_stats  -- {"downloaded": int, "failed": int, "skipped": int}
    scoped_events -- the events that were processed this run (for per-event detail)
    """
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    total_bytes = sum(
        (IMAGES_DIR / r["image_file"]).stat().st_size
        for r in rows if (IMAGES_DIR / r["image_file"]).exists()
    )
    print()
    print("=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total images on disk : {len(rows)}")
    print(f"  Storage              : {total_bytes / 1e6:.1f} MB")
    if run_stats:
        print()
        print(f"  This run:")
        print(f"    Downloaded : {run_stats['downloaded']}")
        print(f"    Failed     : {run_stats['failed']}")
        print(f"    Skipped    : {run_stats['skipped']}  (already on disk or duplicate URL)")
    print()
    processed_cats = {e["category"] for e in (scoped_events or EVENTS)}
    for cat in CATEGORIES:
        if cat not in processed_cats:
            continue
        n   = cat_counts.get(cat, 0)
        bar = "#" * (n // 2)
        pct = f"{n/(TARGET_PER_EVENT*10)*100:.0f}%" if n else "0%"
        flag = " OK" if n >= MIN_TARGET else " LOW"
        print(f"  {cat:<14} {n:>4}  {pct:>4}  {bar}{flag}")

    # Per-event breakdown (only for scoped events this run)
    if scoped_events:
        print()
        print(f"  Per-event (this run):")
        print(f"  {'Event':<40} {'Cat':<10} {'On disk':>7} {'Min':>4}")
        print(f"  {'-'*40} {'-'*10} {'-'*7} {'-'*4}")
        for ev in scoped_events:
            n = sum(1 for r in rows if r["event_name"] == ev["event_name"])
            ok_marker = "OK" if n >= MIN_TARGET else "LOW" if n > 0 else "MISS"
            print(f"  {ev['event_name']:<40} {ev['category']:<10} {n:>7}  [{ok_marker}]")

    missing = [e["event_name"] for e in (scoped_events or EVENTS)
               if not any(r["event_name"] == e["event_name"] for r in rows)]
    if missing:
        print(f"\n  Missing / empty events ({len(missing)}):")
        for nm in missing:
            print(f"    x  {nm}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical disaster images from Wikimedia Commons (v4)"
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Run query audit (count candidates for all events) without downloading",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without any network calls or downloads",
    )
    parser.add_argument(
        "--category", choices=CATEGORIES,
        help="Only process events in this category",
    )
    parser.add_argument(
        "--all-categories", action="store_true",
        help="Process all 5 categories (default: flood + cyclone + earthquake)",
    )
    parser.add_argument(
        "--event",
        help="Only process the event with this id (e.g. flood_kerala_2018)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the 80%% gate check and download even if coverage is low",
    )
    args = parser.parse_args()

    for cat in CATEGORIES:
        (IMAGES_DIR / cat).mkdir(parents=True, exist_ok=True)

    # ── Scope filter ────────────────────────────────────────────────────────
    events = EVENTS
    if args.event:
        events = [e for e in events if e["id"] == args.event]
        if not events:
            log.error("Event id %r not found.", args.event)
            sys.exit(1)
    elif args.category:
        events = [e for e in events if e["category"] == args.category]
        log.info("Scoped to %d %s events", len(events), args.category)
    elif not args.all_categories:
        events = [e for e in events if e["category"] in DEFAULT_CATEGORIES]
        log.info("Default scope: %s  (%d events)", DEFAULT_CATEGORIES, len(events))

    # ── AUDIT mode ─────────────────────────────────────────────────────────
    if args.audit:
        log.info("=== AUDIT MODE ===  %d events", len(events))
        results = run_audit(events)
        _write_audit_csv(results)
        _write_candidate_xlsx(results)
        _print_audit_report(results)
        gate_ok = _check_80pct_gate(results)
        if gate_ok:
            log.info("Gate PASSED. Run without --audit to start download.")
        else:
            log.warning("Gate FAILED. Fix low-coverage events or use --force to override.")
        return

    # ── DRY-RUN mode ───────────────────────────────────────────────────────
    if args.dry_run:
        log.info("=== DRY-RUN ===  %d events x %d target = %d images",
                 len(events), TARGET_PER_EVENT, len(events) * TARGET_PER_EVENT)
        checkpoint = _load_checkpoint()
        all_rows: list[dict] = []
        for i, event in enumerate(events, 1):
            log.info("[%d/%d] %s (%d)", i, len(events), event["event_name"], event["year"])
            rows, _ = _process_event(event, checkpoint, dry_run=True)
            all_rows.extend(rows)
        _print_report(all_rows, scoped_events=events)
        return

    # ── DOWNLOAD mode ───────────────────────────────────────────────────────
    # Load prior audit best-queries if available
    best_queries: dict[str, str] = {}
    if AUDIT_CSV.exists():
        try:
            with open(AUDIT_CSV, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    best_queries[row["event_id"]] = row["best_query"]
            log.info("Loaded %d best-queries from audit CSV", len(best_queries))
        except Exception:
            pass

    checkpoint = _load_checkpoint()
    log.info("Checkpoint: %d URLs already downloaded", len(checkpoint))

    log.info("=== DOWNLOAD ===  %d events  (target %d/event, min %d/event)",
             len(events), TARGET_PER_EVENT, MIN_TARGET)

    all_rows: list[dict] = []
    total_dl = total_fail = total_skip = 0

    for i, event in enumerate(events, 1):
        log.info("[%d/%d] %s (%d) [%s]",
                 i, len(events), event["event_name"], event["year"], event["category"])
        bq = best_queries.get(event["id"])
        rows, stats = _process_event(event, checkpoint, dry_run=False, best_query=bq)
        all_rows.extend(rows)
        total_dl   += stats["downloaded"]
        total_fail += stats["failed"]
        total_skip += stats["skipped"]

        # Checkpoint log after each event
        log.info("  [checkpoint] event=%s  dl=%d  fail=%d  skip=%d  total_dl=%d",
                 event["id"], stats["downloaded"], stats["failed"],
                 stats["skipped"], total_dl)

        if i < len(events):
            log.info("  Waiting %.0fs before next event ...", EVENT_DELAY)
            time.sleep(EVENT_DELAY)

    run_stats = {"downloaded": total_dl, "failed": total_fail, "skipped": total_skip}

    log.info("Writing output files ...")
    _write_events_json(all_rows)
    _write_metadata_csv(all_rows)
    _write_summary_xlsx(all_rows)
    _print_report(all_rows, run_stats=run_stats, scoped_events=events)
    log.info("Done. Total downloaded=%d  failed=%d  skipped=%d",
             total_dl, total_fail, total_skip)


if __name__ == "__main__":
    main()
