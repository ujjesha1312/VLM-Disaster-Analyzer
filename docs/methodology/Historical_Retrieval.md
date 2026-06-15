# Historical Disaster Retrieval Module

**VLM Disaster Analyzer — FAISS Similarity Search**

> Source of truth: [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md) §4  
> Related: [System_Architecture.md](../architecture/System_Architecture.md) · [Dataset_Preparation.md](Dataset_Preparation.md) · [Results_Summary.md](../reports/Results_Summary.md)

---

## Purpose

When a disaster image is analyzed, the system retrieves the top-5 most visually similar historical disaster events from a curated database of 30 Indian and global events. This provides:

- **Contextual precedents** — past events with known outcomes
- **Scale reference** — casualty figures and affected population estimates
- **Operational intelligence** — what response was effective historically
- **Research grounding** — links satellite imagery to documented disaster records

This is Stage 3 of the three-stage production pipeline and runs in under 100 ms.

---

## Architecture

```
Query Image
     │
     ▼
CLIP get_image_features()
     │  512-dim unit-normalized vector
     ▼
FAISS IndexFlatIP.search(query_vec, top_k)
     │  inner product = cosine similarity (unit vectors)
     ▼
Filter by category (optional)
     │
     ▼
Top-k results with similarity %
{ event, year, location, category,
  similarity, casualties, affected_population,
  description, damage_usd_billion, source }
```

CLIP's embedding from Stage 1 is reused — no second forward pass.

---

## Database

**File:** `datasets/historical/historical_events.json`  
**Count:** 30 events  
**Categories:** Flood (7), Cyclone (7), Wildfire (6), Earthquake (5), Landslide (5)  
**Focus:** India-focused with major global benchmarks

### Event Record Schema

```json
{
  "id": "flood_kerala_2018",
  "name": "Kerala Floods",
  "year": 2018,
  "category": "flood",
  "location": "Kerala, India",
  "description": "Worst floods in nearly a century affecting 14 districts.",
  "casualties": 483,
  "affected_population": "5.4 million",
  "damage_usd_billion": 3.0,
  "source": "NDMA India",
  "image_filename": "kerala_floods_2018.jpg",
  "wikipedia_search": "2018 Kerala floods"
}
```

### Complete Event List

#### Flood (7 events)
| Event | Year | Location | Casualties |
|---|---|---|---|
| Kerala Floods | 2018 | Kerala, India | 483 |
| Assam Floods | 2020 | Assam, India | 123 |
| Assam Floods | 2022 | Assam, India | 193 |
| Bihar Floods | 2017 | Bihar, India | 514 |
| Uttarakhand Floods | 2013 | Uttarakhand, India | 5,700+ |
| Pakistan Super Floods | 2022 | Pakistan | 1,739 |
| Bangladesh Floods | 2017 | Bangladesh | 114 |

#### Cyclone (7 events)
| Event | Year | Location | Casualties |
|---|---|---|---|
| Cyclone Fani | 2019 | Odisha, India | 89 |
| Cyclone Amphan | 2020 | West Bengal, India | 128 |
| Cyclone Biparjoy | 2023 | Gujarat, India | 2 |
| Cyclone Yaas | 2021 | Odisha/WB, India | 19 |
| Cyclone Tauktae | 2021 | Gujarat, India | 155 |
| Cyclone Nargis | 2008 | Myanmar | 138,000+ |
| Typhoon Haiyan | 2013 | Philippines | 6,300+ |

#### Wildfire (6 events)
| Event | Year | Location | Casualties |
|---|---|---|---|
| Uttarakhand Wildfires | 2021 | Uttarakhand, India | 4 |
| Uttarakhand Wildfires | 2024 | Uttarakhand, India | 5 |
| Himachal Pradesh Wildfires | 2023 | Himachal Pradesh, India | 3 |
| Camp Fire | 2018 | California, USA | 85 |
| Amazon Fires | 2019 | Brazil | — |
| Australian Black Summer | 2019–20 | Australia | 34 |

#### Earthquake (5 events)
| Event | Year | Location | Casualties |
|---|---|---|---|
| Nepal Earthquake | 2015 | Nepal | 8,964 |
| Gujarat Earthquake | 2001 | Gujarat, India | 20,000+ |
| Haiti Earthquake | 2010 | Haiti | 230,000+ |
| Turkey–Syria Earthquakes | 2023 | Turkey/Syria | 59,000+ |
| Sikkim Earthquake | 2023 | Sikkim, India | 40 |

#### Landslide (5 events)
| Event | Year | Location | Casualties |
|---|---|---|---|
| Wayanad Landslide | 2024 | Kerala, India | 400+ |
| Kedarnath Landslide | 2013 | Uttarakhand, India | 5,700+ |
| Pune Landslide | 2014 | Maharashtra, India | 151 |
| Manipur Landslide | 2022 | Manipur, India | 37 |
| Joshimath Subsidence | 2023 | Uttarakhand, India | — |

---

## Index Construction

### 1. Download reference images

```bash
python scripts/download_historical_images.py
```

This script:
- Maps each event ID to a Wikipedia article title
- Calls the Wikipedia `pageimages` API for CC-licensed thumbnails
- Downloads images to `datasets/historical/images/<category>/`
- Automatically triggers `src/retrieval/build_index.py` on completion

Options:
```bash
python scripts/download_historical_images.py --skip-existing  # Skip already downloaded
python scripts/download_historical_images.py --no-build       # Download only, don't build index
```

### 2. Build FAISS index

```bash
python src/retrieval/build_index.py
# or with dry run (no writes):
python src/retrieval/build_index.py --dry-run
```

The builder:
1. Loads `datasets/historical/historical_events.json`
2. For each event with an existing image file, runs CLIP `get_image_features()`
3. L2-normalizes each 512-dim vector (`features / features.norm()`)
4. Inserts into `faiss.IndexFlatIP` (inner product index)
5. Serializes to `datasets/historical/index/disaster.index`
6. Writes metadata to `datasets/historical/index/metadata.json`

Events with missing image files are skipped and logged — the index is built from whatever images are available.

### Index files

```
datasets/historical/
├── historical_events.json          ← Event database (30 records)
├── images/
│   ├── flood/                      ← Reference images by category
│   ├── cyclone/
│   ├── wildfire/
│   ├── earthquake/
│   └── landslide/
└── index/
    ├── disaster.index              ← FAISS binary index
    └── metadata.json               ← Event metadata for search results
```

---

## Search Mechanics

`src/retrieval/search.py` — lazy singleton, loaded on first call.

### `find_similar_events(image_path, top_k=5, category_filter=None)`

1. Embed query image: `clip_model.embed_image(image_path)` → 512-dim unit vector
2. FAISS search: `_index.search(query_vec, fetch_k)` → (distances, indices)
   - `IndexFlatIP` with unit vectors: distance = cosine similarity ∈ [0, 1]
3. If `category_filter` set: fetch `top_k × 5` candidates, filter, take top `top_k`
4. Convert similarity to percentage: `round(float(dist) * 100, 1)`
5. Return list of event dicts with similarity added

### Similarity thresholds (frontend rendering)

| Similarity | Color | Interpretation |
|---|---|---|
| ≥ 80% | Emerald green | High visual match |
| 65–80% | Amber | Moderate match |
| < 65% | Muted gray | Low match |

### `index_status()`

Returns:
```json
{
  "index_built": true,
  "event_count": 28,
  "index_path": "datasets/historical/index/disaster.index"
}
```

---

## API Endpoints

### `POST /predict/similar`

Upload a disaster image → receive top-k similar historical events.

**Request:**
```
POST /predict/similar
Content-Type: multipart/form-data

file=<image>
top_k=5          (optional, 1–20, default 5)
category=flood   (optional, filter by category)
```

**Response:**
```json
{
  "similar_events": [
    {
      "event": "Kerala Floods",
      "year": 2018,
      "location": "Kerala, India",
      "category": "flood",
      "description": "Worst floods in nearly a century...",
      "similarity": 87.3,
      "casualties": 483,
      "affected_population": "5.4 million",
      "damage_usd_billion": 3.0,
      "source": "NDMA India"
    }
  ],
  "index_status": { "index_built": true, "event_count": 28 },
  "top_k": 5
}
```

### `GET /retrieval/status`

```json
{
  "index_built": true,
  "event_count": 28,
  "index_path": "datasets/historical/index/disaster.index"
}
```

---

## Frontend Integration

The `SimilarEventsCard` component in `frontend/src/App.jsx` renders when the unified pipeline returns `similar_events.length > 0`.

Each card displays:
- Rank badge (1st, 2nd, …)
- Event name + year
- Location
- One-sentence description
- Color-coded similarity bar (emerald / amber / gray)
- Casualty and affected population figures

The card renders nothing when `similar_events` is empty — full backward compatibility when the index is not built.

---

## Fault Tolerance

Stage 3 in `backend/services/disaster_service.py`:

```python
similar_events = []
if ENABLE_RETRIEVAL:
    try:
        from retrieval.search import find_similar_events
        similar_events = await asyncio.to_thread(
            find_similar_events, image_path, 5, final_type.lower()
        )
    except Exception as _retrieval_err:
        log.debug("[Retrieval] Skipped: %s", _retrieval_err)
```

Scenarios that return `[]` silently:
- FAISS index not built
- Index file corrupted
- `ENABLE_RETRIEVAL=false`
- Image cannot be embedded

Stage 1 and Stage 2 output is unaffected in all cases.

---

*For the full pipeline see [System_Architecture.md](../architecture/System_Architecture.md)*  
*For dataset details see [Dataset_Preparation.md](Dataset_Preparation.md)*
