# Post-AGI Society Dashboard

A knowledge graph dashboard that curates, analyzes, and visualizes the intellectual landscape around the societal, economic, and existential implications of Artificial General Intelligence (AGI). It collects artifacts (books, papers, essays, tweets, interviews, news, research, projections) and the relationships between them, computes sentiment and projection plausibility, and presents everything in an interactive graph with animated analytics.

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Frontend   │────▶│   FastAPI (Python)│────▶│   SQLite DB  │
│  (vanilla JS)│◀────│   uvicorn :8200  │◀────│  (data/ags.db)│
└──────────────┘     └──────────────────┘     └──────────────┘
                            │
                    ┌───────┴────────┐
                    │   APScheduler   │
                    │  (daily 4AM)    │
                    └────────────────┘
```

- **Backend**: FastAPI with SQLAlchemy ORM, SQLite database (WAL mode)
- **Frontend**: Single-page HTML/CSS/JS with vis-network (graph), Chart.js (charts), Font Awesome (icons)
- **Scraping**: feedparser (RSS), requests (Nitter tweets), arXiv API XML
- **Scheduling**: APScheduler runs ingestion daily at 04:00 UTC
- **Containerization**: Docker + docker-compose with named volume `ags_data` for persistence

---

## Data Model

### Artifact — Core entity
| Field | Type | Description |
|---|---|---|
| id | Integer | Primary key |
| title | String(500) | Artifact title |
| artifact_type | Enum | `book`, `paper`, `essay`, `tweet`, `interview`, `news`, `research`, `projection` |
| url | String(1000) | Source URL |
| description | Text | Summary / abstract |
| content | Text | Full text (paper abstract, diarized interview concepts) |
| source | String(200) | Origin (e.g. "arXiv", "LessWrong", "@samaltman") |
| author | String(200) | Author name |
| date_published | DateTime | Publication date |
| tags | String(500) | Comma-separated tags |

### Sentiment — Per-artifact sentiment scores
| Field | Type | Description |
|---|---|---|
| score | Float | -1 (very negative) to +1 (very positive) |
| label | Enum | `very_negative`, `negative`, `neutral`, `positive`, `very_positive` |
| source | String | `manual`, `initial`, `auto-scraper`, `auto-nitter`, `auto-arxiv` |

### Projection — Future-facing predictions
| Field | Type | Description |
|---|---|---|
| projection_type | Enum | `utopian`, `dystopian`, `cautionary`, `accelerationist`, `neutral` |
| confidence | Float | 0 to 1 |
| timeframe | String | e.g. "2040", "2027-2030" |
| summary | Text | Short description of the projection |

### Relationship — Directed graph edges
| Field | Type | Description |
|---|---|---|
| relationship_type | Enum | `references`, `supports`, `contradicts`, `builds_upon`, `responds_to`, `similar_to` |
| description | String(500) | Context for the relationship |

---

## Data Sources & Ingestion

### Manual Seed Data (63 artifacts, 74 relationships)
Curated collection spanning the full AGI discourse spectrum:
- **Books**: Bostrom, Kurzweil, Tegmark, Russell, Christian, Ord, Barrat, Gawdat, Metz
- **Papers**: GPT-4, Scaling Laws, Chinchilla, Constitutional AI, DPO, Weak-to-Strong, Sleeper Agents, WMDP, PRM, RLAIF, Gemini, HH-RLHF, Data Scaling
- **Essays**: Yudkowsky (Lethalities), Aschenbrenner (Situational Awareness), Altman, Sutton (Bitter Lesson), Scott Alexander (Moloch), Cotra (Bio Anchors), Amodei (Loving Grace)
- **Interviews**: Lex Fridman (Altman, LeCun, Yudkowsky), 80,000 Hours (Cotra), JRE (Yampolskiy), Dwarkesh Patel (Aschenbrenner), Ezra Klein (Amodei), Bankless (Vitalik)
- All with pre-computed sentiments, projections, and cross-referenced relationships

### Auto-Ingestion (RSS + Nitter + arXiv)
Runs on demand (Fetch button) or daily at 04:00 UTC:

| Source | Type | Details |
|---|---|---|
| Hacker News RSS | news | Frontpage posts |
| arXiv cs.AI RSS | paper | AI papers |
| LessWrong RSS | essay | Alignment discussions |
| AI Alignment Forum RSS | research | Technical alignment |
| AI News RSS | news | Industry coverage |
| ML Mastery RSS | research | ML tutorials |
| OpenAI Blog RSS | research | Frontier lab updates |
| Nitter (13 accounts) | tweet | samaltman, elonmusk, ylecun, demishassabis, karpathy, shanelegg, geoffreyhinton, nickbostrom, raoul_poe, ESYudkowsky, daniela_amodei, darioduck, metaculus |
| arXiv API (7 queries) | paper | AGI, interpretability, scaling laws, RLHF, governance, superalignment, AI capabilities |

### Auto-Processing Pipeline
Each newly ingested artifact goes through:

1. **Relevance filtering** — matched against 36 post-AGI keywords
2. **Sentiment scoring** — regex keyword counting (positive vs negative word lists)
3. **Tag extraction** — matching keywords converted to tags
4. **Projection detection** — regex for timeline patterns (`by 2030`, `within 5 years`) and forecast language (`predict`, `likely`, `probability`) → creates projection with type, confidence, timeframe, summary
5. **Auto-relationships** — compares against all existing artifacts:
   - ≥2 shared tags → `similar_to`
   - Same author → `similar_to`
   - Title mentioned in description → `references`
   - Capped at 8 edges per new artifact

---

## Plausibility Computation

The **Plausibility tab** shows an animated visualization of how plausible each projection type is, based on collected evidence.

### Data Source
`GET /api/plausibility` aggregates across all artifacts with projections:

```python
for each projection_type (utopian, dystopian, cautionary, accelerationist, neutral):
    avg_confidence = mean of all projection.confidence values
    avg_sentiment  = mean of all sentiment scores for artifacts that have this projection type
    plausibility   = avg_confidence * 0.6 + max(0, avg_sentiment) * 0.4
```

### Formula

```
plausibility(type) = confidence_weight * 0.6 + max(0, sentiment_weight) * 0.4
```

- **Confidence (60%)** — how definitively the sources state their projections. Higher confidence = more definitive claims.
- **Sentiment (40%)** — only positive sentiment contributes (via `max(0, avg_sentiment)`). Negative sentiment does not reduce plausibility — it simply contributes 0 on the sentiment axis. This means:
  - Cautionary/dystopian projections (negative sentiment) are scored primarily on confidence alone
  - Utopian/accelerationist projections (positive sentiment) get a boost from both confidence and sentiment

### Overall Plausibility
```
overall = avg_conf_all * 0.6 + max(0, avg_sent_all) * 0.4
```
Represents the aggregate credibility-weighted outlook across all collected projections.

### Current Scores (seed data)
| Type | Plausibility | Confidence | Sentiment | Count |
|---|---|---|---|---|
| Utopian | 71.7% | 71.7% | +0.72 | 6 |
| Accelerationist | 56.4% | 70.9% | +0.35 | 11 |
| Dystopian | 49.5% | 82.5% | -0.77 | 6 |
| Cautionary | 43.0% | 71.7% | -0.32 | 9 |
| Neutral | 40.5% | 57.5% | +0.15 | 12 |
| **Overall** | **47.7%** | **69.1%** | **+0.16** | **44** |

---

## Frontend Features

### Main View: Knowledge Graph
- **vis-network** interactive graph with zoom, pan, physics simulation
- Nodes colored by artifact type with **colored glow shadows**
- Node size scales with sentiment magnitude + confidence
- Hover tooltip, click to open detail overlay
- Type filter buttons, text search, graph fit controls
- **Dark/Light theme toggle** persisted in localStorage

### Detail Overlay
- Full description, source link with "Open Source" button
- Tags, sentiment scores, projection cards
- Graph connections (clickable — navigates to linked node)
- **Similar artifacts** with color-coded similarity ring, shared tags, and "View in Graph" button
- **Diarized concepts** rendered as styled cards for interviews

### Plausibility Tab
- **Animated SVG ring** for overall plausibility (count-up from 0)
- **Per-type cards** sorted by plausibility descending, each with:
  - Small SVG ring fill (animated stroke-dashoffset)
  - Confidence fill bar (animated width)
  - Count-up number animations (eased cubic-bezier via requestAnimationFrame)
  - Definition line + aggregated projection summaries
  - List of contributing artifacts
- Staggered card entrance animations (200ms + 180ms offset per card)

### Right Panel (Collapsible)
- **Add** — manual artifact creation with sentiment and projection fields
- **Relate** — create relationships between any two artifacts
- **Database** — re-seed button (clears all data and reloads from seed)

### Fetch Button
- Manual trigger for web scraper (runs in background thread)
- Polls `/api/ingest/status` until completion
- "Last fetch" timestamp shown in header stats bar
- APScheduler runs automatically daily at 04:00 UTC

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/artifacts` | List artifacts (filter by type, tag, search) |
| GET | `/api/artifacts/{id}` | Artifact detail with sentiments, projections, relationships |
| POST | `/api/artifacts` | Create artifact (optional sentiment/projection) |
| DELETE | `/api/artifacts/{id}` | Delete artifact |
| GET | `/api/artifacts/{id}/similar` | Similar artifacts with scores and shared tags |
| GET | `/api/graph` | Full graph data (nodes + edges for vis-network) |
| GET | `/api/stats` | Aggregate stats (counts, avg sentiment, projection breakdown) |
| GET | `/api/plausibility` | Per-type plausibility with summaries and supporting artifacts |
| POST | `/api/relationships` | Create relationship |
| POST | `/api/ingest/run` | Trigger scraper (runs in background) |
| GET | `/api/ingest/status` | Last run, items fetched, total runs |
| POST | `/api/reseed` | Clear database and reload from seed |
| GET | `/api/search` | Full-text search across title/description/author/tags |
| GET | `/` | Serve dashboard |

---

## Running the Project

### Docker (Recommended)
```bash
docker compose up -d --build
# Dashboard at http://localhost:8200
# Persistent volume ags_data stores SQLite database
```

### Local Development
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8200 --reload
# Dashboard at http://localhost:8200
# Database at data/ags.db
```

### First Start
The database is auto-seeded on first startup with 63 artifacts and 74 relationships. To reset:
```bash
rm -f data/ags.db
# Restart the server
```
Or use the **Re-seed Database** button in the right panel.

---

## Project Structure

```
post-AGI-society/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes, scheduler, CORS
│   ├── database.py      # SQLAlchemy engine (WAL mode + busy_timeout)
│   ├── models.py        # ORM: Artifact, Sentiment, Projection, Relationship
│   ├── scraper.py       # RSS/Nitter/arXiv ingestion pipeline
│   └── seed_data.py     # 63 curated artifacts + 74 relationships
├── data/                # SQLite database (gitignored, persisted via Docker volume)
├── static/
│   └── index.html       # Single-page dashboard (all CSS/JS inline)
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component | Library |
|---|---|
| Backend framework | FastAPI 0.115 |
| ASGI server | Uvicorn 0.34 |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite 3 (WAL mode) |
| Graph visualization | vis-network 9.1 |
| Charts | Chart.js 4.4 |
| Icons | Font Awesome 6.7 |
| RSS parsing | feedparser 6.0 |
| HTTP | requests + httpx |
| Scheduling | APScheduler 3.10 |
| Containerization | Docker + docker-compose |
