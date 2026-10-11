import os
import re
import atexit
import threading
from collections import defaultdict
from datetime import datetime, timezone, timedelta, datetime as dt
from typing import Optional

import logging

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from sqlalchemy import func
from apscheduler.schedulers.background import BackgroundScheduler

from .config import (
    ADMIN_TOKEN,
    CORS_ORIGINS,
    DATA_DIR,
    INGEST_ENABLED,
    INGEST_HOUR,
    INGEST_MINUTE,
    SEED_ON_STARTUP,
)
from .api_security import require_admin, safe_url
from .database import init_db, get_db
from .models import Artifact, Sentiment, Projection, Relationship, ArtifactType, SecurityAssessment
from .seed_data import seed_database
from .scraper import run_ingestion
from . import security as sec_agent

logger = logging.getLogger("ags")
logging.basicConfig(level=logging.INFO)

ingestion_stats = {"last_run": None, "items_fetched": 0, "total_runs": 0}
ingestion_lock = threading.Lock()

os.makedirs(DATA_DIR, exist_ok=True)

from contextlib import asynccontextmanager

app = FastAPI(title="Post-AGI Society Dashboard")


# CORS: wildcard origin is incompatible with credentials (and unsafe), so we
# only enable credentials when explicit origins are configured via env.
_wildcard = CORS_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)
if _wildcard and ADMIN_TOKEN:
    logger.warning(
        "CORS is wide open ('*') while AGS_ADMIN_TOKEN is set; "
        "configure AGS_CORS_ORIGINS for your real front-end origins."
    )

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

    from fastapi.responses import FileResponse
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))


def run_scheduled_ingestion():
    if not ingestion_lock.acquire(blocking=False):
        return
    db = next(get_db())
    try:
        count = run_ingestion(db)
        ingestion_stats["last_run"] = datetime.now(timezone.utc).isoformat()
        ingestion_stats["items_fetched"] += count
        ingestion_stats["total_runs"] += 1
    finally:
        db.close()
        ingestion_lock.release()

scheduler = BackgroundScheduler()
if INGEST_ENABLED:
    scheduler.add_job(run_scheduled_ingestion, "cron", hour=INGEST_HOUR, minute=INGEST_MINUTE)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))


async def on_start():
    init_db()
    _ensure_security_columns()
    db = next(get_db())
    try:
        if SEED_ON_STARTUP:
            seed_database(db)
    finally:
        db.close()


# Modern lifespan handler (replaces deprecated @app.on_event("startup")).
@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await on_start()
    yield


app.router.lifespan_context = _lifespan


class ArtifactCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    artifact_type: str
    url: Optional[str] = None
    description: Optional[str] = Field(None, max_length=20000)
    content: Optional[str] = Field(None, max_length=200000)
    source: Optional[str] = Field(None, max_length=300)
    author: Optional[str] = Field(None, max_length=300)
    date_published: Optional[datetime] = None
    tags: Optional[str] = Field(None, max_length=2000)
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[str] = Field(None, max_length=50)
    projection_type: Optional[str] = Field(None, max_length=50)
    projection_confidence: Optional[float] = Field(None, ge=0, le=1)
    projection_timeframe: Optional[str] = Field(None, max_length=100)
    projection_summary: Optional[str] = Field(None, max_length=5000)

    @field_validator("artifact_type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        allowed = {t.value for t in ArtifactType}
        if v not in allowed:
            raise ValueError(f"artifact_type must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator("url")
    @classmethod
    def _valid_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        if safe_url(v) is None:
            raise ValueError("url must be a valid http(s) URL")
        return v.strip()


class RelationshipCreate(BaseModel):
    source_id: int
    target_id: int
    relationship_type: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=2000)


class SecurityAssessRequest(BaseModel):
    product_name: Optional[str] = ""
    product_url: Optional[str] = ""
    exposure: Optional[str] = "confidential_data"
    use_case: Optional[str] = ""
    workflow_text: Optional[str] = ""
    doc_urls: Optional[list[str]] = []
    focus: Optional[list[str]] = ["accidental_copy", "misclassification"]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/artifacts")
def list_artifacts(
    artifact_type: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    from sqlalchemy.orm import joinedload
    q = db.query(Artifact).options(joinedload(Artifact.sentiments))

    if artifact_type:
        q = q.filter(Artifact.artifact_type == artifact_type)
    if tag:
        q = q.filter(Artifact.tags.contains(tag))
    if search:
        q = q.filter(
            Artifact.title.ilike(f"%{search}%")
            | Artifact.description.ilike(f"%{search}%")
            | Artifact.author.ilike(f"%{search}%")
        )

    total = q.count()
    items = q.order_by(Artifact.created_at.desc()).offset(offset).limit(limit).all()

    result = []
    for a in items:
        d = _artifact_json(a)
        sents = a.sentiments
        if sents:
            d["sentiment"] = round(sum(s.score for s in sents) / len(sents), 2)
        else:
            d["sentiment"] = None
        result.append(d)

    return {"total": total, "items": result}


@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: int, db: Session = Depends(get_db)):
    a = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if not a:
        raise HTTPException(404, "Artifact not found")

    sentiments = db.query(Sentiment).filter(Sentiment.artifact_id == a.id).all()
    projections = db.query(Projection).filter(Projection.artifact_id == a.id).all()

    out_rels = (
        db.query(Relationship, Artifact)
        .join(Artifact, Relationship.target_id == Artifact.id)
        .filter(Relationship.source_id == a.id)
        .all()
    )
    in_rels = (
        db.query(Relationship, Artifact)
        .join(Artifact, Relationship.source_id == Artifact.id)
        .filter(Relationship.target_id == a.id)
        .all()
    )

    result = _artifact_json(a)
    result["sentiments"] = [
        {"score": s.score, "label": s.label, "source": s.source} for s in sentiments
    ]
    result["projections"] = [
        {"type": p.projection_type, "confidence": p.confidence,
         "timeframe": p.timeframe, "summary": p.summary} for p in projections
    ]
    result["relationships"] = [
        {"id": r.Relationship.id, "direction": "outgoing",
         "target_id": r.Relationship.target_id, "target_title": r.Artifact.title,
         "type": r.Relationship.relationship_type, "description": r.Relationship.description}
        for r in out_rels
    ] + [
        {"id": r.Relationship.id, "direction": "incoming",
         "source_id": r.Relationship.source_id, "source_title": r.Artifact.title,
         "type": r.Relationship.relationship_type, "description": r.Relationship.description}
        for r in in_rels
    ]
    return result


@app.post("/api/artifacts", dependencies=[Depends(require_admin)])
def create_artifact(data: ArtifactCreate, db: Session = Depends(get_db)):
    artifact = Artifact(
        title=data.title,
        artifact_type=data.artifact_type,
        url=data.url,
        description=data.description,
        content=data.content,
        source=data.source,
        author=data.author,
        date_published=data.date_published or datetime.now(timezone.utc),
        tags=data.tags,
    )
    db.add(artifact)
    db.flush()

    if data.sentiment_score is not None:
        db.add(Sentiment(
            artifact_id=artifact.id,
            score=data.sentiment_score,
            label=data.sentiment_label or "neutral",
            source="manual",
        ))

    if data.projection_type:
        db.add(Projection(
            artifact_id=artifact.id,
            projection_type=data.projection_type,
            confidence=data.projection_confidence or 0.5,
            timeframe=data.projection_timeframe,
            summary=data.projection_summary,
        ))

    db.commit()
    db.refresh(artifact)
    return _artifact_json(artifact)


@app.delete("/api/artifacts/{artifact_id}", dependencies=[Depends(require_admin)])
def delete_artifact(artifact_id: int, db: Session = Depends(get_db)):
    a = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if not a:
        raise HTTPException(404, "Artifact not found")
    db.delete(a)
    db.commit()
    return {"ok": True}


@app.get("/api/graph")
def get_graph(
    db: Session = Depends(get_db),
    limit: int = Query(1500, ge=1, le=5000),
    type: Optional[str] = None,
    q: Optional[str] = None,
):
    """Graph payload, bounded and filterable so the network view stays fast.

    - `limit` caps node count (default 1500) — prevents unbounded payloads.
    - `type` filters by artifact_type (comma-separated list allowed).
    - `q` is a cheap substring filter on title/tags, applied before the cap.
    Edges are pruned to only reference returned nodes.
    """
    node_query = db.query(Artifact)
    if type:
        types = [t.strip() for t in type.split(",") if t.strip()]
        if types:
            node_query = node_query.filter(Artifact.artifact_type.in_(types))
    if q:
        like = f"%{q.strip()}%"
        node_query = node_query.filter(
            (Artifact.title.ilike(like)) | (Artifact.tags.ilike(like))
        )
    # Newest first so the cap keeps the most relevant/recent portion of the graph.
    artifacts = node_query.order_by(Artifact.date_published.desc().nullslast()).limit(limit).all()
    node_ids = {a.id for a in artifacts}
    relationships = db.query(Relationship).filter(
        Relationship.source_id.in_(node_ids), Relationship.target_id.in_(node_ids)
    ).all()

    # Bulk aggregates in 2 queries (was N+1: 2 queries per artifact).
    sent_rows = db.query(Sentiment.artifact_id, Sentiment.score).all()
    sent_sum: dict = defaultdict(float)
    sent_cnt: dict = defaultdict(int)
    for aid, score in sent_rows:
        sent_sum[aid] += score
        sent_cnt[aid] += 1

    proj_rows = db.query(Projection.artifact_id, Projection.confidence).all()
    conf_max: dict = defaultdict(float)
    for aid, conf in proj_rows:
        if conf > conf_max[aid]:
            conf_max[aid] = conf

    nodes = []
    for a in artifacts:
        avg_sentiment = sent_sum[a.id] / sent_cnt[a.id] if sent_cnt[a.id] else 0
        max_confidence = conf_max[a.id]

        nodes.append({
            "id": a.id,
            "label": a.title[:60],
            "title": a.title,
            "type": a.artifact_type,
            "sentiment": round(avg_sentiment, 2),
            "confidence": round(max_confidence, 2),
            "author": a.author or "",
            "date": a.date_published.isoformat() if a.date_published else "",
            "tags": a.tags or "",
        })

    edges = []
    for r in relationships:
        if r.source_id not in node_ids or r.target_id not in node_ids:
            continue
        edges.append({
            "id": r.id,
            "from": r.source_id,
            "to": r.target_id,
            "label": r.relationship_type,
            "title": r.description or "",
        })

    return {"nodes": nodes, "edges": edges}


def _parse_tag_set(tags: Optional[str]) -> set:
    if not tags:
        return set()
    return set(t.strip().lower() for t in tags.split(",") if t.strip())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@app.get("/api/graph/clusters")
def get_graph_clusters(
    mode: str = Query("category", pattern="^(category|similarity)$"),
    threshold: float = Query(0.25, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Return cluster centroids for the graph view.

    - mode=category: one centroid per artifact_type.
    - mode=similarity: tag-Jaccard union-find components (size >= 2).
      Each cluster carries a centroid label derived from shared/common tags.
    """
    artifacts = db.query(Artifact).all()

    if mode == "category":
        grouped: dict = defaultdict(list)
        for a in artifacts:
            grouped[a.artifact_type or "unknown"].append(a.id)
        clusters = []
        for atype, member_ids in sorted(grouped.items()):
            # centroid: most frequent tags within this type
            tag_counts: dict = defaultdict(int)
            for a in artifacts:
                if a.id in member_ids:
                    for t in _parse_tag_set(a.tags):
                        tag_counts[t] += 1
            top_tags = sorted(tag_counts, key=lambda t: -tag_counts[t])[:3]
            clusters.append({
                "id": f"category:{atype}",
                "label": f"{atype} ({len(member_ids)})",
                "mode": "category",
                "centroid": atype,
                "centroid_tags": top_tags,
                "member_ids": sorted(member_ids),
                "size": len(member_ids),
            })
        return {"mode": mode, "clusters": clusters}

    # mode == "similarity": union-find over tag Jaccard similarity
    ids = [a.id for a in artifacts]
    tag_map = {a.id: _parse_tag_set(a.tags) for a in artifacts}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if _jaccard(tag_map[ids[i]], tag_map[ids[j]]) >= threshold:
                union(ids[i], ids[j])

    groups: dict = defaultdict(list)
    for i in ids:
        # only group artifacts that actually have tags; untagged stay solo
        if not tag_map[i]:
            continue
        groups[find(i)].append(i)

    clusters = []
    for idx, (root, member_ids) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        if len(member_ids) < 2:
            continue
        # centroid tags: intersection if non-empty else most common
        sets = [tag_map[m] for m in member_ids]
        inter = set.intersection(*sets) if sets else set()
        if inter:
            centroid_tags = sorted(inter)
        else:
            counts: dict = defaultdict(int)
            for s in sets:
                for t in s:
                    counts[t] += 1
            centroid_tags = sorted(counts, key=lambda t: (-counts[t], t))[:3]
        centroid_label = ", ".join(centroid_tags[:3]) if centroid_tags else f"group {idx + 1}"
        clusters.append({
            "id": f"similarity:{idx}",
            "label": f"{centroid_label} ({len(member_ids)})",
            "mode": "similarity",
            "centroid": centroid_label,
            "centroid_tags": centroid_tags,
            "member_ids": sorted(member_ids),
            "size": len(member_ids),
        })
    return {"mode": mode, "threshold": threshold, "clusters": clusters}


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(Artifact).count()
    by_type = (
        db.query(Artifact.artifact_type, func.count(Artifact.id))
        .group_by(Artifact.artifact_type)
        .all()
    )

    sentiments = db.query(Sentiment.score).all()
    avg_sentiment = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0

    projections = db.query(Projection).all()
    proj_by_type = {}
    for p in projections:
        proj_by_type[p.projection_type] = proj_by_type.get(p.projection_type, 0) + 1
    avg_confidence = (
        sum(p.confidence for p in projections) / len(projections) if projections else 0
    )

    return {
        "total_artifacts": total,
        "by_type": dict(by_type),
        "avg_sentiment": round(avg_sentiment, 3),
        "projections": {
            "by_type": proj_by_type,
            "avg_confidence": round(avg_confidence, 3),
            "total": len(projections),
        },
    }


TIMEFRAME_RE = re.compile(r"(\d{4})(?:\s*[-–]\s*(\d{4}))?")


def parse_timeframe(tf: Optional[str]) -> Optional[tuple]:
    if not tf:
        return None
    m = TIMEFRAME_RE.search(tf)
    if m:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        return (start, end)
    return None


def _compute_plausibility(projections_list, sentiments_dict, artifacts_list,
                          year_from=None, year_to=None,
                          source_year_from=None, source_year_to=None):
    artifacts_map = {a.id: a for a in artifacts_list}
    type_data = {}
    for p in projections_list:
        # Filter by projection timeframe
        tf_range = parse_timeframe(p.timeframe)
        if year_from is not None or year_to is not None:
            if tf_range is None:
                continue
        if year_from is not None and tf_range[1] < year_from:
            continue
        if year_to is not None and tf_range[0] > year_to:
            continue

        # Filter by source artifact publication year
        if source_year_from is not None or source_year_to is not None:
            art = artifacts_map.get(p.artifact_id)
            if art and art.date_published:
                pub_year = art.date_published.year
                if source_year_from is not None and pub_year < source_year_from:
                    continue
                if source_year_to is not None and pub_year > source_year_to:
                    continue
            else:
                continue  # skip if no date when filtering by source year
        t = p.projection_type
        if t not in type_data:
            type_data[t] = {"confs": [], "sents": [], "artifacts": set(), "summaries": []}
        type_data[t]["confs"].append(p.confidence)
        type_data[t]["artifacts"].add(p.artifact_id)
        if p.summary:
            type_data[t]["summaries"].append(p.summary)

    for aid, s in sentiments_dict.items():
        for t, d in type_data.items():
            if aid in d["artifacts"]:
                d["sents"].append(s.score)

    results = {}
    for ptype, d in type_data.items():
        avg_conf = sum(d["confs"]) / len(d["confs"]) if d["confs"] else 0
        avg_sent = sum(d["sents"]) / len(d["sents"]) if d["sents"] else 0
        plausibility = avg_conf * 0.6 + max(0, avg_sent) * 0.4
        artifact_titles = [a.title for a in artifacts_list if a.id in d["artifacts"]]
        results[ptype] = {
            "count": len(d["confs"]),
            "avg_confidence": round(avg_conf, 3),
            "avg_sentiment": round(avg_sent, 3),
            "plausibility": round(min(1, max(0, plausibility)), 3),
            "artifacts": artifact_titles[:15],
            "summaries": d["summaries"][:6],
        }

    return results


@app.get("/api/plausibility")
def get_plausibility(
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
    source_year_from: Optional[int] = Query(None),
    source_year_to: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    artifacts = db.query(Artifact).all()
    all_sentiments = {s.artifact_id: s for s in db.query(Sentiment).all()}
    all_projections = db.query(Projection).all()

    results = _compute_plausibility(
        all_projections, all_sentiments, artifacts,
        year_from, year_to, source_year_from, source_year_to,
    )

    filtered_projections = [
        p for p in all_projections
        if _projection_in_filter(p, artifacts, year_from, year_to, source_year_from, source_year_to)
    ]

    overall_conf = sum(p.confidence for p in filtered_projections) / len(filtered_projections) if filtered_projections else 0
    overall_sent = sum(s.score for s in all_sentiments.values()) / len(all_sentiments) if all_sentiments else 0
    overall_plaus = overall_conf * 0.6 + max(0, overall_sent) * 0.4

    return {
        "types": results,
        "overall": {
            "total_projections": len(filtered_projections),
            "avg_confidence": round(overall_conf, 3),
            "avg_sentiment": round(overall_sent, 3),
            "plausibility": round(min(1, max(0, overall_plaus)), 3),
        },
        "year_range": {"projection": {"from": year_from, "to": year_to},
                       "source": {"from": source_year_from, "to": source_year_to}},
    }


def _projection_in_filter(p, artifacts, year_from, year_to, source_year_from, source_year_to):
    artifacts_map = {a.id: a for a in artifacts}
    if year_from is not None or year_to is not None:
        tf_range = parse_timeframe(p.timeframe)
        if tf_range is None:
            return False
        if year_from is not None and tf_range[1] < year_from:
            return False
        if year_to is not None and tf_range[0] > year_to:
            return False
    if source_year_from is not None or source_year_to is not None:
        art = artifacts_map.get(p.artifact_id)
        if art and art.date_published:
            py = art.date_published.year
            if source_year_from is not None and py < source_year_from:
                return False
            if source_year_to is not None and py > source_year_to:
                return False
        else:
            return False
    return True


def _projection_in_year_range(p, year_from, year_to):
    tf_range = parse_timeframe(p.timeframe)
    if tf_range is None:
        return False
    if year_from is not None and tf_range[1] < year_from:
        return False
    if year_to is not None and tf_range[0] > year_to:
        return False
    return True


@app.get("/api/plausibility/agentic")
def get_plausibility_agentic(
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
    source_year_from: Optional[int] = Query(None),
    source_year_to: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Agentic prediction view: scout → forecaster → critic deliberation.

    Same filters and response shape as classical `/api/plausibility`
    (per-type plausibility/confidence/sentiment/count plus overall), with
    agent extras per type: rationale, disagreement IQR, reliability grade,
    caveats — plus the A2A trace and method note.
    """
    from .agents import run_prediction_workflow

    artifacts = db.query(Artifact).all()
    art_map = {a.id: a for a in artifacts}
    wf = run_prediction_workflow(db, year_from, year_to, source_year_from, source_year_to)

    types: dict = {}
    for ptype, f in wf["forecasts"].items():
        pool = wf["pools"].get(ptype, [])
        titles = []
        summaries = []
        for item in sorted(pool, key=lambda i: -i["weight"])[:15]:
            a = art_map.get(item["artifact_id"])
            if a and a.title not in titles:
                titles.append(a.title)
            if item.get("summary") and len(summaries) < 6:
                summaries.append(item["summary"])
        rev = wf["reviews"].get(ptype, {})
        types[ptype] = {
            "count": f["n"],
            "avg_confidence": f["wmedian_confidence"],
            "avg_sentiment": f["wmedian_sentiment"],
            "plausibility": f["score"],
            "artifacts": titles,
            "summaries": summaries,
            "rationale": f["rationale"],
            "disagreement_iqr": f["disagreement_iqr"],
            "reliability": rev.get("reliability", "low"),
            "caveats": rev.get("caveats", []),
            "auto_share": f["auto_share"],
            "stale_share": f["stale_share"],
        }
    scores = [v["plausibility"] for v in types.values()]
    return {
        "mode": "agentic",
        "types": types,
        "overall": {
            "total_projections": sum(v["count"] for v in types.values()),
            "avg_confidence": round(sum(v["avg_confidence"] for v in types.values()) / len(types), 3) if types else 0,
            "avg_sentiment": round(sum(v["avg_sentiment"] for v in types.values()) / len(types), 3) if types else 0,
            "plausibility": round(sum(scores) / len(scores), 3) if scores else 0,
        },
        "method": ("evidence-scout pools dated projections with recency/source weights; "
                   "forecaster takes weighted medians with disagreement penalties, staleness "
                   "discounts and whistleblower boosts; critic grades reliability and caveats."),
        "skipped_by_filters": wf["skipped_by_filters"],
        "a2a_task_id": wf["task_id"],
        "a2a_trace": wf["a2a_trace"],
        "year_range": {"projection": {"from": year_from, "to": year_to},
                       "source": {"from": source_year_from, "to": source_year_to}},
    }


@app.get("/api/plausibility/trend")
def plausibility_trend(
    year_start: int = Query(2020),
    year_end: int = Query(2100),
    step: int = Query(5),
    mode: str = Query("projection_year"),
    db: Session = Depends(get_db),
):
    artifacts = db.query(Artifact).all()
    artifacts_map = {a.id: a for a in artifacts}
    all_sentiments = {s.artifact_id: s for s in db.query(Sentiment).all()}
    all_projections = db.query(Projection).all()
    all_types = sorted(set(p.projection_type for p in all_projections))

    years = list(range(year_start, year_end + 1, step))
    if years[-1] != year_end:
        years.append(year_end)

    type_series = {t: [] for t in all_types}
    for yr in years:
        if mode == "source_year":
            sel = [
                p for p in all_projections
                if _projection_published_in_year(p, artifacts_map, yr)
            ]
        else:
            sel = [p for p in all_projections if _projection_covers_year(p, yr)]
        yr_results = _compute_plausibility(sel, all_sentiments, artifacts)
        for t in all_types:
            val = yr_results.get(t, {}).get("plausibility", 0)
            type_series[t].append(val)

    return {
        "years": years,
        "series": type_series,
        "mode": mode,
    }


def _projection_covers_year(p, year):
    tf_range = parse_timeframe(p.timeframe)
    if tf_range is None:
        return False
    return tf_range[0] <= year <= tf_range[1]


def _projection_published_in_year(p, artifacts_map, year):
    art = artifacts_map.get(p.artifact_id)
    if art and art.date_published:
        return art.date_published.year == year
    return False


# ------------------------------------------------------------ timeline ---
# "Where are we toward AGI, and what bridges the gap" — computed live from
# the knowledge graph on every call, so the tab stays fresh as ingestion
# (daily 04:00 UTC + manual Fetch) adds artifacts, projections, and warnings.

# Gap dimensions: what it takes to bridge to safe AGI. Each dimension is a
# live tag query over artifacts plus a statement of what "bridged" requires.
TIMELINE_GAPS = [
    {"id": "alignment", "title": "Alignment & control",
     "tags": ["alignment", "superalignment", "control-problem", "deceptive-alignment",
              "goal-misgeneralization", "mesa-optimization", "rlaif", "scalable-oversight"],
     "bridged_when": "Formal guarantees (not empirical tuning) that superhuman systems pursue intended goals; deception detectable before deployment.",
     "why_open": "Sleeper-agent and mesa-optimization results show deception persisting through RLHF; whistleblower testimony reports alignment deprioritized under product pressure."},
    {"id": "evaluations", "title": "Evaluations & monitoring",
     "tags": ["benchmark", "evaluations", "measurement", "interpretability", "wmdp",
              "elk", "redteam", "audit"],
     "bridged_when": "Pre-deployment safety cases with third-party audits, published evals, and post-deployment monitoring as deployment preconditions.",
     "why_open": "Benchmarks underestimate capability growth; dangerous dual-use knowledge (WMDP) already measurable in frontier models without gating releases."},
    {"id": "governance", "title": "Governance & policy",
     "tags": ["governance", "regulation", "policy", "eu-act", "pause", "preparedness",
              "agi-readiness", "coordination"],
     "bridged_when": "Binding international oversight of frontier training and deployment, plus institutional AGI readiness (incident response, technical state capacity).",
     "why_open": "EU AI Act sets precedent but targets narrow risk tiers; readiness memos warn neither labs nor governments can currently absorb AGI-speed change."},
    {"id": "scaling", "title": "Capabilities & scaling trajectory",
     "tags": ["scaling", "scaling-laws", "compute", "frontier", "capabilities",
              "reasoning", "multimodal", "data-scarcity"],
     "bridged_when": "A shared, evidence-based picture of how fast capabilities arrive — the timeline both camps plan against.",
     "why_open": "Short-timeline (2027–2030) and long-timeline (2040–2050+) camps cite the same scaling curves; data scarcity vs. synthetic data is the open variable."},
    {"id": "safety-culture", "title": "Safety culture & right to warn",
     "tags": ["whistleblower", "safety-culture", "right-to-warn", "ethics",
              "moral-policy", "retaliation"],
     "bridged_when": "Durable insider-warning channels: no silencing agreements, anonymous reporting to boards/regulators, legal protection for good-faith disclosure.",
     "why_open": "From Gebru (2020) to superalignment resignations (2024), internal ethics channels repeatedly failed without external power — the Right to Warn demands are still unmet."},
]

CAPABILITY_SIGNAL_TAGS = ["capabilities", "frontier", "scaling", "benchmarks",
                          "reasoning", "multimodal", "alphafold", "breakthrough"]


def _clean_text(text: str | None, limit: int = 0) -> str:
    """Strip HTML leaked in by RSS/auto-ingested summaries for clean display."""
    import re as _re
    import html as _html
    s = _re.sub(r"<[^>]+>", " ", _html.unescape(text or ""))
    s = _re.sub(r"<[^>]*$", "", s)  # trailing unterminated tag from truncation
    s = _re.sub(r"\s+", " ", s).strip()
    return s[:limit] if limit and len(s) > limit else s


# Year bounds for forecast statistics: excludes parser noise (e.g. "1000"
# matched inside "1000 words") while keeping genuine retrospective forecasts.
FORECAST_YEAR_MIN, FORECAST_YEAR_MAX = 2000, 2100


def _artifact_matches_tags(artifact_tags: str | None, wanted: list[str]) -> bool:
    have = set(t.strip().lower() for t in (artifact_tags or "").split(",") if t.strip())
    return bool(have & set(wanted))


@app.get("/api/timeline")
def get_timeline(db: Session = Depends(get_db)):
    """Where we are toward AGI + what bridges the gap. Fully derived from the
    live knowledge graph: median forecast year, horizon bands, per-gap evidence
    counts, latest capability signals and whistleblower warnings, and ingestion
    freshness — so the Timeline tab updates itself as the graph grows."""
    now_year = datetime.now(timezone.utc).year
    artifacts = db.query(Artifact).all()
    projections = db.query(Projection).all()
    sentiments = {s.artifact_id: s.score for s in db.query(Sentiment).all()}
    art_map = {a.id: a for a in artifacts}

    def _mini(a: Artifact) -> dict:
        return {"id": a.id, "title": a.title, "artifact_type": a.artifact_type,
                "author": a.author,
                "date": a.date_published.isoformat() if a.date_published else None,
                "tags": a.tags or "", "url": a.url or ""}

    # Forecast distribution from parseable projection timeframes.
    forecasts = []
    for p in projections:
        tf = parse_timeframe(p.timeframe)
        if not tf:
            continue
        if tf[0] < FORECAST_YEAR_MIN or tf[0] > FORECAST_YEAR_MAX:
            continue  # parser noise, not a real forecast horizon
        mid = (tf[0] + tf[1]) / 2
        a = art_map.get(p.artifact_id)
        forecasts.append({"year_mid": mid, "start": tf[0], "end": tf[1],
                          "type": p.projection_type, "confidence": p.confidence,
                          "summary": _clean_text(p.summary, 240),
                          "artifact": _mini(a) if a else None,
                          "sentiment": sentiments.get(p.artifact_id)})
    forecasts.sort(key=lambda f: f["year_mid"])
    mids = sorted(f["year_mid"] for f in forecasts)

    def _pct(vals: list[float], p: float):
        if not vals:
            return None
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(p / 100 * len(vals)))]

    type_counts: dict[str, int] = {}
    for f in forecasts:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1

    # Horizon bands: achieved, near (0–3y), mid (3–7y), far (7y+), undated.
    bands = {"achieved": [], "near": [], "mid": [], "far": [], "undated": []}
    for p in projections:
        a = art_map.get(p.artifact_id)
        entry = {"type": p.projection_type, "confidence": p.confidence,
                 "timeframe": p.timeframe, "summary": _clean_text(p.summary, 240),
                 "artifact": _mini(a) if a else None}
        tf = parse_timeframe(p.timeframe)
        if not tf or tf[0] < FORECAST_YEAR_MIN or tf[0] > FORECAST_YEAR_MAX:
            bands["undated"].append(entry)
        elif tf[1] <= now_year:
            bands["achieved"].append(entry)
        elif tf[0] <= now_year + 3:
            bands["near"].append(entry)
        elif tf[0] <= now_year + 7:
            bands["mid"].append(entry)
        else:
            bands["far"].append(entry)
    for key in bands:
        bands[key].sort(key=lambda e: -(e["confidence"] or 0))
        bands[key] = bands[key][:10]

    # Latest capability signals (last 24 months preferred, fallback to latest).
    cutoff = now_year - 2
    sigs = [a for a in artifacts
            if _artifact_matches_tags(a.tags, CAPABILITY_SIGNAL_TAGS)]
    sigs.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
              reverse=True)
    recent_sigs = [a for a in sigs
                   if a.date_published and a.date_published.year >= cutoff][:8]
    if len(recent_sigs) < 4:
        recent_sigs = sigs[:8]

    # Whistleblower counterweight: latest warnings.
    warns = [a for a in artifacts if _artifact_matches_tags(a.tags, ["whistleblower"])]
    warns.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
               reverse=True)

    # Gap dimensions with live evidence.
    gaps = []
    for g in TIMELINE_GAPS:
        matched = [a for a in artifacts if _artifact_matches_tags(a.tags, g["tags"])]
        matched.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
                     reverse=True)
        recent = [a for a in matched
                  if a.date_published and a.date_published.year >= now_year - 2]
        neg = [sentiments.get(a.id, 0) for a in matched if a.id in sentiments]
        avg_sent = round(sum(neg) / len(neg), 2) if neg else 0.0
        gaps.append({"id": g["id"], "title": g["title"],
                     "bridged_when": g["bridged_when"], "why_open": g["why_open"],
                     "status": "open",
                     "evidence_count": len(matched),
                     "recent_count": len(recent),
                     "avg_sentiment": avg_sent,
                     "key_items": [_mini(a) for a in matched[:4]]})

    # --- Progress-per-gap: how much of each bridge has been built recently.
    # Heuristic live metric: share of a gap's tagged artifacts published in the
    # last 24 months (activity momentum) blended with evidence-volume coverage
    # against a soft target. Recomputed on every call as ingestion grows.
    def _gap_progress(matched: list[Artifact]) -> int:
        if not matched:
            return 0
        recent = [a for a in matched
                  if a.date_published and a.date_published.year >= now_year - 2]
        momentum = len(recent) / len(matched)          # 0..1 freshness share
        coverage = min(1.0, len(matched) / 25.0)       # soft target: 25 items
        return max(3, min(95, round(100 * (0.55 * coverage + 0.45 * momentum))))

    # --- Chronological master timeline: merge projections + capability signals
    # + warnings into one dated stream so users can scrub history → today → future.
    def _dt_key(dt):
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    chrono: list[dict] = []
    for p in projections:
        a = art_map.get(p.artifact_id)
        tf = parse_timeframe(p.timeframe)
        if not tf or tf[0] < FORECAST_YEAR_MIN or tf[0] > FORECAST_YEAR_MAX:
            continue
        chrono.append({
            "kind": "projection", "year_mid": (tf[0] + tf[1]) / 2,
            "start": tf[0], "end": tf[1], "timeframe": p.timeframe,
            "label": p.projection_type, "confidence": p.confidence,
            "summary": _clean_text(p.summary, 240),
            "source_date": _dt_key(a.date_published).isoformat() if a else None,
            "artifact": _mini(a) if a else None,
            "sentiment": sentiments.get(p.artifact_id),
        })
    for a in artifacts:
        if _artifact_matches_tags(a.tags, ["whistleblower"]):
            kind = "warning"
        elif _artifact_matches_tags(a.tags, CAPABILITY_SIGNAL_TAGS):
            kind = "signal"
        else:
            continue
        chrono.append({
            "kind": kind, "year_mid": _dt_key(a.date_published).year or None,
            "start": None, "end": None, "timeframe": None,
            "label": a.artifact_type, "confidence": None,
            "summary": _clean_text(a.description or "", 240),
            "source_date": _dt_key(a.date_published).isoformat() if a.date_published else None,
            "artifact": _mini(a),
            "sentiment": sentiments.get(a.id),
        })
    chrono.sort(key=lambda e: (e["year_mid"] is None, e["year_mid"] or 0))
    for e in chrono:
        e["past"] = bool(e["year_mid"] and e["year_mid"] <= now_year)
    chrono = chrono[:400]

    # Enrich each gap with a live progress estimate (see _gap_progress above).
    for g_out, g_def in zip(gaps, TIMELINE_GAPS):
        matched_full = [art for art in artifacts
                        if _artifact_matches_tags(art.tags, g_def["tags"])]
        g_out["progress"] = _gap_progress(matched_full)

    return {
        "now_year": now_year,
        "where_we_are": {
            "forecast_count": len(forecasts),
            "median_agi_year": _pct(mids, 50),
            "spread_p25": _pct(mids, 25),
            "spread_p75": _pct(mids, 75),
            "min_year": mids[0] if mids else None,
            "max_year": mids[-1] if mids else None,
            "type_distribution": type_counts,
            "by_year": _timeline_histogram(mids),
        },
        "chrono": chrono,
        "horizon_bands": bands,
        "gaps": gaps,
        "recent_capability_signals": [_mini(a) for a in recent_sigs],
        "latest_warnings": [_mini(a) for a in warns[:5]],
        "warning_count": len(warns),
        "freshness": {
            "total_artifacts": len(artifacts),
            "total_projections": len(projections),
            "last_ingest": ingestion_stats.get("last_run"),
            "ingest_runs": ingestion_stats.get("total_runs", 0),
        },
    }


# ------------------------------------------------- professions ---
# Profession-capture tracker: what AI has already taken over vs. what it is
# projected to consume. Fully evidence-driven — every status and confidence
# score is recomputed live from the current graph on each call, so Fetch /
# ingestion runs move the numbers with no manual edits.

PROFESSIONS = [
    {"id": "software", "name": "Software engineering", "icon": "fa-code",
     "tags": ["code", "software", "programming", "copilot", "coding", "swe-bench", "agents"],
     "blurb": "Code generation, debugging, and agentic software tasks."},
    {"id": "support", "name": "Customer support", "icon": "fa-headset",
     "tags": ["support", "chatbot", "call-center", "helpdesk", "service"],
     "blurb": "Tier-1 support chats, call handling, and ticket triage."},
    {"id": "translation", "name": "Translation", "icon": "fa-language",
     "tags": ["translation", "multilingual", "language"],
     "blurb": "Routine professional translation across major languages."},
    {"id": "writing", "name": "Copywriting & content", "icon": "fa-pen-nib",
     "tags": ["copywriting", "content", "marketing", "journalism", "writing"],
     "blurb": "Marketing copy, SEO content, drafts, and routine reporting."},
    {"id": "data-entry", "name": "Data entry & processing", "icon": "fa-keyboard",
     "tags": ["data-entry", "ocr", "document", "extraction", "rpa", "automation"],
     "blurb": "Form processing, OCR pipelines, and back-office data work."},
    {"id": "design", "name": "Graphic design & media", "icon": "fa-palette",
     "tags": ["design", "image-generation", "diffusion", "video", "multimodal", "art"],
     "blurb": "Stock imagery, concept art, layouts, and short-form video."},
    {"id": "legal", "name": "Paralegal & legal research", "icon": "fa-scale-balanced",
     "tags": ["legal", "paralegal", "contracts", "discovery", "compliance"],
     "blurb": "Document review, discovery, contract drafting assistance."},
    {"id": "finance", "name": "Accounting & analysis", "icon": "fa-calculator",
     "tags": ["accounting", "bookkeeping", "auditing", "finance", "analyst"],
     "blurb": "Bookkeeping, reconciliation, and first-pass financial analysis."},
    {"id": "diagnostics", "name": "Medical diagnostics", "icon": "fa-stethoscope",
     "tags": ["radiology", "diagnostics", "alphafold", "drug-discovery", "pathology", "medical"],
     "blurb": "Image-based diagnosis support, protein folding, drug pipelines."},
    {"id": "tutoring", "name": "Tutoring & education", "icon": "fa-graduation-cap",
     "tags": ["tutoring", "education", "teaching", "edtech"],
     "blurb": "Personalized tutoring, grading assistance, curriculum drafts."},
    {"id": "driving", "name": "Driving & delivery", "icon": "fa-truck-fast",
     "tags": ["autonomous", "self-driving", "delivery", "logistics", "transport"],
     "blurb": "Highway autonomy, warehouse robotics, last-mile pilots."},
    {"id": "manufacturing", "name": "Manufacturing & inspection", "icon": "fa-industry",
     "tags": ["manufacturing", "robotics", "inspection", "quality", "factory"],
     "blurb": "Visual inspection, pick-and-place, predictive maintenance."},
    {"id": "science", "name": "Scientific research", "icon": "fa-flask",
     "tags": ["science", "research", "discovery", "breakthrough", "alphafold", "materials"],
     "blurb": "Literature synthesis, hypothesis generation, lab automation."},
    {"id": "music", "name": "Music & creative writing", "icon": "fa-music",
     "tags": ["music", "fiction", "screenwriting", "creative", "poetry"],
     "blurb": "Functional music, first drafts, and assisted composition."},
]

FUTURE_WORK_KEYWORDS = ["job", "employment", "unemploy", "automat", "labor", "labour",
                        "workforce", "post-work", "economic-activity", "displace",
                        "replace human", "replace majority", "occupation"]


@app.get("/api/timeline/professions")
def get_profession_capture(db: Session = Depends(get_db)):
    """What AI has captured so far vs. projected future consumption.

    Statuses and confidence scores are recomputed live from the graph:
    evidence counts, recency, source-type diversity, and linked projection
    confidence — so every Fetch/ingestion run revisits them.
    """
    now_year = datetime.now(timezone.utc).year
    artifacts = db.query(Artifact).all()
    projections = db.query(Projection).all()
    art_map = {a.id: a for a in artifacts}

    def _mini(a: Artifact) -> dict:
        return {"id": a.id, "title": a.title, "artifact_type": a.artifact_type,
                "date": a.date_published.isoformat() if a.date_published else None}

    def _age_years(a: Artifact):
        if not a.date_published:
            return None
        try:
            base = a.date_published
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - base).total_seconds() / 31536000
        except Exception:
            return None

    captured = []
    for prof in PROFESSIONS:
        matched = [a for a in artifacts if _artifact_matches_tags(a.tags, prof["tags"])]
        matched.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
                     reverse=True)
        n = len(matched)
        recent = [a for a in matched if (_age_years(a) or 99) <= 2]
        types = {a.artifact_type for a in matched}
        linked = [pr for pr in projections
                  if art_map.get(pr.artifact_id) in matched]
        avg_conf = (sum(pr.confidence for pr in linked) / len(linked)) if linked else 0.0
        if n >= 4 and recent:
            status = "captured"
        elif n >= 2:
            status = "emerging"
        elif n == 1:
            status = "exposed"
        else:
            status = "monitored"
        score = min(95, 15 + 9 * min(n, 6) + (8 if recent else 0)
                    + (7 if len(types) >= 3 else 0) + round(10 * avg_conf))
        captured.append({
            "id": prof["id"], "name": prof["name"], "icon": prof["icon"],
            "blurb": prof["blurb"], "status": status, "confidence": score,
            "evidence_count": n, "recent_count": len(recent),
            "source_types": sorted(types),
            "linked_projections": len(linked),
            "components": {"base": 15, "evidence": 9 * min(n, 6),
                           "recency": 8 if recent else 0,
                           "diversity": 7 if len(types) >= 3 else 0,
                           "projection": round(10 * avg_conf)},
            "key_items": [_mini(a) for a in matched[:3]],
        })
    captured.sort(key=lambda c: (-c["confidence"], -c["evidence_count"]))

    # Future consumption: workforce/automation projections with horizons.
    future = []
    for pr in projections:
        a = art_map.get(pr.artifact_id)
        hay = f"{pr.summary or ''} {pr.timeframe or ''} {(a.title if a else '')} {(a.tags if a else '')}".lower()
        if not any(k in hay for k in FUTURE_WORK_KEYWORDS):
            continue
        tf = parse_timeframe(pr.timeframe)
        profs = [pf["name"] for pf in PROFESSIONS
                 if _artifact_matches_tags(a.tags if a else "", pf["tags"])]
        future.append({"type": pr.projection_type, "confidence": pr.confidence,
                       "timeframe": pr.timeframe, "start": tf[0] if tf else None,
                       "summary": _clean_text(pr.summary, 220),
                       "professions": profs[:3], "artifact": _mini(a) if a else None})
    future.sort(key=lambda e: ((e["start"] or 9999), -(e["confidence"] or 0)))

    return {
        "now_year": now_year,
        "captured": captured,
        "captured_count": sum(1 for c in captured if c["status"] == "captured"),
        "future": future[:15],
        "future_count": len(future),
        "freshness": {"total_artifacts": len(artifacts),
                      "total_projections": len(projections),
                      "last_ingest": ingestion_stats.get("last_run")},
        "method": ("Status from live evidence (4+ items incl. recent = captured, "
                   "2+ = emerging, 1 = exposed, else monitored). Confidence = 15 base + "
                   "9/evidence item (max 6) + 8 recency + 7 source diversity + "
                   "10x linked projection confidence, capped at 95. Recomputed on every call."),
    }


HUMAN_EDGE = [
    {"id": "moral", "name": "Moral judgment & accountability", "icon": "fa-scale-balanced",
     "tags": ["ethics", "moral-policy", "accountability", "liability"],
     "threat_tags": ["autonomous", "agents", "decision"],
     "irreducible": "Someone must answer for outcomes. Responsibility cannot be outsourced — only exercised.",
     "blurb": "Courts, boards, and commanders deciding right from wrong."},
    {"id": "oversight", "name": "AI safety oversight & auditing", "icon": "fa-shield-halved",
     "tags": ["superalignment", "evaluations", "audit", "redteam", "interpretability", "oversight"],
     "threat_tags": ["capabilities", "scaling", "automation"],
     "irreducible": "Only intelligence outside the system can check the system. Oversight collapses if the overseer is the model itself.",
     "blurb": "Independent evaluators, auditors, and interpretability researchers."},
    {"id": "science-taste", "name": "Scientific taste & paradigm shifts", "icon": "fa-lightbulb",
     "tags": ["science", "discovery", "breakthrough", "paradigm"],
     "threat_tags": ["automated", "scaling", "agents"],
     "irreducible": "Choosing which questions matter is a value judgment, not an optimization.",
     "blurb": "Framing problems, abandoning dead paradigms, betting on the unproven."},
    {"id": "care", "name": "Empathic care", "icon": "fa-hand-holding-heart",
     "tags": ["care", "therapy", "nursing", "empathy", "health", "childcare"],
     "threat_tags": ["chatbot", "companion", "automation"],
     "irreducible": "Being cared for by someone who could choose otherwise. Simulated empathy has no moral weight.",
     "blurb": "Therapy, nursing, childcare, eldercare, hospice."},
    {"id": "trades", "name": "Embodied skilled trades", "icon": "fa-screwdriver-wrench",
     "tags": ["trades", "dexterity", "plumbing", "electrical", "construction", "craft"],
     "threat_tags": ["robotics", "humanoid", "automation"],
     "irreducible": "Unstructured physical world + human homes. Dexterity plus judgment in places robots rarely go.",
     "blurb": "Electricians, plumbers, builders, repair generalists."},
    {"id": "law", "name": "Law & adjudication", "icon": "fa-gavel",
     "tags": ["legal", "law", "courts", "judges", "justice"],
     "threat_tags": ["contracts", "discovery", "automation"],
     "irreducible": "Legitimacy requires judgment by peers under law — a sentence computed is not justice delivered.",
     "blurb": "Judges, juries, and the legitimacy of the bench."},
    {"id": "meaning", "name": "Creative authorship & meaning", "icon": "fa-feather-pointed",
     "tags": ["creative", "authorship", "fiction", "art", "meaning"],
     "threat_tags": ["image-generation", "content", "automation"],
     "irreducible": "Art matters because a human meant it. Audiences pay for intention, not output.",
     "blurb": "Authors, artists, and makers whose intent is the product."},
    {"id": "diplomacy", "name": "Diplomacy & trust negotiation", "icon": "fa-handshake",
     "tags": ["diplomacy", "negotiation", "trust", "coordination", "treaty"],
     "threat_tags": ["agents", "automation", "surveillance"],
     "irreducible": "Peace is credible commitment between minds that can defect. Machines cannot promise.",
     "blurb": "Negotiators, mediators, and treaty builders."},
    {"id": "spirit", "name": "Spiritual & religious life", "icon": "fa-place-of-worship",
     "tags": ["religion", "spiritual", "faith", "pastoral"],
     "threat_tags": ["companion", "chatbot", "content"],
     "irreducible": "Faith is a relationship with the transcendent through human community — not content delivery.",
     "blurb": "Clergy, pastoral care, and communal practice."},
    {"id": "mentorship", "name": "Education & human formation", "icon": "fa-person-chalkboard",
     "tags": ["education", "teaching", "mentorship", "formation"],
     "threat_tags": ["tutoring", "edtech", "automation"],
     "irreducible": "Becoming a person happens between people. A tutor can inform; only a mentor can form.",
     "blurb": "Teachers and mentors shaping judgment, not just knowledge."},
]


@app.get("/api/timeline/human-edge")
def get_human_edge(db: Session = Depends(get_db)):
    """Technologies and sectors that still need human intelligence.

    Resilience scores recomputed live: human-side discourse evidence minus
    automation encroachment, plus cautionary-projection support — so Fetch /
    ingestion revisits every number.
    """
    artifacts = db.query(Artifact).all()
    projections = db.query(Projection).all()
    art_map = {a.id: a for a in artifacts}

    def _mini(a: Artifact) -> dict:
        return {"id": a.id, "title": a.title, "artifact_type": a.artifact_type,
                "date": a.date_published.isoformat() if a.date_published else None}

    def _recent(a: Artifact) -> bool:
        if not a.date_published:
            return False
        try:
            base = a.date_published
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - base).total_seconds() < 2 * 31536000
        except Exception:
            return False

    areas = []
    for he in HUMAN_EDGE:
        human = [a for a in artifacts if _artifact_matches_tags(a.tags, he["tags"])]
        encroach = [a for a in artifacts if _artifact_matches_tags(a.tags, he["threat_tags"])]
        human.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
                   reverse=True)
        encroach.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
                      reverse=True)
        linked = [pr for pr in projections if art_map.get(pr.artifact_id) in human]
        caution = [pr for pr in linked if pr.projection_type in ("cautionary", "neutral")]
        avg_c = (sum(pr.confidence for pr in caution) / len(caution)) if caution else 0.0
        hn, en = len(human), len(encroach)
        score = max(10, min(95, 45 + 8 * min(hn, 4) + (6 if any(_recent(a) for a in human) else 0)
                            + round(10 * avg_c) - 8 * min(en, 4)))
        status = "durable" if score >= 60 else ("contested" if score >= 40 else "eroding")
        areas.append({
            "id": he["id"], "name": he["name"], "icon": he["icon"],
            "blurb": he["blurb"], "irreducible": he["irreducible"],
            "status": status, "resilience": score,
            "human_evidence": hn, "encroachment": en,
            "recent_human": any(_recent(a) for a in human),
            "cautionary_support": round(avg_c, 2),
            "components": {"base": 45, "discourse": 8 * min(hn, 4),
                           "recency": 6 if any(_recent(a) for a in human) else 0,
                           "projection": round(10 * avg_c),
                           "encroachment": -8 * min(en, 4)},
            "key_items": [_mini(a) for a in human[:3]],
            "watch_signals": [_mini(a) for a in encroach[:3]],
            "supporting": [{"type": pr.projection_type, "confidence": pr.confidence,
                            "timeframe": pr.timeframe,
                            "summary": _clean_text(pr.summary, 160),
                            "artifact": _mini(art_map[pr.artifact_id]) if art_map.get(pr.artifact_id) else None}
                           for pr in sorted(caution, key=lambda pr: -(pr.confidence or 0))[:3]],
        })
    areas.sort(key=lambda x: (-x["resilience"], -x["human_evidence"]))
    return {
        "now_year": datetime.now(timezone.utc).year,
        "areas": areas,
        "durable_count": sum(1 for x in areas if x["status"] == "durable"),
        "contested_count": sum(1 for x in areas if x["status"] == "contested"),
        "freshness": {"total_artifacts": len(artifacts),
                      "total_projections": len(projections),
                      "last_ingest": ingestion_stats.get("last_run")},
        "method": ("Resilience = 45 base + 8/human-discourse item (max 4) + 6 recent + "
                   "10x cautionary projection confidence − 8/encroaching item (max 4), "
                   "clamped 10–95. Durable ≥60, contested 40–59, eroding <40. Recomputed on every call."),
    }


SAFETY_TAGS = ["safety", "alignment", "superalignment", "xrisk", "existential-risk",
               "control-problem", "deceptive-alignment", "goal-misgeneralization",
               "mesa-optimization", "interpretability", "evaluations", "redteam", "audit",
               "oversight", "sleeper-agents", "backdoor", "jailbreak", "refusal",
               "uncontrollability", "misalignment", "corrigibility", "wmdp", "elk",
               "rlaif", "scalable-oversight", "moral-policy", "right-to-warn"]


@app.get("/api/timeline/safety")
def get_safety_dimension(db: Session = Depends(get_db)):
    """AI safety as a cross-collection dimension: every artifact type's safety
    content, corpus sentiment vs. overall, top safety projections, and insider
    warnings — recomputed live on each call."""
    artifacts = db.query(Artifact).all()
    projections = db.query(Projection).all()
    sentiments = {s.artifact_id: s.score for s in db.query(Sentiment).all()}
    art_map = {a.id: a for a in artifacts}

    def _mini(a: Artifact) -> dict:
        return {"id": a.id, "title": a.title, "artifact_type": a.artifact_type,
                "author": a.author,
                "date": a.date_published.isoformat() if a.date_published else None,
                "sentiment": sentiments.get(a.id)}

    def _recent(a: Artifact) -> bool:
        if not a.date_published:
            return False
        try:
            base = a.date_published
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - base).total_seconds() < 2 * 31536000
        except Exception:
            return False

    safety = [a for a in artifacts if _artifact_matches_tags(a.tags, SAFETY_TAGS)]
    safety.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True)
    by_type: dict[str, dict] = {}
    for a in safety:
        bucket = by_type.setdefault(a.artifact_type or "unknown",
                                    {"count": 0, "recent": 0, "top": []})
        bucket["count"] += 1
        if _recent(a):
            bucket["recent"] += 1
        if len(bucket["top"]) < 3:
            bucket["top"].append(_mini(a))
    sents = [sentiments[a.id] for a in safety if a.id in sentiments]
    all_sents = list(sentiments.values())
    avg_safety = round(sum(sents) / len(sents), 2) if sents else 0.0
    avg_all = round(sum(all_sents) / len(all_sents), 2) if all_sents else 0.0

    safety_ids = {a.id for a in safety}
    top_projs = sorted(
        (pr for pr in projections if pr.artifact_id in safety_ids),
        key=lambda pr: (-(pr.confidence or 0)))[:8]
    proj_items = []
    for pr in top_projs:
        a = art_map.get(pr.artifact_id)
        proj_items.append({"type": pr.projection_type, "confidence": pr.confidence,
                           "timeframe": pr.timeframe,
                           "summary": _clean_text(pr.summary, 180),
                           "artifact": _mini(a) if a else None})
    wb = [a for a in safety if "whistleblower" in (a.tags or "").lower()]
    wb.sort(key=lambda a: a.date_published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True)
    return {
        "now_year": datetime.now(timezone.utc).year,
        "total": len(safety),
        "recent_count": sum(1 for a in safety if _recent(a)),
        "by_type": by_type,
        "collection_types": len(by_type),
        "avg_sentiment_safety": avg_safety,
        "avg_sentiment_overall": avg_all,
        "sentiment_gap": round(avg_safety - avg_all, 2),
        "top_projections": proj_items,
        "whistleblower_count": len(wb),
        "latest_warnings": [_mini(a) for a in wb[:4]],
        "latest_items": [_mini(a) for a in safety[:6]],
        "freshness": {"total_artifacts": len(artifacts),
                      "total_projections": len(projections),
                      "last_ingest": ingestion_stats.get("last_run")},
        "method": ("Safety corpus = artifacts tagged with 27 safety keywords across all "
                   "9 collection types. Sentiment gap = safety avg − overall avg "
                   "(negative = safety discourse runs more concerned). Recomputed on every call."),
    }


def _timeline_histogram(mids: list[float]) -> list[dict]:
    """Forecast counts per 5-year bucket for the tab chart."""
    if not mids:
        return []
    lo = int(min(mids)) // 5 * 5
    hi = int(max(mids)) // 5 * 5
    buckets = []
    for start in range(lo, hi + 1, 5):
        c = sum(1 for m in mids if start <= m < start + 5)
        buckets.append({"bucket": f"{start}–{start + 4}", "count": c})
    return buckets


# ------------------------------------------------------ frontier race ---
# 6th tab: who is racing to crown the first AGI/ASI system. Curated snapshot
# (public info, Oct 2026 — sizes of closed labs are estimates) enriched live
# with the graph's own median AGI year so the lab race can be read against
# community forecasts on every call.
FRONTIER_LABS = [
    {"id": "openai", "organization": "OpenAI", "country": "USA", "flag": "🇺🇸",
     "flagship_model": "GPT-6 Astra", "model_family": "GPT-6 (Astra/Sol/Luna, computer-use)",
     "size_params": "Undisclosed (~1–2T MoE class, rumored)", "size_class": "XL",
     "org_size": "~3,000 staff · ~$20B+ raised", "compute": "Stargate + Azure (>500k H100 equiv.)",
     "license": "Proprietary — API + ChatGPT, no weights", "license_type": "proprietary",
     "crown_score": 96, "agi_target": "2028", "agi_mid": 2028, "asi_target": "2032–35", "asi_mid": 2033,
     "agi_confidence": 0.85, "stance": "Altman: 'taste of AGI' via computer-use agents; superintelligence this decade.",
     "last_milestone": "GPT-6 Astra + Sol/Luna computer-use agents (Sep 2026)"},
    {"id": "anthropic", "organization": "Anthropic", "country": "USA", "flag": "🇺🇸",
     "flagship_model": "Claude Opus 5.5", "model_family": "Claude 5 (constitutional, agentic)",
     "size_params": "Undisclosed (dense/MoE, frontier class)", "size_class": "XL",
     "org_size": "~1,500 staff · ~$18B raised", "compute": "AWS + Google TPU (Project Rainier)",
     "license": "Proprietary — API + Claude apps, no weights", "license_type": "proprietary",
     "crown_score": 94, "agi_target": "2027–28", "agi_mid": 2028, "asi_target": "2031–34", "asi_mid": 2032,
     "agi_confidence": 0.88, "stance": "Amodei: 'powerful AI' 2026–27; responsible-scaling + ASL tiers.",
     "last_milestone": "Opus 5.5 + Sonnet 5.5 coding crowns (Sep 2026)"},
    {"id": "deepmind", "organization": "Google DeepMind", "country": "UK / USA", "flag": "🇬🇧",
     "flagship_model": "Gemini 4 Argon", "model_family": "Gemini 4 (multimodal MoE)",
     "size_params": "Undisclosed (frontier MoE)", "size_class": "XL",
     "org_size": "~2,000+ research staff · Alphabet", "compute": "TPU v6 Trillium pods (in-house)",
     "license": "Proprietary — Vertex AI + AI Studio API", "license_type": "proprietary",
     "crown_score": 93, "agi_target": "2029–30", "agi_mid": 2030, "asi_target": "2033–36", "asi_mid": 2034,
     "agi_confidence": 0.8, "stance": "Hassabis: AGI ~2030 with scientific discovery as the test; AlphaFold lineage.",
     "last_milestone": "Gemini 4 Argon frontier rollout + 3.8 Flash (2026)"},
    {"id": "nvidia", "organization": "NVIDIA", "country": "USA", "flag": "🇺🇸",
     "flagship_model": "Nemotron 3 Ultra", "model_family": "Nemotron 3 MoE (agentic)",
     "size_params": "550B MoE (open) · Nano 30B + Lightning single-GPU", "size_class": "L",
     "org_size": "Big Tech · GPU maker", "compute": "In-house DGX Cloud",
     "license": "Open — NVIDIA Open Model License", "license_type": "open-weights",
     "crown_score": 83, "agi_target": "2031+", "agi_mid": 2031, "asi_target": "2036+", "asi_mid": 2037,
     "agi_confidence": 0.5, "stance": "Full-stack play: open agentic models pull GPU demand; Nemotron 4 1T in dev.",
     "last_milestone": "Nemotron 3 Ultra 550B + Nano 30B open (2026)"},
    {"id": "xai", "organization": "xAI", "country": "USA", "flag": "🇺🇸",
     "flagship_model": "Grok 4.7", "model_family": "Grok 4 (Colossus-trained)",
     "size_params": "Undisclosed (200k+ H100 cluster class)", "size_class": "XL",
     "org_size": "~500 staff · ~$12B raised", "compute": "Colossus 1–2 (200k+ H100s)",
     "license": "Proprietary API (Grok-1 weights Apache-2.0 legacy)", "license_type": "proprietary",
     "crown_score": 88, "agi_target": "2027", "agi_mid": 2027, "asi_target": "2030–33", "asi_mid": 2031,
     "agi_confidence": 0.9, "stance": "Musk: AGI imminent; fastest-compute strategy; 'maximum truth-seeking'.",
     "last_milestone": "Grok 4.7 coding crown + Copilot integration (Sep 2026)"},
    {"id": "meta", "organization": "Meta", "country": "USA", "flag": "🇺🇸",
     "flagship_model": "Muse Spark 1.2", "model_family": "Muse (MSL frontier) · Llama open lineage",
     "size_params": "Undisclosed (frontier class; Llama 4 Maverick 400B/17B still open)", "size_class": "XL",
     "org_size": "MSL + 1,000+ researchers · Big Tech", "compute": "2× 24k-H100 clusters + MSL build-out",
     "license": "Open weights pledged for Spark 1.2 (pending drop) · Llama Community live", "license_type": "hybrid",
     "crown_score": 86, "agi_target": "2030–32", "agi_mid": 2031, "asi_target": "2035+", "asi_mid": 2036,
     "agi_confidence": 0.62, "stance": "MSL bets on personal superintelligence; open-weights pledge as US-leadership play.",
     "last_milestone": "Muse Spark 1.2 + Glimmer 30B open drop (Aug 2026)"},
    {"id": "deepseek", "organization": "DeepSeek", "country": "China", "flag": "🇨🇳",
     "flagship_model": "DeepSeek V4-Pro", "model_family": "DeepSeek V4 MoE (1M ctx)",
     "size_params": "1.6T total / 49B active (MoE) · 1M ctx", "size_class": "XL",
     "org_size": "~200 staff · High-Flyer backed", "compute": "~50k Hopper-class (export-constrained)",
     "license": "Open weights — MIT", "license_type": "open-weights",
     "crown_score": 88, "agi_target": "2028–29", "agi_mid": 2029, "asi_target": "2032–35", "asi_mid": 2033,
     "agi_confidence": 0.78, "stance": "Efficiency-first open lab; V4-Pro 2nd among open weights.",
     "last_milestone": "V4-Pro/Flash + Vision open drops (2026)"},
    {"id": "qwen", "organization": "Alibaba Qwen", "country": "China", "flag": "🇨🇳",
     "flagship_model": "Qwen3.8-Max", "model_family": "Qwen3.8 MoE (262K ctx)",
     "size_params": "2.4T total / 95B active (MoE)", "size_class": "XL",
     "org_size": "Alibaba DAMO + Tongyi Lab", "compute": "Alibaba Cloud hyperscale",
     "license": "Open weights w/ commercial restrictions (custom)", "license_type": "hybrid",
     "crown_score": 86, "agi_target": "2029", "agi_mid": 2029, "asi_target": "2033–36", "asi_mid": 2034,
     "agi_confidence": 0.72, "stance": "Max-class goes open (with commercial strings); Qwen4 commercial next.",
     "last_milestone": "3.8-Max + 27B open drops (Aug 2026)"},
    {"id": "moonshot", "organization": "Moonshot AI", "country": "China", "flag": "🇨🇳",
     "flagship_model": "Kimi K3", "model_family": "Kimi K3 MoE (native vision, 1M ctx)",
     "size_params": "2.8T total (aggressive MoE) · 1M ctx", "size_class": "XL",
     "org_size": "Startup · ~$1B+ raised", "compute": "Domestic + leased hyperscale",
     "license": "Open weights — Modified MIT", "license_type": "open-weights",
     "crown_score": 85, "agi_target": "2029–30", "agi_mid": 2030, "asi_target": "2034+", "asi_mid": 2035,
     "agi_confidence": 0.68, "stance": "World's largest open-weights drop; agent-first lab.",
     "last_milestone": "Kimi K3 2.8T open release (Jul 2026)"},
    {"id": "zhipu", "organization": "Zhipu AI (GLM)", "country": "China", "flag": "🇨🇳",
     "flagship_model": "GLM-5.2", "model_family": "GLM-5 MoE (agentic engineering)",
     "size_params": "744B MoE class", "size_class": "L",
     "org_size": "~1,000 staff · state-backed", "compute": "Domestic clusters",
     "license": "Dual — open GLM-4 weights + proprietary API", "license_type": "hybrid",
     "crown_score": 80, "agi_target": "2030", "agi_mid": 2030, "asi_target": "2035+", "asi_mid": 2036,
     "agi_confidence": 0.62, "stance": "HKEX-listed; vendor-claim near-Opus coding.",
     "last_milestone": "GLM-5/5.2 open + HKEX listing (2026)"},
    {"id": "tencent", "organization": "Tencent Hunyuan", "country": "China", "flag": "🇨🇳",
     "flagship_model": "Hunyuan Hy4 Preview", "model_family": "Hunyuan Hy MoE (multimodal)",
     "size_params": "Undisclosed (Hy3: 295B/21B MoE)", "size_class": "M",
     "org_size": "Big Tech division", "compute": "Tencent Cloud hyperscale",
     "license": "Hy4-Preview reportedly open · Cloud API", "license_type": "hybrid",
     "crown_score": 74, "agi_target": "2030–32", "agi_mid": 2031, "asi_target": "2035+", "asi_mid": 2036,
     "agi_confidence": 0.55, "stance": "Ecosystem play: WeChat + cloud distribution over raw frontier.",
     "last_milestone": "Hy4 Preview (Aug 2026); Hy3 295B open"},
    {"id": "baidu", "organization": "Baidu", "country": "China", "flag": "🇨🇳",
     "flagship_model": "Ernie 5.1", "model_family": "Ernie 5 (efficient multimodal)",
     "size_params": "Undisclosed", "size_class": "M",
     "org_size": "Big Tech division", "compute": "Baidu AI Cloud",
     "license": "Proprietary — Qianfan API", "license_type": "proprietary",
     "crown_score": 70, "agi_target": "2031+", "agi_mid": 2032, "asi_target": "2036+", "asi_mid": 2037,
     "agi_confidence": 0.52, "stance": "Efficiency play: flagship at 6% pretrain cost; conservative dates.",
     "last_milestone": "Ernie 5.1 top-5 leaderboard (May 2026)"},
    {"id": "mistral", "organization": "Mistral AI", "country": "France", "flag": "🇫🇷",
     "flagship_model": "Mistral Large 4", "model_family": "Large (granular MoE, natively multimodal)",
     "size_params": "~1.05T total / ~52B active (MoE) + 1.6B vision encoder · 1M-token context", "size_class": "L",
     "org_size": "~150 staff · ~$2B raised", "compute": "European HPC + partners",
     "license": "Open weights — MNPL (Large) / API preview on Mistral Studio", "license_type": "open-weights",
     "crown_score": 80, "agi_target": "2031+", "agi_mid": 2031, "asi_target": "2036+", "asi_mid": 2037,
     "agi_confidence": 0.55, "stance": "EU sovereignty champion; Large 4 granular-MoE bet at 1T scale, natively multimodal.",
     "last_milestone": "Large 4 1T-MoE open-weight drop, 1M context (2026)"},
    {"id": "cohere", "organization": "Cohere", "country": "Canada", "flag": "🇨🇦",
     "flagship_model": "Command A+", "model_family": "Command A+ MoE (enterprise)",
     "size_params": "218B total / 25B active (MoE)", "size_class": "M",
     "org_size": "~400 staff · ~$1B raised", "compute": "Oracle + CoreWeave partnerships",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "crown_score": 70, "agi_target": "2032+", "agi_mid": 2033, "asi_target": "2038+", "asi_mid": 2038,
     "agi_confidence": 0.45, "stance": "Enterprise language AI; no AGI claim — productivity moat.",
     "last_milestone": "Command A+ MoE open under Apache-2.0 (May 2026)"},
    {"id": "ssi", "organization": "Safe Superintelligence (SSI)", "country": "USA / Israel", "flag": "🇺🇸",
     "flagship_model": "SSI-1 (stealth)", "model_family": "Undisclosed (superintelligence-direct)",
     "size_params": "Undisclosed (stealth)", "size_class": "XL?",
     "org_size": "Small elite team · NVIDIA $5B strategic partner", "compute": "Undisclosed (NVIDIA-partnered)",
     "license": "Proprietary — stealth, no release yet", "license_type": "proprietary",
     "crown_score": 79, "agi_target": "2028–30", "agi_mid": 2029, "asi_target": "2030–33 (direct)", "asi_mid": 2031,
     "agi_confidence": 0.75, "stance": "NVIDIA-backed straight-shot SSI; first model reportedly due 2026.",
     "last_milestone": "NVIDIA $5B partnership (Jul 2026)"},
]

FRONTIER_TIMELINE = [
    {"year": 2017, "label": "Transformer", "kind": "milestone", "detail": "Vaswani et al. — the architecture every contender still scales."},
    {"year": 2020, "label": "GPT-3 · scaling laws", "kind": "milestone", "detail": "175B few-shot + Kaplan/McCandlish scaling curves launch the race."},
    {"year": 2022, "label": "ChatGPT moment", "kind": "milestone", "detail": "OpenAI turns LLMs into a consumer crown; 100M users in 2 months."},
    {"year": 2023, "label": "GPT-4 · Claude · Llama", "kind": "milestone", "detail": "Three-pole race forms: closed frontier (US) vs open weights (Meta)."},
    {"year": 2024, "label": "Reasoning + video", "kind": "milestone", "detail": "o1 chain-of-thought, Sora, Claude 3.5 Sonnet coding lead."},
    {"year": 2025, "label": "Agents + open shock", "kind": "milestone", "detail": "DeepSeek-R1/Qwen3/Kimi-K2 prove open weights race at 1/10th cost; GPT-5, Claude 4, Gemini 2.5 ship agents."},
    {"year": 2026, "label": "Crown fight (now)", "kind": "now", "detail": "GPT-5.2 vs Opus 4.5 vs Gemini 3 vs Grok 4.1 — coding/agent benchmarks trade the lead monthly."},
    {"year": 2028, "label": "AGI window (short camp)", "kind": "agi", "detail": "Musk/Amodei/Altman targets cluster 2027–29: autonomous R&D + full job-task automation."},
    {"year": 2030, "label": "AGI window (consensus)", "kind": "agi", "detail": "Median lab target ~2029–30; Hassabis/consensus view. Graph community median shown live below."},
    {"year": 2033, "label": "Early ASI", "kind": "asi", "detail": "Median lab ASI target ~2033–34: self-improving systems, science acceleration."},
    {"year": 2038, "label": "Full ASI horizon", "kind": "asi", "detail": "Long-tail lab targets 2036–38+: superintelligent fleets, governance test."},
]

FRONTIER_PHASES = [
    {"phase": "Narrow → General copilots", "span": "2020–2025", "label": "done",
     "detail": "Chat, code, multimodal assistants. Benchmarks saturated; agents emerge."},
    {"phase": "Agentic frontier race", "span": "2025–2028", "label": "now",
     "detail": "SWE/agentic crowns rotate monthly. Compute (Colossus, Stargate, TPU pods) is the moat; open weights close the gap."},
    {"phase": "AGI contention", "span": "2027–2031", "label": "agi",
     "detail": "Lab medians point here. Definition fight: autonomous researcher + economic-task generality, not just chat."},
    {"phase": "ASI transition", "span": "2031–2038+", "label": "asi",
     "detail": "Self-improvement loops + science acceleration. Safety-culture and governance gaps decide whether it is steered."},
]


# GPU needed to RUN each flagship (inference, not training). Estimates for
# full-weight serving at usable context (FP8/BF16) plus cheapest practical
# quantized local run (Q4/HQQ/AWQ via vLLM/SGLang/Ollama). API-only labs have
# no local path — weights are closed. Oct 2026 snapshot; quant tooling moves fast.
FRONTIER_RUN_COMPUTE = {
    "openai": {"run_tier": "api-only",
               "run_gpus": "No local run — weights closed",
               "run_vram": "n/a",
               "run_note": "Datacenter replica est. 8× H100 80GB per shard; only via API/ChatGPT."},
    "anthropic": {"run_tier": "api-only",
                  "run_gpus": "No local run — weights closed",
                  "run_vram": "n/a",
                  "run_note": "AWS/TPU serving fleet; only via API/Claude apps."},
    "deepmind": {"run_tier": "api-only",
                 "run_gpus": "No local run — weights closed",
                 "run_vram": "n/a",
                 "run_note": "TPU-pod serving; only via Vertex AI / AI Studio."},
    "xai": {"run_tier": "api-only",
            "run_gpus": "No local run (flagship) — Grok-1 314B legacy weights need ~4× H100",
            "run_vram": "n/a",
            "run_note": "Grok 4.7 via API/X Premium; legacy Grok-1 open weights are not the frontier model."},
    "meta": {"run_tier": "datacenter",
             "run_gpus": "FP8 ~5–6× H100 80GB · Q4 ~3× H100 / ~9× RTX 4090 24GB",
             "run_vram": "~200–400 GB",
             "run_note": "Figures for Llama 4 Maverick (runnable open flagship); Muse Spark 1.2 weights pledged, pending drop."},
    "deepseek": {"run_tier": "datacenter",
                 "run_gpus": "FP8 ~1.6 TB → ~20× H100 80GB · Q4 ~800 GB → ~10× H100 / ~34× RTX 4090",
                 "run_vram": "~800 GB–1.6 TB",
                 "run_note": "V4-Pro 1.6T MoE (49B active); MLA + 1M ctx need headroom beyond weights."},
    "qwen": {"run_tier": "workstation",
             "run_gpus": "FP8 ~2.4 TB → ~30× H100 80GB · Q4 ~1.2 TB → ~15× H100 / ~50× RTX 4090",
             "run_vram": "~1.2–2.4 TB",
             "run_note": "3.8-Max 2.4T MoE (95B active); Max-class open weights need multi-node."},
    "moonshot": {"run_tier": "datacenter",
                 "run_gpus": "FP8 ~2.8 TB → ~36× H100 80GB · Q4 ~1.4 TB → ~18× H100 / ~60× RTX 4090",
                 "run_vram": "~1.4–2.8 TB",
                 "run_note": "Kimi K3 2.8T aggressive MoE; heaviest open run on the board — multi-node only."},
    "zhipu": {"run_tier": "workstation",
              "run_gpus": "FP8 ~744 GB → ~10× H100 80GB · Q4 ~370 GB → ~5× H100 / ~16× RTX 4090",
              "run_vram": "~370–744 GB",
              "run_note": "GLM-5 744B MoE class; agentic-engineering tuned."},
    "tencent": {"run_tier": "datacenter",
                "run_gpus": "Hy3 295B Q4 ~150 GB → ~2× H100 / ~7× RTX 4090 (Hy4 TBD)",
                "run_vram": "~150 GB+",
                "run_note": "Hy3 weights open on GitHub; Hy4-Preview reportedly open, sizing TBD."},
    "baidu": {"run_tier": "api-only",
              "run_gpus": "No local run — weights closed",
              "run_vram": "n/a",
              "run_note": "Ernie 5.1 via Qianfan API only (4.5 family weights are open)."},
    "mistral": {"run_tier": "datacenter",
                "run_gpus": "FP8 ~1 TB → ~13× H100 80GB · Q4 ~525 GB → ~7× H100 / ~22× RTX 4090",
                "run_vram": "~525 GB–1 TB",
                "run_note": "1.05T granular MoE (52B active): burns like a 50B dense per token, but full weights must fit in VRAM."},
    "cohere": {"run_tier": "workstation",
               "run_gpus": "FP8 ~436 GB → ~6× H100 80GB · Q4 ~110 GB → ~2× H100 / ~5× RTX 4090",
               "run_vram": "~110–436 GB",
               "run_note": "Command A+ 218B MoE (25B active), Apache-2.0; enterprise build."},
    "nvidia": {"run_tier": "datacenter",
               "run_gpus": "FP8 ~550 GB → ~7× H100 80GB · Q4 ~275 GB → ~4× H100 / ~12× RTX 4090",
               "run_vram": "~275–550 GB",
               "run_note": "Long-run-agent tuned; Nano 30B runs on 1 GPU (see desktop subtab)."},
    "ssi": {"run_tier": "api-only",
            "run_gpus": "Unknown — stealth, no weights released",
            "run_vram": "n/a",
            "run_note": "No public model to run yet; watch for release-license change."},
}

# Task shapes behind the per-model GPU + power table. Multipliers apply to the
# numeric anchors below (Q4 single-user chat = 1×, full-precision fleet for
# serving). Power model: H100 SXM 700W each + 500W host per ≤8-GPU node.
RUN_TASK_DEFS = [
    {"id": "chat", "name": "Chat · 8k ctx",
     "desc": "Single-user assistant, short context, Q4 quantized — the cheapest way to run it."},
    {"id": "agent", "name": "Agentic coding · 32k + tools",
     "desc": "Sustained reasoning loop with tool calls; ~1.5× GPUs for KV cache + steady boost clocks."},
    {"id": "longctx", "name": "Long-context · 128k RAG",
     "desc": "Big KV cache dominates — ~2× GPUs/VRAM over chat."},
    {"id": "serve", "name": "Serving · 64 concurrent",
     "desc": "Throughput fleet: 4 full-precision replicas behind a router."},
]

# Numeric anchors per locally-runnable lab: Q4 chat GPU count, full-precision
# replica size, Q4 VRAM GB, FP8 VRAM GB, RTX-4090 Q4 equivalent count.
RUN_ANCHORS = {
    "meta": {"q4_h100": 3, "fp8_h100": 6, "q4_gb": 200, "fp8_gb": 400, "r4090": 9},
    "deepseek": {"q4_h100": 10, "fp8_h100": 20, "q4_gb": 800, "fp8_gb": 1600, "r4090": 34},
    "qwen": {"q4_h100": 15, "fp8_h100": 30, "q4_gb": 1200, "fp8_gb": 2400, "r4090": 50},
    "moonshot": {"q4_h100": 18, "fp8_h100": 36, "q4_gb": 1400, "fp8_gb": 2800, "r4090": 60},
    "zhipu": {"q4_h100": 5, "fp8_h100": 10, "q4_gb": 370, "fp8_gb": 744, "r4090": 16},
    "mistral": {"q4_h100": 7, "fp8_h100": 13, "q4_gb": 525, "fp8_gb": 1050, "r4090": 22},
    "cohere": {"q4_h100": 2, "fp8_h100": 6, "q4_gb": 110, "fp8_gb": 436, "r4090": 5},
    "tencent": {"q4_h100": 2, "fp8_h100": 8, "q4_gb": 150, "fp8_gb": 590, "r4090": 7},
    "nvidia": {"q4_h100": 4, "fp8_h100": 7, "q4_gb": 275, "fp8_gb": 550, "r4090": 12},
}

API_REPLICA_H100 = 8  # est. GPUs per provider-side flagship replica


# Sortable model specs: total/active params (B), architecture, weight-file size
# on disk (GB, Q4 vs FP8/BF16). Null = undisclosed (sorts last, shows "—").
# est=true marks press/analyst estimates, displayed with a "~".
FRONTIER_MODEL_SPECS = {
    "openai": {"params_total_b": None, "params_active_b": None, "params_est": False,
               "arch": "undisclosed", "disk_q4_gb": None, "disk_fp8_gb": None},
    "anthropic": {"params_total_b": None, "params_active_b": None, "params_est": False,
                  "arch": "undisclosed", "disk_q4_gb": None, "disk_fp8_gb": None},
    "deepmind": {"params_total_b": None, "params_active_b": None, "params_est": False,
                 "arch": "undisclosed", "disk_q4_gb": None, "disk_fp8_gb": None},
    "xai": {"params_total_b": None, "params_active_b": None, "params_est": False,
            "arch": "undisclosed", "disk_q4_gb": None, "disk_fp8_gb": None},
    "meta": {"params_total_b": 400, "params_active_b": 17, "params_est": False,
             "arch": "moe", "disk_q4_gb": 200, "disk_fp8_gb": 400},
    "deepseek": {"params_total_b": 1600, "params_active_b": 49, "params_est": False,
                 "arch": "moe", "disk_q4_gb": 800, "disk_fp8_gb": 1600},
    "qwen": {"params_total_b": 2400, "params_active_b": 95, "params_est": True,
             "arch": "moe", "disk_q4_gb": 1200, "disk_fp8_gb": 2400},
    "moonshot": {"params_total_b": 2800, "params_active_b": None, "params_est": True,
                 "arch": "moe", "disk_q4_gb": 1400, "disk_fp8_gb": 2800},
    "zhipu": {"params_total_b": 744, "params_active_b": None, "params_est": True,
              "arch": "moe", "disk_q4_gb": 370, "disk_fp8_gb": 744, "disk_est": True},
    "tencent": {"params_total_b": 295, "params_active_b": 21, "params_est": False,
                "arch": "moe", "disk_q4_gb": 150, "disk_fp8_gb": 590, "disk_est": True},
    "baidu": {"params_total_b": None, "params_active_b": None, "params_est": False,
              "arch": "undisclosed", "disk_q4_gb": None, "disk_fp8_gb": None},
    "mistral": {"params_total_b": 1050, "params_active_b": 52, "params_est": True,
                "arch": "moe", "disk_q4_gb": 525, "disk_fp8_gb": 1050},
    "cohere": {"params_total_b": 218, "params_active_b": 25, "params_est": False,
               "arch": "moe", "disk_q4_gb": 110, "disk_fp8_gb": 436},
    "nvidia": {"params_total_b": 550, "params_active_b": None, "params_est": False,
               "arch": "moe", "disk_q4_gb": 275, "disk_fp8_gb": 550},
    "ssi": {"params_total_b": None, "params_active_b": None, "params_est": False,
            "arch": "undisclosed", "disk_q4_gb": None, "disk_fp8_gb": None},
}

# AI harness / tool adoption race: the coding agents developers actually live
# in. adoption_score blends work-usage share (2026 dev surveys), revenue scale
# and benchmark leadership of the models each harness serves. Oct 2026 snapshot.
FRONTIER_HARNESSES = [
    {"id": "claude-code", "tool": "Claude Code", "vendor": "Anthropic", "country": "USA", "flag": "🇺🇸",
     "harness_type": "CLI agent", "license": "Proprietary (free tier + Pro/Max/API)", "license_type": "proprietary",
     "adoption": "39% of devs at work (2026 surveys) — #1 in actual use",
     "bench": "Tops SWE-bench via Opus 5.5-class models",
     "adoption_score": 95, "last_milestone": "Terminal agent + IDE hooks (2026)"},
    {"id": "copilot", "tool": "GitHub Copilot", "vendor": "Microsoft / GitHub", "country": "USA", "flag": "🇺🇸",
     "harness_type": "Completion + agent mode", "license": "Proprietary (subscription)", "license_type": "proprietary",
     "adoption": "~29% of devs; largest installed base (JetBrains Jan 2026)",
     "bench": "Agent mode on frontier GPT/Claude models",
     "adoption_score": 90, "last_milestone": "Agent mode + mission control (2025–26)"},
    {"id": "cursor", "tool": "Cursor", "vendor": "Anysphere", "country": "USA", "flag": "🇺🇸",
     "harness_type": "AI-native IDE", "license": "Proprietary (~$4B ARR)", "license_type": "proprietary",
     "adoption": "AI-IDE standard-setter; $4B annualized revenue pace",
     "bench": "Model-agnostic, ships newest frontier models first",
     "adoption_score": 88, "last_milestone": "Background agents + cloud (2025–26)"},
    {"id": "codex", "tool": "OpenAI Codex", "vendor": "OpenAI", "country": "USA", "flag": "🇺🇸",
     "harness_type": "CLI + cloud agents", "license": "Proprietary (ChatGPT/API)", "license_type": "proprietary",
     "adoption": "Fastest-growing CLI agent on GPT-6-class models",
     "bench": "SWE tasks at ~1/20th legacy-agent cost",
     "adoption_score": 82, "last_milestone": "Codex CLI + cloud fleet (2025–26)"},
    {"id": "windsurf", "tool": "Windsurf", "vendor": "Cognition", "country": "USA", "flag": "🇺🇸",
     "harness_type": "AI IDE + agents", "license": "Proprietary", "license_type": "proprietary",
     "adoption": "Strong pro-team footprint post-Cognition deal",
     "bench": "Cascade agents on frontier models",
     "adoption_score": 72, "last_milestone": "Cognition acquisition, agent push (2025)"},
    {"id": "cline", "tool": "Cline", "vendor": "Community open-source", "country": "Global", "flag": "🌍",
     "harness_type": "IDE extension", "license": "Open — Apache-2.0", "license_type": "open-weights",
     "adoption": "Top open agent extension; bring-your-own-key to any frontier model",
     "bench": "Community evals track closed tools within points",
     "adoption_score": 70, "last_milestone": "Multi-model marketplace (2025–26)"},
    {"id": "devin", "tool": "Devin", "vendor": "Cognition", "country": "USA", "flag": "🇺🇸",
     "harness_type": "Autonomous cloud agent", "license": "Proprietary (seat/API)", "license_type": "proprietary",
     "adoption": "Enterprise autonomous-SWE pioneer; premium pricing",
     "bench": "End-to-end repo tasks, async fleet",
     "adoption_score": 66, "last_milestone": "Enterprise fleet + Windsurf synergy (2025–26)"},
    {"id": "gemini-assist", "tool": "Gemini Code Assist", "vendor": "Google", "country": "USA", "flag": "🇺🇸",
     "harness_type": "IDE + cloud", "license": "Proprietary (free tier)", "license_type": "proprietary",
     "adoption": "Wide free-tier reach via 1M-token context window",
     "bench": "Gemini 4 long-context refactors",
     "adoption_score": 64, "last_milestone": "Free tier + 1M context (2025–26)"},
    {"id": "aider", "tool": "Aider", "vendor": "Community open-source", "country": "Global", "flag": "🌍",
     "harness_type": "CLI pair-programmer", "license": "Open — Apache-2.0", "license_type": "open-weights",
     "adoption": "Terminal purists' pick; git-native edit loop",
     "bench": "Polyglot benchmark stalwart on open models",
     "adoption_score": 60, "last_milestone": "Voice + image input (2025–26)"},
    {"id": "lingma", "tool": "Tongyi Lingma", "vendor": "Alibaba", "country": "China", "flag": "🇨🇳",
     "harness_type": "IDE plugin", "license": "Proprietary (free)", "license_type": "proprietary",
     "adoption": "Dominant in CN dev workflows on Qwen models",
     "bench": "Qwen3-Coder tuned completions",
     "adoption_score": 58, "last_milestone": "Qwen3-Coder integration (2025–26)"},
]

# Average-home-use profiles behind the desktop subtab (all on consumer GPUs).
DESK_PROFILES = [
    {"id": "regular", "name": "Regular query",
     "desc": "~1 h active inference/day: drafting, Q&A, short code help."},
    {"id": "agentic", "name": "Agentic use",
     "desc": "~4 h/day sustained: coding-agent loops, tool calls, long context."},
]

# Personal-desktop models: open, <240B params, runnable on a home rig.
# gpus = typical 24GB-class cards (RTX 3090/4090); capability = editorial 0–100;
# home_score = capability − 6/extra-GPU − 3 custom-license + 2 tiny-MoE bonus.
FRONTIER_DESKTOPS = [
    {"id": "qwen3-32b", "model": "Qwen3-32B", "vendor": "Alibaba Qwen", "country": "China", "flag": "🇨🇳",
     "params_b": 32, "active_b": None, "arch": "dense",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "disk_q4_gb": 19, "home_gpus": "1× RTX 4090 24GB", "gpus": 1, "min_vram": "20 GB",
     "capability": 92, "blurb": "The 32B sweet spot: near-70B quality on a single card.",
     "keys": ["qwen3-32b", "qwen3 32b"]},
    {"id": "qwen3-30b-a3b", "model": "Qwen3-30B-A3B", "vendor": "Alibaba Qwen", "country": "China", "flag": "🇨🇳",
     "params_b": 30, "active_b": 3, "arch": "moe",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "disk_q4_gb": 18, "home_gpus": "1× RTX 3090/4090 24GB", "gpus": 1, "min_vram": "20 GB",
     "capability": 86, "blurb": "MoE efficiency pick: 3B active per token, sips power.",
     "keys": ["qwen3-30b", "30b-a3b", "qwen3 30b"]},
    {"id": "llama33-70b", "model": "Llama 3.3 70B", "vendor": "Meta", "country": "USA", "flag": "🇺🇸",
     "params_b": 70, "active_b": None, "arch": "dense",
     "license": "Open weights — Llama Community (custom)", "license_type": "open-weights",
     "disk_q4_gb": 40, "home_gpus": "2× RTX 4090 24GB", "gpus": 2, "min_vram": "2× 24 GB",
     "capability": 90, "blurb": "Proven 70B workhorse; needs a 2-GPU rig.",
     "keys": ["llama 3.3", "llama 70b"]},
    {"id": "r1-distill-70b", "model": "DeepSeek-R1-Distill-Llama-70B", "vendor": "DeepSeek", "country": "China", "flag": "🇨🇳",
     "params_b": 70, "active_b": None, "arch": "dense",
     "license": "Open weights — MIT", "license_type": "open-weights",
     "disk_q4_gb": 40, "home_gpus": "2× RTX 4090 24GB", "gpus": 2, "min_vram": "2× 24 GB",
     "capability": 87, "blurb": "Open reasoning traces at home; thinks before it speaks.",
     "keys": ["r1-distill", "r1 distill", "deepseek-r1"]},
    {"id": "scout-109b", "model": "Llama 4 Scout", "vendor": "Meta", "country": "USA", "flag": "🇺🇸",
     "params_b": 109, "active_b": 17, "arch": "moe",
     "license": "Open weights — Llama Community (custom)", "license_type": "open-weights",
     "disk_q4_gb": 60, "home_gpus": "2× RTX 5090 32GB (or 3× 4090)", "gpus": 2, "min_vram": "~64 GB",
     "capability": 88, "blurb": "Biggest home MoE: 109B knowledge, 17B active burn.",
     "keys": ["llama 4 scout", "llama scout"]},
    {"id": "devstral-24b", "model": "Devstral Small 24B", "vendor": "Mistral AI", "country": "France", "flag": "🇫🇷",
     "params_b": 24, "active_b": None, "arch": "dense",
     "license": "Open weights — MNPL", "license_type": "open-weights",
     "disk_q4_gb": 14, "home_gpus": "1× RTX 4090 (fits 16 GB cards too)", "gpus": 1, "min_vram": "16 GB",
     "capability": 84, "blurb": "Agentic-coding tuned; the home code-agent pick.",
     "keys": ["devstral"]},
    {"id": "glm-32b", "model": "GLM-Z1-32B", "vendor": "Zhipu AI", "country": "China", "flag": "🇨🇳",
     "params_b": 32, "active_b": None, "arch": "dense",
     "license": "Open weights — MIT", "license_type": "open-weights",
     "disk_q4_gb": 19, "home_gpus": "1× RTX 4090 24GB", "gpus": 1, "min_vram": "20 GB",
     "capability": 82, "blurb": "Bilingual reasoning 32B; MIT-licensed and tidy.",
     "keys": ["glm-z1", "glm-4-32b", "glm 32b"]},
    {"id": "gemma3-27b", "model": "Gemma 3 27B", "vendor": "Google", "country": "USA", "flag": "🇺🇸",
     "params_b": 27, "active_b": None, "arch": "dense",
     "license": "Open weights — Gemma (custom, free)", "license_type": "hybrid",
     "disk_q4_gb": 16, "home_gpus": "1× RTX 4090 24GB", "gpus": 1, "min_vram": "18 GB",
     "capability": 82, "blurb": "Multimodal 27B; polite, compact, single-card.",
     "keys": ["gemma 3", "gemma3"]},
    {"id": "mistral-small-24b", "model": "Mistral Small 3.1 24B", "vendor": "Mistral AI", "country": "France", "flag": "🇫🇷",
     "params_b": 24, "active_b": None, "arch": "dense",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "disk_q4_gb": 14, "home_gpus": "1× RTX 4090 (fits 16 GB cards too)", "gpus": 1, "min_vram": "16 GB",
     "capability": 80, "blurb": "Vision-capable 24B; Apache-2.0, no strings.",
     "keys": ["mistral small", "small 3.1"]},
    {"id": "phi4-14b", "model": "Phi-4 14B", "vendor": "Microsoft", "country": "USA", "flag": "🇺🇸",
     "params_b": 14, "active_b": None, "arch": "dense",
     "license": "Open weights — MIT", "license_type": "open-weights",
     "disk_q4_gb": 9, "home_gpus": "1× RTX 3090 (even 12 GB cards)", "gpus": 1, "min_vram": "12 GB",
     "capability": 78, "blurb": "Smallest serious desk model; runs almost anywhere.",
     "keys": ["phi-4", "phi 4"]},
    {"id": "qwen38-27b", "model": "Qwen3.8-27B", "vendor": "Alibaba Qwen", "country": "China", "flag": "🇨🇳",
     "params_b": 27, "active_b": None, "arch": "dense",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "disk_q4_gb": 15, "home_gpus": "1× RTX 4090 (fits 16GB cards too)", "gpus": 1, "min_vram": "16 GB",
     "capability": 93, "blurb": "Aug-2026 open drop: agentic dense VLM, 262K ctx, single-GPU flagship.",
     "keys": ["qwen3.8-27b", "qwen3.8 27b", "qwen 3.8 27b"]},
    {"id": "glimmer-30b", "model": "Muse Glimmer 30B", "vendor": "Meta", "country": "USA", "flag": "🇺🇸",
     "params_b": 30, "active_b": None, "arch": "dense",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "disk_q4_gb": 17, "home_gpus": "1× RTX 3090/4090 24GB", "gpus": 1, "min_vram": "20 GB",
     "capability": 85, "blurb": "Meta's always-on local-agent 30B; most permissive Meta license yet.",
     "keys": ["muse glimmer", "glimmer 30b"]},
    {"id": "nano-30b", "model": "Nemotron 3 Nano 30B", "vendor": "NVIDIA", "country": "USA", "flag": "🇺🇸",
     "params_b": 30, "active_b": None, "arch": "hybrid",
     "license": "Open — NVIDIA Open Model License", "license_type": "open-weights",
     "disk_q4_gb": 17, "home_gpus": "1× RTX 3090/4090 24GB", "gpus": 1, "min_vram": "20 GB",
     "capability": 83, "blurb": "Hybrid-reasoning 30B built for long-running agents.",
     "keys": ["nemotron 3 nano", "nemotron nano"]},
    {"id": "mixtral-8x22b", "model": "Mixtral 8x22B", "vendor": "Mistral AI", "country": "France", "flag": "🇫🇷",     "params_b": 141, "active_b": 39, "arch": "moe",
     "license": "Open weights — Apache-2.0", "license_type": "open-weights",
     "disk_q4_gb": 75, "home_gpus": "3× RTX 4090 (rich-home rig)", "gpus": 3, "min_vram": "3× 24 GB",
     "capability": 84, "blurb": "Veteran big MoE; quality costs a third card.",
     "keys": ["mixtral", "8x22b"]},
]


def _desk_use(gpus: int) -> dict:
    """Home power/cost for the two average-use profiles.

    Active draw ≈ 320W per 24GB card under inference + 150W host; regular =
    1 h/day active, agentic = 4 h/day. Power at $0.15/kWh.
    """
    draw = round(gpus * 0.32 + 0.15, 2)
    out = {}
    for pid, hours in (("regular", 1), ("agentic", 4)):
        kwh_day = round(draw * hours, 2)
        kwh_mo = round(kwh_day * 30, 1)
        out[pid] = {"draw_kw": draw, "kwh_day": kwh_day, "kwh_mo": kwh_mo,
                    "cost_mo": round(kwh_mo * 0.15, 2)}
    return out


def _desk_score(m: dict) -> float:
    s = m["capability"] - (m["gpus"] - 1) * 6
    if m["license_type"] not in ("open-weights",):
        s -= 3
    elif "custom" in m["license"].lower() or "gemma" in m["license"].lower() or "mnpl" in m["license"].lower():
        s -= 3
    if m["arch"] == "moe" and m["params_b"] <= 35:
        s += 2
    return round(s, 1)


def _kw(n_h100: int) -> str:
    import math as _m
    nodes = max(1, _m.ceil(n_h100 / 8))
    w = n_h100 * 700 + nodes * 500
    return f"~{w / 1000:.1f} kW draw · ~{w * 24 / 1000:.0f} kWh/day"


def _frontier_run_tasks(lab_id: str, tier: str) -> list:
    """Per-task GPU + VRAM + power rows for one lab (estimates, see method)."""
    import math as _m
    if tier not in ("workstation", "datacenter") or lab_id not in RUN_ANCHORS:
        rep = f"{API_REPLICA_H100}× H100"
        return [
            {"id": "chat", "task": "Chat · 8k ctx",
             "gpus": "Shared shard (≈1/20 replica, provider-side)", "vram": "provider-side",
             "power": "~0.3 kW per active user",
             "note": "You pay per token; provider amortizes one replica over many users."},
            {"id": "agent", "task": "Agentic coding · 32k + tools",
             "gpus": f"1 dedicated replica ({rep})", "vram": "provider-side",
             "power": _kw(API_REPLICA_H100),
             "note": "Long reasoning traces pin a full replica per active agent."},
            {"id": "longctx", "task": "Long-context · 128k RAG",
             "gpus": f"2 replicas (16× H100)", "vram": "provider-side",
             "power": _kw(16),
             "note": "KV cache for 128k+ context spills across a second replica."},
            {"id": "serve", "task": "Serving · 64 concurrent",
             "gpus": "8 replicas (64× H100)", "vram": "provider-side",
             "power": _kw(64),
             "note": "Fleet behind a router; autoscaled, not yours to power."},
        ]
    a = RUN_ANCHORS[lab_id]
    q, f, r4090 = a["q4_h100"], a["fp8_h100"], a["r4090"]
    agent_n = _m.ceil(q * 1.5)
    long_n = q * 2
    serve_n = f * 4
    return [
        {"id": "chat", "task": "Chat · 8k ctx",
         "gpus": f"{q}× H100 80GB Q4 (or ~{r4090}× RTX 4090)",
         "vram": f"~{a['q4_gb']} GB",
         "power": _kw(q),
         "note": "Cheapest local run; single user, quantized."},
        {"id": "agent", "task": "Agentic coding · 32k + tools",
         "gpus": f"{agent_n}× H100 80GB Q4",
         "vram": f"~{int(a['q4_gb'] * 1.3)} GB",
         "power": _kw(agent_n),
         "note": "Sustained boost clocks + tool-call KV retention."},
        {"id": "longctx", "task": "Long-context · 128k RAG",
         "gpus": f"{long_n}× H100 80GB Q4",
         "vram": f"~{a['q4_gb'] * 2} GB",
         "power": _kw(long_n),
         "note": "KV cache at 128k dominates VRAM — add one GPU headroom."},
        {"id": "serve", "task": "Serving · 64 concurrent",
         "gpus": f"{serve_n}× H100 80GB FP8 (4 replicas)",
         "vram": f"~{a['fp8_gb'] * 4} GB fleet",
         "power": _kw(serve_n),
         "note": "Full-precision fleet; router + redundancy included in count."},
    ]


# Live-adoption match terms: counted against artifact title/description/tags on
# every call, so buzz, momentum and the harness crown move with ingestion.
FRONTIER_BUZZ_KEYS = {
    "openai": ["openai", "gpt-6", "gpt-5", "astra", "sam altman", "stargate"],
    "anthropic": ["anthropic", "claude opus", "claude sonnet", "amodei"],
    "deepmind": ["deepmind", "gemini 4", "gemini 3", "hassabis", "vertex ai"],
    "xai": ["xai", "grok 4", "grok-1", "colossus", "elon musk"],
    "meta": ["llama 4", "llama", "maverick", "muse spark", "muse glimmer", "meta superintelligence", "zuckerberg"],
    "deepseek": ["deepseek", "deepseek v4", "deepseek-r1", "high-flyer"],
    "qwen": ["qwen", "qwen3", "qwen3.8", "qwen max", "qwen4", "tongyi"],
    "moonshot": ["moonshot", "kimi k3", "kimi k2", "kimi"],
    "zhipu": ["zhipu", "glm-5", "glm-4", "z.ai", "autoglm"],
    "tencent": ["hunyuan", "hy4", "hy3", "tencent ai"],
    "baidu": ["ernie", "ernie 5", "qianfan", "baidu ai"],
    "mistral": ["mistral large", "mistral ai", "le chat", "mnpl"],
    "cohere": ["cohere", "command a", "command r"],
    "nvidia": ["nemotron", "nvidia ai", "jensen huang"],
    "ssi": ["safe superintelligence", "ssi", "sutskever"],
}

HARNESS_BUZZ_KEYS = {
    "claude-code": ["claude code"],
    "copilot": ["github copilot", "copilot"],
    "cursor": ["cursor ide", "cursor editor", "cursor ai", "anysphere"],
    "codex": ["codex cli", "openai codex"],
    "windsurf": ["windsurf"],
    "cline": ["cline ai", "cline bot", "cline x"],
    "devin": ["devin ai", "cognition devin"],
    "gemini-assist": ["code assist", "gemini code"],
    "aider": ["aider"],
    "lingma": ["lingma", "tongyi lingma"],
}


def _buzz(artifacts, sentiments: dict, keys: list) -> dict:
    """Live graph buzz for a lab/harness: mentions, 90-day recency, sentiment."""
    import time as _t
    now = _t.time()
    mentions = 0
    recent = 0
    sents = []
    for a in artifacts:
        hay = f"{a.title or ''} {a.description or ''} {a.tags or ''}".lower()
        if not any(k in hay for k in keys):
            continue
        mentions += 1
        if a.id in sentiments:
            sents.append(sentiments[a.id])
        base = a.date_published or a.created_at
        if base:
            try:
                ts = base.timestamp() if hasattr(base, "timestamp") else 0
                if now - ts < 90 * 86400:
                    recent += 1
            except Exception:
                pass
    avg_sent = round(sum(sents) / len(sents), 2) if sents else 0.0
    if mentions == 0:
        momentum = "quiet"
    elif recent >= 3:
        momentum = "rising"
    elif recent >= 1:
        momentum = "active"
    else:
        momentum = "steady"
    return {"mentions": mentions, "recent_90d": recent,
            "sentiment": avg_sent, "momentum": momentum}


# Home-rig presets for the desktop configurator. usable_vram ≈ what a model
# can actually allocate after OS/framework overhead (unified-memory rigs share
# with the OS; split-VRAM rigs lose a slice to sharding).
FRONTIER_RIGS = [
    {"id": "dgx-spark", "name": "1× DGX Spark 128GB",
     "gpus": "1× GB10 Grace-Blackwell", "vram_total": 128, "usable_vram": 110,
     "gpu_count": 1, "bw_gbs": 300,
     "note": "Unified memory — model shares with OS; ~110GB usable. Multi-GPU sharding not needed."},
    {"id": "2x5080", "name": "2× RTX 5080 (32GB)",
     "gpus": "2× 16GB", "vram_total": 32, "usable_vram": 30,
     "gpu_count": 2, "bw_gbs": 960,
     "note": "Split VRAM — needs a tensor-parallel backend (vLLM/SGLang); single-GPU backends see 16GB only."},
    {"id": "1x5090", "name": "1× RTX 5090 32GB",
     "gpus": "1× 32GB GDDR7", "vram_total": 32, "usable_vram": 30,
     "gpu_count": 1, "bw_gbs": 1792,
     "note": "32GB on one card — fits 70B Q4s solo, no sharding needed."},
    {"id": "2x5090", "name": "2× RTX 5090 (64GB)",
     "gpus": "2× 32GB GDDR7", "vram_total": 64, "usable_vram": 60,
     "gpu_count": 2, "bw_gbs": 1792,
     "note": "Split VRAM — tensor-parallel backend; the home heavyweight."},
    {"id": "1x4090", "name": "1× RTX 4090 24GB",
     "gpus": "1× 24GB", "vram_total": 24, "usable_vram": 22,
     "gpu_count": 1, "bw_gbs": 1008,
     "note": "The classic home-lab card."},
    {"id": "2x4090", "name": "2× RTX 4090 (48GB)",
     "gpus": "2× 24GB", "vram_total": 48, "usable_vram": 45,
     "gpu_count": 2, "bw_gbs": 1008,
     "note": "Split VRAM — tensor-parallel backend; fits most 70B Q4s."},
    {"id": "mac-128", "name": "Mac Studio 128GB",
     "gpus": "Apple Silicon unified", "vram_total": 128, "usable_vram": 100,
     "gpu_count": 1, "bw_gbs": 400,
     "note": "Unified memory via MLX; huge capacity, slower than CUDA."},
]


# Residential electricity rates behind the desktop energy-cost selector.
# Representative 2026 all-in residential tariffs (energy + taxes + fees).
# Users can override with their own rate; costs = kWh × rate, computed live.
ENERGY_RATES = [
    {"id": "nl", "country": "Netherlands", "currency": "EUR", "symbol": "€", "per_kwh": 0.28},
    {"id": "de", "country": "Germany", "currency": "EUR", "symbol": "€", "per_kwh": 0.36},
    {"id": "fr", "country": "France", "currency": "EUR", "symbol": "€", "per_kwh": 0.24},
    {"id": "uk", "country": "United Kingdom", "currency": "GBP", "symbol": "£", "per_kwh": 0.26},
    {"id": "us", "country": "United States", "currency": "USD", "symbol": "$", "per_kwh": 0.18},
    {"id": "ca", "country": "Canada", "currency": "CAD", "symbol": "C$", "per_kwh": 0.16},
    {"id": "au", "country": "Australia", "currency": "AUD", "symbol": "A$", "per_kwh": 0.34},
    {"id": "cn", "country": "China", "currency": "CNY", "symbol": "¥", "per_kwh": 0.55},
    {"id": "in", "country": "India", "currency": "INR", "symbol": "₹", "per_kwh": 8.0},
    {"id": "jp", "country": "Japan", "currency": "JPY", "symbol": "¥", "per_kwh": 31.0},
]


@app.get("/api/frontier-race")
def get_frontier_race(db: Session = Depends(get_db)):
    """6th-tab payload: frontier labs racing to crown AGI/ASI.

    Static curated race card (org, country, size, model, operational license,
    GPU needed to run) enriched live with the graph's median AGI forecast so
    lab targets can be read against community evidence on every call.
    """
    now_year = datetime.now(timezone.utc).year
    projections = db.query(Projection).all()
    mids = []
    for p in projections:
        tf = parse_timeframe(p.timeframe)
        if not tf:
            continue
        if tf[0] < FORECAST_YEAR_MIN or tf[0] > FORECAST_YEAR_MAX:
            continue
        mids.append((tf[0] + tf[1]) / 2)
    mids.sort()
    graph_median = mids[len(mids) // 2] if mids else None

    agi_mids = sorted(l["agi_mid"] for l in FRONTIER_LABS)
    asi_mids = sorted(l["asi_mid"] for l in FRONTIER_LABS)
    artifacts = db.query(Artifact).all()
    sent_map = {s.artifact_id: s.score for s in db.query(Sentiment).all()}
    labs = []
    for l in sorted(FRONTIER_LABS, key=lambda l: -l["crown_score"]):
        merged = dict(l)
        merged.update(FRONTIER_RUN_COMPUTE.get(l["id"], {}))
        merged.update(FRONTIER_MODEL_SPECS.get(l["id"], {}))
        merged["run_tasks"] = _frontier_run_tasks(l["id"], merged.get("run_tier", "api-only"))
        merged["buzz"] = _buzz(artifacts, sent_map, FRONTIER_BUZZ_KEYS.get(l["id"], [l["organization"].lower()]))
        labs.append(merged)
    # Harness crown is LIVE: base adoption blended with 90-day graph buzz and
    # sentiment, re-sorted on every call — Fetch/ingestion moves the crown.
    harnesses = []
    for h in FRONTIER_HARNESSES:
        m = dict(h)
        b = _buzz(artifacts, sent_map, HARNESS_BUZZ_KEYS.get(h["id"], [h["tool"].lower()]))
        m["live_mentions"] = b["mentions"]
        m["live_recent"] = b["recent_90d"]
        m["live_sentiment"] = b["sentiment"]
        m["momentum"] = b["momentum"]
        m["live_adoption"] = round(h["adoption_score"] + min(15, 3 * b["recent_90d"])
                                   + 5 * max(0, b["sentiment"]), 1)
        harnesses.append(m)
    harnesses.sort(key=lambda h: -h["live_adoption"])
    for rank, h in enumerate(harnesses, start=1):
        h["rank"] = rank
    # Desktop subtab (<240B home rigs): scored once on capability-per-hassle,
    # then nudged live by graph buzz like the harness crown.
    desktops = []
    for m in FRONTIER_DESKTOPS:
        d2 = dict(m)
        d2["use"] = _desk_use(m["gpus"])
        d2["home_score"] = _desk_score(m)
        b = _buzz(artifacts, sent_map, m.get("keys", [m["model"].lower()]))
        d2["live_mentions"] = b["mentions"]
        d2["live_recent"] = b["recent_90d"]
        d2["live_sentiment"] = b["sentiment"]
        d2["momentum"] = b["momentum"]
        d2["live_home"] = round(d2["home_score"] + min(8, 2 * b["recent_90d"])
                                + 3 * max(0, b["sentiment"]), 1)
        desktops.append(d2)
    desktops.sort(key=lambda m: -m["live_home"])
    for rank, m in enumerate(desktops, start=1):
        m["rank"] = rank
    for rank, lab in enumerate(labs, start=1):
        lab["rank"] = rank
    open_n = sum(1 for l in FRONTIER_LABS if l["license_type"] in ("open-weights", "hybrid"))
    local_n = sum(1 for l in labs if l.get("run_tier") in ("workstation", "datacenter"))
    countries = sorted({l["country"] for l in FRONTIER_LABS})
    return {
        "now_year": now_year,
        "live": True,
        "labs": labs,
        "leader": labs[0] if labs else None,
        "harnesses": harnesses,
        "harness_leader": harnesses[0] if harnesses else None,
        "desktops": desktops,
        "desktop_leader": desktops[0] if desktops else None,
        "desk_profiles": DESK_PROFILES,
        "rigs": FRONTIER_RIGS,
        "energy_rates": ENERGY_RATES,
        "summary": {
            "lab_count": len(FRONTIER_LABS),
            "country_count": len(countries),
            "countries": countries,
            "open_count": open_n,
            "closed_count": len(FRONTIER_LABS) - open_n,
            "local_run_count": local_n,
            "api_only_count": len(FRONTIER_LABS) - local_n,
            "median_lab_agi": agi_mids[len(agi_mids) // 2] if agi_mids else None,
            "earliest_lab_agi": min(agi_mids) if agi_mids else None,
            "latest_lab_agi": max(agi_mids) if agi_mids else None,
            "median_lab_asi": asi_mids[len(asi_mids) // 2] if asi_mids else None,
            "graph_median_agi": graph_median,
            "graph_forecast_count": len(mids),
            "harness_count": len(harnesses),
            "harness_types": sorted({h["harness_type"] for h in harnesses}),
            "harness_open_count": sum(1 for h in harnesses if h["license_type"] == "open-weights"),
        },
        "timeline": FRONTIER_TIMELINE,
        "roadmap": FRONTIER_PHASES,
        "run_task_defs": RUN_TASK_DEFS,
        "scale": {"min_year": 2017, "max_year": 2040},
        "freshness": {"total_projections": len(projections),
                      "last_ingest": ingestion_stats.get("last_run")},
        "method": ("crown_score blends public benchmark leadership, shipped capability and compute scale "
                   "(Oct 2026 snapshot — closed-lab sizes are estimates). AGI/ASI mids are each lab's stated "
                   "or inferred target window midpoint. Run-GPU figures are inference estimates: full-weight "
                   "FP8/BF16 serving vs cheapest practical Q4 local run (vLLM/SGLang/Ollama); add headroom for "
                   "long context. Per-task power = GPUs × 700W (H100 SXM TDP) + 500W host per ≤8-GPU node, as "
                   "24h kWh; API-only rows are provider-side replica estimates, not your electricity bill. "
                   "LIVE layer: buzz counts artifact mentions per lab/harness with 90-day recency and sentiment; "
                   "harness live_adoption = base score + up to 15 buzz points + up to 5 sentiment points, "
                   "re-ranked every call. Desktop home_score = capability − 6/extra-GPU − 3 custom-license "
                   "+ 2 tiny-MoE bonus, nudged live by buzz (max 8 + 3). Rig fit: Q4 need ≈ disk + 4GB, "
                   "FP16 need ≈ 2× params + 6GB; ✅ fits usable VRAM, ⚠️ within 20% over (drop to Q3/offload), "
                   "❌ beyond that. TPS ≈ 0.5 × rig bandwidth ÷ active-weight bytes per token (Q4 decode-bound), "
                   "shown as a ±30% range — estimates, not benchmarks. Energy costs = kWh × the selected country tariff "
                   "(representative residential incl. taxes; override with your own). "
                   "Graph median AGI is recomputed live from projections."),
        "caveats": ("Closed-lab parameter counts and compute are undisclosed — sizes marked ~ are press/analyst "
                    "estimates. Run-GPU needs vary with quantization, context length and framework; treat them "
                    "as planning bands, not vendor specs. Buzz reflects this graph's corpus (curated + ingested), "
                    "not the whole internet — press Fetch or wait for the daily run to freshen it. Licenses change "
                    "per release; check the vendor model card before reuse."),
    }


# ---------------------------------------------------------- discoveries ---
# Attention queue + new arrivals + influential findings. All derived live from
# the graph so the Discoveries landing view stays fresh as ingestion runs.

def _mini_with_sent(a: Artifact, sent_map: dict) -> dict:
    d = {"id": a.id, "title": a.title, "artifact_type": a.artifact_type,
         "author": a.author,
         "date": a.date_published.isoformat() if a.date_published else None,
         "created": a.created_at.isoformat() if a.created_at else None,
         "tags": a.tags or ""}
    if a.id in sent_map:
        d["sentiment"] = round(sent_map[a.id], 2)
    return d


@app.get("/api/discoveries")
def get_discoveries(db: Session = Depends(get_db)):
    """Items needing immediate attention, newest arrivals, and the most
    influential findings — ranked live from projections, recency, graph degree,
    and whistleblower signals."""
    now = datetime.now(timezone.utc)
    artifacts = db.query(Artifact).all()
    projections = db.query(Projection).all()
    sent_map = {s.artifact_id: s.score for s in db.query(Sentiment).all()}

    degree: dict[int, int] = {}
    for r in db.query(Relationship).all():
        degree[r.source_id] = degree.get(r.source_id, 0) + 1
        degree[r.target_id] = degree.get(r.target_id, 0) + 1

    def age_days(a: Artifact) -> float:
        base = a.date_published or a.created_at
        if not base:
            return 9999.0
        try:
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return max(0.0, (now - base).total_seconds() / 86400)
        except Exception:
            return 9999.0

    # --- Attention queue -------------------------------------------------
    attention = []
    for p in projections:
        if p.projection_type not in ("dystopian", "cautionary") or (p.confidence or 0) < 0.6:
            continue
        a = next((x for x in artifacts if x.id == p.artifact_id), None)
        if not a:
            continue
        age = age_days(a)
        urgency = round((p.confidence or 0) * 10
                        + (2 if p.projection_type == "dystopian" else 1)
                        + (2 if age <= 90 else (1 if age <= 365 else 0)), 1)
        reasons = [f"{p.projection_type} outlook at {int((p.confidence or 0) * 100)}% confidence"]
        if age <= 90:
            reasons.append("published in the last 90 days")
        if "whistleblower" in (a.tags or "").lower():
            reasons.append("corroborated by insider testimony")
            urgency += 1
        attention.append({"kind": "risk-projection", "urgency": urgency,
                          "reasons": reasons, "timeframe": p.timeframe,
                          "summary": _clean_text(p.summary, 220), "artifact": _mini_with_sent(a, sent_map)})
    for a in artifacts:
        if "whistleblower" not in (a.tags or "").lower():
            continue
        if age_days(a) > 730:
            continue
        attention.append({"kind": "whistleblower", "urgency": 9.0,
                          "reasons": ["insider warning from a frontier lab"],
                          "timeframe": None, "summary": _clean_text(a.description, 220),
                          "artifact": _mini_with_sent(a, sent_map)})
    seen_art = set()
    deduped = []
    for item in sorted(attention, key=lambda x: -x["urgency"]):
        aid = item["artifact"]["id"]
        if aid in seen_art:
            continue
        seen_art.add(aid)
        deduped.append(item)
    attention = deduped[:8]

    # --- New arrivals -----------------------------------------------------
    def _recency_key(a: Artifact):
        base = a.created_at or a.date_published
        if not base:
            return datetime.min.replace(tzinfo=timezone.utc)
        return base if base.tzinfo else base.replace(tzinfo=timezone.utc)

    fresh = sorted(artifacts, key=_recency_key, reverse=True)[:10]
    new_arrivals = []
    for a in fresh:
        age = age_days(a)
        new_arrivals.append({**_mini_with_sent(a, sent_map),
                             "age_label": "today" if age < 1 else (f"{int(age)}d ago" if age < 30 else (f"{int(age / 30)}mo ago" if age < 730 else "archive")),
                             "is_new": age <= 14,
                             "connections": degree.get(a.id, 0)})

    # --- Influential findings ----------------------------------------------
    ranked = sorted(artifacts, key=lambda a: (-degree.get(a.id, 0),
                                              -(abs(sent_map.get(a.id, 0)) or 0)))[:8]
    influential = []
    for a in ranked:
        projs = [{"type": p.projection_type, "confidence": p.confidence}
                 for p in projections if p.artifact_id == a.id]
        influential.append({**_mini_with_sent(a, sent_map),
                            "degree": degree.get(a.id, 0),
                            "projections": projs[:2]})

    def _age_seconds(a: Artifact) -> float:
        base = a.created_at
        if not base:
            return 10 ** 12
        try:
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return (now - base).total_seconds()
        except Exception:
            return 10 ** 12

    week_new = sum(1 for a in artifacts if _age_seconds(a) < 7 * 86400)

    # --- Analytics & storytelling -----------------------------------------
    def _pub_dt(a: Artifact):
        base = a.date_published or a.created_at
        if not base:
            return None
        try:
            return base if base.tzinfo else base.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    pub_dts = [d for d in (_pub_dt(a) for a in artifacts) if d]

    # Pulse: weekly ingestion counts over the last 12 weeks (+ previous 4 for deltas)
    weeks = []
    for k in range(15, -1, -1):
        w_end = now - timedelta(days=7 * k)
        w_start = w_end - timedelta(days=7)
        c = sum(1 for a in artifacts
                if a.created_at and w_start <= (a.created_at if a.created_at.tzinfo
                                                else a.created_at.replace(tzinfo=timezone.utc)) < w_end)
        weeks.append({"week_start": w_start.strftime("%Y-%m-%d"), "label": w_start.strftime("%b %d"),
                      "count": c})
    recent4 = sum(w["count"] for w in weeks[-4:])
    prior4 = sum(w["count"] for w in weeks[-8:-4])
    delta_pct = round((recent4 - prior4) / prior4 * 100) if prior4 else (100 if recent4 else 0)

    # Outlook mix across all projections
    type_counts: dict[str, int] = {}
    for p in projections:
        type_counts[p.projection_type] = type_counts.get(p.projection_type, 0) + 1

    # Weighted discourse mood (-1..+1)
    scores = [s.score for s in db.query(Sentiment).all() if s.score is not None]
    mean_sent = round(sum(scores) / len(scores), 3) if scores else 0.0

    # Topic clusters from tags of recent (<=365d) artifacts
    tag_recs: dict[str, list] = {}
    for a in artifacts:
        age = age_days(a)
        if age > 365:
            continue
        for t in re.split(r"[,;]\s*|\s+", (a.tags or "").lower()):
            t = t.strip(" .\"'")
            if len(t) < 3 or t in ("the", "and", "with", "from", "new"):
                continue
            e = tag_recs.setdefault(t, [0, 0.0, 0.0])
            e[0] += 1
            e[1] += age
            e[2] += abs(sent_map.get(a.id, 0) or 0)
    topics = sorted(
        [{"topic": t, "count": v[0], "avg_age_days": round(v[1] / v[0]),
          "heat": round(v[0] * (1 + v[2] / v[0])),
          "trend": "new" if v[1] / v[0] < 60 else ("rising" if v[1] / v[0] < 180 else "steady")}
         for t, v in tag_recs.items() if v[0] >= 2],
        key=lambda x: -x["heat"])[:10]

    # Velocity: monthly publication volume over last 12 months
    vel_months = []
    for k in range(11, -1, -1):
        m_end = now - timedelta(days=30 * k)
        m_start = now - timedelta(days=30 * (k + 1))
        c = sum(1 for d in pub_dts if m_start <= d < m_end)
        vel_months.append({"month": m_end.strftime("%b"), "label": m_end.strftime("%b %y"), "count": c})

    # --- Narrative generator: one coherent paragraph about what changed ----
    total = len(artifacts)
    parts = [f"The archive now holds {total} artifact" + ("" if total == 1 else "s") +
             f" · {len(projections)} outlooks · {sum(degree.values()) // 2 if degree else 0} links."]
    if recent4:
        cmp_txt = (f"a {abs(delta_pct)}% " + ("acceleration" if delta_pct >= 0 else "slowdown")
                   + " vs the previous month") if prior4 else "as ingestion ramps up"
        parts.append(f"{recent4} arrived in the last 4 weeks ({cmp_txt}).")
    top_t = topics[0] if topics else None
    if top_t:
        parts.append(f"'{top_t['topic']}' is the hottest thread — {top_t['count']} recent items, {top_t['trend']}.")
    if attention:
        lead = attention[0]
        la = lead["artifact"]
        parts.append(f"Top of the watchlist: “{_clean_text(la['title'], 60)}” — "
                     f"{', '.join(lead['reasons'][:2])}"
                     + (f", horizon {lead['timeframe']}" if lead.get("timeframe") else "") + ".")
    if scores:
        tone = "alarmed" if mean_sent < -0.3 else ("cautious" if mean_sent < -0.05
               else ("bullish" if mean_sent > 0.2 else "mixed"))
        parts.append(f"Discourse reads {tone} (mean sentiment {mean_sent:+.2f}).")
    dyst = type_counts.get("dystopian", 0)
    utop = type_counts.get("utopian", 0)
    if dyst or utop:
        parts.append(f"Outlook mix tilts toward "
                     + ("risk" if dyst >= utop else "promise")
                     + f" ({dyst} dystopian vs {utop} utopian projections).")

    return {"attention": attention, "attention_count": len(attention),
            "new_arrivals": new_arrivals, "new_this_week": week_new,
            "influential": influential,
            "max_degree": max(degree.values()) if degree else 0,
            "story": " ".join(parts),
            "pulse": {"weeks": weeks[4:], "recent4": recent4, "prior4": prior4,
                      "delta_pct": delta_pct},
            "totals": {"artifacts": total, "projections": len(projections),
                       "links": sum(degree.values()) // 2 if degree else 0,
                       "whistleblower": sum(1 for a in artifacts
                                            if "whistleblower" in (a.tags or "").lower())},
            "outlook_mix": [{"type": t, "count": c} for t, c in sorted(type_counts.items(),
                                                                       key=lambda x: -x[1])],
            "sentiment": {"mean": mean_sent, "n": len(scores)},
            "topics": topics,
            "velocity": vel_months,
            "generated_at": now.isoformat()}


@app.post("/api/relationships", dependencies=[Depends(require_admin)])
def create_relationship(data: RelationshipCreate, db: Session = Depends(get_db)):
    src = db.query(Artifact).filter(Artifact.id == data.source_id).first()
    tgt = db.query(Artifact).filter(Artifact.id == data.target_id).first()
    if not src or not tgt:
        raise HTTPException(404, "Source or target artifact not found")

    rel = Relationship(
        source_id=data.source_id,
        target_id=data.target_id,
        relationship_type=data.relationship_type,
        description=data.description,
    )
    db.add(rel)
    db.commit()
    db.refresh(rel)
    return {
        "id": rel.id,
        "source_id": rel.source_id,
        "target_id": rel.target_id,
        "relationship_type": rel.relationship_type,
        "description": rel.description,
    }


@app.get("/api/ingest/status")
def ingest_status():
    return ingestion_stats


@app.post("/api/ingest/run", dependencies=[Depends(require_admin)])
def run_scraper_manual(db: Session = Depends(get_db)):
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(429, "Ingestion already in progress")
    ingestion_lock.release()

    def _background_ingest():
        if not ingestion_lock.acquire(blocking=False):
            return
        db_bg = next(get_db())
        try:
            count = run_ingestion(db_bg)
            ingestion_stats["last_run"] = datetime.now(timezone.utc).isoformat()
            ingestion_stats["items_fetched"] += count
            ingestion_stats["total_runs"] += 1
        finally:
            db_bg.close()
            ingestion_lock.release()

    import threading
    threading.Thread(target=_background_ingest, daemon=True).start()
    return {"ingested": -1, "status": "started", "total_runs": ingestion_stats["total_runs"]}


@app.post("/api/reseed", dependencies=[Depends(require_admin)])
def reseed(db: Session = Depends(get_db)):
    db.query(Relationship).delete()
    db.query(Projection).delete()
    db.query(Sentiment).delete()
    db.query(Artifact).delete()
    db.commit()
    seed_database(db)
    return {"ok": True, "seeded": db.query(Artifact).count()}


@app.get("/api/artifacts/{artifact_id}/similar")
def similar_artifacts(artifact_id: int, limit: int = Query(8, le=20), db: Session = Depends(get_db)):
    a = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if not a:
        raise HTTPException(404, "Artifact not found")

    a_tags = set(t.strip().lower() for t in (a.tags or "").split(",") if t.strip())
    all_others = db.query(Artifact).filter(Artifact.id != artifact_id).all()

    scored = []
    for other in all_others:
        score = 0.0
        reasons = []
        shared_tags = []

        o_tags = set(t.strip().lower() for t in (other.tags or "").split(",") if t.strip())
        if a_tags and o_tags:
            overlap = a_tags & o_tags
            union = a_tags | o_tags
            jaccard = len(overlap) / len(union) if union else 0
            if jaccard > 0:
                score += jaccard * 0.4
                shared_tags = list(overlap)
                reasons.append(f"{len(overlap)} shared tags")

        if other.author and a.author and other.author.lower() == a.author.lower():
            score += 0.3
            reasons.append("same author")

        if other.artifact_type == a.artifact_type:
            score += 0.15
            reasons.append("same type")

        rels = (
            db.query(Relationship)
            .filter(
                ((Relationship.source_id == a.id) & (Relationship.target_id == other.id))
                | ((Relationship.target_id == a.id) & (Relationship.source_id == other.id))
            )
            .all()
        )
        if rels:
            score += 0.3
            reasons.append(f"connected via {rels[0].relationship_type}")

        if other.description and a.description:
            a_words = set(a.description.lower().split())
            o_words = set(other.description.lower().split())
            word_overlap = len(a_words & o_words)
            if word_overlap > 10:
                score += 0.15
                reasons.append("content overlap")

        if score > 0:
            scored.append((round(score, 2), reasons, other, shared_tags))

    scored.sort(key=lambda x: -x[0])
    top = scored[:limit]

    return {
        "items": [
            {
                **_artifact_json(s),
                "similarity": sc,
                "reasons": reasons,
                "shared_tags": st,
            }
            for sc, reasons, s, st in top
        ]
    }


@app.get("/api/search")
def search(q: str = Query(...), db: Session = Depends(get_db)):
    results = (
        db.query(Artifact)
        .filter(
            Artifact.title.ilike(f"%{q}%")
            | Artifact.description.ilike(f"%{q}%")
            | Artifact.author.ilike(f"%{q}%")
            | Artifact.tags.ilike(f"%{q}%")
        )
        .limit(20)
        .all()
    )
    return {"items": [_artifact_json(a) for a in results]}


@app.post("/api/security/assess")
def security_assess(data: SecurityAssessRequest, db: Session = Depends(get_db)):
    """AI Security Engineering and Evaluation Agent — run a deep assessment.

    Runs the agent-to-agent workflow (orchestrator → research-collector →
    threat-intel → report-writer) with agentic in-app search, so the report's
    executive summary and Known Exploits section cite related research from
    this app's own knowledge graph.
    """
    import json as _json

    _ensure_security_columns()
    result = sec_agent.build_assessment(
        product_name=data.product_name or "Target product",
        product_url=data.product_url or "",
        exposure=data.exposure or "confidential_data",
        use_case=data.use_case or "",
        workflow_text=data.workflow_text or "",
        doc_urls=data.doc_urls or [],
        focus=data.focus or [],
        db=db,
    )
    rec = SecurityAssessment(
        product_name=result["product_name"],
        product_url=result["product_url"],
        exposure=result["exposure"],
        use_case=data.use_case or "",
        overall_pct=result["overall_pct"],
        posture=result["posture"],
        markdown=result["markdown"],
        diagrams_json=_json.dumps(result["diagrams"]),
        threats_json=_json.dumps(result["threats"]),
        evidence_json=_json.dumps({
            "evidence": result.get("evidence", []),
            "queries_run": result.get("queries_run", []),
            "known_exploits": result.get("known_exploits", []),
        }),
        a2a_trace_json=_json.dumps({
            "task_id": result.get("a2a_task_id", ""),
            "trace": result.get("a2a_trace", []),
        }),
        version=1,
        status="draft",
        hypotheses_json=_json.dumps([]),
        reviews_json=_json.dumps([]),
        ratings_json=_json.dumps([]),
        redteam_json=_json.dumps({}),
        scenarios_json=_json.dumps([]),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {
        "id": rec.id,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        **result,
    }


def _security_json(rec: SecurityAssessment):
    import json as _json

    def _load(raw, default):
        try:
            return _json.loads(raw) if raw else default
        except Exception:
            return default

    diagrams = _load(rec.diagrams_json, {})
    threats = _load(rec.threats_json, [])
    evidence_blob = _load(getattr(rec, "evidence_json", None), {})
    trace_blob = _load(getattr(rec, "a2a_trace_json", None), {})
    return {
        "id": rec.id,
        "product_name": rec.product_name,
        "product_url": rec.product_url,
        "exposure": rec.exposure,
        "use_case": rec.use_case,
        "overall_pct": rec.overall_pct,
        "posture": rec.posture,
        "markdown": rec.markdown,
        "diagrams": diagrams,
        "threats": threats,
        "evidence": evidence_blob.get("evidence", []) if isinstance(evidence_blob, dict) else [],
        "queries_run": evidence_blob.get("queries_run", []) if isinstance(evidence_blob, dict) else [],
        "known_exploits": evidence_blob.get("known_exploits", []) if isinstance(evidence_blob, dict) else [],
        "a2a_task_id": trace_blob.get("task_id", "") if isinstance(trace_blob, dict) else "",
        "a2a_trace": trace_blob.get("trace", []) if isinstance(trace_blob, dict) else [],
        "version": getattr(rec, "version", 1) or 1,
        "parent_id": getattr(rec, "parent_id", None),
        "status": getattr(rec, "status", "draft") or "draft",
        "hypotheses": _load(getattr(rec, "hypotheses_json", None), []),
        "reviews": _load(getattr(rec, "reviews_json", None), []),
        "ratings": _load(getattr(rec, "ratings_json", None), []),
        "redteam": _load(getattr(rec, "redteam_json", None), {}),
        "scenarios": _load(getattr(rec, "scenarios_json", None), []),
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
    }


def _ensure_security_columns():
    """Add evidence/a2a + research-collaboration columns to pre-existing SQLite DBs."""
    try:
        from sqlalchemy import text as _text
        from .database import engine as _engine
        with _engine.begin() as conn:
            cols = [r[1] for r in conn.execute(_text("PRAGMA table_info(security_assessments)")).fetchall()]
            for name, ddl in [
                ("evidence_json", "TEXT"), ("a2a_trace_json", "TEXT"),
                ("version", "INTEGER"), ("parent_id", "INTEGER"),
                ("status", "VARCHAR(50)"), ("hypotheses_json", "TEXT"),
                ("reviews_json", "TEXT"), ("ratings_json", "TEXT"),
                ("redteam_json", "TEXT"), ("scenarios_json", "TEXT"),
            ]:
                if name not in cols:
                    conn.execute(_text(f"ALTER TABLE security_assessments ADD COLUMN {name} {ddl}"))
    except Exception:
        pass  # table may not exist yet on first boot; init_db creates it with columns


@app.get("/.well-known/agents")
def well_known_agents():
    """A2A agent registry (discovery document)."""
    from .agents import get_agent_cards, PROTOCOL

    return {"protocol": PROTOCOL, "agents": get_agent_cards()}


@app.get("/api/agents/cards")
def agent_cards():
    """List A2A agent cards (capabilities + endpoints)."""
    from .agents import get_agent_cards, PROTOCOL

    return {"protocol": PROTOCOL, "agents": get_agent_cards()}


class AgentInvokeRequest(BaseModel):
    to: str = "research-collector"
    intent: str = "collect_research"
    payload: Optional[dict] = None
    task_id: Optional[str] = None


@app.post("/api/agents/invoke")
def agent_invoke(data: AgentInvokeRequest, db: Session = Depends(get_db)):
    """Invoke one A2A protocol hop directly (debugging / agent-to-agent calls)."""
    from .agents import new_envelope, dispatch

    env = new_envelope(
        "api-caller", data.to, data.intent, data.payload or {},
        task_id=data.task_id, note="direct invoke",
    )
    try:
        from .sandbox import audit as _audit
        _audit("api-caller", f"{data.to}.{data.intent}", data.to, "invoked")
    except Exception:
        pass
    try:
        res = dispatch(env, db)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return res


@app.get("/api/agents/protection")
def agent_protection():
    """Protection posture: OpenShell supervision detection + guardrail policy."""
    from .sandbox import protection_status
    return protection_status()


@app.get("/api/agents/ledger")
def agent_ledger(limit: int = Query(50, le=200)):
    """Ledger entries (hash-chained, persistent), newest first."""
    from .ledger import recent
    return {"items": recent(limit)}


@app.get("/api/agents/ledger/verify")
def agent_ledger_verify():
    """Verify the full hash chain. Poll count/head to detect tail truncation."""
    from .ledger import verify
    return verify()


@app.get("/api/agents/audit")
def agent_audit(limit: int = Query(50, le=200)):
    """Recent agent tool-use audit trail (fetch/invoke, newest first)."""
    from .sandbox import recent_audit
    return {"items": recent_audit(limit)}


@app.get("/api/security/assessments")
def security_list(db: Session = Depends(get_db)):
    recs = db.query(SecurityAssessment).order_by(SecurityAssessment.id.desc()).limit(50).all()
    return {"items": [
        {k: v for k, v in _security_json(r).items() if k != "markdown"} | {"threat_count": len(_security_json(r)["threats"])}
        for r in recs
    ]}


@app.get("/api/security/assessments/{assessment_id}")
def security_get(assessment_id: int, db: Session = Depends(get_db)):
    rec = db.query(SecurityAssessment).filter(SecurityAssessment.id == assessment_id).first()
    if not rec:
        raise HTTPException(404, "Assessment not found")
    return _security_json(rec)


@app.get("/api/security/assessments/{assessment_id}/markdown")
def security_markdown(assessment_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import PlainTextResponse

    rec = db.query(SecurityAssessment).filter(SecurityAssessment.id == assessment_id).first()
    if not rec:
        raise HTTPException(404, "Assessment not found")
    return PlainTextResponse(rec.markdown, media_type="text/markdown")


@app.get("/api/security/assessments/{assessment_id}/pdf")
def security_pdf(assessment_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import Response

    rec = db.query(SecurityAssessment).filter(SecurityAssessment.id == assessment_id).first()
    if not rec:
        raise HTTPException(404, "Assessment not found")
    try:
        pdf = sec_agent.build_pdf(rec.markdown, title=f"AI Security Assessment — {rec.product_name}")
    except RuntimeError as exc:
        raise HTTPException(501, str(exc))
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=security-assessment-{rec.id}.pdf"})


# ------------------------------------------------- security analytics ---
# PhD-research endpoints: charts data, coverage/gap analysis, what-if lab,
# comparison, exports, kill-chains, red-team, hypotheses, reviews, ratings.

def _get_assessment_or_404(assessment_id: int, db: Session) -> SecurityAssessment:
    rec = db.query(SecurityAssessment).filter(SecurityAssessment.id == assessment_id).first()
    if not rec:
        raise HTTPException(404, "Assessment not found")
    return rec


@app.get("/api/security/coverage")
def security_coverage():
    """Static framework reference: OWASP LLM 2025, NIST AI RMF, LINDDUN,
    per-threat mappings, kill-chain templates, red-team suite."""
    from . import security_analytics as ana
    return {
        "owasp_llm_2025": ana.OWASP_LLM_2025,
        "nist_ai_rmf": ana.NIST_AI_RMF_FUNCTIONS,
        "linddun": ana.LINDDUN_CATEGORIES,
        "threat_frameworks": ana.THREAT_FRAMEWORKS,
        "kill_chain_templates": ana.KILL_CHAINS,
        "redteam_suite": ana.REDTEAM_SUITE,
    }


@app.get("/api/security/assessments/{assessment_id}/analytics")
def security_analytics(assessment_id: int, db: Session = Depends(get_db)):
    """Charts + gap-analysis + uncertainty payload for one assessment."""
    import json as _json
    from . import security_analytics as ana
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    data = _security_json(rec)
    threats = data["threats"]
    return {
        "id": rec.id,
        "overall_pct": data["overall_pct"],
        "severity_distribution": ana.severity_distribution(threats),
        "stride_breakdown": ana.stride_breakdown(threats),
        "heatmap": ana.heatmap_grid(threats),
        "framework_coverage": ana.framework_coverage(threats),
        "evidence_grades": ana.grade_evidence_list(data["evidence"]),
        "sensitivity": ana.sensitivity_tornado(threats),
        "monte_carlo": ana.monte_carlo(threats),
        "killchains": ana.compose_killchains(threats),
    }


class WhatIfRequest(BaseModel):
    likelihood_delta: Optional[dict[str, int]] = None


@app.post("/api/security/assessments/{assessment_id}/whatif")
def security_whatif(assessment_id: int, data: WhatIfRequest, db: Session = Depends(get_db)):
    """Counterfactual lab: apply per-threat likelihood deltas, recompute overall."""
    from . import security_analytics as ana
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    threats = _security_json(rec)["threats"]
    return {"id": rec.id, **ana.recompute_whatif(threats, data.likelihood_delta or {})}


@app.get("/api/security/compare")
def security_compare(ids: str = Query(...), db: Session = Depends(get_db)):
    """Compare assessments: overall trend + per-threat score deltas."""
    _ensure_security_columns()
    try:
        wanted = [int(x) for x in ids.split(",") if x.strip()][:8]
    except ValueError:
        raise HTTPException(400, "ids must be comma-separated integers")
    recs = [db.query(SecurityAssessment).filter(SecurityAssessment.id == i).first() for i in wanted]
    recs = [r for r in recs if r]
    if not recs:
        raise HTTPException(404, "No matching assessments")
    payloads = [_security_json(r) for r in recs]
    tids = sorted({t["id"] for p in payloads for t in p["threats"]})
    matrix = []
    for tid in tids:
        row = {"id": tid, "title": next((t["title"] for p in payloads for t in p["threats"] if t["id"] == tid), tid)}
        for p in payloads:
            hit = next((t for t in p["threats"] if t["id"] == tid), None)
            row[f"a{p['id']}"] = hit["score"] if hit else None
        scores = [row[f"a{p['id']}"] for p in payloads if row[f"a{p['id']}"] is not None]
        row["delta"] = (scores[-1] - scores[0]) if len(scores) >= 2 else 0
        matrix.append(row)
    return {
        "assessments": [{"id": p["id"], "product_name": p["product_name"],
                         "overall_pct": p["overall_pct"], "status": p["status"],
                         "version": p["version"], "created_at": p["created_at"]} for p in payloads],
        "threat_matrix": matrix,
    }


@app.get("/api/security/assessments/{assessment_id}/killchain")
def security_killchain(assessment_id: int, db: Session = Depends(get_db)):
    from . import security_analytics as ana
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    return {"id": rec.id, "killchains": ana.compose_killchains(_security_json(rec)["threats"])}


@app.get("/api/security/assessments/{assessment_id}/threats.csv")
def security_threats_csv(assessment_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import Response
    import csv, io
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "title", "stride", "owasp", "likelihood", "impact", "score", "severity"])
    for t in _security_json(rec)["threats"]:
        w.writerow([t["id"], t["title"], t["stride"], t["owasp"],
                    t["likelihood"], t["impact"], t["score"], t["severity"]])
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=threats-{rec.id}.csv"})


@app.get("/api/security/assessments/{assessment_id}/evidence.bib")
def security_evidence_bib(assessment_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import PlainTextResponse
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    lines = []
    for e in _security_json(rec)["evidence"]:
        key = f"artifact{e.get('artifact_id', 'x')}"
        title = (e.get("title") or "untitled").replace("{", "").replace("}", "")
        year = str(e.get("date_published") or "")[:4] or "nodate"
        lines.append(f"@misc{{{key},\n  title={{{title}}},\n  year={{{year}}},\n  note={{artifact #{e.get('artifact_id')}}}\n}}")
    return PlainTextResponse("\n\n".join(lines) or "% no evidence", media_type="text/plain")


@app.get("/api/security/assessments/{assessment_id}/export.zip")
def security_export_zip(assessment_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import Response
    import csv, io, json as _json, zipfile
    from . import security_analytics as ana
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    data = _security_json(rec)
    analytics = {"severity": ana.severity_distribution(data["threats"]),
                 "framework_coverage": ana.framework_coverage(data["threats"]),
                 "monte_carlo": ana.monte_carlo(data["threats"])}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.md", data["markdown"] or "")
        z.writestr("assessment.json", _json.dumps(data, indent=2, default=str))
        z.writestr("analytics.json", _json.dumps(analytics, indent=2, default=str))
        tbuf = io.StringIO()
        w = csv.writer(tbuf)
        w.writerow(["id", "title", "stride", "owasp", "likelihood", "impact", "score", "severity"])
        for t in data["threats"]:
            w.writerow([t["id"], t["title"], t["stride"], t["owasp"],
                        t["likelihood"], t["impact"], t["score"], t["severity"]])
        z.writestr("threats.csv", tbuf.getvalue())
        z.writestr("diagrams.json", _json.dumps(data["diagrams"], indent=2))
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=security-assessment-{rec.id}.zip"})


class RedTeamSubmit(BaseModel):
    results: list[dict] = []


@app.post("/api/security/assessments/{assessment_id}/redteam")
def security_redteam(assessment_id: int, data: RedTeamSubmit, db: Session = Depends(get_db)):
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from . import security_analytics as ana
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    suite_ids = {s["id"] for s in ana.REDTEAM_SUITE}
    cleaned = [{"id": r.get("id"), "verdict": (r.get("verdict") or "").lower(),
                "notes": r.get("notes", "")}
               for r in (data.results or []) if r.get("id") in suite_ids
               and (r.get("verdict") or "").lower() in ("pass", "fail")]
    blob = {"suite_version": "1.0", "results": cleaned,
            "logged_at": _dt.now(_tz.utc).isoformat()}
    rec.redteam_json = _json.dumps(blob)
    db.commit()
    passed = sum(1 for r in cleaned if r["verdict"] == "pass")
    return {"id": rec.id, "total": len(cleaned), "passed": passed,
            "failed": len(cleaned) - passed, "redteam": blob}


class HypothesesSubmit(BaseModel):
    hypotheses: list[dict] = []


@app.post("/api/security/assessments/{assessment_id}/hypotheses")
def security_hypotheses(assessment_id: int, data: HypothesesSubmit, db: Session = Depends(get_db)):
    import json as _json
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    cleaned = [{"id": h.get("id") or f"H{i+1}", "statement": h.get("statement", ""),
                "prediction": h.get("prediction", ""), "status": h.get("status", "open"),
                "observation": h.get("observation", "")}
               for i, h in enumerate(data.hypotheses or [])]
    rec.hypotheses_json = _json.dumps(cleaned)
    db.commit()
    return {"id": rec.id, "hypotheses": cleaned}


class ReviewSubmit(BaseModel):
    reviewer: str = ""
    verdict: str = "comment"
    comments: str = ""


@app.post("/api/security/assessments/{assessment_id}/reviews")
def security_reviews(assessment_id: int, data: ReviewSubmit, db: Session = Depends(get_db)):
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    try:
        reviews = _json.loads(rec.reviews_json) if rec.reviews_json else []
    except Exception:
        reviews = []
    reviews.append({"reviewer": data.reviewer or "anonymous",
                    "verdict": data.verdict, "comments": data.comments,
                    "at": _dt.now(_tz.utc).isoformat()})
    rec.reviews_json = _json.dumps(reviews)
    verdicts = [r["verdict"] for r in reviews]
    if verdicts.count("approve") >= 2:
        rec.status = "published"
    elif "request-changes" in verdicts:
        rec.status = "changes-requested"
    elif reviews:
        rec.status = "in-review"
    db.commit()
    return {"id": rec.id, "status": rec.status, "reviews": reviews}


class RatingSubmit(BaseModel):
    rater: str = ""
    likelihood: Optional[dict[str, int]] = None
    impact: Optional[dict[str, int]] = None


@app.post("/api/security/assessments/{assessment_id}/ratings")
def security_ratings(assessment_id: int, data: RatingSubmit, db: Session = Depends(get_db)):
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from . import security_analytics as ana
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    try:
        ratings = _json.loads(rec.ratings_json) if rec.ratings_json else []
    except Exception:
        ratings = []
    ratings.append({"rater": data.rater or "anonymous",
                    "likelihood": data.likelihood or {}, "impact": data.impact or {},
                    "at": _dt.now(_tz.utc).isoformat()})
    rec.ratings_json = _json.dumps(ratings)
    db.commit()
    out: dict = {"id": rec.id, "ratings_count": len(ratings)}
    if len(ratings) >= 2:
        tids = sorted(set(ratings[0].get("likelihood", {})) | set(ratings[1].get("likelihood", {})))
        la = [int(ratings[0].get("likelihood", {}).get(t, 3)) for t in tids]
        lb = [int(ratings[1].get("likelihood", {}).get(t, 3)) for t in tids]
        ia = [int(ratings[0].get("impact", {}).get(t, 3)) for t in tids]
        ib = [int(ratings[1].get("impact", {}).get(t, 3)) for t in tids]
        out["kappa_likelihood"] = ana.cohen_kappa(la, lb, categories=[1, 2, 3, 4, 5])
        out["kappa_impact"] = ana.cohen_kappa(ia, ib, categories=[1, 2, 3, 4, 5])
    return out


@app.post("/api/security/assessments/{assessment_id}/clone")
def security_clone(assessment_id: int, db: Session = Depends(get_db)):
    import json as _json
    _ensure_security_columns()
    rec = _get_assessment_or_404(assessment_id, db)
    child = SecurityAssessment(
        product_name=rec.product_name, product_url=rec.product_url,
        exposure=rec.exposure, use_case=rec.use_case,
        overall_pct=rec.overall_pct, posture=rec.posture, markdown=rec.markdown,
        diagrams_json=rec.diagrams_json, threats_json=rec.threats_json,
        evidence_json=rec.evidence_json, a2a_trace_json=rec.a2a_trace_json,
        version=(rec.version or 1) + 1, parent_id=rec.id, status="draft",
        hypotheses_json=rec.hypotheses_json, reviews_json=_json.dumps([]),
        ratings_json=_json.dumps([]), redteam_json=_json.dumps({}),
        scenarios_json=rec.scenarios_json,
    )
    db.add(child)
    db.commit()
    db.refresh(child)
    return _security_json(child)


def _artifact_json(a: Artifact):
    return {
        "id": a.id,
        "title": a.title,
        "artifact_type": a.artifact_type,
        "url": a.url,
        "description": a.description,
        "content": a.content,
        "source": a.source,
        "author": a.author,
        "date_published": a.date_published.isoformat() if a.date_published else None,
        "tags": a.tags,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }
