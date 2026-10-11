import os
import re
import atexit
import threading
from collections import defaultdict
from datetime import datetime, timezone, datetime as dt
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

    return {
        "now_year": now_year,
        "where_we_are": {
            "forecast_count": len(forecasts),
            "median_agi_year": _pct(mids, 50),
            "spread_p25": _pct(mids, 25),
            "spread_p75": _pct(mids, 75),
            "type_distribution": type_counts,
            "by_year": _timeline_histogram(mids),
        },
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
    return {"attention": attention, "attention_count": len(attention),
            "new_arrivals": new_arrivals, "new_this_week": week_new,
            "influential": influential,
            "max_degree": max(degree.values()) if degree else 0,
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
