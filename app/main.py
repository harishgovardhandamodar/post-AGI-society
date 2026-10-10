import os
import re
import atexit
import threading
from collections import defaultdict
from datetime import datetime, timezone, datetime as dt
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from apscheduler.schedulers.background import BackgroundScheduler

from .database import init_db, get_db
from .models import Artifact, Sentiment, Projection, Relationship, ArtifactType, SecurityAssessment
from .seed_data import seed_database
from .scraper import run_ingestion
from . import security as sec_agent

ingestion_stats = {"last_run": None, "items_fetched": 0, "total_runs": 0}
ingestion_lock = threading.Lock()

os.makedirs("data", exist_ok=True)

app = FastAPI(title="Post-AGI Society Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
scheduler.add_job(run_scheduled_ingestion, "cron", hour=4, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))


@app.on_event("startup")
def on_start():
    init_db()
    _ensure_security_columns()
    db = next(get_db())
    try:
        seed_database(db)
    finally:
        db.close()


class ArtifactCreate(BaseModel):
    title: str
    artifact_type: str
    url: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    date_published: Optional[datetime] = None
    tags: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    projection_type: Optional[str] = None
    projection_confidence: Optional[float] = None
    projection_timeframe: Optional[str] = None
    projection_summary: Optional[str] = None


class RelationshipCreate(BaseModel):
    source_id: int
    target_id: int
    relationship_type: str
    description: Optional[str] = None


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


@app.post("/api/artifacts")
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


@app.delete("/api/artifacts/{artifact_id}")
def delete_artifact(artifact_id: int, db: Session = Depends(get_db)):
    a = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if not a:
        raise HTTPException(404, "Artifact not found")
    db.delete(a)
    db.commit()
    return {"ok": True}


@app.get("/api/graph")
def get_graph(db: Session = Depends(get_db)):
    artifacts = db.query(Artifact).all()
    relationships = db.query(Relationship).all()

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


@app.post("/api/relationships")
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


@app.post("/api/ingest/run")
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


@app.post("/api/reseed")
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
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
    }


def _ensure_security_columns():
    """Add evidence/a2a columns to pre-existing SQLite DBs (fresh DBs already have them)."""
    try:
        from sqlalchemy import text as _text
        from .database import engine as _engine
        with _engine.begin() as conn:
            cols = [r[1] for r in conn.execute(_text("PRAGMA table_info(security_assessments)")).fetchall()]
            if "evidence_json" not in cols:
                conn.execute(_text("ALTER TABLE security_assessments ADD COLUMN evidence_json TEXT"))
            if "a2a_trace_json" not in cols:
                conn.execute(_text("ALTER TABLE security_assessments ADD COLUMN a2a_trace_json TEXT"))
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
        res = dispatch(env, db)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return res


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
