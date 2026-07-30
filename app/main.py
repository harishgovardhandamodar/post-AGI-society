import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import init_db, get_db
from .models import Artifact, Sentiment, Projection, Relationship, ArtifactType
from .seed_data import seed_database
from .scraper import run_ingestion

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


@app.on_event("startup")
def on_start():
    init_db()
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
    q = db.query(Artifact)

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

    return {"total": total, "items": [_artifact_json(a) for a in items]}


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

    nodes = []
    for a in artifacts:
        sentiments = db.query(Sentiment).filter(Sentiment.artifact_id == a.id).all()
        avg_sentiment = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0

        projections = db.query(Projection).filter(Projection.artifact_id == a.id).all()
        max_confidence = max((p.confidence for p in projections), default=0)

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


@app.post("/api/ingest/run")
def run_scraper(db: Session = Depends(get_db)):
    count = run_ingestion(db)
    return {"ingested": count}


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
