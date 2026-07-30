from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
import enum

from .database import Base


class ArtifactType(str, enum.Enum):
    NEWS = "news"
    TWEET = "tweet"
    PAPER = "paper"
    BOOK = "book"
    INTERVIEW = "interview"
    PROJECTION = "projection"
    RESEARCH = "research"


class SentimentLabel(str, enum.Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ProjectionType(str, enum.Enum):
    UTOPIAN = "utopian"
    DYSTOPIAN = "dystopian"
    NEUTRAL = "neutral"
    ACCELERATIONIST = "accelerationist"
    CAUTIONARY = "cautionary"


class RelationshipType(str, enum.Enum):
    REFERENCES = "references"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    BUILD_UPON = "builds_upon"
    RESPONDS_TO = "responds_to"
    SIMILAR_TO = "similar_to"


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    artifact_type = Column(String(50), nullable=False, index=True)
    url = Column(String(1000), nullable=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    source = Column(String(200), nullable=True)
    author = Column(String(200), nullable=True)
    date_published = Column(DateTime, nullable=True)
    tags = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    sentiments = relationship("Sentiment", back_populates="artifact", cascade="all, delete-orphan")
    projections = relationship("Projection", back_populates="artifact", cascade="all, delete-orphan")
    outgoing_relationships = relationship(
        "Relationship", foreign_keys="Relationship.source_id",
        back_populates="source", cascade="all, delete-orphan"
    )
    incoming_relationships = relationship(
        "Relationship", foreign_keys="Relationship.target_id",
        back_populates="target", cascade="all, delete-orphan"
    )


class Sentiment(Base):
    __tablename__ = "sentiments"

    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False)
    score = Column(Float, nullable=False)
    label = Column(String(50), nullable=False)
    source = Column(String(100), nullable=True)

    artifact = relationship("Artifact", back_populates="sentiments")


class Projection(Base):
    __tablename__ = "projections"

    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False)
    projection_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False, default=0.5)
    timeframe = Column(String(50), nullable=True)
    summary = Column(Text, nullable=True)

    artifact = relationship("Artifact", back_populates="projections")


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(Integer, ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    description = Column(String(500), nullable=True)

    source = relationship("Artifact", foreign_keys=[source_id], back_populates="outgoing_relationships")
    target = relationship("Artifact", foreign_keys=[target_id], back_populates="incoming_relationships")
