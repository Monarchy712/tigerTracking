"""SQLAlchemy ORM models for tiger tracking."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    input_dir: Mapped[str] = mapped_column(String(512))
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    blank_removed: Mapped[int] = mapped_column(Integer, default=0)
    tiger_detected: Mapped[int] = mapped_column(Integer, default=0)
    new_individuals: Mapped[int] = mapped_column(Integer, default=0)
    auto_matched: Mapped[int] = mapped_column(Integer, default=0)
    pending_review: Mapped[int] = mapped_column(Integer, default=0)
    bytes_quarantined: Mapped[int] = mapped_column(Integer, default=0)
    estimated_time_saved_sec: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="running")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    images: Mapped[list[ImageRecord]] = relationship(back_populates="run")
    occupancy_snapshots: Mapped[list[OccupancySnapshot]] = relationship(back_populates="run")
    alerts: Mapped[list[Alert]] = relationship(back_populates="run")


class ImageRecord(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("processing_runs.id"), nullable=True)
    original_path: Mapped[str] = mapped_column(String(512))
    working_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    flank_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    station_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_blank: Mapped[bool] = mapped_column(Boolean, default=False)
    blank_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    quarantine_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    has_tiger: Mapped[bool] = mapped_column(Boolean, default=False)
    tiger_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)   # [x, y, w, h] body box
    animal_count: Mapped[int] = mapped_column(Integer, default=0)        # animals above threshold
    anomaly_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # per-frame anomaly signals

    run: Mapped[ProcessingRun | None] = relationship(back_populates="images")
    sightings: Mapped[list[Sighting]] = relationship(back_populates="image")


class Tiger(Base):
    __tablename__ = "tigers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tiger_code: Mapped[str] = mapped_column(String(32), unique=True)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    representatives: Mapped[list[TigerRepresentative]] = relationship(back_populates="tiger")
    sightings: Mapped[list[Sighting]] = relationship(back_populates="tiger")
    occupancy_snapshots: Mapped[list[OccupancySnapshot]] = relationship(back_populates="tiger")


class TigerRepresentative(Base):
    __tablename__ = "tiger_representatives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tiger_id: Mapped[int] = mapped_column(ForeignKey("tigers.id"))
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    flank_path: Mapped[str] = mapped_column(String(512))
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tiger: Mapped[Tiger] = relationship(back_populates="representatives")


class Sighting(Base):
    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tiger_id: Mapped[int] = mapped_column(ForeignKey("tigers.id"))
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("processing_runs.id"), nullable=True)
    station_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_type: Mapped[str] = mapped_column(String(32), default="auto")

    tiger: Mapped[Tiger] = relationship(back_populates="sightings")
    image: Mapped[ImageRecord] = relationship(back_populates="sightings")


class MatchReview(Base):
    __tablename__ = "match_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    candidate_tiger_id: Mapped[int | None] = mapped_column(ForeignKey("tigers.id"), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    reviewer_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class OccupancySnapshot(Base):
    __tablename__ = "occupancy_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("processing_runs.id"))
    tiger_id: Mapped[int] = mapped_column(ForeignKey("tigers.id"))
    centroid_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    centroid_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_sq_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    capture_count: Mapped[int] = mapped_column(Integer, default=0)
    station_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    home_range_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped[ProcessingRun] = relationship(back_populates="occupancy_snapshots")
    tiger: Mapped[Tiger] = relationship(back_populates="occupancy_snapshots")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("processing_runs.id"))
    tiger_id: Mapped[int | None] = mapped_column(ForeignKey("tigers.id"), nullable=True)
    alert_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_survey_artifact: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped[ProcessingRun] = relationship(back_populates="alerts")


class Station(Base):
    __tablename__ = "stations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    zone: Mapped[str] = mapped_column(String(32), default="core")
    is_village_adjacent: Mapped[bool] = mapped_column(Boolean, default=False)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)   # village/livestock interface
    is_waterhole: Mapped[bool] = mapped_column(Boolean, default=False)   # perennial water source
    # Forest-department administrative units, carried through to M-STrIPES exports.
    range_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    beat: Mapped[str | None] = mapped_column(String(128), nullable=True)
    compartment: Mapped[str | None] = mapped_column(String(64), nullable=True)


engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _apply_column_migrations()


# Columns added after the first release. SQLite has no ALTER-if-not-exists, so
# they are patched onto existing databases here instead of via a migration tool.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "tiger_representatives": {"embedding_json": "TEXT"},
    "images": {
        "bbox_json": "TEXT",
        "animal_count": "INTEGER DEFAULT 0",
        "anomaly_json": "TEXT",
    },
    "stations": {
        "is_sensitive": "BOOLEAN DEFAULT 0",
        "is_waterhole": "BOOLEAN DEFAULT 0",
        "range_name": "VARCHAR(128)",
        "beat": "VARCHAR(128)",
        "compartment": "VARCHAR(64)",
    },
}


def _apply_column_migrations() -> None:
    """Add newer columns to databases created by an older schema version."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table, columns in _ADDED_COLUMNS.items():
        if table not in existing_tables:
            continue
        present = {col["name"] for col in inspector.get_columns(table)}
        missing = {name: ddl for name, ddl in columns.items() if name not in present}
        if not missing:
            continue
        with engine.connect() as conn:
            for name, ddl in missing.items():
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
            conn.commit()


def get_session():
    return SessionLocal()
