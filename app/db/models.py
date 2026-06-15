"""
ORM Models — WeldVision v2
Tables:
  inspection_sessions  — one row per video upload (the "job")
  inspection_frames    — one row per trimmed/stitched image pair sent to Gemini
  frame_defects        — one row per defect found in a frame
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# inspection_sessions
# ---------------------------------------------------------------------------
class InspectionSession(Base):
    __tablename__ = "inspection_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)

    # Mobile app form inputs
    object_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    object_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    scan_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    side: Mapped[str | None] = mapped_column(String(64), nullable=True)
    welding_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    welding_position: Mapped[str | None] = mapped_column(String(64), nullable=True)
    remarks: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Video metadata
    video_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)         # S3 URL of original video
    video_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Processing outputs
    frames_extracted: Mapped[int] = mapped_column(Integer, default=0)
    compile_chart_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Aggregate stats (denormalised for fast reads)
    avg_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_defects_found: Mapped[int] = mapped_column(Integer, default=0)
    overall_compliance_aws: Mapped[bool] = mapped_column(Boolean, default=False)
    overall_compliance_iso: Mapped[bool] = mapped_column(Boolean, default=False)
    pass_count: Mapped[int] = mapped_column(Integer, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    review_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    processing_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="processing")
    # status values: processing | completed | failed

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    frames: Mapped[List[InspectionFrame]] = relationship(
        "InspectionFrame", back_populates="session", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# inspection_frames
# ---------------------------------------------------------------------------
class InspectionFrame(Base):
    __tablename__ = "inspection_frames"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("inspection_sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True
    )

    # Frame identity
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # image_label = "{object_id}{frame_index+1}"  e.g. "A1", "A2"
    image_label: Mapped[str] = mapped_column(String(64), nullable=False)

    # Two source frames that were stitched together before being sent to Gemini
    # e.g. frame A1 + A2 → stitched → Gemini analyses the pair
    source_frame_a_label: Mapped[str | None] = mapped_column(String(64), nullable=True)  # e.g. "A1"
    source_frame_b_label: Mapped[str | None] = mapped_column(String(64), nullable=True)  # e.g. "A2"

    # Timestamps of the two raw frames in the video
    timestamp_a_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp_b_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # S3 URLs
    raw_frame_a_url: Mapped[str | None] = mapped_column(Text, nullable=True)   # trimmed frame A
    raw_frame_b_url: Mapped[str | None] = mapped_column(Text, nullable=True)   # trimmed frame B
    stitched_image_url: Mapped[str | None] = mapped_column(Text, nullable=True) # A+B side-by-side
    annotated_image_url: Mapped[str | None] = mapped_column(Text, nullable=True) # annotated result

    # Gemini output summary
    overall_result: Mapped[str | None] = mapped_column(String(16), nullable=True)
    weld_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    defect_count: Mapped[int] = mapped_column(Integer, default=0)
    defect_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    standards_compliance: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommendations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    # Relationships
    session: Mapped[InspectionSession] = relationship("InspectionSession", back_populates="frames")
    defects: Mapped[List[FrameDefect]] = relationship(
        "FrameDefect", back_populates="frame", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# frame_defects
# ---------------------------------------------------------------------------
class FrameDefect(Base):
    __tablename__ = "frame_defects"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    frame_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("inspection_frames.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)  # denormalised

    defect_id: Mapped[str] = mapped_column(String(32), nullable=False)
    defect_type: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Bounding box (relative coords 0–1)
    bb_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_width: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_height: Mapped[float | None] = mapped_column(Float, nullable=True)

    length_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    width_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    position: Mapped[str | None] = mapped_column(String(256), nullable=True)
    standards_reference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    frame: Mapped[InspectionFrame] = relationship("InspectionFrame", back_populates="defects")


# ---------------------------------------------------------------------------
# point_cloud_scans  — stores uploaded .plv files from 3D/LiDAR scanners
# ---------------------------------------------------------------------------
class PointCloudScan(Base):
    __tablename__ = "point_cloud_scans"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Unique identifier for this upload (UUID)
    scan_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)

    # Weld object this scan belongs to (groups multiple scans together)
    object_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Optional link to an InspectionSession (same weld job)
    # Not a foreign key constraint — soft link only, keeps tables independent
    linked_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    # Form metadata from the mobile app
    object_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    scan_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    side: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scanner_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # File metadata
    original_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Point Cloud Measurements (populated after analysis)
    length_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    width_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    point_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mesh_s3_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # S3 storage
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)  # for presigned URL generation
    s3_url: Mapped[str] = mapped_column(Text, nullable=False)          # public-style URL

    # Lifecycle
    # status values: uploaded | processing | analyzed | error
    status: Mapped[str] = mapped_column(String(32), default="uploaded", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

