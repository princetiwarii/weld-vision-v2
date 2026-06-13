from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DefectSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class OverallResult(str, Enum):
    PASS   = "pass"
    FAIL   = "fail"
    REVIEW = "review"


# ---------------------------------------------------------------------------
# Defect building blocks
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x:      float = Field(..., description="Top-left x as fraction of image width (0–1)")
    y:      float = Field(..., description="Top-left y as fraction of image height (0–1)")
    width:  float = Field(..., description="Box width as fraction of image width (0–1)")
    height: float = Field(..., description="Box height as fraction of image height (0–1)")


class Defect(BaseModel):
    defect_id:           str
    type:                str
    label:               Optional[str]   = None
    severity:            DefectSeverity
    description:         str
    confidence:          float = Field(..., ge=0.0, le=1.0)
    bounding_box:        Optional[BoundingBox] = None
    length_mm:           Optional[float] = None
    depth_mm:            Optional[float] = None
    width_mm:            Optional[float] = None
    count:               Optional[int]   = None
    estimated_count:     Optional[str]   = None
    position:            Optional[str]   = None
    standards_reference: Optional[str]   = None
    recommendation:      Optional[str]   = None


class WeldingStandardsCompliance(BaseModel):
    standard:  str
    grade:     Optional[str] = None
    compliant: bool
    notes:     Optional[str] = None


# ---------------------------------------------------------------------------
# Per-frame (stitched pair) result
# ---------------------------------------------------------------------------
class FramePairResult(BaseModel):
    frame_index:          int
    image_label:          str               # e.g. "A1" (the pair label)
    source_frame_a_label: str               # e.g. "A1"
    source_frame_b_label: Optional[str]     # e.g. "A2" (None if odd last frame)
    timestamp_a_seconds:  float
    timestamp_b_seconds:  Optional[float]

    # S3 URLs
    raw_frame_a_url:     str
    raw_frame_b_url:     Optional[str]
    stitched_image_url:  str
    annotated_image_url: str

    # Gemini outputs
    overall_result:       OverallResult
    weld_quality_score:   float
    defects:              List[Defect]
    defect_summary:       Dict[str, int]
    standards_compliance: List[WeldingStandardsCompliance]
    recommendations:      List[str]
    model_notes:          Optional[str] = None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class DefectStatEntry(BaseModel):
    defect_type:        str
    total_count:        int
    frames_affected:    int
    avg_confidence:     float
    severity_breakdown: Dict[str, int]
    avg_length_mm:      Optional[float] = None
    avg_depth_mm:       Optional[float] = None


class StatisticalSummary(BaseModel):
    total_frames_analyzed:  int
    total_defects_found:    int
    defect_type_stats:      List[DefectStatEntry]
    avg_quality_score:      float
    min_quality_score:      float
    max_quality_score:      float
    pass_count:             int
    fail_count:             int
    review_count:           int
    most_severe_defect:     Optional[str] = None
    overall_compliance_aws: bool
    overall_compliance_iso: bool
    top_recommendations:    List[str]


# ---------------------------------------------------------------------------
# Main response
# ---------------------------------------------------------------------------
class VideoInspectionResponse(BaseModel):
    success:                 bool = True
    message:                 str
    session_id:              str

    # Mobile app form fields echoed back
    object_id:               str
    object_name:             Optional[str] = None
    scan_number:             Optional[str] = None
    side:                    Optional[str] = None

    # Video info
    video_filename:          str
    video_url:               str
    video_duration_seconds:  Optional[float] = None
    frames_extracted:        int             # raw frames pulled from video
    frame_pairs_analyzed:    int             # stitched pairs sent to Gemini

    # Outputs
    compile_chart_url:       str
    per_pair_results:        List[FramePairResult]
    statistical_summary:     StatisticalSummary

    processing_time_seconds: float
    analyzed_at:             datetime


# ---------------------------------------------------------------------------
# Session retrieval
# ---------------------------------------------------------------------------
class FrameUrlSummary(BaseModel):
    """Per-frame URL summary returned inside session listings."""
    frame_index:         int
    image_label:         str
    stitched_image_url:  Optional[str]
    annotated_image_url: Optional[str]
    overall_result:      Optional[str]
    weld_quality_score:  Optional[float]


class SessionSummary(BaseModel):
    """Lightweight summary for listing sessions."""
    session_id:              str
    object_id:               str
    object_name:             Optional[str]
    scan_number:             Optional[str]
    side:                    Optional[str]
    video_filename:          str
    video_url:               str
    frames_extracted:        int
    avg_quality_score:       Optional[float]
    total_defects_found:     int
    overall_compliance_aws:  bool
    overall_compliance_iso:  bool
    status:                  str
    compile_chart_url:       Optional[str]
    created_at:              datetime
    completed_at:            Optional[datetime]
    frames:                  Optional[List[FrameUrlSummary]] = None


class SessionDetailResponse(BaseModel):
    """Full detail of a completed session — fetched from Postgres."""
    success:                bool = True
    session:                SessionSummary
    per_pair_results:       List[FramePairResult]
    statistical_summary:    Optional[StatisticalSummary] = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail:  Optional[Any] = None


# ---------------------------------------------------------------------------
# Re-Inspection (fresh Gemini run on existing stitched images)
# ---------------------------------------------------------------------------
class ReinspectFrameResult(BaseModel):
    frame_index:         int
    image_label:         str
    stitched_image_url:  str            # original URL from DB (unchanged)
    annotated_image_url: str            # freshly generated new S3 URL
    overall_result:       OverallResult
    weld_quality_score:   float
    defects:              List[Defect]
    defect_summary:       Dict[str, int]
    standards_compliance: List[WeldingStandardsCompliance]
    recommendations:      List[str]
    model_notes:          Optional[str] = None


class ReinspectResponse(BaseModel):
    success:            bool = True
    object_id:          str
    session_id:         str             # session whose stitched images were re-used
    frames_reinspected: int
    compile_chart_url:  str             # freshly generated S3 URL
    per_frame_results:  List[ReinspectFrameResult]


# ---------------------------------------------------------------------------
# Point Cloud (.plv) schemas
# ---------------------------------------------------------------------------
class PointCloudUploadResponse(BaseModel):
    """Returned after a successful .plv file upload."""
    success:           bool = True
    scan_id:           str
    object_id:         str
    object_name:       Optional[str] = None
    scan_number:       Optional[str] = None
    side:              Optional[str] = None
    scanner_model:     Optional[str] = None
    linked_session_id: Optional[str] = None
    original_filename: str
    file_size_bytes:   int
    s3_url:            str
    status:            str
    created_at:        datetime


class PointCloudScanSummary(BaseModel):
    """Lightweight summary used in list responses."""
    scan_id:           str
    object_id:         str
    object_name:       Optional[str]
    scan_number:       Optional[str]
    side:              Optional[str]
    scanner_model:     Optional[str]
    linked_session_id: Optional[str]
    original_filename: str
    file_size_bytes:   Optional[int]
    s3_url:            str
    status:            str
    created_at:        datetime


class PointCloudListResponse(BaseModel):
    """List of all PLV scans for an object_id."""
    success:   bool = True
    object_id: str
    count:     int
    scans:     List[PointCloudScanSummary]


class PointCloudDownloadResponse(BaseModel):
    """Presigned download URL for a specific PLV scan (valid 1 hour)."""
    success:           bool = True
    scan_id:           str
    original_filename: str
    download_url:      str         # presigned S3 URL, valid for 1 hour
    expires_in_seconds: int = 3600
