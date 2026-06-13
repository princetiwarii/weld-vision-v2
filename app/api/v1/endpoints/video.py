"""
POST /api/v1/inspections/video

Mobile app sends:
  - All trimmed images (extracted on the Swift side from the video)
  - Form fields: object_id, object_name, scan_number, side, etc.
  - segment_length_cm: physical length of each image segment in cm

Backend pipeline:
  1. Validate all images are actual weld images (reject non-weld images)
  2. Ensure object-level S3 folder exists:  inspections/{object_id}/
  3. Label each image: {object_id}{N}  →  A1, A2, A3 ...
  4. Keep raw bytes in memory only — NOT uploaded to S3
  5. Stitch consecutive pairs: (A1+A2), (A3+A4) ...
     - Each stitched image carries a continuous physical cm scale bar
  6. Upload ONLY stitched images to S3
  7. Run Gemini AI analysis on each stitched pair
  8. Annotate stitched images with defect overlays and upload to S3
  9. Build compile chart with continuous cm rulers per cell
  10. Upload compile chart to S3
  11. Persist session + frames to PostgreSQL
  12. Return JSON with S3 URLs

Non-weld image policy:
  - After reading all uploaded images, each image is validated via Gemini
  - If ANY image is not a weld image → entire batch is rejected with 422
  - This protects the pipeline from accidental or erroneous uploads
"""
import uuid
import time
import math
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.db.database import get_db
from app.db.models import InspectionSession, InspectionFrame, FrameDefect
from app.schemas.inspection import (
    VideoInspectionResponse, FramePairResult, StatisticalSummary,
    OverallResult, Defect, BoundingBox, WeldingStandardsCompliance,
    DefectSeverity,
)
from app.services.image_stitcher import stitch_pair
from app.services.gemini_service import gemini_service
from app.services.annotation_service import annotate_image
from app.services.s3_service import s3_service
from app.services.stats_service import compute_statistics
from app.services.compile_chart import build_compile_chart

router = APIRouter()

ALLOWED_MIME  = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 20 * 1024 * 1024   # 20 MB per image
MAX_IMAGES    = 50


def _make_label(object_id: str, n: int) -> str:
    """e.g. object_id='A', n=1  →  'A1'"""
    return f"{object_id.upper()}{n}"


async def _persist_session(
    db, session_id, object_id, object_name, scan_number,
    side, welding_type, welding_position, remarks,
    total_images,
) -> InspectionSession:
    session = InspectionSession(
        session_id=session_id,
        object_id=object_id,
        object_name=object_name,
        scan_number=scan_number,
        side=side,
        welding_type=welding_type,
        welding_position=welding_position,
        remarks=remarks,
        video_filename=f"{object_id}_scan",
        video_url="",
        frames_extracted=total_images,
        status="processing",
    )
    db.add(session)
    await db.flush()
    return session


async def _persist_frame(db, session_id: str, result: FramePairResult) -> InspectionFrame:
    frame = InspectionFrame(
        session_id=session_id,
        frame_index=result.frame_index,
        image_label=result.image_label,
        source_frame_a_label=result.source_frame_a_label,
        source_frame_b_label=result.source_frame_b_label,
        timestamp_a_seconds=result.timestamp_a_seconds,
        timestamp_b_seconds=result.timestamp_b_seconds,
        raw_frame_a_url=result.raw_frame_a_url,
        raw_frame_b_url=result.raw_frame_b_url,
        stitched_image_url=result.stitched_image_url,
        annotated_image_url=result.annotated_image_url,
        overall_result=result.overall_result.value,
        weld_quality_score=result.weld_quality_score,
        defect_count=len(result.defects),
        defect_summary=result.defect_summary,
        standards_compliance=[s.model_dump() for s in result.standards_compliance],
        recommendations=result.recommendations,
        model_notes=result.model_notes,
    )
    db.add(frame)
    await db.flush()

    for d in result.defects:
        bb = d.bounding_box
        db.add(FrameDefect(
            frame_id=frame.id,
            session_id=session_id,
            defect_id=d.defect_id,
            defect_type=d.type,
            severity=d.severity.value,
            description=d.description,
            confidence=d.confidence,
            bb_x=bb.x if bb else None,
            bb_y=bb.y if bb else None,
            bb_width=bb.width if bb else None,
            bb_height=bb.height if bb else None,
            length_mm=d.length_mm,
            depth_mm=d.depth_mm,
            width_mm=d.width_mm,
            position=d.position,
            standards_reference=d.standards_reference,
            recommendation=d.recommendation,
        ))
    return frame


@router.post(
    "/video",
    response_model=VideoInspectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit trimmed weld images for inspection",
    description="""
The **Swift app** extracts frames from the recorded video and sends them here as images.

### What this endpoint receives
- `images` — all trimmed frames (JPEG/PNG, in order, max 50)
- `object_id` — used to label each image: `A` → `A1`, `A2`, `A3` …
- `object_name`, `scan_number`, `side` — metadata from the inspection form

### Non-weld image validation
Every uploaded image is validated by Gemini AI to confirm it shows a welded joint.
If any image is NOT a weld image, the entire batch is rejected with HTTP 422.

### S3 Storage Structure (stitched + annotated + chart only — raw frames NOT stored)
```
inspections/
  {object_id}/
    {session_id}/
      frames/
        stitched/   A1.jpg, A3.jpg ...
        annotated/  A1_annotated.jpg ...
      chart/
        compile_chart.jpg
```

### Processing pipeline
1. All images validated as weld images (rejected if not)
2. Each image labelled `{object_id}{N}` (in memory only)
3. Consecutive pairs stitched with continuous cm scale bar
4. Stitched pairs uploaded to S3
5. Gemini AI analyzes each stitched pair for defects
6. Defect annotations drawn and uploaded to S3
7. Compile chart built and uploaded
8. Session + frames persisted to PostgreSQL
9. Full JSON response returned
    """,
)
async def analyze_weld_images(
    images: List[UploadFile] = File(..., description="Trimmed weld frames in order (JPEG/PNG, max 50)"),
    object_id:          str            = Form(...,  description="Weld object ID, e.g. 'A'. Frames labelled A1, A2 …"),
    object_name:        Optional[str]  = Form(None, description="e.g. Pipe Joint A"),
    scan_number:        Optional[str]  = Form(None, description="Scan / inspection number"),
    side:               Optional[str]  = Form(None, description="Side of weld being inspected"),
    welding_type:       Optional[str]  = Form(None, description="e.g. Fillet Weld, Butt Weld"),
    welding_position:   Optional[str]  = Form(None, description="e.g. Flat, Vertical, Overhead"),
    remarks:            Optional[str]  = Form(None),
    db: AsyncSession = Depends(get_db),
):
    t_start    = time.time()
    session_id = str(uuid.uuid4())
    object_id  = object_id.strip().upper()

    logger.info(
        f"[{session_id}] Inspection START — "
        f"object_id={object_id!r} images={len(images)}"
    )

    # -----------------------------------------------------------------------
    # Step 1 — Read and basic-validate all images
    # -----------------------------------------------------------------------
    if not images:
        raise HTTPException(status_code=422, detail="Upload at least 1 image.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"Maximum {MAX_IMAGES} images allowed.")

    image_data: List[tuple] = []   # (label, raw_bytes, mime_type)
    for i, img in enumerate(images):
        if img.content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=422,
                detail=f"Image {i+1} ({img.filename}): unsupported type '{img.content_type}'. Use JPEG or PNG.",
            )
        raw = await img.read()
        if len(raw) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Image {i+1} ({img.filename}) exceeds 20 MB limit.",
            )
        label = _make_label(object_id, i + 1)
        image_data.append((label, raw, img.content_type))
        logger.debug(f"[{session_id}] Received frame {label} ({len(raw)} bytes)")

    total_images = len(image_data)

    # -----------------------------------------------------------------------
    # Step 2 — Weld image validation (all images must be weld images)
    # Validate every image before starting the pipeline.
    # Rejected images are listed clearly in the error response.
    # -----------------------------------------------------------------------
    logger.info(f"[{session_id}] Validating {total_images} images as weld images…")
    non_weld_images = []

    for label, raw, mime in image_data:
        is_weld, reason = gemini_service.validate_weld_image(raw, mime_type=mime or "image/jpeg")
        if not is_weld:
            non_weld_images.append({"label": label, "reason": reason})
            logger.warning(f"[{session_id}] Non-weld image detected: {label} — {reason}")

    if non_weld_images:
        detail = {
            "error": "Non-weld images detected. All uploaded images must show a welded joint.",
            "rejected_images": non_weld_images,
            "advice": (
                "Please ensure all images show the weld bead on metal base material. "
                "Remove any non-weld images and resubmit."
            ),
        }
        logger.error(f"[{session_id}] Batch rejected — {len(non_weld_images)} non-weld image(s)")
        raise HTTPException(status_code=422, detail=detail)

    logger.info(f"[{session_id}] All {total_images} images validated as weld images ✓")

    # -----------------------------------------------------------------------
    # Step 3 — Ensure object-level S3 folder
    # -----------------------------------------------------------------------
    s3_service.ensure_object_folder(object_id)
    logger.info(f"[{session_id}] S3 object folder ensured: inspections/{object_id}/")

    # -----------------------------------------------------------------------
    # Step 4 — Create DB session row
    # -----------------------------------------------------------------------
    db_session = await _persist_session(
        db, session_id, object_id, object_name, scan_number, side,
        welding_type, welding_position, remarks, total_images,
    )

    # -----------------------------------------------------------------------
    # Step 5 — Pipeline: stitch → Gemini → annotate → chart → DB finalize
    # -----------------------------------------------------------------------
    num_pairs = 0
    seg_len   = 20.0
    pair_results: List[FramePairResult] = []
    chart_url = ""
    summary   = None
    elapsed   = 0.0

    try:
        num_pairs              = math.ceil(total_images / 2)
        stitched_bytes_list: List[bytes] = []
        annotated_bytes_list: List[bytes] = []
        pair_cm_ranges: List[tuple]       = []

        for pair_idx in range(num_pairs):
            i_a = pair_idx * 2
            i_b = i_a + 1

            label_a, raw_a, _ = image_data[i_a]

            has_b   = i_b < total_images
            label_b = image_data[i_b][0] if has_b else None
            raw_b   = image_data[i_b][1] if has_b else None

            # Cumulative cm positions
            start_cm = i_a * seg_len
            len_a    = seg_len
            len_b    = seg_len if has_b else 0.0
            end_cm   = start_cm + len_a + len_b
            pair_cm_ranges.append((start_cm, end_cm) if seg_len > 0 else None)

            # Stitch
            stitched = stitch_pair(
                raw_a, raw_b,
                start_cm_a=start_cm,
                length_cm_a=len_a,
                length_cm_b=len_b,
            )

            # Upload stitched image to S3
            stitch_key   = f"inspections/{object_id}/{session_id}/frames/stitched/{label_a}.jpg"
            stitched_url = s3_service.upload_bytes(stitched, stitch_key, "image/jpeg")

            logger.info(
                f"[{session_id}] Stitched pair {label_a}"
                + (f"+{label_b}" if label_b else " (single)")
                + f"  [{start_cm:.1f}–{end_cm:.1f} cm]"
                + f"  [{pair_idx + 1}/{num_pairs}]"
            )

            # Gemini analysis
            result: FramePairResult = gemini_service.analyze_pair(
                stitched_bytes=stitched,
                frame_index=pair_idx,
                image_label=label_a,
                source_frame_a_label=label_a,
                source_frame_b_label=label_b,
                timestamp_a=float(i_a),
                timestamp_b=float(i_b) if has_b else None,
                raw_frame_a_url="",
                raw_frame_b_url=None,
                stitched_image_url=stitched_url,
                annotated_image_url="",
                start_cm=start_cm,
                length_cm=len_a + len_b,
            )

            # Annotate + upload
            annotated_bytes = annotate_image(stitched, result.defects)
            ann_key = f"inspections/{object_id}/{session_id}/frames/annotated/{label_a}_annotated.jpg"
            annotated_url = s3_service.upload_bytes(annotated_bytes, ann_key, "image/jpeg")
            result.annotated_image_url = annotated_url

            pair_results.append(result)
            stitched_bytes_list.append(stitched)
            annotated_bytes_list.append(annotated_bytes)

            await _persist_frame(db, session_id, result)

        # -------------------------------------------------------------------
        # Statistics + compile chart
        # -------------------------------------------------------------------
        summary = compute_statistics(pair_results)

        chart_title = (
            f"WeldVision — {object_name or object_id} "
            f"| Scan {scan_number or 'N/A'} "
            f"| Side: {side or 'N/A'} "
            f"| {total_images} frames → {num_pairs} pairs"
            + (f" | {seg_len * total_images:.0f} cm total" if seg_len > 0 else "")
        )
        chart_bytes = build_compile_chart(
            annotated_bytes_list, pair_results, summary, chart_title,
            pair_cm_ranges=pair_cm_ranges,
        )
        chart_key = f"inspections/{object_id}/{session_id}/chart/compile_chart.jpg"
        chart_url = s3_service.upload_bytes(chart_bytes, chart_key, "image/jpeg")
        logger.info(f"[{session_id}] Compile chart uploaded → {chart_key}")

        # -------------------------------------------------------------------
        # Finalise DB row
        # -------------------------------------------------------------------
        elapsed = round(time.time() - t_start, 2)

        db_session.compile_chart_url       = chart_url
        db_session.avg_quality_score       = summary.avg_quality_score
        db_session.total_defects_found     = summary.total_defects_found
        db_session.overall_compliance_aws  = summary.overall_compliance_aws
        db_session.overall_compliance_iso  = summary.overall_compliance_iso
        db_session.pass_count              = summary.pass_count
        db_session.fail_count              = summary.fail_count
        db_session.review_count            = summary.review_count
        db_session.processing_time_seconds = elapsed
        db_session.status                  = "completed"
        db_session.completed_at            = datetime.now(timezone.utc)

        await db.commit()
        logger.info(f"[{session_id}] ✓ Done in {elapsed}s — {total_images} images, {num_pairs} pairs")

    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception(f"[{session_id}] Pipeline failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inspection pipeline failed: {str(exc)}",
        )

    return VideoInspectionResponse(
        message=(
            f"Inspection complete. {total_images} images processed, "
            f"{num_pairs} stitched pairs analyzed by Gemini."
            + (f" Scale: 0–{seg_len * total_images:.0f} cm." if seg_len > 0 else "")
        ),
        session_id=session_id,
        object_id=object_id,
        object_name=object_name,
        scan_number=scan_number,
        side=side,
        video_filename=f"{object_id}_scan",
        video_url="",
        video_duration_seconds=None,
        frames_extracted=total_images,
        frame_pairs_analyzed=num_pairs,
        compile_chart_url=chart_url,
        per_pair_results=pair_results,
        statistical_summary=summary,
        processing_time_seconds=elapsed,
        analyzed_at=datetime.now(timezone.utc),
    )
