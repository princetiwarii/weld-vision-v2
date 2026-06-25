"""
POST /api/v1/inspections/video

Mobile app sends:
  - All trimmed images (extracted on the Swift side from the video)
  - Form fields: object_id, object_name, scan_number, side, etc.
  - segment_length_cm: physical length of each image segment in cm

Backend pipeline:
  1. Ensure object-level S3 folder exists:  inspections/{object_id}/
  2. Label each image: {object_id}{N}  →  A1, A2, A3 ...
  3. Keep raw bytes in memory only — NOT uploaded to S3
  4. Stitch consecutive pairs: (A1+A2), (A3+A4) ...
  5. Upload ONLY stitched images to S3
  6. Persist session + pending frames to PostgreSQL
  7. Return JSON with status="pending"
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
from app.db.models import InspectionSession, InspectionFrame
from app.schemas.inspection import VideoUploadResponse
from app.services.image_stitcher import stitch_pair
from app.services.s3_service import s3_service

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
        status="pending",
    )
    db.add(session)
    await db.flush()
    return session


async def _persist_pending_frame(
    db, session_id: str, frame_index: int, image_label: str,
    source_frame_a_label: str, source_frame_b_label: str | None,
    timestamp_a: float, timestamp_b: float | None,
    stitched_image_url: str
) -> InspectionFrame:
    frame = InspectionFrame(
        session_id=session_id,
        frame_index=frame_index,
        image_label=image_label,
        source_frame_a_label=source_frame_a_label,
        source_frame_b_label=source_frame_b_label,
        timestamp_a_seconds=timestamp_a,
        timestamp_b_seconds=timestamp_b,
        raw_frame_a_url="",
        raw_frame_b_url="",
        stitched_image_url=stitched_image_url,
        annotated_image_url="",
        overall_result="review",
        weld_quality_score=0.0,
        defect_count=0,
        defect_summary={},
        standards_compliance=[],
        recommendations=[],
        model_notes="",
    )
    db.add(frame)
    await db.flush()
    return frame


@router.post(
    "/video",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit trimmed weld images for stitching and storage",
    description="""
The **Swift app** extracts frames from the recorded video and sends them here as images.
This endpoint stitches the images and saves them to S3 and PostgreSQL.
Gemini analysis is deferred to the GET API to prevent timeouts.

### Processing pipeline
1. Each image labelled `{object_id}{N}` (in memory only)
2. Consecutive pairs stitched with continuous cm scale bar
3. Stitched pairs uploaded to S3
4. Session + pending frames persisted to PostgreSQL
5. Basic JSON response returned immediately
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
        f"[{session_id}] Upload START — "
        f"object_id={object_id!r} images={len(images)}"
    )

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

    await s3_service.ensure_object_folder(object_id)
    logger.info(f"[{session_id}] S3 object folder ensured: inspections/{object_id}/")

    db_session = await _persist_session(
        db, session_id, object_id, object_name, scan_number, side,
        welding_type, welding_position, remarks, total_images,
    )

    num_pairs = 0
    seg_len   = 20.0
    elapsed   = 0.0

    try:
        num_pairs = math.ceil(total_images / 2)

        for pair_idx in range(num_pairs):
            i_a = pair_idx * 2
            i_b = i_a + 1

            label_a, raw_a, _ = image_data[i_a]

            has_b   = i_b < total_images
            label_b = image_data[i_b][0] if has_b else None
            raw_b   = image_data[i_b][1] if has_b else None

            start_cm = i_a * seg_len
            len_a    = seg_len
            len_b    = seg_len if has_b else 0.0
            end_cm   = start_cm + len_a + len_b

            stitched = stitch_pair(
                raw_a, raw_b,
                start_cm_a=start_cm,
                length_cm_a=len_a,
                length_cm_b=len_b,
            )

            stitch_key   = f"inspections/{object_id}/{session_id}/frames/stitched/{label_a}.jpg"
            stitched_url = await s3_service.upload_bytes(stitched, stitch_key, "image/jpeg")

            logger.info(
                f"[{session_id}] Stitched pair {label_a}"
                + (f"+{label_b}" if label_b else " (single)")
                + f"  [{start_cm:.1f}–{end_cm:.1f} cm]"
                + f"  [{pair_idx + 1}/{num_pairs}]"
            )

            await _persist_pending_frame(
                db=db,
                session_id=session_id,
                frame_index=pair_idx,
                image_label=label_a,
                source_frame_a_label=label_a,
                source_frame_b_label=label_b,
                timestamp_a=float(i_a),
                timestamp_b=float(i_b) if has_b else None,
                stitched_image_url=stitched_url,
            )

        elapsed = round(time.time() - t_start, 2)
        await db.commit()
        logger.info(f"[{session_id}] OK: Done in {elapsed}s — {total_images} images, {num_pairs} pairs")

    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception(f"[{session_id}] Pipeline failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload pipeline failed: {str(exc)}",
        )

    return VideoUploadResponse(
        message=f"Upload complete. {total_images} images processed, {num_pairs} stitched pairs saved.",
        session_id=session_id,
        object_id=object_id,
        frames_extracted=total_images,
        frame_pairs_stitched=num_pairs,
        status="pending"
    )
