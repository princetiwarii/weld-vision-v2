"""
GET /api/v1/inspections/reinspect/{object_id}
=============================================
Fetches the stitched images from a previous session for this object_id,
sends them FRESH to Gemini AI for a brand-new inspection, annotates the
results, builds a new compile chart, uploads everything to S3, and returns
the new URLs.

The original database records are NOT modified — this is a read-only
re-analysis operation.
"""
import asyncio
from typing import Optional
import httpx

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.db.database import get_db
from app.db.models import InspectionSession, InspectionFrame
from app.schemas.inspection import (
    ReinspectFrameResult,
    ReinspectResponse,
    OverallResult,
    Defect,
    BoundingBox,
    DefectSeverity,
    WeldingStandardsCompliance,
)
from app.services.gemini_service import gemini_service
from app.services.annotation_service import annotate_image
from app.services.s3_service import s3_service
from app.services.compile_chart import build_compile_chart
from app.schemas.inspection import FramePairResult

router = APIRouter()


def _s3_key_from_url(url: str) -> str:
    """
    Extract the S3 object key from a public S3 URL.
    URL format: https://{bucket}.s3.{region}.amazonaws.com/{key}
    """
    # Strip the scheme + host portion, leaving just the key path
    # e.g. "https://mybucket.s3.ap-south-1.amazonaws.com/inspections/A/xyz/frame.jpg"
    # → "inspections/A/xyz/frame.jpg"
    parts = url.split(".amazonaws.com/", 1)
    if len(parts) == 2:
        return parts[1]
    # Fallback: strip leading slash if present
    from urllib.parse import urlparse
    return urlparse(url).path.lstrip("/")


def _download_from_s3(url: str) -> bytes:
    """
    Download bytes from S3 using a presigned URL so it works even
    when the bucket does NOT have public-read ACL enabled.
    """
    key = _s3_key_from_url(url)
    try:
        # Generate a short-lived presigned URL (60 seconds is plenty)
        presigned_url = s3_service.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": s3_service.bucket, "Key": key},
            ExpiresIn=60,
        )
        import urllib.request
        with urllib.request.urlopen(presigned_url, timeout=30) as resp:
            return resp.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download stitched image from S3 (key={key}): {e}",
        )


@router.get(
    "/reinspect/{object_id}",
    response_model=ReinspectResponse,
    summary="Fresh re-inspection of all stitched images for an object_id",
    description=(
        "Fetches the stitched weld images from a previous inspection session "
        "for the given object_id, sends them to Gemini AI for a completely fresh "
        "analysis, annotates the results, and builds a new compile chart. "
        "The original DB records are not modified. "
        "By default uses the most recent completed session; pass `session_id` "
        "to target a specific one."
    ),
)
async def reinspect_object(
    object_id: str,
    session_id: Optional[str] = Query(
        None,
        description="Target a specific session_id. If omitted, the most recent completed session is used.",
    ),
    db: AsyncSession = Depends(get_db),
):
    object_id = object_id.upper()

    # ------------------------------------------------------------------
    # 1. Resolve the session to re-inspect
    # ------------------------------------------------------------------
    if session_id:
        q = (
            select(InspectionSession)
            .where(
                InspectionSession.object_id == object_id,
                InspectionSession.session_id == session_id,
            )
            .options(selectinload(InspectionSession.frames))
        )
    else:
        q = (
            select(InspectionSession)
            .where(
                InspectionSession.object_id == object_id,
                InspectionSession.status == "completed",
            )
            .order_by(desc(InspectionSession.created_at))
            .limit(1)
            .options(selectinload(InspectionSession.frames))
        )

    result = await db.execute(q)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No completed inspection session found for object_id '{object_id}'"
                + (f" with session_id '{session_id}'" if session_id else "")
                + ". Run a video inspection first."
            ),
        )

    frames = sorted(session.frames, key=lambda f: f.frame_index)

    if not frames:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session.session_id}' has no frames stored.",
        )

    # Filter to only frames that have a stitched image URL
    frames_with_image = [f for f in frames if f.stitched_image_url]
    if not frames_with_image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session.session_id}' has no stitched image URLs stored.",
        )

    logger.info(
        f"Re-inspection started | object_id={object_id} | session={session.session_id} | "
        f"frames={len(frames_with_image)}"
    )

    # ------------------------------------------------------------------
    # 2. Download all stitched images concurrently from S3
    # ------------------------------------------------------------------
    logger.info("Downloading stitched images from S3…")
    stitched_bytes_list: list[bytes] = []
    for frame in frames_with_image:
        stitched_bytes_list.append(_download_from_s3(frame.stitched_image_url))
    logger.info(f"Downloaded {len(stitched_bytes_list)} stitched images")

    # ------------------------------------------------------------------
    # 3. Run fresh Gemini analysis on each image (in sequence to avoid
    #    hammering the API; change to asyncio.gather if you want parallel)
    # ------------------------------------------------------------------
    frame_results: list[ReinspectFrameResult] = []
    annotated_bytes_list: list[bytes] = []

    for i, (frame, raw_bytes) in enumerate(zip(frames_with_image, stitched_bytes_list)):
        logger.info(f"Gemini re-analysis [{i+1}/{len(frames_with_image)}] frame={frame.image_label}")

        # Fresh Gemini call — reuse the existing service
        pair_result: FramePairResult = gemini_service.analyze_pair(
            stitched_bytes=raw_bytes,
            frame_index=frame.frame_index,
            image_label=frame.image_label,
            source_frame_a_label=frame.source_frame_a_label or frame.image_label,
            source_frame_b_label=frame.source_frame_b_label,
            timestamp_a=frame.timestamp_a_seconds or 0.0,
            timestamp_b=frame.timestamp_b_seconds,
            raw_frame_a_url=frame.raw_frame_a_url or "",
            raw_frame_b_url=frame.raw_frame_b_url,
            stitched_image_url=frame.stitched_image_url or "",
            annotated_image_url="",  # placeholder — we'll fill this after upload
        )

        # Annotate the image with the new Gemini results
        annotated = annotate_image(raw_bytes, pair_result.defects)
        annotated_bytes_list.append(annotated)

        # Upload new annotated image to S3
        s3_key = (
            f"inspections/{object_id}/reinspect/{session.session_id}/"
            f"annotated_{frame.frame_index:03d}_{frame.image_label}.jpg"
        )
        new_annotated_url = s3_service.upload_bytes(annotated, s3_key, "image/jpeg")
        logger.info(f"Uploaded re-annotated image → {new_annotated_url}")

        frame_results.append(
            ReinspectFrameResult(
                frame_index=frame.frame_index,
                image_label=frame.image_label,
                stitched_image_url=frame.stitched_image_url or "",
                annotated_image_url=new_annotated_url,
                overall_result=pair_result.overall_result,
                weld_quality_score=pair_result.weld_quality_score,
                defects=pair_result.defects,
                defect_summary=pair_result.defect_summary,
                standards_compliance=pair_result.standards_compliance,
                recommendations=pair_result.recommendations,
                model_notes=pair_result.model_notes,
            )
        )

    # ------------------------------------------------------------------
    # 4. Build a fresh compile chart from the new annotated images
    # ------------------------------------------------------------------
    logger.info("Building new compile chart…")

    # Build minimal FramePairResult list for compile_chart (stats not needed)
    pair_results_for_chart = [
        FramePairResult(
            frame_index=fr.frame_index,
            image_label=fr.image_label,
            source_frame_a_label=fr.image_label,
            source_frame_b_label=None,
            timestamp_a_seconds=0.0,
            timestamp_b_seconds=None,
            raw_frame_a_url="",
            raw_frame_b_url=None,
            stitched_image_url=fr.stitched_image_url,
            annotated_image_url=fr.annotated_image_url,
            overall_result=fr.overall_result,
            weld_quality_score=fr.weld_quality_score,
            defects=fr.defects,
            defect_summary=fr.defect_summary,
            standards_compliance=fr.standards_compliance,
            recommendations=fr.recommendations,
            model_notes=fr.model_notes,
        )
        for fr in frame_results
    ]

    # Compute real summary from reinspection results
    from app.services.stats_service import compute_statistics
    real_summary = compute_statistics(pair_results_for_chart)

    from datetime import datetime, timezone as _tz
    reinspect_ts = datetime.now(_tz.utc).strftime("%d %b %Y")

    chart_bytes = build_compile_chart(
        annotated_images=annotated_bytes_list,
        results=pair_results_for_chart,
        summary=real_summary,
        title=f"Re-Inspection — {object_id}  |  {reinspect_ts}",
    )

    # Upload new compile chart
    chart_key = (
        f"inspections/{object_id}/reinspect/{session.session_id}/compile_chart.jpg"
    )
    compile_chart_url = s3_service.upload_bytes(chart_bytes, chart_key, "image/jpeg")
    logger.info(f"Compile chart uploaded → {compile_chart_url}")

    # ------------------------------------------------------------------
    # 5. Return response (DB is NOT modified)
    # ------------------------------------------------------------------
    logger.info(
        f"Re-inspection complete | object_id={object_id} | "
        f"frames={len(frame_results)} | chart={compile_chart_url}"
    )

    return ReinspectResponse(
        success=True,
        object_id=object_id,
        session_id=session.session_id,
        frames_reinspected=len(frame_results),
        compile_chart_url=compile_chart_url,
        per_frame_results=frame_results,
    )
