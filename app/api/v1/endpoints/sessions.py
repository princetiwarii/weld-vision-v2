"""
GET /api/v1/inspections/sessions          — list all sessions (paginated)
GET /api/v1/inspections/sessions/{id}     — full detail of one session (triggers AI if pending)
GET /api/v1/inspections/sessions/object/{object_id} — all scans for one object
"""
import time
from datetime import datetime, timezone
from typing import Optional, List
import json
import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from loguru import logger
from pydantic import BaseModel

from app.db.database import get_db, AsyncSessionLocal
from app.db.models import InspectionSession, InspectionFrame, FrameDefect
from app.schemas.inspection import (
    SessionSummary, SessionDetailResponse, FramePairResult,
    OverallResult, Defect, DefectSeverity, BoundingBox,
    WeldingStandardsCompliance, StatisticalSummary, DefectStatEntry,
    FrameUrlSummary,
)
from app.services.stats_service import compute_statistics
from app.services.gemini_service import gemini_service, build_rich_text_output, build_object_summary_table
from app.services.annotation_service import annotate_image
from app.services.s3_service import s3_service
from app.services.compile_chart import build_compile_chart

router = APIRouter()


def _orm_defects_to_schema(db_defects: List[FrameDefect]) -> List[Defect]:
    out = []
    for d in db_defects:
        try:
            bb = None
            if d.bb_x is not None:
                bb = BoundingBox(x=d.bb_x, y=d.bb_y, width=d.bb_width, height=d.bb_height)
            out.append(Defect(
                defect_id=d.defect_id,
                type=d.defect_type,
                label=d.defect_type,  # Fallback to type since label isn't in DB
                estimated_count=None, # Not in DB
                severity=DefectSeverity(d.severity),
                description=d.description or "",
                confidence=d.confidence,
                bounding_box=bb,
                length_mm=d.length_mm,
                depth_mm=d.depth_mm,
                width_mm=d.width_mm,
                position=d.position,
                standards_reference=d.standards_reference,
                recommendation=d.recommendation,
            ))
        except Exception as e:
            logger.warning(f"Failed to map defect {d.defect_id}: {e}")
            continue
    return out


def _orm_frame_to_schema(f: InspectionFrame) -> FramePairResult:
    defects = _orm_defects_to_schema(f.defects)
    standards = []
    for s in (f.standards_compliance or []):
        try:
            standards.append(WeldingStandardsCompliance(**s))
        except Exception:
            pass

    return FramePairResult(
        frame_index=f.frame_index,
        image_label=f.image_label,
        source_frame_a_label=f.source_frame_a_label or "",
        source_frame_b_label=f.source_frame_b_label,
        timestamp_a_seconds=f.timestamp_a_seconds or 0.0,
        timestamp_b_seconds=f.timestamp_b_seconds,
        raw_frame_a_url=f.raw_frame_a_url or "",
        raw_frame_b_url=f.raw_frame_b_url,
        stitched_image_url=f.stitched_image_url or "",
        annotated_image_url=f.annotated_image_url or "",
        overall_result=OverallResult(f.overall_result or "review"),
        weld_quality_score=f.weld_quality_score or 0.0,
        defects=defects,
        defect_summary=f.defect_summary or {},
        standards_compliance=standards,
        recommendations=f.recommendations or [],
        model_notes=f.model_notes,
    )


def _orm_session_to_summary(s: InspectionSession, include_frames: bool = False) -> SessionSummary:
    frames_data = None
    if include_frames and s.frames:
        frames_data = [
            FrameUrlSummary(
                frame_index=f.frame_index,
                image_label=f.image_label,
                stitched_image_url=f.stitched_image_url,
                annotated_image_url=f.annotated_image_url,
                overall_result=f.overall_result,
                weld_quality_score=f.weld_quality_score,
            )
            for f in sorted(s.frames, key=lambda x: x.frame_index)
        ]
    return SessionSummary(
        session_id=s.session_id,
        object_id=s.object_id,
        object_name=s.object_name,
        scan_number=s.scan_number,
        side=s.side,
        video_filename=s.video_filename,
        video_url=s.video_url,
        frames_extracted=s.frames_extracted,
        avg_quality_score=s.avg_quality_score,
        total_defects_found=s.total_defects_found,
        overall_compliance_aws=s.overall_compliance_aws,
        overall_compliance_iso=s.overall_compliance_iso,
        status=s.status,
        compile_chart_url=s.compile_chart_url,
        created_at=s.created_at,
        completed_at=s.completed_at,
        frames=frames_data,
    )


# ---------------------------------------------------------------------------
# List sessions
# ---------------------------------------------------------------------------
@router.get(
    "/sessions",
    summary="List all inspection sessions (most recent first)",
)
async def list_sessions(
    limit:  int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
):
    q = select(InspectionSession).order_by(desc(InspectionSession.created_at))
    if status_filter:
        q = q.where(InspectionSession.status == status_filter)
    q = q.offset(offset).limit(limit)

    result = await db.execute(q)
    sessions = result.scalars().all()

    return {
        "success": True,
        "count": len(sessions),
        "sessions": [_orm_session_to_summary(s) for s in sessions],
    }


class ManualAnnotationUpdate(BaseModel):
    annotated_image_url: str
    gemini_json_output: Optional[dict] = None

# ---------------------------------------------------------------------------
# Unified Endpoint to Override Images
# ---------------------------------------------------------------------------
@router.put(
    "/sessions/object/{object_id}/override-images",
    summary="Replace all annotated images and compiled chart for an object",
)
async def override_images(
    object_id: str,
    compiled_chart: UploadFile = File(...),
    annotated_images: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Overwrites the S3 images and DB URLs for an existing session's frames and chart.
    """
    q = (
        select(InspectionSession)
        .where(InspectionSession.object_id == object_id.upper())
        .options(selectinload(InspectionSession.frames))
        .order_by(desc(InspectionSession.created_at))
        .limit(1)
    )
    result = await db.execute(q)
    session = result.scalar_one_or_none()

    import uuid

    # Ensure annotated_images is a list even if only one file is uploaded
    if not isinstance(annotated_images, list):
        annotated_images = [annotated_images]

    if not session:
        # Create a new session on the fly
        session_id = str(uuid.uuid4())
        session = InspectionSession(
            session_id=session_id,
            object_id=object_id.upper(),
            scan_number="Manual",
            video_filename="Manual Override",
            status="completed",
        )
        db.add(session)
        await db.flush()

        frames = []
        # Create a frame for each annotated image provided
        for i in range(len(annotated_images)):
            frame = InspectionFrame(
                session_id=session.session_id,
                frame_index=i,
                image_label=f"{object_id.upper()}{i+1}_manual",
                annotated_image_url="",
            )
            db.add(frame)
            frames.append(frame)
        await db.flush()
    else:
        frames = sorted(session.frames, key=lambda x: x.frame_index)

    # 1. Update Compiled Chart
    chart_bytes = await compiled_chart.read()
    chart_key = f"inspections/{session.object_id}/{session.session_id}/chart/compile_chart_manual.jpg"
    chart_url = await s3_service.upload_bytes(chart_bytes, chart_key, compiled_chart.content_type)
    session.compile_chart_url = chart_url
    session.status = "completed"

    # 2. Update Annotated Images
    # Map uploaded images by their base filename (e.g. "18GG1.jpg" -> "18GG1")
    uploaded_files_map = {}
    for f in annotated_images:
        if f.filename:
            # Strip extension to get base name and uppercase it for robust matching
            base_name = f.filename.rsplit('.', 1)[0].upper()
            uploaded_files_map[base_name] = f
            # Also map the full filename just in case
            uploaded_files_map[f.filename.upper()] = f

    frames_updated = 0
    for frame in frames:
        label_upper = frame.image_label.upper()

        # Look for a match
        matched_file = None
        for key, file_obj in uploaded_files_map.items():
            if label_upper == key or label_upper in key:
                matched_file = file_obj
                break

        if matched_file:
            img_bytes = await matched_file.read()
            key = f"inspections/{session.object_id}/{session.session_id}/frames/annotated/{frame.image_label}_manual.jpg"
            url = await s3_service.upload_bytes(img_bytes, key, matched_file.content_type)
            frame.annotated_image_url = url
            frames_updated += 1
        else:
            # Delete frames that do not have a corresponding uploaded image
            await db.delete(frame)

    await db.commit()

    return {
        "success": True,
        "session_id": session.session_id,
        "compile_chart_url": session.compile_chart_url,
        "frames_updated": frames_updated
    }


# ---------------------------------------------------------------------------
# Get session by object_id (MANUAL results - does NOT trigger AI)
# ---------------------------------------------------------------------------
@router.get(
    "/sessions/object/{object_id}/manual-results",
    summary="Get all inspection sessions for a specific object_id (strictly returns DB data without triggering AI)",
)
async def get_manual_sessions_by_object(
    object_id: str,
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(InspectionSession)
        .where(InspectionSession.object_id == object_id.upper())
        .options(
            selectinload(InspectionSession.frames)
            .selectinload(InspectionFrame.defects)
        )
        .order_by(desc(InspectionSession.created_at))
    )
    result = await db.execute(q)
    sessions = result.scalars().all()

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No sessions found for object_id '{object_id}'",
        )

    session_details = []

    for session in sessions:
        frames = sorted(session.frames, key=lambda x: x.frame_index)

        # Auto-finalize removed. Session will be finalized when the compiled chart is manually uploaded via PUT.

        pair_results = [_orm_frame_to_schema(f) for f in frames]
        summary = compute_statistics(pair_results) if pair_results and session.status == "completed" else None

        session_details.append(
            SessionDetailResponse(
                session=_orm_session_to_summary(session),
                compile_chart_url=session.compile_chart_url,
                per_pair_results=pair_results,
                statistical_summary=summary,
            )
        )

    return {
        "success": True,
        "object_id": object_id.upper(),
        "count": len(session_details),
        "sessions": session_details,
    }


# ---------------------------------------------------------------------------
# API 1: Inspect and Calculate Measurements
# ---------------------------------------------------------------------------
@router.get(
    "/sessions/object/{object_id}/inspect",
    summary="Inspects all frames of an object, reading the scale and calculating measurements via Gemini 2.5 Flash",
)
async def inspect_object(
    object_id: str,
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(InspectionSession)
        .where(InspectionSession.object_id == object_id.upper())
        .options(
            selectinload(InspectionSession.frames)
            .selectinload(InspectionFrame.defects)
        )
        .order_by(desc(InspectionSession.created_at))
    )
    result = await db.execute(q)
    sessions = result.scalars().all()

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No sessions found for object_id '{object_id}'",
        )

    updated_sessions = []

    for session in sessions:
        session.status = "processing"
        await db.commit()

        frames = sorted(session.frames, key=lambda x: x.frame_index)
        pair_results = []
        raw_gemini_outputs = []

        for frame in frames:
            if not frame.stitched_image_url or ".amazonaws.com/" not in frame.stitched_image_url:
                continue

            stitch_key = frame.stitched_image_url.split(".amazonaws.com/")[1]
            try:
                stitched_bytes = await s3_service.download_bytes(stitch_key)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                continue

            # API 1 specific: Call the new inspect_with_measurements
            raw_result = await gemini_service.inspect_with_measurements(
                image_bytes=stitched_bytes,
                model_name="gemini-2.5-flash"
            )

            # Map the raw JSON back into DB models
            frame.overall_result = raw_result.get("overall_result", "review")
            frame.weld_quality_score = float(raw_result.get("weld_quality_score", 0.0))
            frame.defect_summary = raw_result.get("defect_summary", {})
            frame.standards_compliance = raw_result.get("standards_compliance", [])
            frame.recommendations = raw_result.get("recommendations", [])
            frame.model_notes = raw_result.get("model_notes", "")

            # Clear old defects
            for d in frame.defects:
                await db.delete(d)

            defects_parsed = []
            for d in raw_result.get("defects", []):
                bb = d.get("bounding_box", {})
                # Handle label and estimated_count by prepending to description
                desc_text = d.get("description", "")
                if d.get("estimated_count"):
                    desc_text = f"Count: {d.get('estimated_count')} | " + desc_text
                
                new_defect = FrameDefect(
                    frame_id=frame.id,
                    session_id=session.session_id,
                    defect_id=d.get("defect_id", str(uuid.uuid4())[:8]),
                    defect_type=d.get("type", "Unknown"),
                    severity=d.get("severity", "medium"),
                    description=desc_text,
                    confidence=d.get("confidence", 1.0),
                    bb_x=bb.get("x") if bb else None,
                    bb_y=bb.get("y") if bb else None,
                    bb_width=bb.get("width") if bb else None,
                    bb_height=bb.get("height") if bb else None,
                    length_mm=d.get("length_mm"),
                    depth_mm=d.get("depth_mm"),
                    width_mm=d.get("width_mm"),
                    position=d.get("location_description")
                )
                db.add(new_defect)
                defects_parsed.append(new_defect)

            # Update frame relationship so it can be passed to schema
            frame.defects = defects_parsed

            pair_results.append(_orm_frame_to_schema(frame))
            raw_gemini_outputs.append(raw_result)

        # Recompute stats
        if pair_results:
            summary = compute_statistics(pair_results)
            session.avg_quality_score = summary.avg_quality_score
            session.total_defects_found = summary.total_defects_found
            session.overall_compliance_aws = summary.overall_compliance_aws
            session.overall_compliance_iso = summary.overall_compliance_iso
            session.pass_count = summary.pass_count
            session.fail_count = summary.fail_count
            session.review_count = summary.review_count

        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)
        await db.commit()

        # Build rich per-image text outputs with measurements
        frames_out = []
        for frame, ro in zip(frames, raw_gemini_outputs):
            rich = build_rich_text_output(
                image_label=frame.image_label,
                stitched_image_url=frame.stitched_image_url or "",
                raw_result=ro,
            )
            frames_out.append(rich)

        # Build cross-image summary table
        object_summary = build_object_summary_table(frames_out) if frames_out else {}

        summary_dict = {}
        if pair_results:
            summary_dict = compute_statistics(pair_results).model_dump()

        updated_sessions.append({
            "session_id": session.session_id,
            "status": session.status,
            "images_analyzed": len(frames_out),
            "per_image_results": frames_out,
            "object_summary_table": object_summary,
            "statistical_summary": summary_dict,
        })

    return {
        "success": True,
        "object_id": object_id.upper(),
        "sessions": updated_sessions
    }

# ---------------------------------------------------------------------------
# API 2: Generate Annotated Images (Python Drawing)
# ---------------------------------------------------------------------------
@router.get(
    "/sessions/object/{object_id}/generate-images",
    summary="Generates annotated images using Python to accurately draw defects from JSON",
)
async def generate_images_for_object(
    object_id: str,
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(InspectionSession)
        .where(InspectionSession.object_id == object_id.upper())
        .where(InspectionSession.status == "completed")
        .options(
            selectinload(InspectionSession.frames)
            .selectinload(InspectionFrame.defects)
        )
        .order_by(desc(InspectionSession.created_at))
    )
    result = await db.execute(q)
    sessions = result.scalars().all()

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No completed sessions found for object_id '{object_id}'. Call /inspect first.",
        )

    updated_sessions = []
    for session in sessions:
        frames = sorted(session.frames, key=lambda x: x.frame_index)

        annotated_bytes_list = []
        pair_results = []
        pair_cm_ranges = []
        seg_len = 20.0
        images_generated = 0

        for frame in frames:
            start_cm = frame.frame_index * seg_len * 2
            len_a = seg_len
            len_b = seg_len if frame.source_frame_b_label else 0.0
            end_cm = start_cm + len_a + len_b
            pair_cm_ranges.append((start_cm, end_cm) if seg_len > 0 else None)

            if not frame.stitched_image_url or ".amazonaws.com/" not in frame.stitched_image_url:
                logger.warning(f"[{session.session_id}] Missing stitched URL for frame {frame.frame_index}")
                annotated_bytes_list.append(None)
                continue

            stitch_key = frame.stitched_image_url.split(".amazonaws.com/")[1]
            try:
                stitched_bytes = await s3_service.download_bytes(stitch_key)
            except Exception as e:
                logger.warning(f"[{session.session_id}] Failed to download stitched URL for frame {frame.frame_index}: {e}")
                annotated_bytes_list.append(None)
                continue

            # Load defects from DB and convert to schema
            pair_result: FramePairResult = _orm_frame_to_schema(frame)

            # API 2 specific: We DO NOT call Gemini again. We trust the JSON coordinates
            # and draw using Python for 100% precision.
            loop = asyncio.get_running_loop()
            annotated_bytes = await loop.run_in_executor(None, annotate_image, stitched_bytes, pair_result.defects, pair_result.overall_result.value)
            ann_key = f"inspections/{session.object_id}/{session.session_id}/frames/annotated/{frame.image_label}_annotated.jpg"
            annotated_url = await s3_service.upload_bytes(annotated_bytes, ann_key, "image/jpeg")

            frame.annotated_image_url = annotated_url
            pair_result.annotated_image_url = annotated_url

            pair_results.append(pair_result)
            annotated_bytes_list.append(annotated_bytes)
            images_generated += 1

        if images_generated > 0:
            summary = compute_statistics(pair_results)
            total_images = session.frames_extracted
            num_pairs = len(frames)

            chart_title = (
                f"WeldVision — {session.object_name or session.object_id} "
                f"| Scan {session.scan_number or 'N/A'} "
                f"| Side: {session.side or 'N/A'} "
                f"| {total_images} frames → {num_pairs} pairs"
                + (f" | {seg_len * total_images:.0f} cm total" if seg_len > 0 else "")
            )

            if any(b is not None for b in annotated_bytes_list):
                loop = asyncio.get_running_loop()
                chart_bytes = await loop.run_in_executor(
                    None,
                    lambda: build_compile_chart(
                        annotated_bytes_list, pair_results, summary, chart_title,
                        pair_cm_ranges=pair_cm_ranges,
                    )
                )
                chart_key = f"inspections/{session.object_id}/{session.session_id}/chart/compile_chart.jpg"
                chart_url = await s3_service.upload_bytes(chart_bytes, chart_key, "image/jpeg")
                session.compile_chart_url = chart_url

        await db.commit()
        updated_sessions.append({
            "session_id": session.session_id,
            "compile_chart_url": session.compile_chart_url,
            "images_generated": images_generated,
            "frames": [{"frame_index": f.frame_index, "annotated_image_url": f.annotated_image_url} for f in frames]
        })

    return {
        "success": True,
        "object_id": object_id.upper(),
        "sessions": updated_sessions
    }
