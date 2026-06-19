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
from app.services.gemini_service import gemini_service
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
        except Exception:
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


async def _run_ai_pipeline_background(session_id: str):
    """
    Background worker that fetches stitched images, runs Gemini, annotates, and compiles charts.
    """
    logger.info(f"[{session_id}] Background AI Pipeline starting...")
    t_start = time.time()

    async with AsyncSessionLocal() as db:
        try:
            # Re-fetch session within the background db context
            q = (
                select(InspectionSession)
                .where(InspectionSession.session_id == session_id)
                .options(
                    selectinload(InspectionSession.frames)
                    .selectinload(InspectionFrame.defects)
                )
            )
            result = await db.execute(q)
            session = result.scalar_one_or_none()
            if not session:
                logger.error(f"[{session_id}] Background task aborted: Session not found")
                return

            frames = sorted(session.frames, key=lambda x: x.frame_index)
            pair_results = []
            annotated_bytes_list = []
            pair_cm_ranges = []
            seg_len = 20.0

            for frame in frames:
                if ".amazonaws.com/" not in frame.stitched_image_url:
                    logger.warning(f"[{session_id}] Invalid stitched URL: {frame.stitched_image_url}")
                    continue
                
                stitch_key = frame.stitched_image_url.split(".amazonaws.com/")[1]
                stitched_bytes = await s3_service.download_bytes(stitch_key)
                logger.info(f"[{session_id}] Downloaded {frame.image_label} for analysis")

                start_cm = frame.frame_index * seg_len * 2
                len_a = seg_len
                len_b = seg_len if frame.source_frame_b_label else 0.0
                end_cm = start_cm + len_a + len_b
                pair_cm_ranges.append((start_cm, end_cm) if seg_len > 0 else None)

                # Run Gemini
                pair_result: FramePairResult = await gemini_service.analyze_pair(
                    stitched_bytes=stitched_bytes,
                    frame_index=frame.frame_index,
                    image_label=frame.image_label,
                    source_frame_a_label=frame.source_frame_a_label,
                    source_frame_b_label=frame.source_frame_b_label,
                    timestamp_a=frame.timestamp_a_seconds,
                    timestamp_b=frame.timestamp_b_seconds,
                    raw_frame_a_url=frame.raw_frame_a_url,
                    raw_frame_b_url=frame.raw_frame_b_url,
                    stitched_image_url=frame.stitched_image_url,
                    annotated_image_url="",
                    start_cm=start_cm,
                    length_cm=len_a + len_b,
                )

                # Annotate & Upload
                loop = asyncio.get_running_loop()
                annotated_bytes = await loop.run_in_executor(None, annotate_image, stitched_bytes, pair_result.defects)
                ann_key = f"inspections/{session.object_id}/{session_id}/frames/annotated/{frame.image_label}_annotated.jpg"
                annotated_url = await s3_service.upload_bytes(annotated_bytes, ann_key, "image/jpeg")
                pair_result.annotated_image_url = annotated_url

                pair_results.append(pair_result)
                annotated_bytes_list.append(annotated_bytes)

                # Update Frame in DB
                frame.annotated_image_url = annotated_url
                frame.overall_result = pair_result.overall_result.value
                frame.weld_quality_score = pair_result.weld_quality_score
                frame.defect_count = len(pair_result.defects)
                frame.defect_summary = pair_result.defect_summary
                frame.standards_compliance = [s.model_dump() for s in pair_result.standards_compliance]
                frame.recommendations = pair_result.recommendations
                frame.model_notes = pair_result.model_notes

                for d in pair_result.defects:
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

            # Compile Chart
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
            
            loop = asyncio.get_running_loop()
            chart_bytes = await loop.run_in_executor(
                None, 
                lambda: build_compile_chart(
                    annotated_bytes_list, pair_results, summary, chart_title,
                    pair_cm_ranges=pair_cm_ranges,
                )
            )
            chart_key = f"inspections/{session.object_id}/{session_id}/chart/compile_chart.jpg"
            chart_url = await s3_service.upload_bytes(chart_bytes, chart_key, "image/jpeg")

            # Update Session
            elapsed = round(time.time() - t_start, 2)
            session.compile_chart_url = chart_url
            session.avg_quality_score = summary.avg_quality_score
            session.total_defects_found = summary.total_defects_found
            session.overall_compliance_aws = summary.overall_compliance_aws
            session.overall_compliance_iso = summary.overall_compliance_iso
            session.pass_count = summary.pass_count
            session.fail_count = summary.fail_count
            session.review_count = summary.review_count
            session.processing_time_seconds = elapsed
            session.completed_at = datetime.now(timezone.utc)

            # Check if session was manually overridden while AI was running
            current_status = await db.scalar(select(InspectionSession.status).where(InspectionSession.session_id == session_id))
            if current_status == "completed":
                logger.warning(f"[{session_id}] AI Pipeline aborting save: session already marked completed (likely manual override).")
                await db.rollback()
                return

            await db.commit()
            logger.info(f"[{session_id}] ✓ AI Pipeline done in {elapsed}s")

        except Exception as exc:
            await db.rollback()
            logger.exception(f"[{session_id}] AI Pipeline failed during GET: {exc}")
            
            # Optionally mark session as failed
            q = select(InspectionSession).where(InspectionSession.session_id == session_id)
            res = await db.execute(q)
            failed_session = res.scalar_one_or_none()
            if failed_session:
                failed_session.status = "failed"
                await db.commit()


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


# ---------------------------------------------------------------------------
# Get session by object_id (all scans for one weld object)
# ---------------------------------------------------------------------------
@router.get(
    "/sessions/object/{object_id}",
    summary="Get all inspection sessions for a specific object_id (triggers background AI if pending)",
)
async def sessions_by_object(
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
    status_changed = False

    for session in sessions:
        if session.status == "pending":
            logger.info(f"[{session.session_id}] Session is pending. Running AI Pipeline synchronously...")
            session.status = "processing"
            await db.commit()
            
            sid = str(session.session_id)
            await _run_ai_pipeline_background(sid)
            
            # Expire just THIS session to force a fresh pull of URLs from the DB
            db.expire(session)
            
            # Re-fetch the session after background task completes
            q_refresh = (
                select(InspectionSession)
                .where(InspectionSession.session_id == sid)
                .options(
                    selectinload(InspectionSession.frames)
                    .selectinload(InspectionFrame.defects)
                )
            )
            res = await db.execute(q_refresh)
            session = res.scalar_one_or_none()
            if not session:
                continue

        pair_results = [_orm_frame_to_schema(f) for f in sorted(session.frames, key=lambda x: x.frame_index)]
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
# Get full session detail (Runs AI if pending)
# ---------------------------------------------------------------------------
@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    summary="Get full detail of one inspection session (triggers background AI if pending)",
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(InspectionSession)
        .where(InspectionSession.session_id == session_id)
        .options(
            selectinload(InspectionSession.frames)
            .selectinload(InspectionFrame.defects)
        )
    )
    result  = await db.execute(q)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    if session.status == "pending":
        logger.info(f"[{session_id}] Session is pending. Running AI Pipeline synchronously...")
        session.status = "processing"
        await db.commit()
        
        await _run_ai_pipeline_background(session_id)
        
        # Expire just THIS session to force a fresh pull of URLs from the DB
        db.expire(session)
        
        # Re-fetch session
        res = await db.execute(q)
        session = res.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session lost during processing.")

    pair_results = [_orm_frame_to_schema(f) for f in sorted(session.frames, key=lambda x: x.frame_index)]
    summary      = compute_statistics(pair_results) if pair_results and session.status == "completed" else None

    return SessionDetailResponse(
        session=_orm_session_to_summary(session),
        compile_chart_url=session.compile_chart_url,
        per_pair_results=pair_results,
        statistical_summary=summary,
    )


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
    sorted_images = sorted(annotated_images, key=lambda f: f.filename)

    for i, frame in enumerate(frames):
        if i < len(sorted_images):
            img_file = sorted_images[i]
            img_bytes = await img_file.read()
            key = f"inspections/{session.object_id}/{session.session_id}/frames/annotated/{frame.image_label}_manual.jpg"
            url = await s3_service.upload_bytes(img_bytes, key, img_file.content_type)
            frame.annotated_image_url = url
        else:
            # Delete extra frames if fewer images are uploaded
            await db.delete(frame)

    await db.commit()

    return {
        "success": True,
        "session_id": session.session_id,
        "compile_chart_url": session.compile_chart_url,
        "frames_updated": min(len(frames), len(sorted_images))
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

