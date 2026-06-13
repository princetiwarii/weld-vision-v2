# """
# GET /api/v1/inspections/sessions          — list all sessions (paginated)
# GET /api/v1/inspections/sessions/{id}     — full detail of one session
# GET /api/v1/inspections/sessions/object/{object_id} — all scans for one object
# """
# from typing import Optional, List
# from fastapi import APIRouter, Depends, HTTPException, Query, status
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select, desc
# from sqlalchemy.orm import selectinload
# from loguru import logger

# from app.db.database import get_db
# from app.db.models import InspectionSession, InspectionFrame, FrameDefect
# from app.schemas.inspection import (
#     SessionSummary, SessionDetailResponse, FramePairResult,
#     OverallResult, Defect, DefectSeverity, BoundingBox,
#     WeldingStandardsCompliance, StatisticalSummary, DefectStatEntry,
# )
# from app.services.stats_service import compute_statistics

# router = APIRouter()


# def _orm_defects_to_schema(db_defects: List[FrameDefect]) -> List[Defect]:
#     out = []
#     for d in db_defects:
#         try:
#             bb = None
#             if d.bb_x is not None:
#                 bb = BoundingBox(x=d.bb_x, y=d.bb_y, width=d.bb_width, height=d.bb_height)
#             out.append(Defect(
#                 defect_id=d.defect_id,
#                 type=d.defect_type,
#                 severity=DefectSeverity(d.severity),
#                 description=d.description or "",
#                 confidence=d.confidence,
#                 bounding_box=bb,
#                 length_mm=d.length_mm,
#                 depth_mm=d.depth_mm,
#                 width_mm=d.width_mm,
#                 position=d.position,
#                 standards_reference=d.standards_reference,
#                 recommendation=d.recommendation,
#             ))
#         except Exception:
#             continue
#     return out


# def _orm_frame_to_schema(f: InspectionFrame) -> FramePairResult:
#     defects = _orm_defects_to_schema(f.defects)
#     standards = []
#     for s in (f.standards_compliance or []):
#         try:
#             standards.append(WeldingStandardsCompliance(**s))
#         except Exception:
#             pass

#     return FramePairResult(
#         frame_index=f.frame_index,
#         image_label=f.image_label,
#         source_frame_a_label=f.source_frame_a_label or "",
#         source_frame_b_label=f.source_frame_b_label,
#         timestamp_a_seconds=f.timestamp_a_seconds or 0.0,
#         timestamp_b_seconds=f.timestamp_b_seconds,
#         raw_frame_a_url=f.raw_frame_a_url or "",
#         raw_frame_b_url=f.raw_frame_b_url,
#         stitched_image_url=f.stitched_image_url or "",
#         annotated_image_url=f.annotated_image_url or "",
#         overall_result=OverallResult(f.overall_result or "review"),
#         weld_quality_score=f.weld_quality_score or 0.0,
#         defects=defects,
#         defect_summary=f.defect_summary or {},
#         standards_compliance=standards,
#         recommendations=f.recommendations or [],
#         model_notes=f.model_notes,
#     )


# def _orm_session_to_summary(s: InspectionSession) -> SessionSummary:
#     return SessionSummary(
#         session_id=s.session_id,
#         object_id=s.object_id,
#         object_name=s.object_name,
#         scan_number=s.scan_number,
#         side=s.side,
#         video_filename=s.video_filename,
#         video_url=s.video_url,
#         frames_extracted=s.frames_extracted,
#         avg_quality_score=s.avg_quality_score,
#         total_defects_found=s.total_defects_found,
#         overall_compliance_aws=s.overall_compliance_aws,
#         overall_compliance_iso=s.overall_compliance_iso,
#         status=s.status,
#         compile_chart_url=s.compile_chart_url,
#         created_at=s.created_at,
#         completed_at=s.completed_at,
#     )


# # ---------------------------------------------------------------------------
# # List sessions
# # ---------------------------------------------------------------------------
# @router.get(
#     "/sessions",
#     summary="List all inspection sessions (most recent first)",
# )
# async def list_sessions(
#     limit:  int = Query(20, ge=1, le=100),
#     offset: int = Query(0, ge=0),
#     status_filter: Optional[str] = Query(None, alias="status"),
#     db: AsyncSession = Depends(get_db),
# ):
#     q = select(InspectionSession).order_by(desc(InspectionSession.created_at))
#     if status_filter:
#         q = q.where(InspectionSession.status == status_filter)
#     q = q.offset(offset).limit(limit)

#     result = await db.execute(q)
#     sessions = result.scalars().all()

#     return {
#         "success": True,
#         "count": len(sessions),
#         "sessions": [_orm_session_to_summary(s) for s in sessions],
#     }


# # ---------------------------------------------------------------------------
# # Get session by object_id (all scans for one weld object)
# # ---------------------------------------------------------------------------
# @router.get(
#     "/sessions/object/{object_id}",
#     summary="Get all inspection sessions for a specific object_id",
# )
# async def sessions_by_object(
#     object_id: str,
#     db: AsyncSession = Depends(get_db),
# ):
#     q = (
#         select(InspectionSession)
#         .where(InspectionSession.object_id == object_id.upper())
#         .order_by(desc(InspectionSession.created_at))
#     )
#     result = await db.execute(q)
#     sessions = result.scalars().all()

#     if not sessions:
#         raise HTTPException(
#             status_code=404,
#             detail=f"No sessions found for object_id '{object_id}'",
#         )

#     return {
#         "success": True,
#         "object_id": object_id.upper(),
#         "count": len(sessions),
#         "sessions": [_orm_session_to_summary(s) for s in sessions],
#     }


# # ---------------------------------------------------------------------------
# # Get full session detail
# # ---------------------------------------------------------------------------
# @router.get(
#     "/sessions/{session_id}",
#     response_model=SessionDetailResponse,
#     summary="Get full detail of one inspection session (frames + defects)",
# )
# async def get_session(
#     session_id: str,
#     db: AsyncSession = Depends(get_db),
# ):
#     q = (
#         select(InspectionSession)
#         .where(InspectionSession.session_id == session_id)
#         .options(
#             selectinload(InspectionSession.frames)
#             .selectinload(InspectionFrame.defects)
#         )
#     )
#     result  = await db.execute(q)
#     session = result.scalar_one_or_none()

#     if not session:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Session '{session_id}' not found.",
#         )

#     pair_results = [_orm_frame_to_schema(f) for f in sorted(session.frames, key=lambda x: x.frame_index)]
#     summary      = compute_statistics(pair_results) if pair_results else None

#     return SessionDetailResponse(
#         session=_orm_session_to_summary(session),
#         per_pair_results=pair_results,
#         statistical_summary=summary,
#     )


"""
GET /api/v1/inspections/sessions          — list all sessions (paginated)
GET /api/v1/inspections/sessions/{id}     — full detail of one session
GET /api/v1/inspections/sessions/object/{object_id} — all scans for one object
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.db.database import get_db
from app.db.models import InspectionSession, InspectionFrame, FrameDefect
from app.schemas.inspection import (
    SessionSummary, SessionDetailResponse, FramePairResult,
    OverallResult, Defect, DefectSeverity, BoundingBox,
    WeldingStandardsCompliance, StatisticalSummary, DefectStatEntry,
    FrameUrlSummary,
)
from app.services.stats_service import compute_statistics

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
    summary="Get all inspection sessions for a specific object_id",
)
async def sessions_by_object(
    object_id: str,
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(InspectionSession)
        .where(InspectionSession.object_id == object_id.upper())
        .options(selectinload(InspectionSession.frames))
        .order_by(desc(InspectionSession.created_at))
    )
    result = await db.execute(q)
    sessions = result.scalars().all()

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No sessions found for object_id '{object_id}'",
        )

    return {
        "success": True,
        "object_id": object_id.upper(),
        "count": len(sessions),
        "sessions": [_orm_session_to_summary(s, include_frames=True) for s in sessions],
    }


# ---------------------------------------------------------------------------
# Get full session detail
# ---------------------------------------------------------------------------
@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    summary="Get full detail of one inspection session (frames + defects)",
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

    pair_results = [_orm_frame_to_schema(f) for f in sorted(session.frames, key=lambda x: x.frame_index)]
    summary      = compute_statistics(pair_results) if pair_results else None

    return SessionDetailResponse(
        session=_orm_session_to_summary(session),
        per_pair_results=pair_results,
        statistical_summary=summary,
    )

