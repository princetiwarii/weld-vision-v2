"""
Point Cloud (.plv) File Upload API
====================================
POST /api/v1/pointclouds/upload
    Accept a .plv file + form metadata → upload to S3 → save to PostgreSQL

GET /api/v1/pointclouds/{object_id}
    List all PLV scans stored for an object_id (most recent first)

GET /api/v1/pointclouds/download/{scan_id}
    Return a presigned S3 download URL (valid 1 hour) for a specific scan

No .plv content is parsed — this is purely a storage and retrieval API.
AI analysis of point cloud data will be added in a future iteration.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from loguru import logger

from app.db.database import get_db
from app.db.models import PointCloudScan
from app.schemas.inspection import (
    PointCloudUploadResponse,
    PointCloudScanSummary,
    PointCloudListResponse,
    PointCloudDownloadResponse,
)
from app.services.s3_service import s3_service

router = APIRouter()

# Maximum allowed file size: 500 MB
MAX_PLV_SIZE_BYTES = 500 * 1024 * 1024

# Accepted file extensions
ALLOWED_EXTENSIONS = {".plv"}


def _validate_plv_file(filename: str, size: int) -> None:
    """Validate that the uploaded file is a .plv file within size limits."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only .plv files are accepted. Got: '{ext or 'no extension'}'",
        )
    if size > MAX_PLV_SIZE_BYTES:
        mb = size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {mb:.1f} MB. Maximum allowed is 500 MB.",
        )


def _orm_to_summary(scan: PointCloudScan) -> PointCloudScanSummary:
    return PointCloudScanSummary(
        scan_id=scan.scan_id,
        object_id=scan.object_id,
        object_name=scan.object_name,
        scan_number=scan.scan_number,
        side=scan.side,
        scanner_model=scan.scanner_model,
        linked_session_id=scan.linked_session_id,
        original_filename=scan.original_filename,
        file_size_bytes=scan.file_size_bytes,
        s3_url=scan.s3_url,
        status=scan.status,
        created_at=scan.created_at,
    )


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    response_model=PointCloudUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .plv point cloud file",
    description=(
        "Accepts a `.plv` point cloud file from a 3D/LiDAR scanner along with "
        "form metadata. The file is uploaded to AWS S3 and metadata is persisted "
        "to PostgreSQL. No content parsing is performed — storage only. "
        "Maximum file size: **500 MB**."
    ),
)
async def upload_point_cloud(
    file: UploadFile = File(..., description="The .plv point cloud file"),
    object_id: str = Form(..., description="Weld object ID (e.g. 'A'). Used to group scans."),
    object_name: Optional[str] = Form(None, description="Human-readable object name"),
    scan_number: Optional[str] = Form(None, description="Scan / inspection number"),
    side: Optional[str] = Form(None, description="Side of the weld being scanned"),
    scanner_model: Optional[str] = Form(None, description="Scanner make/model (e.g. 'Faro Focus S150')"),
    notes: Optional[str] = Form(None, description="Any additional notes"),
    linked_session_id: Optional[str] = Form(
        None,
        description=(
            "Optional: link this PLV scan to an existing InspectionSession "
            "(session_id from a previous video inspection of the same weld)."
        ),
    ),
    db: AsyncSession = Depends(get_db),
):
    object_id = object_id.strip().upper()
    scan_id = str(uuid.uuid4())
    filename = file.filename or f"{object_id}_scan.plv"

    logger.info(
        f"PLV upload START | object_id={object_id} | file={filename} | scan_id={scan_id}"
    )

    # Read the file into memory
    raw_bytes = await file.read()
    file_size = len(raw_bytes)

    # Validate
    _validate_plv_file(filename, file_size)

    # Upload to S3
    s3_key = f"pointclouds/{object_id}/{scan_id}/{filename}"
    s3_url = s3_service.upload_bytes(
        data=raw_bytes,
        key=s3_key,
        content_type="application/octet-stream",
    )
    logger.info(f"PLV uploaded to S3 → {s3_key} ({file_size / 1024 / 1024:.2f} MB)")

    # Persist to DB
    db_scan = PointCloudScan(
        scan_id=scan_id,
        object_id=object_id,
        object_name=object_name,
        scan_number=scan_number,
        side=side,
        scanner_model=scanner_model,
        notes=notes,
        linked_session_id=linked_session_id,
        original_filename=filename,
        file_size_bytes=file_size,
        s3_key=s3_key,
        s3_url=s3_url,
        status="uploaded",
    )
    db.add(db_scan)
    await db.commit()
    await db.refresh(db_scan)

    logger.info(f"PLV upload DONE | scan_id={scan_id} | object_id={object_id}")

    return PointCloudUploadResponse(
        success=True,
        scan_id=scan_id,
        object_id=object_id,
        object_name=object_name,
        scan_number=scan_number,
        side=side,
        scanner_model=scanner_model,
        linked_session_id=linked_session_id,
        original_filename=filename,
        file_size_bytes=file_size,
        s3_url=s3_url,
        status="uploaded",
        created_at=db_scan.created_at,
    )


# ---------------------------------------------------------------------------
# GET /{object_id}  — list all scans for an object
# ---------------------------------------------------------------------------
@router.get(
    "/{object_id}",
    response_model=PointCloudListResponse,
    summary="List all point cloud scans for an object_id",
)
async def list_point_clouds(
    object_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    object_id = object_id.upper()

    q = (
        select(PointCloudScan)
        .where(PointCloudScan.object_id == object_id)
        .order_by(desc(PointCloudScan.created_at))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(q)
    scans = result.scalars().all()

    if not scans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No point cloud scans found for object_id '{object_id}'.",
        )

    return PointCloudListResponse(
        success=True,
        object_id=object_id,
        count=len(scans),
        scans=[_orm_to_summary(s) for s in scans],
    )


# ---------------------------------------------------------------------------
# GET /download/{scan_id}  — presigned download URL
# ---------------------------------------------------------------------------
@router.get(
    "/download/{scan_id}",
    response_model=PointCloudDownloadResponse,
    summary="Get a presigned download URL for a .plv scan (valid 1 hour)",
    description=(
        "Since the S3 bucket is private, this endpoint generates a temporary "
        "presigned URL that allows the caller to download the `.plv` file "
        "directly from S3 for up to 1 hour."
    ),
)
async def download_point_cloud(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
):
    q = select(PointCloudScan).where(PointCloudScan.scan_id == scan_id)
    result = await db.execute(q)
    scan = result.scalar_one_or_none()

    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Point cloud scan '{scan_id}' not found.",
        )

    try:
        presigned_url = s3_service.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": s3_service.bucket, "Key": scan.s3_key},
            ExpiresIn=3600,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for scan_id={scan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not generate download URL: {e}",
        )

    logger.info(f"Presigned URL generated for scan_id={scan_id} | key={scan.s3_key}")

    return PointCloudDownloadResponse(
        success=True,
        scan_id=scan_id,
        original_filename=scan.original_filename,
        download_url=presigned_url,
        expires_in_seconds=3600,
    )
