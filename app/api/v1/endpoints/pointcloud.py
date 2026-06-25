"""
Point Cloud (.ply / .plv) File Upload API
====================================
POST /api/v1/pointclouds/upload
    Accept a .ply file + form metadata → upload to S3 → save to PostgreSQL

GET /api/v1/pointclouds/{object_id}
    List all point cloud scans stored for an object_id (most recent first)

GET /api/v1/pointclouds/download/{scan_id}
    Return a presigned S3 download URL (valid 1 hour) for a specific scan

.ply files will be processed automatically to calculate bounding box dimensions and generate a mesh.
"""
import os
import uuid
import tempfile
import asyncio
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
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Accepted file extensions
ALLOWED_EXTENSIONS = {".plv", ".ply"}


def _validate_pointcloud_file(filename: str, size: int) -> None:
    """Validate that the uploaded file is a .ply or .plv file within size limits."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only .ply and .plv files are accepted. Got: '{ext or 'no extension'}'",
        )
    if size > MAX_FILE_SIZE_BYTES:
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
        length_mm=scan.length_mm,
        width_mm=scan.width_mm,
        height_mm=scan.height_mm,
        point_count=scan.point_count,
        mesh_s3_url=scan.mesh_s3_url,
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
    summary="Upload a .ply point cloud file",
    description=(
        "Accepts a `.ply` point cloud file from a 3D/LiDAR scanner along with "
        "form metadata. The file is uploaded to AWS S3 and metadata is persisted "
        "to PostgreSQL. `.ply` files will be automatically analyzed to extract dimensions. "
        "Maximum file size: **500 MB**."
    ),
)
async def upload_point_cloud(
    file: UploadFile = File(..., description="The .ply point cloud file"),
    object_id: str = Form(..., description="Weld object ID (e.g. 'A'). Used to group scans."),
    object_name: Optional[str] = Form(None, description="Human-readable object name"),
    scan_number: Optional[str] = Form(None, description="Scan / inspection number"),
    side: Optional[str] = Form(None, description="Side of the weld being scanned"),
    scanner_model: Optional[str] = Form(None, description="Scanner make/model (e.g. 'Faro Focus S150')"),
    notes: Optional[str] = Form(None, description="Any additional notes"),
    linked_session_id: Optional[str] = Form(
        None,
        description=(
            "Optional: link this scan to an existing InspectionSession "
            "(session_id from a previous video inspection of the same weld)."
        ),
    ),
    is_large_object: bool = Form(False, description="Skip table/support plane filtering if True"),
    db: AsyncSession = Depends(get_db),
):
    object_id = object_id.strip().upper()
    scan_id = str(uuid.uuid4())
    filename = file.filename or f"{object_id}_scan.ply"

    logger.info(
        f"Point cloud upload START | object_id={object_id} | file={filename} | scan_id={scan_id}"
    )

    # Read the file into memory
    raw_bytes = await file.read()
    file_size = len(raw_bytes)

    # Validate
    _validate_pointcloud_file(filename, file_size)

    # PROCESS POINT CLOUD using Open3D
    measurements = {}
    mesh_s3_url = None
    if filename.lower().endswith(".ply"):
        from app.services.pointcloud_service import process_point_cloud
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_ply_path = os.path.join(temp_dir, filename)
            with open(temp_ply_path, "wb") as f:
                f.write(raw_bytes)
            
            try:
                measurements, mesh_path = await asyncio.to_thread(
                    process_point_cloud,
                    ply_path=temp_ply_path,
                    is_large_object=is_large_object,
                    output_dir=temp_dir
                )
                
                # Upload mesh if generated
                if mesh_path and os.path.exists(mesh_path):
                    with open(mesh_path, "rb") as mf:
                        mesh_bytes = mf.read()
                    
                    mesh_s3_key = f"pointclouds/{object_id}/{scan_id}/mesh_{filename}"
                    mesh_s3_url = await s3_service.upload_bytes(
                        data=mesh_bytes,
                        key=mesh_s3_key,
                        content_type="application/octet-stream",
                    )
            except Exception as e:
                logger.error(f"Failed to process point cloud: {e}")

    # Upload to S3
    s3_key = f"pointclouds/{object_id}/{scan_id}/{filename}"
    s3_url = await s3_service.upload_bytes(
        data=raw_bytes,
        key=s3_key,
        content_type="application/octet-stream",
    )
    logger.info(f"Point cloud uploaded to S3 → {s3_key} ({file_size / 1024 / 1024:.2f} MB)")

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
        length_mm=measurements.get("length_mm"),
        width_mm=measurements.get("width_mm"),
        height_mm=measurements.get("height_mm"),
        point_count=measurements.get("point_count"),
        mesh_s3_url=mesh_s3_url,
        s3_key=s3_key,
        s3_url=s3_url,
        status="uploaded",
    )
    db.add(db_scan)
    await db.commit()
    await db.refresh(db_scan)

    logger.info(f"Point cloud upload DONE | scan_id={scan_id} | object_id={object_id}")

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
        length_mm=measurements.get("length_mm"),
        width_mm=measurements.get("width_mm"),
        height_mm=measurements.get("height_mm"),
        point_count=measurements.get("point_count"),
        mesh_s3_url=mesh_s3_url,
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
    summary="Get a presigned download URL for a .ply scan (valid 1 hour)",
    description=(
        "Since the S3 bucket is private, this endpoint generates a temporary "
        "presigned URL that allows the caller to download the `.ply` file "
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
