from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from loguru import logger

from app.db.database import get_db
from app.db.models import InspectionSession, PointCloudScan
from app.services.s3_service import s3_service
from pydantic import BaseModel

router = APIRouter()

class ObjectDeleteResponse(BaseModel):
    success: bool = True
    message: str
    deleted_sessions: int
    deleted_pointclouds: int
    deleted_s3_objects: int

@router.delete("/{object_id}", response_model=ObjectDeleteResponse)
async def delete_object(object_id: str, db: AsyncSession = Depends(get_db)):
    """
    Deletes an object completely from the system.
    This includes:
    1. Wiping all associated files in AWS S3 (videos, stitched images, point clouds).
    2. Deleting all InspectionSession and PointCloudScan records from the database.
       (Cascades to frames and defects).
    """
    logger.info(f"Initiating full deletion for object_id: {object_id}")

    # 1. S3 Deletion
    # We delete everything under the object's prefixes.
    s3_deleted_count = 0
    inspection_prefix = f"inspections/{object_id}/"
    pointcloud_prefix = f"pointclouds/{object_id}/"

    try:
        s3_deleted_count += s3_service.delete_prefix(inspection_prefix)
        s3_deleted_count += s3_service.delete_prefix(pointcloud_prefix)
    except Exception as e:
        logger.error(f"Error deleting S3 objects for {object_id}: {e}")
        # Proceed with DB deletion even if S3 fails partially, 
        # or we could fail entirely. Let's proceed to prevent DB orphans.

    # 2. DB Deletion
    try:
        # Count sessions before deleting
        stmt_sessions = select(InspectionSession).where(InspectionSession.object_id == object_id)
        result_sessions = await db.execute(stmt_sessions)
        sessions_to_delete = result_sessions.scalars().all()
        session_count = len(sessions_to_delete)

        # Count point clouds before deleting
        stmt_scans = select(PointCloudScan).where(PointCloudScan.object_id == object_id)
        result_scans = await db.execute(stmt_scans)
        scans_to_delete = result_scans.scalars().all()
        scan_count = len(scans_to_delete)

        # Delete from DB
        # We can just use delete() statement directly
        await db.execute(delete(InspectionSession).where(InspectionSession.object_id == object_id))
        await db.execute(delete(PointCloudScan).where(PointCloudScan.object_id == object_id))
        
        await db.commit()

        logger.info(f"Object {object_id} successfully wiped. Sessions: {session_count}, Scans: {scan_count}, S3 Objects: {s3_deleted_count}")

        return ObjectDeleteResponse(
            message=f"Object {object_id} successfully wiped.",
            deleted_sessions=session_count,
            deleted_pointclouds=scan_count,
            deleted_s3_objects=s3_deleted_count
        )

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete DB records for {object_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete database records: {e}"
        )
