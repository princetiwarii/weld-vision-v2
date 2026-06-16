import asyncio
from datetime import datetime, timezone
from sqlalchemy import select, delete
from loguru import logger

from app.db.database import AsyncSessionLocal
from app.db.models import InspectionSession, PointCloudScan
from app.services.s3_service import s3_service

async def cleanup_before_date(target_date: datetime):
    logger.info(f"Starting cleanup for records created before {target_date.isoformat()}")
    
    async with AsyncSessionLocal() as db:
        try:
            # 1. Fetch old sessions
            stmt_sessions = select(InspectionSession).where(InspectionSession.created_at < target_date)
            result_sessions = await db.execute(stmt_sessions)
            old_sessions = result_sessions.scalars().all()
            
            s3_deleted_count = 0
            
            # Delete S3 files for old sessions
            for session in old_sessions:
                # Prefix for a specific session's S3 files
                prefix = f"inspections/{session.object_id}/{session.session_id}/"
                logger.info(f"Deleting S3 prefix: {prefix}")
                try:
                    s3_deleted_count += s3_service.delete_prefix(prefix)
                except Exception as e:
                    logger.error(f"Failed to delete S3 prefix {prefix}: {e}")
            
            # 2. Fetch old point cloud scans
            stmt_scans = select(PointCloudScan).where(PointCloudScan.created_at < target_date)
            result_scans = await db.execute(stmt_scans)
            old_scans = result_scans.scalars().all()
            
            # Delete S3 files for old scans
            for scan in old_scans:
                prefix = f"pointclouds/{scan.object_id}/{scan.scan_id}/"
                logger.info(f"Deleting S3 prefix: {prefix}")
                try:
                    s3_deleted_count += s3_service.delete_prefix(prefix)
                except Exception as e:
                    logger.error(f"Failed to delete S3 prefix {prefix}: {e}")
            
            # 3. Delete from Database
            sessions_deleted = len(old_sessions)
            scans_deleted = len(old_scans)
            
            if sessions_deleted > 0:
                # This will cascade and delete associated InspectionFrames and FrameDefects
                await db.execute(delete(InspectionSession).where(InspectionSession.created_at < target_date))
                logger.info(f"Deleted {sessions_deleted} InspectionSession records.")
                
            if scans_deleted > 0:
                await db.execute(delete(PointCloudScan).where(PointCloudScan.created_at < target_date))
                logger.info(f"Deleted {scans_deleted} PointCloudScan records.")
            
            await db.commit()
            
            logger.info(f"Cleanup completed! Deleted {sessions_deleted} sessions, {scans_deleted} scans, and {s3_deleted_count} S3 objects.")
            
        except Exception as e:
            await db.rollback()
            logger.exception(f"Cleanup failed: {e}")

if __name__ == "__main__":
    # Define the target date: Before 12th June 2026 (UTC)
    target_dt = datetime(2026, 6, 12, tzinfo=timezone.utc)
    
    # Run the async cleanup function
    asyncio.run(cleanup_before_date(target_dt))
