import asyncio
import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from app.db.database import AsyncSessionLocal
from app.db.models import InspectionSession
from app.services.s3_service import S3Service

async def main():
    print("Starting data cleanup...")
    
    # "till 17th june after that i want to keep"
    # This means anything created before 2026-06-18 00:00:00 UTC should be deleted.
    cutoff_date = datetime(2026, 6, 18, 0, 0, 0, tzinfo=timezone.utc)
    print(f"Cutoff date: {cutoff_date}")
    
    s3_service = S3Service()
    
    async with AsyncSessionLocal() as session:
        # Find all old sessions
        query = select(InspectionSession).where(InspectionSession.created_at < cutoff_date)
        result = await session.execute(query)
        old_sessions = result.scalars().all()
        
        print(f"Found {len(old_sessions)} sessions to delete.")
        
        for idx, s in enumerate(old_sessions, 1):
            print(f"[{idx}/{len(old_sessions)}] Deleting session {s.session_id} (created at {s.created_at})")
            
            # 1. Delete S3 files
            # S3 prefix is: inspections/{object_id}/{session_id}/
            # The object_id is saved uppercase in S3 but let's just use s.object_id
            prefix = f"inspections/{s.object_id.upper()}/{s.session_id}/"
            deleted_count = s3_service.delete_prefix(prefix)
            print(f"  - Deleted {deleted_count} files from S3 ({prefix})")
            
            # 2. Delete from DB
            await session.delete(s)
            
        # Commit DB changes
        await session.commit()
        print("Cleanup finished successfully!")

if __name__ == "__main__":
    asyncio.run(main())
