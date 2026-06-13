from fastapi import APIRouter
from app.api.v1.endpoints import video, sessions, reinspect, pointcloud, objects

api_router = APIRouter()
api_router.include_router(video.router,       prefix="/inspections",  tags=["Video Inspection"])
api_router.include_router(sessions.router,    prefix="/inspections",  tags=["Sessions"])
api_router.include_router(reinspect.router,   prefix="/inspections",  tags=["Re-Inspection"])
api_router.include_router(pointcloud.router,  prefix="/pointclouds",  tags=["Point Cloud"])
api_router.include_router(objects.router,     prefix="/objects",      tags=["Objects"])
