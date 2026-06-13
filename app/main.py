from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time

from app.core.config import settings
from app.core.logger import setup_logging
from app.db.database import create_tables
from app.api.v1.router import api_router

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} [{settings.APP_ENV}]")
    await create_tables()
    logger.info("Database tables ensured ✓")
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="WeldVision API v2",
    version="2.0.0",
    description="""
## WeldVision API — AI-Powered Weld Defect Inspection (v2)

Designed for the **WeldVision mobile app** (iOS/Swift).

### Workflow
1. Mobile app collects form details (object_id, object_name, scan_number, side)
2. App records inspection video while moving along the weld
3. App POSTs video to `/api/v1/inspections/video`
4. Backend extracts uniformly-spaced frames, stitches pairs, runs Gemini AI inspection
5. All results stored in **PostgreSQL** and uploaded to **S3**
6. Response contains all URLs + full analysis JSON

### Key Endpoints
| Route | Description |
|---|---|
| `POST /api/v1/inspections/video` | Upload video → full inspection pipeline |
| `GET /api/v1/inspections/sessions` | List all past sessions |
| `GET /api/v1/inspections/sessions/{id}` | Full detail of one session |
| `GET /api/v1/inspections/sessions/object/{object_id}` | All scans for one weld object |

### Image Naming Convention
Frames are labelled `{object_id}{N}` — e.g. object_id=`A` → frames `A1`, `A2`, `A3` …  
Pairs are stitched: `A1+A2` → one Gemini call, `A3+A4` → next call, etc.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t        = time.time()
    response = await call_next(request)
    ms       = round((time.time() - t) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(ms)
    return response


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"success": False, "message": "Not found"})


@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.exception(f"Unhandled 500 on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error"},
    )


app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Health"])
async def root():
    return {"message": f"{settings.APP_NAME} is running", "docs": "/docs", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "service": settings.APP_NAME, "version": "2.0.0"}
