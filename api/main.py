"""
PolarGrid AI — FastAPI Backend
RESTful API for energy management system operations.
SIH 2026 • Problem Statement 26061 • Ministry of Earth Sciences / NCPOR
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import HealthResponse
from api.routes.forecast import router as forecast_router
from api.routes.dispatch import router as dispatch_router
from api.routes.optimization import router as optimization_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FastAPI App ===
app = FastAPI(
    title="PolarGrid AI API",
    description="AI-Driven Smart Energy Management System for Polar Research Stations (Maitri Antarctic Station)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """System health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        project="PolarGrid AI",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# Include modular routers
app.include_router(forecast_router)
app.include_router(dispatch_router)
app.include_router(optimization_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
