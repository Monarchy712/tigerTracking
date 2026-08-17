"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import ensure_data_dirs
from src.db.models import init_db

app = FastAPI(
    title="Tiger Tracking System",
    description="Camera trap pipeline: blank filtering, individual ID, occupancy mapping, alerts",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    ensure_data_dirs()
    init_db()


@app.get("/")
def root():
    return {
        "service": "Tiger Tracking System",
        "docs": "/docs",
        "endpoints": {
            "run_pipeline": "POST /api/v1/pipeline/run",
            "tigers": "GET /api/v1/tigers",
            "reviews": "GET /api/v1/reviews/pending",
            "alerts": "GET /api/v1/runs/{run_id}/alerts",
            "occupancy_map": "GET /api/v1/exports/{run_id}/map",
        },
    }
