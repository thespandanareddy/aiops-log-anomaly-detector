"""
FastAPI Application Factory for AIOps Log Anomaly Detector.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from db.database import db
from src.ingestion.dataset_downloader import ensure_benchmarks_available, create_synthetic_dev_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aiops-detector")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing AIOps Anomaly Detector application...")
    db.init_db()
    create_synthetic_dev_dataset()
    logger.info("Database and sample datasets initialized successfully.")
    yield
    logger.info("AIOps Anomaly Detector shutting down.")


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    app = FastAPI(
        title="Drift-Aware, Explainable AIOps Log Anomaly Detector",
        description="CPU-friendly unsupervised log anomaly detection and explainability system.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/")
    def root():
        return {
            "name": "Drift-Aware, Explainable AIOps Log Anomaly Detector API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/api/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
