"""
Phidias Articulation MVP - FastAPI Main Application

This is the entry point for the backend API server.
Provides endpoints for:
- GLB file upload and parsing
- USD export with physics schemas
- File serving for frontend

Run with: uvicorn app.main:app --reload --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Phidias Articulation API",
    description="GLB to USD articulation pipeline for NVIDIA Isaac Sim",
    version="1.0.0"
)

# Configure CORS for frontend access
# In production, replace "*" with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload and output directories exist
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.on_event("startup")
async def startup_event():
    """Log startup and check dependencies."""
    logger.info("Starting Phidias Articulation API...")
    
    # Check if USD is available
    try:
        from pxr import Usd, UsdPhysics
        logger.info("USD libraries loaded successfully")
    except ImportError as e:
        logger.warning(f"USD libraries not fully available: {e}")
    
    # Check if trimesh is available
    try:
        import trimesh
        logger.info(f"Trimesh version: {trimesh.__version__}")
    except ImportError:
        logger.error("Trimesh not available - GLB parsing will fail")
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Phidias Articulation API...")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Phidias Articulation API",
        "version": "1.0.0",
        "description": "GLB to USD articulation pipeline",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
