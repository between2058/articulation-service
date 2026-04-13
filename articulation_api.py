"""
Articulation Service — FastAPI Entry Point

Provides endpoints for:
- GLB file upload and parsing (with material extraction)
- USDA export with physics schemas + UsdPreviewSurface materials
- USDZ export (packaged with textures)
- File serving for generated outputs and uploaded models

Run with: uvicorn articulation_api:app --host 0.0.0.0 --port 52071
"""

import os
import json
import uuid
import logging
import logging.handlers
import datetime
from pathlib import Path
from shutil import copy2

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pxr import Sdf

from models.schemas import (
    ArticulationData,
    UploadResponse,
    ExportResponse,
)
from services.glb_parser import glb_parser
from services.usd_builder import usd_builder
from services.physics_injector import physics_injector
from services.usdz_packager import usdz_packager

# =============================================================================
# Logging
# =============================================================================

os.makedirs("logs", exist_ok=True)


class TaiwanFormatter(logging.Formatter):
    """Formatter using UTC+8 timestamps."""
    _TZ = datetime.timezone(datetime.timedelta(hours=8))

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{record.msecs:03.0f}"


class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "GET /health" not in record.getMessage()


def _rotating_handler(filename, formatter):
    h = logging.handlers.TimedRotatingFileHandler(
        f"logs/{filename}", when="midnight", backupCount=14, encoding="utf-8",
    )
    h.setFormatter(formatter)
    return h


_fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_access_fmt = TaiwanFormatter("%(asctime)s %(message)s")

logger = logging.getLogger("articulation_api")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(_rotating_handler("app.log", _fmt))
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
logger.addHandler(_console)

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
logging.getLogger("uvicorn.access").addHandler(
    _rotating_handler("access.log", _access_fmt)
)
logging.getLogger("uvicorn").addHandler(_rotating_handler("uvicorn.log", _fmt))

# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="Phidias Articulation Service",
    description="GLB parsing (with materials) and USD export (with UsdPreviewSurface)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Startup / Shutdown
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Phidias Articulation Service...")
    try:
        from pxr import Usd, UsdPhysics, UsdShade
        logger.info("USD libraries loaded successfully")
    except ImportError as e:
        logger.warning(f"USD libraries not fully available: {e}")
    try:
        import trimesh
        logger.info(f"Trimesh version: {trimesh.__version__}")
    except ImportError:
        logger.error("Trimesh not available — GLB parsing will fail")
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Phidias Articulation Service...")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "articulation-service"}


@app.post("/api/parse-glb", response_model=UploadResponse)
async def parse_glb(file: UploadFile = File(...)):
    """
    Upload a GLB file, parse its mesh structure and extract materials.

    Returns parts list with geometry info and PBR material data.
    """
    if not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Only .glb files are supported")

    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    filepath = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        logger.info(f"Saved uploaded file: {filepath} ({len(content)} bytes)")

        parts = glb_parser.parse_glb(str(filepath))
        model_url = f"/api/models/{safe_filename}"

        return UploadResponse(
            success=True,
            message=f"Successfully parsed {len(parts)} parts from {file.filename}",
            filename=safe_filename,
            model_url=model_url,
            parts=parts,
        )

    except ValueError as e:
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        logger.error(f"parse-glb failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Parse failed: {str(e)}")


@app.post("/api/export-usda", response_model=ExportResponse)
async def export_usda(
    file: UploadFile = File(...),
    articulation: str = Form(...),
):
    """
    Accept a GLB file + articulation JSON, export as USDA.

    The articulation JSON string must conform to ArticulationData schema.
    The resulting USDA contains UsdPreviewSurface materials + PhysX schemas.
    """
    return await _export(file, articulation, fmt="usda")


@app.post("/api/export-usdz", response_model=ExportResponse)
async def export_usdz(
    file: UploadFile = File(...),
    articulation: str = Form(...),
):
    """
    Accept a GLB file + articulation JSON, export as USDZ.

    USDZ packages the USDA + textures into a single zip archive.
    """
    return await _export(file, articulation, fmt="usdz")


async def _export(
    file: UploadFile,
    articulation_json: str,
    fmt: str,
) -> ExportResponse:
    """
    Shared export logic for USDA and USDZ.

    Follows the exact flow from articulation-editor bugfix--usdz-texture:
    1. Save GLB
    2. get_all_mesh_data (extracts textures from BIN chunk + returns MaterialInfo)
    3. extract_textures again (cached) to get filename list
    4. Build texture_mapping: old_path -> simple sequential name
    5. Update MaterialInfo texture filenames to mapped names
    6. build_from_parts (USD references ./texture_name.png)
    7. inject_physics
    8. save_stage
    9. Copy textures to USDA directory
    10. Package USDZ with ARKit packager
    11. Clean up temp files
    """
    if not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Only .glb files are supported")

    try:
        art_data = ArticulationData.model_validate_json(articulation_json)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid articulation JSON: {e}"
        )

    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    glb_path = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()
        with open(glb_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved GLB for export: {glb_path} ({len(content)} bytes)")

        # ---- Step 1: Get mesh data (triggers texture extraction internally) ----
        mesh_data = glb_parser.get_all_mesh_data(str(glb_path))

        if not mesh_data:
            raise HTTPException(
                status_code=400, detail="No mesh data found in GLB file"
            )

        # ---- Step 2: Extract textures (cached from step 1) ----
        glb_name = glb_path.stem
        texture_dir = OUTPUT_DIR / glb_name
        texture_files = []
        texture_mapping = {}

        try:
            extracted_textures, extracted_data = glb_parser.extract_textures(
                str(glb_path), glb_name
            )
            if extracted_textures:
                texture_files = [
                    str(texture_dir / fname)
                    for fname in extracted_textures.values()
                ]
                for idx, fname in enumerate(extracted_textures.values()):
                    old_path = str(texture_dir / fname)
                    new_name = f"texture_{idx}{Path(fname).suffix}"
                    texture_mapping[old_path] = new_name
                    print(f"[INFO] Texture mapping: {fname} -> {new_name}")

                print(f"[INFO] Extracted {len(texture_files)} textures to {texture_dir}")
        except Exception as e:
            print(f"[WARNING] Could not extract textures: {e}")

        # ---- Step 3: Update MaterialInfo texture filenames to mapped names ----
        # (Copied from editor routes.py — handles both MaterialInfo objects and dicts)
        if texture_mapping:
            for part_id, data in mesh_data.items():
                if 'material' not in data:
                    continue
                material = data['material']

                # Base color texture
                if hasattr(material, 'has_base_color_texture') and material.has_base_color_texture and material.base_color_texture:
                    base_color_texture = material.base_color_texture
                    if hasattr(base_color_texture, 'filename') and base_color_texture.filename:
                        old_texture_name = base_color_texture.filename
                        for old_path, new_name in texture_mapping.items():
                            if old_texture_name and old_texture_name in old_path:
                                base_color_texture.filename = new_name
                                print(f"[DEBUG] Updated base color texture: {old_texture_name} -> {new_name}")
                                break

                # Other texture channels
                texture_channels = [
                    'metallic_roughness_texture', 'normal_texture',
                    'occlusion_texture', 'emissive_texture',
                ]
                for channel in texture_channels:
                    attr_name = f'has_{channel}'
                    if hasattr(material, attr_name) and getattr(material, attr_name) and hasattr(material, channel):
                        texture_info = getattr(material, channel)
                        if texture_info and hasattr(texture_info, 'filename') and texture_info.filename:
                            old_texture_name = texture_info.filename
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name in old_path:
                                    texture_info.filename = new_name
                                    print(f"[DEBUG] Updated {channel}: {old_texture_name} -> {new_name}")
                                    break

        # ---- Step 4: Build part info list ----
        part_info = [
            {
                "id": part.id,
                "name": part.name,
                "type": part.type,
                "role": part.role,
                "mobility": part.mobility,
            }
            for part in art_data.parts
        ]

        # ---- Step 5: Build USD stage ----
        base_name = art_data.model_name or "robot"
        usda_filename = f"{base_name}_{unique_id}.usda"

        embed_textures = False  # Always reference externally

        stage, part_paths = usd_builder.build_from_parts(
            filename=usda_filename,
            model_name=art_data.model_name,
            mesh_data=mesh_data,
            part_info=part_info,
            embed_textures=embed_textures,
            texture_files_dir=str(texture_dir) if texture_dir.exists() else None,
        )

        # ---- Step 6: Inject physics ----
        physics_injector.inject_physics(
            stage=stage,
            articulation_data=art_data,
            part_paths=part_paths,
        )

        # ---- Step 7: Save USD stage ----
        usd_path = usd_builder.save_stage(stage)

        # ---- Step 8: Copy textures to USD directory ----
        copied_textures = []
        if texture_files and texture_mapping:
            usd_dir = Path(usd_path).parent
            print(f"[INFO] Copying {len(texture_files)} textures to USD directory: {usd_dir}")

            for old_path, new_name in texture_mapping.items():
                dest_path = usd_dir / new_name
                try:
                    copy2(old_path, dest_path)
                    copied_textures.append(dest_path)
                    print(f"[INFO] Copied texture: {old_path} -> {dest_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to copy texture {old_path} -> {dest_path}: {e}")

        # ---- Step 9: Package USDZ if requested ----
        if fmt == "usdz":
            usdz_filename = f"{base_name}_{unique_id}.usdz"

            try:
                usdz_path = usdz_packager.create_usdz(
                    usd_file_path=usd_path,
                    texture_files=texture_files if texture_files else None,
                    texture_mapping=texture_mapping if texture_mapping else None,
                    output_filename=usdz_filename,
                )

                # Clean up temporary USDA and copied textures
                try:
                    Path(usd_path).unlink()
                    for texture_path in copied_textures:
                        if texture_path.exists():
                            texture_path.unlink()
                            print(f"[INFO] Cleaned up temporary texture: {texture_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up temporary files: {e}")

                logger.info(f"Exported USDZ: {usdz_path}")

            except Exception as e:
                logger.error(f"USDZ packaging failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"USDZ packaging failed: {str(e)}",
                )

            download_url = f"/api/download/{usdz_filename}"
            return ExportResponse(
                success=True,
                message=f"Successfully exported to {usdz_filename}",
                download_url=download_url,
                filename=usdz_filename,
            )

        # USDA response
        download_url = f"/api/download/{usda_filename}"
        return ExportResponse(
            success=True,
            message=f"Successfully exported to {usda_filename}",
            download_url=download_url,
            filename=usda_filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Export failed: {str(e)}"
        )


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download a generated USD/USDZ file."""
    # Prevent directory traversal
    safe = Path(filename).name
    filepath = OUTPUT_DIR / safe

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(filepath),
        filename=safe,
        media_type="application/octet-stream",
    )


@app.get("/api/models/{filename}")
async def serve_model(filename: str):
    """Serve an uploaded GLB model file."""
    safe = Path(filename).name
    filepath = UPLOAD_DIR / safe

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {filename}")

    return FileResponse(
        path=str(filepath),
        filename=safe,
        media_type="model/gltf-binary",
    )


# =============================================================================
# Run directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "articulation_api:app",
        host="0.0.0.0",
        port=52071,
        log_level="info",
    )
