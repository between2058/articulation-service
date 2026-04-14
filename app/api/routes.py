"""
API Routes for Phidias Articulation MVP

Endpoints:
- POST /parse-glb: Upload GLB file, parse and return parts list
- POST /export-usda: Generate USDA from articulation data
- POST /export-usdz: Generate USDZ from articulation data
- POST /debug-glb: Diagnostic report of pipeline stages
- GET /download/{filename}: Download generated USD files
- GET /models/{filename}: Serve uploaded GLB files for frontend viewer
- GET /health: Health check
"""

import json
import os
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import (
    UploadResponse,
    ExportResponse,
    ArticulationData,
)
from app.services.glb_parser import glb_parser
from app.services.usd_builder import usd_builder
from app.services.physics_injector import physics_injector
from app.services.usdz_packager import usdz_packager

logger = logging.getLogger(__name__)

router = APIRouter()

# Directories for file storage
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/parse-glb", response_model=UploadResponse)
async def parse_glb(file: UploadFile = File(...)):
    """
    Upload a GLB file and parse its mesh structure.
    
    The file is saved to the uploads directory and parsed using trimesh
    to extract all mesh parts with their geometric properties.
    
    Returns:
        UploadResponse with parsed parts list and model URL
    """
    # Validate file extension
    if not file.filename.lower().endswith('.glb'):
        raise HTTPException(
            status_code=400, 
            detail="Only .glb files are supported"
        )
    
    # Generate unique filename to avoid collisions
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    filepath = UPLOAD_DIR / safe_filename
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(filepath, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {filepath}")
        
        # Parse the GLB file
        parts = glb_parser.parse_glb(str(filepath))
        
        # Construct model URL for frontend
        model_url = f"/api/models/{safe_filename}"
        
        return UploadResponse(
            success=True,
            message=f"Successfully parsed {len(parts)} parts from {file.filename}",
            filename=safe_filename,
            model_url=model_url,
            parts=parts
        )
        
    except ValueError as e:
        # Clean up file if parsing failed
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up file on any error
        if filepath.exists():
            filepath.unlink()
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def _do_export(
    glb_path: Path,
    glb_filename: str,
    articulation: ArticulationData,
    output_format: str,
) -> ExportResponse:
    """
    Shared export pipeline: mesh extraction → material/texture → USD stage
    → physics schemas → USDA on disk → optional USDZ packaging.

    Body lifted from editor's original /export handler. Substitutions:
    request.articulation.* → articulation.* ; request.output_format →
    output_format ; request.glb_filename → glb_filename. The caller has
    already written the GLB to glb_path.
    """
    try:
        # Generate output filename based on format
        base_name = articulation.model_name or "robot"

        if output_format == "usdz":
            output_filename = f"{base_name}_{str(uuid.uuid4())[:8]}.usdz"
        else:
            output_filename = f"{base_name}_{str(uuid.uuid4())[:8]}.usda"

        # Get mesh data from GLB
        mesh_data = glb_parser.get_all_mesh_data(str(glb_path))

        if not mesh_data:
            raise HTTPException(
                status_code=400,
                detail="No mesh data found in GLB file"
            )

        # Build part info list from articulation data
        part_info = [
            {
                'id': part.id,
                'name': part.name,
                'type': part.type,
                'role': part.role,
                'mobility': part.mobility
            }
            for part in articulation.parts
        ]

        # Build USD stage with meshes
        usda_filename = output_filename.replace('.usdz', '.usda')

        # Extract textures to separate files (not embedded)
        glb_name = Path(glb_filename).stem
        texture_dir = OUTPUT_DIR / glb_name
        texture_files = []
        texture_mapping = {}

        # Extract textures from GLB
        try:
            extracted_textures, extracted_data = glb_parser.extract_textures(str(glb_path), glb_name)
            if extracted_textures:
                texture_files = [str(texture_dir / fname) for fname in extracted_textures.values()]

                for idx, fname in enumerate(extracted_textures.values()):
                    old_path = str(texture_dir / fname)
                    new_name = f"texture_{idx}{Path(fname).suffix}"
                    texture_mapping[old_path] = new_name
                    print(f"[INFO] Texture mapping: {fname} -> {new_name}")

                print(f"[INFO] Extracted {len(texture_files)} textures to {texture_dir}")
        except Exception as e:
            print(f"[WARNING] Could not extract textures: {e}")

        # Build USD stage (textures will be referenced externally, not embedded)
        embed_textures = False

        # Update mesh data to use mapped texture names if USDZ
        if output_format == "usdz" and texture_mapping:
            for part_id, data in mesh_data.items():
                if 'material' in data:
                    material = data['material']
                    if hasattr(material, 'has_base_color_texture') and material.has_base_color_texture and material.base_color_texture:
                        base_color_texture = material.base_color_texture
                        if hasattr(base_color_texture, 'filename') and base_color_texture.filename:
                            old_texture_name = base_color_texture.filename
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name and old_texture_name in old_path:
                                    base_color_texture.filename = new_name
                                    print(f"[DEBUG] Updated base color texture reference for USDZ: {old_texture_name} -> {new_name}")
                                    break
                    elif isinstance(material, dict) and material.get('has_base_color_texture') and material.get('base_color_texture'):
                        base_color_texture = material['base_color_texture']
                        if isinstance(base_color_texture, dict) and 'filename' in base_color_texture:
                            old_texture_name = base_color_texture['filename']
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name and old_texture_name in old_path:
                                    material['base_color_texture']['filename'] = new_name
                                    print(f"[DEBUG] Updated base color texture reference for USDZ: {old_texture_name} -> {new_name}")
                                    break

                    texture_channels = ['metallic_roughness_texture', 'normal_texture', 'occlusion_texture', 'emissive_texture']
                    for channel in texture_channels:
                        attr_name = f'has_{channel}'
                        texture_attr = channel
                        if hasattr(material, attr_name) and getattr(material, attr_name) and hasattr(material, texture_attr):
                            texture_info = getattr(material, texture_attr)
                            if texture_info and hasattr(texture_info, 'filename') and texture_info.filename:
                                old_texture_name = texture_info.filename
                                if old_texture_name:
                                    for old_path, new_name in texture_mapping.items():
                                        if old_texture_name in old_path:
                                            texture_info.filename = new_name
                                            print(f"[DEBUG] Updated {channel} reference for USDZ: {old_texture_name} -> {new_name}")
                                            break
                        elif isinstance(material, dict) and material.get(f'has_{channel}') and material.get(channel):
                            texture_info = material[channel]
                            if isinstance(texture_info, dict) and 'filename' in texture_info:
                                old_texture_name = texture_info['filename']
                                if old_texture_name:
                                    for old_path, new_name in texture_mapping.items():
                                        if old_texture_name in old_path:
                                            material[channel]['filename'] = new_name
                                            print(f"[DEBUG] Updated {channel} reference for USDZ: {old_texture_name} -> {new_name}")
                                            break

        stage, part_paths = usd_builder.build_from_parts(
            filename=usda_filename,
            model_name=articulation.model_name,
            mesh_data=mesh_data,
            part_info=part_info,
            embed_textures=embed_textures,
            texture_files_dir=str(texture_dir) if texture_dir.exists() else None
        )

        # Inject physics schemas
        physics_injector.inject_physics(
            stage=stage,
            articulation_data=articulation,
            part_paths=part_paths
        )

        # Save the USD stage
        usd_path = usd_builder.save_stage(stage)

        # Enhanced validation (optional — script lives only in editor repo;
        # import will fail silently in this service, which is fine)
        try:
            from scripts.validate_usd import validate_physics_usd
            validation_results = validate_physics_usd(usd_path)
            if not validation_results["valid"]:
                logger.warning(f"Validation failed for {usd_path}: {validation_results['errors']}")
            else:
                logger.info(f"Validation passed for {usd_path}")
        except Exception as e:
            logger.warning(f"Validation skipped due to error: {e}")

        # Copy texture files to USD directory (same behavior as editor)
        copied_textures = []
        if texture_files and texture_mapping:
            from shutil import copy2
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

        # If USDZ requested, package it
        final_output_path = usd_path
        if output_format == "usdz":
            usdz_path = usdz_packager.create_usdz(
                usd_file_path=usd_path,
                texture_files=texture_files if texture_files else None,
                texture_mapping=texture_mapping if texture_mapping else None,
                output_filename=output_filename
            )

            try:
                Path(usd_path).unlink()
                for texture_path in copied_textures:
                    if texture_path.exists():
                        texture_path.unlink()
                        print(f"[INFO] Cleaned up temporary texture: {texture_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up temporary files: {e}")

            final_output_path = usdz_path
            logger.info(f"Exported USDZ: {final_output_path}")
        else:
            logger.info(f"Exported USD: {final_output_path}")

        download_url = f"/api/download/{Path(final_output_path).name}"

        return ExportResponse(
            success=True,
            message=f"Successfully exported to {Path(final_output_path).name}",
            download_url=download_url,
            filename=Path(final_output_path).name,
            format=output_format
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


async def _save_and_export(
    file: UploadFile,
    articulation: str,
    output_format: str,
) -> ExportResponse:
    """
    Save the uploaded GLB to UPLOAD_DIR, parse the articulation JSON, then
    delegate to _do_export.
    """
    if not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Only .glb files are supported")

    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    glb_path = UPLOAD_DIR / safe_filename

    content = await file.read()
    with open(glb_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved uploaded file for export: {glb_path} ({len(content)} bytes)")

    try:
        art_payload = json.loads(articulation)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid articulation JSON: {e}")

    try:
        art_data = ArticulationData(**art_payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Articulation schema error: {e}")

    return await _do_export(glb_path, safe_filename, art_data, output_format)


@router.post("/export-usda", response_model=ExportResponse)
async def export_usda(
    file: UploadFile = File(...),
    articulation: str = Form(...),
):
    """Export articulation data to a USDA file. Multipart form-data."""
    return await _save_and_export(file, articulation, "usda")


@router.post("/export-usdz", response_model=ExportResponse)
async def export_usdz(
    file: UploadFile = File(...),
    articulation: str = Form(...),
):
    """Export articulation data to a USDZ file. Multipart form-data."""
    return await _save_and_export(file, articulation, "usdz")


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated USD or USDZ file.

    Args:
        filename: Name of the file to download

    Returns:
        File download response with appropriate MIME type
    """
    filepath = OUTPUT_DIR / filename

    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}"
        )

    # Set appropriate media type based on file extension
    media_type = "application/octet-stream"
    if filename.lower().endswith('.usdz'):
        media_type = "model/vnd.usdz+zip"
    elif filename.lower().endswith('.usda'):
        media_type = "text/plain"

    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type=media_type
    )


@router.get("/models/{filename}")
async def serve_model(filename: str):
    """
    Serve an uploaded GLB model file.
    
    Used by the frontend 3D viewer to load the model.
    
    Args:
        filename: Name of the GLB file
        
    Returns:
        File response with appropriate MIME type
    """
    filepath = UPLOAD_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {filename}"
        )
    
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="model/gltf-binary"
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "phidias-articulation-api"}
