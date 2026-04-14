"""
API Routes for Phidias Articulation MVP

Endpoints:
- POST /upload: Upload GLB file, parse and return parts list
- POST /export: Generate USD from articulation data
- GET /download/{filename}: Download generated USD files
- GET /models/{filename}: Serve uploaded GLB files for frontend viewer
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import (
    UploadResponse,
    ExportRequest,
    ExportResponse,
    Part,
    ArticulationData
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


@router.post("/upload", response_model=UploadResponse)
async def upload_glb(file: UploadFile = File(...)):
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


@router.post("/export", response_model=ExportResponse)
async def export_usd(request: ExportRequest):
    """
    Export articulation data to a physics-enabled USD/USDZ file.

    Takes the GLB filename and articulation data (parts + joints),
    converts to USD with proper physics schemas for Isaac Sim.
    Optionally packages into USDZ format with embedded textures.

    Args:
        request: Export request with GLB filename, articulation data, and output format

    Returns:
        ExportResponse with download URL for the generated file
    """
    glb_path = UPLOAD_DIR / request.glb_filename

    if not glb_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GLB file not found: {request.glb_filename}"
        )

    try:
        # Generate output filename based on format
        base_name = request.articulation.model_name or "robot"
        output_format = request.output_format or "usda"

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
            for part in request.articulation.parts
        ]

        # Build USD stage with meshes
        usda_filename = output_filename.replace('.usdz', '.usda')

        # Extract textures to separate files (not embedded)
        glb_name = Path(request.glb_filename).stem
        texture_dir = OUTPUT_DIR / glb_name
        texture_files = []
        texture_mapping = {}

        # Extract textures from GLB
        try:
            extracted_textures, extracted_data = glb_parser.extract_textures(str(glb_path), glb_name)
            if extracted_textures:
                # Get list of extracted texture files
                texture_files = [str(texture_dir / fname) for fname in extracted_textures.values()]

                # Create mapping: old path -> new name (texture0.png, etc.)
                # For USDZ packaging, textures should be at the root of the archive with simple names
                for idx, fname in enumerate(extracted_textures.values()):
                    old_path = str(texture_dir / fname)
                    # Use simple sequential names for USDZ compatibility
                    new_name = f"texture_{idx}{Path(fname).suffix}"
                    texture_mapping[old_path] = new_name
                    print(f"[INFO] Texture mapping: {fname} -> {new_name}")

                print(f"[INFO] Extracted {len(texture_files)} textures to {texture_dir}")
        except Exception as e:
            print(f"[WARNING] Could not extract textures: {e}")

        # Build USD stage (textures will be referenced externally, not embedded)
        embed_textures = False  # Never embed textures - always reference externally

        # Update mesh data to use mapped texture names if USDZ
        if output_format == "usdz" and texture_mapping:
            # Update material info in mesh_data to use mapped texture names
            # For USDZ, we need to ensure the texture references match what will be in the archive
            for part_id, data in mesh_data.items():
                if 'material' in data:
                    material = data['material']
                    # Handle base color texture - check if it's a MaterialInfo object or dict
                    if hasattr(material, 'has_base_color_texture') and material.has_base_color_texture and material.base_color_texture:
                        # MaterialInfo object
                        base_color_texture = material.base_color_texture
                        if hasattr(base_color_texture, 'filename') and base_color_texture.filename:
                            old_texture_name = base_color_texture.filename
                            # Try to find the full path in mapping
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name and old_texture_name in old_path:
                                    base_color_texture.filename = new_name
                                    print(f"[DEBUG] Updated base color texture reference for USDZ: {old_texture_name} -> {new_name}")
                                    break
                    elif isinstance(material, dict) and material.get('has_base_color_texture') and material.get('base_color_texture'):
                        # Dict format (backward compatibility)
                        base_color_texture = material['base_color_texture']
                        if isinstance(base_color_texture, dict) and 'filename' in base_color_texture:
                            old_texture_name = base_color_texture['filename']
                            # Try to find the full path in mapping
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name and old_texture_name in old_path:
                                    material['base_color_texture']['filename'] = new_name
                                    print(f"[DEBUG] Updated base color texture reference for USDZ: {old_texture_name} -> {new_name}")
                                    break

                    # Handle other texture channels if they exist
                    texture_channels = ['metallic_roughness_texture', 'normal_texture', 'occlusion_texture', 'emissive_texture']
                    for channel in texture_channels:
                        # Check MaterialInfo object format
                        attr_name = f'has_{channel}'
                        texture_attr = channel
                        if hasattr(material, attr_name) and getattr(material, attr_name) and hasattr(material, texture_attr):
                            texture_info = getattr(material, texture_attr)
                            if texture_info and hasattr(texture_info, 'filename') and texture_info.filename:
                                old_texture_name = texture_info.filename
                                if old_texture_name:
                                    # Try to find the full path in mapping
                                    for old_path, new_name in texture_mapping.items():
                                        if old_texture_name in old_path:
                                            texture_info.filename = new_name
                                            print(f"[DEBUG] Updated {channel} reference for USDZ: {old_texture_name} -> {new_name}")
                                            break
                        # Check dict format (backward compatibility)
                        elif isinstance(material, dict) and material.get(f'has_{channel}') and material.get(channel):
                            texture_info = material[channel]
                            if isinstance(texture_info, dict) and 'filename' in texture_info:
                                old_texture_name = texture_info['filename']
                                if old_texture_name:
                                    # Try to find the full path in mapping
                                    for old_path, new_name in texture_mapping.items():
                                        if old_texture_name in old_path:
                                            material[channel]['filename'] = new_name
                                            print(f"[DEBUG] Updated {channel} reference for USDZ: {old_texture_name} -> {new_name}")
                                            break

        stage, part_paths = usd_builder.build_from_parts(
            filename=usda_filename,
            model_name=request.articulation.model_name,
            mesh_data=mesh_data,
            part_info=part_info,
            embed_textures=embed_textures,
            texture_files_dir=str(texture_dir) if texture_dir.exists() else None
        )

        # Inject physics schemas
        physics_injector.inject_physics(
            stage=stage,
            articulation_data=request.articulation,
            part_paths=part_paths
        )

        # Save the USD stage
        usd_path = usd_builder.save_stage(stage)

        # Enhanced validation (optional)
        try:
            from scripts.validate_usd import validate_physics_usd
            validation_results = validate_physics_usd(usd_path)
            if not validation_results["valid"]:
                logger.warning(f"Validation failed for {usd_path}: {validation_results['errors']}")
                # Continue with export but log the issues
            else:
                logger.info(f"Validation passed for {usd_path}")
        except Exception as e:
            logger.warning(f"Validation skipped due to error: {e}")

        # Copy texture files to the same directory as the USD file for both USDA and USDZ
        # This ensures relative paths in the USD file work correctly
        if texture_files and texture_mapping:
            from shutil import copy2
            usd_dir = Path(usd_path).parent
            print(f"[INFO] Copying {len(texture_files)} textures to USD directory: {usd_dir}")

            # Copy each texture file to the USD directory with its mapped name
            copied_textures = []
            for old_path, new_name in texture_mapping.items():
                dest_path = usd_dir / new_name
                try:
                    copy2(old_path, dest_path)
                    copied_textures.append(dest_path)
                    print(f"[INFO] Copied texture: {old_path} -> {dest_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to copy texture {old_path} -> {dest_path}: {e}")

        # If USDZ requested, package it with textures
        final_output_path = usd_path
        if output_format == "usdz":
            # Package USD + textures into USDZ using USD's built-in utility
            usdz_path = usdz_packager.create_usdz(
                usd_file_path=usd_path,
                texture_files=texture_files if texture_files else None,
                texture_mapping=texture_mapping if texture_mapping else None,
                output_filename=output_filename
            )

            # Remove the temporary USD file and copied textures since we now have USDZ
            try:
                Path(usd_path).unlink()
                # Clean up copied texture files
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

        # Construct download URL
        download_url = f"/api/download/{Path(final_output_path).name}"

        return ExportResponse(
            success=True,
            message=f"Successfully exported to {Path(final_output_path).name}",
            download_url=download_url,
            filename=Path(final_output_path).name,
            format=output_format
        )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


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
