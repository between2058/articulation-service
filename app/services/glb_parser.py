"""
GLB Parser Service

Extracts mesh data from GLB (binary glTF) files using trimesh.
Returns structured part information for the frontend to display.

Key assumptions:
- Each mesh node in the GLB represents a selectable part
- Mesh names from GLB are used as part IDs
- Scene graphs are flattened for simplicity in MVP
"""

import os
import json
import struct
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from io import BytesIO

import trimesh
import numpy as np
from PIL import Image

from app.models.schemas import ParsedPart, MaterialInfo, TextureInfo

logger = logging.getLogger(__name__)


class GLBParser:
    """
    Parses GLB files and extracts mesh information.
    
    Uses trimesh library to load binary glTF files and extract
    individual mesh nodes with their geometric properties.
    """
    
    def __init__(self, upload_dir: str = "uploads", output_dir: str = "outputs"):
        """
        Initialize the parser.

        Args:
            upload_dir: Directory where uploaded GLB files are stored
            output_dir: Directory where extracted textures will be saved
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._gltf_cache = {}  # Cache for parsed GLTF JSON data
        self._texture_cache = {}  # Cache for extracted textures

    def _separate_texture_channels(self, image_data: bytes, output_dir: Path, base_name: str) -> Dict[str, TextureInfo]:
        """
        Separate combined PBR texture channels (metallic/roughness/occlusion) into individual textures.

        Args:
            image_data: Raw image data bytes
            output_dir: Directory to save separated textures
            base_name: Base name for output files

        Returns:
            Dict mapping channel names to TextureInfo objects
        """
        try:
            # Load the combined texture
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get image dimensions
            width, height = image.size

            # Split channels
            r, g, b = image.split()

            # Create separate textures:
            # - Metallic from B channel (blue)
            # - Roughness from G channel (green)
            # - Occlusion from R channel (red)

            separated_textures = {}

            # Metallic texture (blue channel)
            metallic_img = Image.new('L', (width, height))
            metallic_img.paste(b)
            metallic_filename = f"{base_name}_metallic.png"
            metallic_path = output_dir / metallic_filename
            metallic_img.save(metallic_path, 'PNG')
            separated_textures['metallic'] = TextureInfo(
                index=0,  # Placeholder index
                filename=metallic_filename,
                source='image/png',
                has_data=True
            )

            # Roughness texture (green channel)
            roughness_img = Image.new('L', (width, height))
            roughness_img.paste(g)
            roughness_filename = f"{base_name}_roughness.png"
            roughness_path = output_dir / roughness_filename
            roughness_img.save(roughness_path, 'PNG')
            separated_textures['roughness'] = TextureInfo(
                index=0,  # Placeholder index
                filename=roughness_filename,
                source='image/png',
                has_data=True
            )

            # Occlusion texture (red channel)
            occlusion_img = Image.new('L', (width, height))
            occlusion_img.paste(r)
            occlusion_filename = f"{base_name}_occlusion.png"
            occlusion_path = output_dir / occlusion_filename
            occlusion_img.save(occlusion_path, 'PNG')
            separated_textures['occlusion'] = TextureInfo(
                index=0,  # Placeholder index
                filename=occlusion_filename,
                source='image/png',
                has_data=True
            )

            logger.info(f"Separated texture channels: {list(separated_textures.keys())}")
            return separated_textures

        except Exception as e:
            logger.warning(f"Failed to separate texture channels: {e}")
            return {}

    def _resolve_texture_source(self, gltf_texture: Dict[str, Any]) -> Optional[int]:
        """
        Return the image index that a GLTF texture should resolve to.

        Prefers the EXT_texture_webp extension's source (for TRELLIS.2 and
        other WebP-only outputs that omit the top-level fallback), then
        falls back to the standard GLTF `source`.

        Returns None if neither is available.
        """
        extensions = gltf_texture.get('extensions') or {}
        webp = extensions.get('EXT_texture_webp') or {}
        if 'source' in webp:
            return webp['source']
        if 'source' in gltf_texture:
            return gltf_texture['source']
        return None

    def _extract_texture_transform(self, texture_info: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Extract texture transform information from KHR_texture_transform extension.

        Args:
            texture_info: GLTF texture info dictionary

        Returns:
            Tuple of (offset, scale, rotation) where:
                - offset: (x, y) translation
                - scale: (x, y) scaling factors
                - rotation: rotation in radians
        """
        # Default values
        offset = (0.0, 0.0)
        scale = (1.0, 1.0)
        rotation = 0.0

        # Check for KHR_texture_transform extension
        if 'extensions' in texture_info and 'KHR_texture_transform' in texture_info['extensions']:
            transform = texture_info['extensions']['KHR_texture_transform']

            # Extract offset
            if 'offset' in transform and len(transform['offset']) >= 2:
                offset = (float(transform['offset'][0]), float(transform['offset'][1]))

            # Extract scale
            if 'scale' in transform and len(transform['scale']) >= 2:
                scale = (float(transform['scale'][0]), float(transform['scale'][1]))

            # Extract rotation (in radians)
            if 'rotation' in transform:
                rotation = float(transform['rotation'])

        return offset, scale, rotation
    
    def parse_glb(self, filepath: str) -> List[ParsedPart]:
        """
        Parse a GLB file and extract all mesh parts.
        
        Args:
            filepath: Path to the GLB file
            
        Returns:
            List of ParsedPart objects with mesh info
            
        Raises:
            FileNotFoundError: If GLB file doesn't exist
            ValueError: If file cannot be parsed as GLB
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"GLB file not found: {filepath}")
        
        logger.info(f"Parsing GLB file: {filepath}")
        
        try:
            # Load the GLB file
            # force='scene' ensures we get a Scene even for single meshes
            scene = trimesh.load(str(filepath), force='scene')
        except Exception as e:
            raise ValueError(f"Failed to parse GLB file: {e}")
        
        parts = []
        
        # Handle both Scene and single Trimesh objects
        if isinstance(scene, trimesh.Scene):
            parts = self._extract_parts_from_scene(scene)
        elif isinstance(scene, trimesh.Trimesh):
            # Single mesh - create a part from it
            part = self._create_part_from_mesh(scene, "mesh_0", "mesh_0")
            parts.append(part)
        else:
            logger.warning(f"Unexpected trimesh type: {type(scene)}")
        
        logger.info(f"Extracted {len(parts)} parts from GLB")
        return parts

    def _parse_gltf_from_glb(self, filepath: Path) -> Dict[str, Any]:
        """
        Parse GLTF JSON directly from GLB file to extract texture and material info.

        Args:
            filepath: Path to GLB file

        Returns:
            Dict containing GLTF JSON data
        """
        filepath = str(filepath)
        if filepath in self._gltf_cache:
            return self._gltf_cache[filepath]

        with open(filepath, 'rb') as f:
            # Read GLB header
            magic = f.read(4)
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]

            # Read JSON chunk
            json_chunk_length = struct.unpack('<I', f.read(4))[0]
            json_chunk_type = f.read(4)

            if json_chunk_type != b'JSON':
                raise ValueError("GLB file does not contain JSON chunk")

            json_data = f.read(json_chunk_length).decode('utf-8')
            gltf = json.loads(json_data)

            # Store in cache
            self._gltf_cache[filepath] = gltf
            return gltf

    def extract_textures(self, filepath: str, output_subdir: Optional[str] = None) -> Tuple[Dict[int, str], Dict[int, bytes]]:
        """
        Extract texture images from GLB file's BIN chunk and save to disk.

        Args:
            filepath: Path to the GLB file
            output_subdir: Optional subdirectory name for organizing textures

        Returns:
            Tuple of (dict mapping texture_index to filename, dict mapping texture_index to raw data)
        """
        filepath = Path(filepath)

        # Check cache first
        cache_key = f"{filepath}_{output_subdir or 'default'}"
        if cache_key in self._texture_cache:
            return self._texture_cache[cache_key]

        # Create output subdirectory if specified
        if output_subdir:
            texture_dir = self.output_dir / output_subdir
            texture_dir.mkdir(parents=True, exist_ok=True)
        else:
            texture_dir = self.output_dir

        gltf = self._parse_gltf_from_glb(filepath)
        extracted_textures = {}
        extracted_data = {}

        if 'images' not in gltf:
            logger.info(f"No images found in {filepath}")
            self._texture_cache[cache_key] = (extracted_textures, extracted_data)
            return extracted_textures, extracted_data

        # Read BIN chunk
        bin_data = self._read_bin_chunk(filepath)

        for i, image in enumerate(gltf['images']):
            try:
                # Get bufferView for this image
                if 'bufferView' not in image:
                    continue

                buffer_view_idx = image['bufferView']
                if buffer_view_idx >= len(gltf['bufferViews']):
                    logger.warning(f"Invalid bufferView index for image {i}")
                    continue

                buffer_view = gltf['bufferViews'][buffer_view_idx]
                offset = buffer_view.get('byteOffset', 0)
                length = buffer_view['byteLength']

                # Extract image data
                if bin_data is None:
                    continue

                if offset + length > len(bin_data):
                    logger.warning(f"Image {i} data exceeds BIN chunk size")
                    continue

                image_data = bin_data[offset:offset + length]

                # Determine file extension from mimeType or uri
                ext = '.png'  # default
                if 'mimeType' in image:
                    if image['mimeType'] == 'image/jpeg':
                        ext = '.jpg'
                    elif image['mimeType'] == 'image/png':
                        ext = '.png'
                    elif image['mimeType'] == 'image/webp':
                        ext = '.webp'

                # Also check uri as fallback
                if 'uri' in image and not image['uri'].startswith('data:'):
                    uri_ext = Path(image['uri']).suffix.lower()
                    if uri_ext in ['.jpg', '.jpeg', '.png', '.webp']:
                        ext = uri_ext

                # Save texture file.
                # WebP is not ARKit-conformant and most USD renderers
                # (Hydra Storm, Blender, Isaac Sim) cannot decode it, so
                # convert to PNG during extraction. PNG/JPEG are kept as-is.
                if ext == '.webp':
                    try:
                        img = Image.open(BytesIO(image_data))
                        if img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGBA' if 'A' in img.mode else 'RGB')
                        texture_filename = f"texture_{i}.png"
                        texture_path = texture_dir / texture_filename
                        img.save(texture_path, format='PNG')
                        with open(texture_path, 'rb') as f:
                            image_data = f.read()
                        logger.info(
                            f"Converted WebP texture {i} to PNG "
                            f"({texture_filename}, {len(image_data)} bytes)"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to convert WebP texture {i}, keeping raw: {e}"
                        )
                        texture_filename = f"texture_{i}{ext}"
                        texture_path = texture_dir / texture_filename
                        with open(texture_path, 'wb') as img_file:
                            img_file.write(image_data)
                else:
                    texture_filename = f"texture_{i}{ext}"
                    texture_path = texture_dir / texture_filename
                    with open(texture_path, 'wb') as img_file:
                        img_file.write(image_data)

                extracted_textures[i] = texture_filename
                extracted_data[i] = image_data  # Store raw data
                logger.info(f"Extracted texture {i}: {texture_filename} ({len(image_data)} bytes)")

            except Exception as e:
                logger.warning(f"Failed to extract texture {i}: {e}")
                continue

        # Cache the results (both filenames and raw data)
        self._texture_cache[cache_key] = (extracted_textures, extracted_data)
        logger.info(f"Extracted {len(extracted_textures)} textures from {filepath}")

        return extracted_textures, extracted_data

    def _read_bin_chunk(self, filepath: Path) -> Optional[bytes]:
        """
        Read the BIN chunk from a GLB file.

        Args:
            filepath: Path to GLB file

        Returns:
            Binary data from BIN chunk or None if not found
        """
        try:
            with open(filepath, 'rb') as f:
                # Read GLB header (12 bytes)
                magic = f.read(4)
                version = struct.unpack('<I', f.read(4))[0]
                length = struct.unpack('<I', f.read(4))[0]

                # Read JSON chunk
                json_chunk_length = struct.unpack('<I', f.read(4))[0]
                json_chunk_type = f.read(4)

                if json_chunk_type != b'JSON':
                    logger.warning("GLB file does not contain JSON chunk")
                    return None

                # Skip JSON data
                f.read(json_chunk_length)

                # Read BIN chunk (if exists)
                if f.tell() < length:
                    bin_chunk_length = struct.unpack('<I', f.read(4))[0]
                    bin_chunk_type = f.read(4)

                    if bin_chunk_type == b'BIN\0':
                        bin_data = f.read(bin_chunk_length)
                        return bin_data
                    else:
                        logger.warning(f"Unexpected chunk type after JSON: {bin_chunk_type}")
                        return None
                else:
                    logger.info("No BIN chunk found (file uses external textures)")
                    return None

        except Exception as e:
            logger.error(f"Failed to read BIN chunk: {e}")
            return None

    def _extract_material_from_gltf(
        self,
        gltf: Dict[str, Any],
        material_index: int,
        extracted_textures: Optional[Dict[int, str]] = None,
        extracted_data: Optional[Dict[int, bytes]] = None
    ) -> MaterialInfo:
        """
        Extract enhanced material information from GLTF JSON with support for advanced features.

        Args:
            gltf: Parsed GLTF JSON
            material_index: Index of the material to extract
            extracted_textures: Dict mapping texture index to filename (from extract_textures)
            extracted_data: Dict mapping texture index to raw binary data (from extract_textures)

        Returns:
            MaterialInfo with enhanced material properties
        """
        if 'materials' not in gltf or material_index >= len(gltf['materials']):
            return MaterialInfo()

        material = gltf['materials'][material_index]
        material_info = MaterialInfo()

        # Check for unlit material extension (KHR_materials_unlit)
        if 'extensions' in material and 'KHR_materials_unlit' in material['extensions']:
            material_info.is_unlit = True

        # Extract PBR properties
        pbr = material.get('pbrMetallicRoughness', {})

        # Base color
        base_color_factor = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
        material_info.diffuse_color = tuple(base_color_factor[:3])  # RGB only
        material_info.alpha = base_color_factor[3] if len(base_color_factor) >= 4 else 1.0

        # Metallic and roughness
        material_info.metallic = pbr.get('metallicFactor', 0.0)
        material_info.roughness = pbr.get('roughnessFactor', 0.5)

        # Emissive properties
        emissive_factor = material.get('emissiveFactor', [0.0, 0.0, 0.0])
        material_info.emissive_factor = tuple(emissive_factor)

        # Check for base color texture
        base_color_texture = pbr.get('baseColorTexture', {})
        if base_color_texture and 'index' in base_color_texture:
            texture_index = base_color_texture['index']
            if 'textures' in gltf and texture_index < len(gltf['textures']):
                gltf_texture = gltf['textures'][texture_index]
                image_index = self._resolve_texture_source(gltf_texture)
                if image_index is not None and 'images' in gltf:
                    if image_index < len(gltf['images']):
                        image = gltf['images'][image_index]

                        # Create base color texture info
                        texture_filename = None
                        has_data = False

                        if extracted_textures and image_index in extracted_textures:
                            texture_filename = extracted_textures[image_index]
                            has_data = image_index in extracted_data if extracted_data else False

                        material_info.has_base_color_texture = True
                        material_info.base_color_texture = TextureInfo(
                            index=image_index,
                            filename=texture_filename,
                            source=image.get('mimeType', 'image/png'),
                            has_data=has_data
                        )

                        # Extract texture transform if available
                        tex_offset, tex_scale, tex_rotation = self._extract_texture_transform(base_color_texture)
                        material_info.tex_coord_offset = tex_offset
                        material_info.tex_coord_scale = tex_scale
                        material_info.tex_coord_rotation = tex_rotation

        # Check for metallic-roughness texture
        mr_texture = pbr.get('metallicRoughnessTexture', {})
        if mr_texture and 'index' in mr_texture:
            texture_index = mr_texture['index']
            if 'textures' in gltf and texture_index < len(gltf['textures']):
                gltf_texture = gltf['textures'][texture_index]
                image_index = self._resolve_texture_source(gltf_texture)
                if image_index is not None and 'images' in gltf:
                    if image_index < len(gltf['images']):
                        image = gltf['images'][image_index]

                        # Create metallic-roughness texture info
                        texture_filename = None
                        has_data = False

                        if extracted_textures and image_index in extracted_textures:
                            texture_filename = extracted_textures[image_index]
                            has_data = image_index in extracted_data if extracted_data else False

                        material_info.has_metallic_roughness_texture = True
                        material_info.metallic_roughness_texture = TextureInfo(
                            index=image_index,
                            filename=texture_filename,
                            source=image.get('mimeType', 'image/png'),
                            has_data=has_data
                        )

        # Check for normal texture
        normal_texture = material.get('normalTexture', {})
        if normal_texture and 'index' in normal_texture:
            texture_index = normal_texture['index']
            if 'textures' in gltf and texture_index < len(gltf['textures']):
                gltf_texture = gltf['textures'][texture_index]
                image_index = self._resolve_texture_source(gltf_texture)
                if image_index is not None and 'images' in gltf:
                    if image_index < len(gltf['images']):
                        image = gltf['images'][image_index]

                        # Create normal texture info
                        texture_filename = None
                        has_data = False

                        if extracted_textures and image_index in extracted_textures:
                            texture_filename = extracted_textures[image_index]
                            has_data = image_index in extracted_data if extracted_data else False

                        material_info.has_normal_texture = True
                        material_info.normal_texture = TextureInfo(
                            index=image_index,
                            filename=texture_filename,
                            source=image.get('mimeType', 'image/png'),
                            has_data=has_data
                        )

        # Check for occlusion texture
        occlusion_texture = material.get('occlusionTexture', {})
        if occlusion_texture and 'index' in occlusion_texture:
            texture_index = occlusion_texture['index']
            if 'textures' in gltf and texture_index < len(gltf['textures']):
                gltf_texture = gltf['textures'][texture_index]
                image_index = self._resolve_texture_source(gltf_texture)
                if image_index is not None and 'images' in gltf:
                    if image_index < len(gltf['images']):
                        image = gltf['images'][image_index]

                        # Create occlusion texture info
                        texture_filename = None
                        has_data = False

                        if extracted_textures and image_index in extracted_textures:
                            texture_filename = extracted_textures[image_index]
                            has_data = image_index in extracted_data if extracted_data else False

                        material_info.has_occlusion_texture = True
                        material_info.occlusion_texture = TextureInfo(
                            index=image_index,
                            filename=texture_filename,
                            source=image.get('mimeType', 'image/png'),
                            has_data=has_data
                        )

        # Check for emissive texture
        emissive_texture = material.get('emissiveTexture', {})
        if emissive_texture and 'index' in emissive_texture:
            texture_index = emissive_texture['index']
            if 'textures' in gltf and texture_index < len(gltf['textures']):
                gltf_texture = gltf['textures'][texture_index]
                image_index = self._resolve_texture_source(gltf_texture)
                if image_index is not None and 'images' in gltf:
                    if image_index < len(gltf['images']):
                        image = gltf['images'][image_index]

                        # Create emissive texture info
                        texture_filename = None
                        has_data = False

                        if extracted_textures and image_index in extracted_textures:
                            texture_filename = extracted_textures[image_index]
                            has_data = image_index in extracted_data if extracted_data else False

                        material_info.has_emissive_texture = True
                        material_info.emissive_texture = TextureInfo(
                            index=image_index,
                            filename=texture_filename,
                            source=image.get('mimeType', 'image/png'),
                            has_data=has_data
                        )

        # Material flags
        material_info.double_sided = material.get('doubleSided', False)
        material_info.alpha_mode = material.get('alphaMode', 'OPAQUE')

        return material_info

    def _extract_parts_from_scene(self, scene: trimesh.Scene) -> List[ParsedPart]:
        """
        Extract parts from a trimesh Scene object.
        
        Iterates through the scene graph to get original GLB node names,
        then extracts the corresponding mesh geometry.
        
        Args:
            scene: Trimesh Scene object
            
        Returns:
            List of ParsedPart objects
        """
        parts = []
        
        # Track used IDs to avoid duplicates
        used_ids = set()
        
        # Build a mapping from geometry key to node name using the scene graph
        geometry_to_node = {}
        try:
            # scene.graph.to_flattened() gives us node info with geometry references
            for node_name in scene.graph.nodes:
                if node_name == scene.graph.base_frame:
                    continue
                try:
                    # Get the transform and geometry name for this node
                    transform, geometry_name = scene.graph.get(node_name)
                    if geometry_name and geometry_name in scene.geometry:
                        # Store node_name -> geometry_name mapping
                        if geometry_name not in geometry_to_node:
                            geometry_to_node[geometry_name] = node_name
                except (ValueError, TypeError):
                    # Node doesn't have geometry reference
                    continue
        except Exception as e:
            logger.warning(f"Could not parse scene graph: {e}")
        
        # Now iterate over geometry, using node names where available
        for i, (geometry_key, mesh) in enumerate(scene.geometry.items()):
            if not isinstance(mesh, trimesh.Trimesh):
                # Skip non-mesh geometry (e.g., point clouds, paths)
                continue
            
            # Try to get the original node name from the scene graph
            # Fall back to geometry key if not found
            node_name = geometry_to_node.get(geometry_key, geometry_key)
            original_name = node_name or f"Part_{i}"
            
            # Generate unique part ID (sanitized for internal use)
            part_id = self._generate_unique_id(original_name, used_ids)
            used_ids.add(part_id)
            
            # Keep original name for display
            part = self._create_part_from_mesh(mesh, part_id, original_name)
            parts.append(part)
        
        return parts
    
    def _create_part_from_mesh(
        self, 
        mesh: trimesh.Trimesh, 
        part_id: str, 
        name: str
    ) -> ParsedPart:
        """
        Create a ParsedPart from a trimesh mesh.
        
        Args:
            mesh: Trimesh mesh object
            part_id: Unique identifier for the part
            name: Display name for the part
            
        Returns:
            ParsedPart with geometric information
        """
        # Get bounding box
        bounds = mesh.bounds if mesh.bounds is not None else np.zeros((2, 3))
        
        # Check if mesh is watertight (closed/manifold)
        # Some meshes might be None or empty, so handle safely
        is_watertight = False
        try:
            if mesh.is_watertight:
                is_watertight = True
        except Exception:
            # If check fails, assume not watertight
            pass
        
        # Extract material information if available
        material_info = None
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            material_info = self._extract_material_info(mesh.visual)

        return ParsedPart(
            id=part_id,
            name=name,
            vertex_count=len(mesh.vertices) if mesh.vertices is not None else 0,
            face_count=len(mesh.faces) if mesh.faces is not None else 0,
            bounds_min=tuple(bounds[0].tolist()),
            bounds_max=tuple(bounds[1].tolist()),
            is_watertight=is_watertight,
            material=material_info
        )
    
    def _generate_unique_id(self, base_name: str, used_names: set) -> str:
        """
        Generate a unique part ID.
        
        Sanitizes the name and adds a suffix if already used.
        
        Args:
            base_name: Preferred name
            used_names: Set of already used names
            
        Returns:
            Unique, sanitized ID string
        """
        # Sanitize: lowercase, replace spaces/special chars with underscore
        sanitized = base_name.lower()
        sanitized = ''.join(c if c.isalnum() else '_' for c in sanitized)
        sanitized = '_'.join(filter(None, sanitized.split('_')))  # Remove empty parts
        
        if not sanitized:
            sanitized = "part"
        
        # Ensure uniqueness
        unique_name = sanitized
        counter = 1
        while unique_name in used_names:
            unique_name = f"{sanitized}_{counter}"
            counter += 1
        
        return unique_name
    
    def _clean_name(self, name: str) -> str:
        """
        Clean up a mesh name for display.
        
        Removes common prefixes/suffixes added by 3D software.
        
        Args:
            name: Raw mesh name
            
        Returns:
            Cleaned display name
        """
        # Remove common suffixes
        for suffix in ['.001', '.002', '_mesh', '_geo', '_Mesh']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Replace underscores with spaces for display
        name = name.replace('_', ' ')
        
        # Capitalize words
        name = name.title()
        
        return name

    def _extract_material_info(self, visual) -> Optional[MaterialInfo]:
        """
        Extract enhanced material information from trimesh visual data.

        Args:
            visual: Trimesh visual object (could be TextureVisual, ColorVisuals, etc.)

        Returns:
            MaterialInfo with enhanced material properties or None if not available
        """
        material_info = MaterialInfo()

        try:
            # Try to get diffuse color (most common attribute)
            if hasattr(visual, 'diffuse') and visual.diffuse is not None:
                # diffuse could be RGB or RGBA
                diffuse = visual.diffuse
                if len(diffuse) >= 3:
                    material_info.diffuse_color = tuple(diffuse[:3])  # RGB only
                    material_info.alpha = diffuse[3] if len(diffuse) == 4 else 1.0

            # Try alternative color sources
            if material_info.diffuse_color == (1.0, 1.0, 1.0):  # Default value
                # Check for vertex colors
                if hasattr(visual, 'colors') and visual.colors is not None:
                    colors = visual.colors
                    if len(colors) > 0:
                        # Use first color as the material color
                        avg_color = np.mean(colors[:, :3], axis=0)
                        material_info.diffuse_color = tuple(avg_color.tolist())

            # Try to get texture if available
            if hasattr(visual, 'material') and visual.material is not None:
                mat = visual.material

                # Handle PBR materials (glTF 2.0 standard)
                if hasattr(mat, 'baseColorFactor'):
                    # PBR material has baseColorFactor (RGB + alpha)
                    base_color = mat.baseColorFactor
                    # Convert from 0-255 range to 0-1 range if needed
                    if max(base_color[:3]) > 1.0:
                        material_info.diffuse_color = (
                            base_color[0]/255.0,
                            base_color[1]/255.0,
                            base_color[2]/255.0
                        )
                    else:
                        material_info.diffuse_color = tuple(base_color[:3])
                    if len(base_color) >= 4:
                        material_info.alpha = base_color[3]

                # Handle texture
                if hasattr(mat, 'image') and mat.image is not None:
                    # Mark that base color texture exists
                    material_info.has_base_color_texture = True
                    material_info.base_color_texture = TextureInfo(
                        index=0,  # Placeholder
                        filename=getattr(mat.image, 'name', getattr(mat.image, 'filename', None)),
                        source="image/png",  # Default assumption
                        has_data=False
                    )

                # Extract PBR-specific properties if available
                if hasattr(mat, 'metallicFactor'):
                    material_info.metallic = mat.metallicFactor
                if hasattr(mat, 'roughnessFactor'):
                    material_info.roughness = mat.roughnessFactor

                # Extract emissive properties if available
                if hasattr(mat, 'emissiveFactor'):
                    material_info.emissive_factor = tuple(mat.emissiveFactor)

            # Check for double-sided flag
            if hasattr(visual, 'material') and visual.material is not None:
                mat = visual.material
                if hasattr(mat, 'doubleSided'):
                    material_info.double_sided = mat.doubleSided

            # If we have no meaningful material info, return None
            if (material_info.diffuse_color == (1.0, 1.0, 1.0) and
                material_info.alpha == 1.0 and
                material_info.metallic == 0.0 and
                material_info.roughness == 0.5 and
                not material_info.has_base_color_texture):
                return None

            return material_info

        except Exception as e:
            logger.warning(f"Failed to extract material info: {e}")
            return None

    def get_mesh_data(self, filepath: str, part_id: str) -> Dict[str, Any]:
        """
        Get detailed mesh data for a specific part.
        
        Used when exporting to USD to get vertex/face data.
        
        Args:
            filepath: Path to GLB file
            part_id: ID of the part to extract
            
        Returns:
            Dict with vertices, faces, and normals
        """
        filepath = Path(filepath)
        scene = trimesh.load(str(filepath), force='scene')
        
        # Find the matching mesh
        if isinstance(scene, trimesh.Trimesh):
            if part_id in ["mesh_0", "part_0"]:
                return self._mesh_to_dict(scene)
        elif isinstance(scene, trimesh.Scene):
            # Search in geometry
            for name, mesh in scene.geometry.items():
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                sanitized_name = self._generate_unique_id(name, set())
                if sanitized_name == part_id or name == part_id:
                    return self._mesh_to_dict(mesh)
        
        raise ValueError(f"Part not found: {part_id}")
    
    def _mesh_to_dict(
        self,
        mesh: trimesh.Trimesh,
        filepath: Optional[Path] = None,
        material_index: Optional[int] = None,
        extracted_textures: Optional[Dict[int, str]] = None,
        extracted_data: Optional[Dict[int, bytes]] = None
    ) -> Dict[str, Any]:
        """
        Convert trimesh to dict with numpy arrays.

        Args:
            mesh: Trimesh mesh
            filepath: Path to source GLB file (optional, for material extraction)
            material_index: Material index from GLTF (optional, for material extraction)
            extracted_textures: Dict mapping texture index to filename (from extract_textures)
            extracted_data: Dict mapping texture index to raw binary data (from extract_textures)

        Returns:
            Dict with mesh data
        """
        result = {
            'vertices': mesh.vertices.astype(np.float32),
            'faces': mesh.faces.astype(np.int32),
            'normals': mesh.vertex_normals.astype(np.float32) if mesh.vertex_normals is not None else None
        }

        # Extract UV coordinates if available
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                # UV coordinates are typically Nx2 or Nx3 arrays
                uv_coords = mesh.visual.uv
                # Ensure we have 2D UV coordinates (U, V)
                if uv_coords.shape[1] >= 2:
                    result['uv_coords'] = uv_coords[:, :2].astype(np.float32)
                    print(f"[DEBUG] Extracted {len(uv_coords)} UV coordinates")

        # Extract material information - prefer GLTF extraction if available
        material_info = None

        if filepath and material_index is not None:
            try:
                filepath_obj = Path(filepath)
                gltf = self._parse_gltf_from_glb(filepath_obj)
                material_info = self._extract_material_from_gltf(gltf, material_index, extracted_textures, extracted_data)
                print(f"[DEBUG] Extracted material from GLTF: {material_info}")
            except Exception as e:
                logger.warning(f"Failed to extract material from GLTF: {e}")
                material_info = None

        # Fallback to trimesh material extraction
        if not material_info and hasattr(mesh, 'visual') and mesh.visual is not None:
            material_info = self._extract_material_info(mesh.visual)
            if material_info:
                print(f"[DEBUG] Extracted material from trimesh: {material_info}")

        if material_info:
            result['material'] = material_info

        return result
    
    def get_all_mesh_data(self, filepath: str, extract_textures: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get mesh data for all parts in a GLB file.

        Args:
            filepath: Path to the GLB file
            extract_textures: If True, extract and save textures to disk

        Returns:
            Dict mapping part_id to mesh data dict with texture references
        """
        filepath = Path(filepath)
        scene = trimesh.load(str(filepath), force='scene')

        result = {}
        used_names = set()

        # Parse GLTF for material indices
        gltf = None
        try:
            gltf = self._parse_gltf_from_glb(filepath)
        except Exception as e:
            logger.warning(f"Could not parse GLTF from GLB: {e}")

        # Extract textures if requested and if we have GLTF data
        extracted_textures = {}
        extracted_data = {}
        if extract_textures and gltf and 'images' in gltf:
            # Use a subdirectory based on the GLB filename (without extension)
            texture_subdir = filepath.stem
            extracted_textures, extracted_data = self.extract_textures(str(filepath), texture_subdir)
            logger.info(f"Extracted {len(extracted_textures)} textures for {filepath.stem}")

        if isinstance(scene, trimesh.Trimesh):
            # For single mesh, try to find material index from GLTF
            material_index = 0  # Default to first material
            result["mesh_0"] = self._mesh_to_dict(scene, filepath, material_index, extracted_textures, extracted_data)
        elif isinstance(scene, trimesh.Scene):
            # Build geometry key to node name mapping (same as _extract_parts_from_scene)
            geometry_to_node = {}
            geometry_to_material = {}  # Track material index for each geometry
            try:
                for node_name in scene.graph.nodes:
                    if node_name == scene.graph.base_frame:
                        continue
                    try:
                        transform, geometry_name = scene.graph.get(node_name)
                        if geometry_name and geometry_name in scene.geometry:
                            if geometry_name not in geometry_to_node:
                                geometry_to_node[geometry_name] = node_name
                    except (ValueError, TypeError):
                        continue
            except Exception:
                pass

            # If we have GLTF data, build mapping from geometry to material index
            if gltf and 'meshes' in gltf:
                for mesh_idx, mesh_def in enumerate(gltf['meshes']):
                    if 'extras' in mesh_def and 'targetNames' in mesh_def['extras']:
                        # This might be a skinned mesh or have target names
                        pass
                    # Look for material index in mesh definition
                    if 'primitives' in mesh_def and len(mesh_def['primitives']) > 0:
                        # For simplicity, use the material of the first primitive
                        primitive = mesh_def['primitives'][0]
                        if 'material' in primitive:
                            material_index = primitive['material']
                            # We need to map this back to geometry name
                            # This is tricky without more info from GLTF
                            # For now, we'll try to match by index

            # Iterate using node names
            for geometry_key, mesh in scene.geometry.items():
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                node_name = geometry_to_node.get(geometry_key, geometry_key)
                original_name = node_name or geometry_key
                part_id = self._generate_unique_id(original_name, used_names)
                used_names.add(part_id)

                # Try to determine material index for this geometry
                # For now, assume materials are assigned in order or use first material
                material_index = 0

                result[part_id] = self._mesh_to_dict(mesh, filepath, material_index, extracted_textures, extracted_data)

        return result


# Singleton instance for use in API routes
glb_parser = GLBParser()
