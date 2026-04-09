"""
GLB Parser Service

Extracts mesh data from GLB (binary glTF) files using trimesh.
Returns structured part information (geometry + PBR materials)
for the frontend to display and for USD export.

Key additions over the articulation-mvp version:
- _extract_material() reads PBR material data from trimesh meshes
  (baseColorFactor, baseColorTexture, metallicFactor, roughnessFactor,
   normalTexture) and returns ParsedMaterial objects.
- get_all_mesh_data() also returns material data per mesh.
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import trimesh
import numpy as np

from models.schemas import ParsedPart, ParsedMaterial

logger = logging.getLogger(__name__)


class GLBParser:
    """
    Parses GLB files and extracts mesh information + PBR materials.

    Uses trimesh library to load binary glTF files and extract
    individual mesh nodes with their geometric properties and
    associated material data.
    """

    def __init__(self, upload_dir: str = "uploads", texture_dir: str = "outputs/textures"):
        """
        Initialize the parser.

        Args:
            upload_dir: Directory where uploaded GLB files are stored
            texture_dir: Directory to save extracted textures
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.texture_dir = Path(texture_dir)
        self.texture_dir.mkdir(parents=True, exist_ok=True)

    def parse_glb(self, filepath: str) -> List[ParsedPart]:
        """
        Parse a GLB file and extract all mesh parts with materials.

        Args:
            filepath: Path to the GLB file

        Returns:
            List of ParsedPart objects with mesh and material info

        Raises:
            FileNotFoundError: If GLB file doesn't exist
            ValueError: If file cannot be parsed as GLB
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"GLB file not found: {filepath}")

        logger.info(f"Parsing GLB file: {filepath}")

        try:
            # force='scene' ensures we get a Scene even for single meshes
            scene = trimesh.load(str(filepath), force='scene')
        except Exception as e:
            raise ValueError(f"Failed to parse GLB file: {e}")

        parts = []

        if isinstance(scene, trimesh.Scene):
            parts = self._extract_parts_from_scene(scene, filepath.stem)
        elif isinstance(scene, trimesh.Trimesh):
            part = self._create_part_from_mesh(scene, "mesh_0", "mesh_0", filepath.stem)
            parts.append(part)
        else:
            logger.warning(f"Unexpected trimesh type: {type(scene)}")

        logger.info(f"Extracted {len(parts)} parts from GLB")
        return parts

    # ------------------------------------------------------------------
    # Scene traversal
    # ------------------------------------------------------------------

    def _extract_parts_from_scene(
        self, scene: trimesh.Scene, model_stem: str
    ) -> List[ParsedPart]:
        """
        Extract parts from a trimesh Scene object.

        Iterates through the scene graph to get original GLB node names,
        then extracts the corresponding mesh geometry and materials.
        """
        parts = []
        used_ids: set = set()

        # Build geometry key -> node name mapping via scene graph
        geometry_to_node = self._build_geometry_node_map(scene)

        for i, (geometry_key, mesh) in enumerate(scene.geometry.items()):
            if not isinstance(mesh, trimesh.Trimesh):
                continue

            node_name = geometry_to_node.get(geometry_key, geometry_key)
            original_name = node_name or f"Part_{i}"

            part_id = self._generate_unique_id(original_name, used_ids)
            used_ids.add(part_id)

            part = self._create_part_from_mesh(mesh, part_id, original_name, model_stem)
            parts.append(part)

        return parts

    def _build_geometry_node_map(self, scene: trimesh.Scene) -> Dict[str, str]:
        """Build a mapping from geometry key to node name using the scene graph."""
        geometry_to_node: Dict[str, str] = {}
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
        except Exception as e:
            logger.warning(f"Could not parse scene graph: {e}")
        return geometry_to_node

    # ------------------------------------------------------------------
    # Part creation
    # ------------------------------------------------------------------

    def _create_part_from_mesh(
        self,
        mesh: trimesh.Trimesh,
        part_id: str,
        name: str,
        model_stem: str,
    ) -> ParsedPart:
        """
        Create a ParsedPart (including material) from a trimesh mesh.
        """
        bounds = mesh.bounds if mesh.bounds is not None else np.zeros((2, 3))

        is_watertight = False
        try:
            if mesh.is_watertight:
                is_watertight = True
        except Exception:
            pass

        material = self._extract_material(mesh, part_id, model_stem)

        return ParsedPart(
            id=part_id,
            name=name,
            vertex_count=len(mesh.vertices) if mesh.vertices is not None else 0,
            face_count=len(mesh.faces) if mesh.faces is not None else 0,
            bounds_min=tuple(bounds[0].tolist()),
            bounds_max=tuple(bounds[1].tolist()),
            is_watertight=is_watertight,
            material=material,
        )

    # ------------------------------------------------------------------
    # Material extraction  (CRITICAL addition)
    # ------------------------------------------------------------------

    def _extract_material(
        self,
        mesh: trimesh.Trimesh,
        part_id: str,
        model_stem: str,
    ) -> Optional[ParsedMaterial]:
        """
        Extract PBR material data from a trimesh mesh.

        Reads the following from the trimesh PBRMaterial (if present):
        - baseColorFactor  (RGBA tuple)
        - baseColorTexture (PIL Image -> saved as PNG)
        - metallicFactor
        - roughnessFactor
        - normalTexture    (PIL Image -> saved as PNG)

        Returns:
            ParsedMaterial or None if mesh has no material
        """
        visual = getattr(mesh, "visual", None)
        if visual is None:
            return None

        mat = getattr(visual, "material", None)
        if mat is None:
            return None

        # trimesh may provide a PBRMaterial or a SimpleMaterial.
        # We attempt to read PBR fields; fall back to simple color.
        mat_name = getattr(mat, "name", None) or "default"

        # --- Base color factor ---------------------------------------------------
        base_color_factor = (0.8, 0.8, 0.8, 1.0)
        pbr_base = getattr(mat, "baseColorFactor", None)
        if pbr_base is not None:
            try:
                # Could be 0-255 int array or 0-1 float array
                arr = np.array(pbr_base, dtype=float)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                if len(arr) == 3:
                    arr = np.append(arr, 1.0)
                base_color_factor = tuple(float(x) for x in arr[:4])
            except Exception:
                pass
        else:
            # Fallback: SimpleMaterial diffuse color
            diffuse = getattr(mat, "diffuse", None)
            if diffuse is not None:
                try:
                    arr = np.array(diffuse, dtype=float)
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                    if len(arr) == 3:
                        arr = np.append(arr, 1.0)
                    base_color_factor = tuple(float(x) for x in arr[:4])
                except Exception:
                    pass

        # --- Metallic / roughness ------------------------------------------------
        metallic_factor = float(getattr(mat, "metallicFactor", 0.0) or 0.0)
        roughness_factor = float(getattr(mat, "roughnessFactor", 0.5) or 0.5)

        # --- Base color texture --------------------------------------------------
        base_color_texture_path: Optional[str] = None
        base_tex = getattr(mat, "baseColorTexture", None)
        if base_tex is not None:
            base_color_texture_path = self._save_texture(
                base_tex, model_stem, part_id, "baseColor"
            )

        # --- Normal texture ------------------------------------------------------
        normal_texture_path: Optional[str] = None
        normal_tex = getattr(mat, "normalTexture", None)
        if normal_tex is not None:
            normal_texture_path = self._save_texture(
                normal_tex, model_stem, part_id, "normal"
            )

        return ParsedMaterial(
            name=mat_name,
            base_color_factor=base_color_factor,
            base_color_texture_path=base_color_texture_path,
            metallic_factor=metallic_factor,
            roughness_factor=roughness_factor,
            normal_texture_path=normal_texture_path,
        )

    def _save_texture(
        self,
        image,
        model_stem: str,
        part_id: str,
        kind: str,
    ) -> Optional[str]:
        """
        Save a PIL Image to the texture directory and return its path.

        The filename includes a content hash to deduplicate identical textures.
        """
        try:
            from PIL import Image as PILImage

            if not isinstance(image, PILImage.Image):
                # trimesh sometimes wraps textures; try to convert
                if hasattr(image, "convert"):
                    image = image.convert("RGBA")
                else:
                    return None

            # Hash image data for dedup
            raw = image.tobytes()
            content_hash = hashlib.md5(raw).hexdigest()[:8]

            filename = f"{model_stem}_{part_id}_{kind}_{content_hash}.png"
            save_path = self.texture_dir / filename

            if not save_path.exists():
                image.save(str(save_path), format="PNG")
                logger.info(f"Saved texture: {save_path}")

            return str(save_path)
        except Exception as e:
            logger.warning(f"Failed to save {kind} texture for {part_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Mesh data extraction (for USD export)
    # ------------------------------------------------------------------

    def get_mesh_data(self, filepath: str, part_id: str) -> Dict[str, Any]:
        """
        Get detailed mesh data for a specific part.

        Used when exporting to USD to get vertex/face data.
        """
        filepath = Path(filepath)
        scene = trimesh.load(str(filepath), force='scene')

        if isinstance(scene, trimesh.Trimesh):
            if part_id in ("mesh_0", "part_0"):
                return self._mesh_to_dict(scene)
        elif isinstance(scene, trimesh.Scene):
            for name, mesh in scene.geometry.items():
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                sanitized_name = self._generate_unique_id(name, set())
                if sanitized_name == part_id or name == part_id:
                    return self._mesh_to_dict(mesh)

        raise ValueError(f"Part not found: {part_id}")

    def get_all_mesh_data(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        Get mesh data + material data for all parts in a GLB file.

        Returns:
            Dict mapping part_id to {vertices, faces, normals, material}
        """
        filepath = Path(filepath)
        scene = trimesh.load(str(filepath), force='scene')
        model_stem = filepath.stem

        result: Dict[str, Dict[str, Any]] = {}
        used_names: set = set()

        if isinstance(scene, trimesh.Trimesh):
            data = self._mesh_to_dict(scene)
            data["material"] = self._extract_material(scene, "mesh_0", model_stem)
            result["mesh_0"] = data
        elif isinstance(scene, trimesh.Scene):
            geometry_to_node = self._build_geometry_node_map(scene)

            for geometry_key, mesh in scene.geometry.items():
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                node_name = geometry_to_node.get(geometry_key, geometry_key)
                original_name = node_name or geometry_key
                part_id = self._generate_unique_id(original_name, used_names)
                used_names.add(part_id)

                data = self._mesh_to_dict(mesh)
                data["material"] = self._extract_material(mesh, part_id, model_stem)
                result[part_id] = data

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mesh_to_dict(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Convert trimesh to dict with numpy arrays."""
        return {
            "vertices": mesh.vertices.astype(np.float32),
            "faces": mesh.faces.astype(np.int32),
            "normals": (
                mesh.vertex_normals.astype(np.float32)
                if mesh.vertex_normals is not None
                else None
            ),
        }

    def _generate_unique_id(self, base_name: str, used_names: set) -> str:
        """Generate a unique, sanitized part ID."""
        sanitized = base_name.lower()
        sanitized = "".join(c if c.isalnum() else "_" for c in sanitized)
        sanitized = "_".join(filter(None, sanitized.split("_")))

        if not sanitized:
            sanitized = "part"

        unique_name = sanitized
        counter = 1
        while unique_name in used_names:
            unique_name = f"{sanitized}_{counter}"
            counter += 1

        return unique_name

    def _clean_name(self, name: str) -> str:
        """Clean up a mesh name for display."""
        for suffix in [".001", ".002", "_mesh", "_geo", "_Mesh"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        name = name.replace("_", " ")
        name = name.title()
        return name


# Singleton instance
glb_parser = GLBParser()
