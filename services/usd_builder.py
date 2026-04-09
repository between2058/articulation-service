"""
USD Builder Service

Builds USD (Universal Scene Description) stages from mesh data.
Creates the scene hierarchy with Xform and Mesh prims, and applies
UsdPreviewSurface materials so models render with correct colors
in NVIDIA Isaac Sim (fixing the white-model bug).

Key additions over the articulation-mvp version:
- _apply_material()    creates UsdShade.Material + UsdPreviewSurface shader
- _create_uv_texture() creates UsdUVTexture node for base color / normal maps
- Materials are bound to mesh prims via UsdShade.MaterialBindingAPI
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# USD imports
from pxr import Usd, UsdGeom, UsdShade, Gf, Vt, Sdf, Ar

from models.schemas import ParsedMaterial

logger = logging.getLogger(__name__)


class USDBuilder:
    """
    Builds USD stages from mesh data with proper PBR materials.

    Creates a well-structured USD file with:
    - Xform hierarchy for parts
    - Mesh prims with geometry
    - UsdPreviewSurface materials (diffuseColor, metallic, roughness, textures)
    - Suitable for physics simulation in Isaac Sim
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the USD builder.

        Args:
            output_dir: Directory to save generated USD files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage creation
    # ------------------------------------------------------------------

    def create_stage(
        self,
        filename: str,
        model_name: str = "Robot",
        up_axis: str = "Z",
        meters_per_unit: float = 1.0,
    ) -> Usd.Stage:
        """
        Create a new USD stage with proper settings.

        Args:
            filename: Output filename (without path)
            model_name: Name for the root prim
            up_axis: Up axis ("Y" or "Z")
            meters_per_unit: Scale factor (1.0 = meters)

        Returns:
            Usd.Stage object
        """
        filepath = self.output_dir / filename

        stage = Usd.Stage.CreateNew(str(filepath))

        # Set stage metadata
        UsdGeom.SetStageUpAxis(
            stage,
            UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y,
        )
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

        # Create root prim
        root_path = f"/{self._sanitize_prim_name(model_name)}"
        root_xform = UsdGeom.Xform.Define(stage, root_path)

        stage.SetDefaultPrim(root_xform.GetPrim())

        logger.info(f"Created USD stage: {filepath}")
        return stage

    # ------------------------------------------------------------------
    # Physics scene
    # ------------------------------------------------------------------

    def add_physics_scene(self, stage: Usd.Stage) -> str:
        """
        Add a PhysicsScene prim to the stage.

        The PhysicsScene is required for simulation and defines
        global physics parameters like gravity.
        """
        from pxr import UsdPhysics

        root_prim = stage.GetDefaultPrim()
        scene_path = root_prim.GetPath().AppendChild("PhysicsScene")

        physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(9.81)

        logger.info(f"Added PhysicsScene at {scene_path}")
        return str(scene_path)

    # ------------------------------------------------------------------
    # Mesh prim
    # ------------------------------------------------------------------

    def add_mesh_prim(
        self,
        stage: Usd.Stage,
        parent_path: str,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        transform: Optional[Tuple[Tuple[float, ...], ...]] = None,
        material: Optional[ParsedMaterial] = None,
    ) -> str:
        """
        Add a mesh geometry prim to the stage, optionally with material.

        Creates an Xform parent with a Mesh child containing the geometry.
        If a ParsedMaterial is provided, a UsdPreviewSurface material is
        created and bound to the mesh prim.

        Args:
            stage: USD stage
            parent_path: Parent prim path
            name: Name for the mesh prim
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices (triangles)
            normals: Optional Nx3 array of vertex normals
            transform: Optional 4x4 transform matrix
            material: Optional ParsedMaterial to apply

        Returns:
            Path to the Xform prim containing the mesh
        """
        safe_name = self._sanitize_prim_name(name)

        # Create Xform for the part
        xform_path = f"{parent_path}/{safe_name}"
        xform = UsdGeom.Xform.Define(stage, xform_path)

        # Apply transform if provided
        if transform is not None:
            xform_op = xform.AddTransformOp()
            xform_op.Set(Gf.Matrix4d(*[item for row in transform for item in row]))

        # Create Mesh prim under the Xform
        mesh_path = f"{xform_path}/mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # Set vertex positions
        points = Vt.Vec3fArray(
            [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        )
        mesh.CreatePointsAttr(points)

        # Set face vertex counts (all triangles = 3)
        face_vertex_counts = Vt.IntArray([3] * len(faces))
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)

        # Set face vertex indices (flattened)
        face_vertex_indices = Vt.IntArray([int(i) for i in faces.flatten()])
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)

        # Set normals if provided
        if normals is not None:
            normal_array = Vt.Vec3fArray(
                [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
            )
            mesh.CreateNormalsAttr(normal_array)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Subdivision scheme = none (render actual triangles)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        # Apply material if provided
        if material is not None:
            self._apply_material(stage, mesh_path, safe_name, material)

        logger.info(
            f"Added mesh '{safe_name}' with {len(vertices)} vertices, "
            f"{len(faces)} faces"
        )
        return xform_path

    # ------------------------------------------------------------------
    # Material application  (CRITICAL addition — fixes white-model bug)
    # ------------------------------------------------------------------

    def _apply_material(
        self,
        stage: Usd.Stage,
        mesh_prim_path: str,
        part_name: str,
        material: ParsedMaterial,
    ) -> None:
        """
        Create a UsdShade.Material with a UsdPreviewSurface shader and
        bind it to the mesh prim.

        Sets diffuseColor from baseColorFactor, metallic, roughness.
        If a base-color texture exists, creates a UsdUVTexture node and
        connects it to the shader's diffuseColor input.

        Args:
            stage: USD stage
            mesh_prim_path: Path to the Mesh prim to bind the material to
            part_name: Sanitized part name (used for material naming)
            material: ParsedMaterial with PBR values
        """
        root_prim = stage.GetDefaultPrim()
        root_path = str(root_prim.GetPath())

        # Create a /Root/Materials scope
        materials_scope_path = f"{root_path}/Materials"
        if not stage.GetPrimAtPath(materials_scope_path).IsValid():
            UsdGeom.Scope.Define(stage, materials_scope_path)

        mat_name = self._sanitize_prim_name(f"Mat_{part_name}")
        mat_path = f"{materials_scope_path}/{mat_name}"

        # Define UsdShade.Material
        usd_material = UsdShade.Material.Define(stage, mat_path)

        # Define UsdPreviewSurface shader
        shader_path = f"{mat_path}/PreviewSurface"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # --- diffuseColor --------------------------------------------------------
        r, g, b = material.base_color_factor[:3]

        if material.base_color_texture_path and Path(material.base_color_texture_path).exists():
            # Create a UsdUVTexture node connected to diffuseColor
            tex_shader = self._create_uv_texture(
                stage, mat_path, "BaseColorTexture", material.base_color_texture_path
            )
            # Connect texture output -> shader diffuseColor
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_shader.ConnectableAPI(), "rgb"
            )
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(r), float(g), float(b))
            )

        # --- opacity (from alpha channel) ----------------------------------------
        alpha = material.base_color_factor[3] if len(material.base_color_factor) > 3 else 1.0
        if alpha < 1.0:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(alpha))

        # --- metallic / roughness ------------------------------------------------
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(material.metallic_factor)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(material.roughness_factor)
        )

        # --- normal map ----------------------------------------------------------
        if material.normal_texture_path and Path(material.normal_texture_path).exists():
            normal_tex_shader = self._create_uv_texture(
                stage, mat_path, "NormalTexture", material.normal_texture_path
            )
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                normal_tex_shader.ConnectableAPI(), "rgb"
            )

        # Connect shader surface output -> material surface
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        usd_material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        # Bind material to the mesh prim
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if mesh_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(usd_material)

        logger.info(
            f"Applied material '{mat_name}' to {mesh_prim_path} "
            f"(diffuse=[{r:.2f},{g:.2f},{b:.2f}], "
            f"metallic={material.metallic_factor}, "
            f"roughness={material.roughness_factor})"
        )

    def _create_uv_texture(
        self,
        stage: Usd.Stage,
        mat_path: str,
        tex_name: str,
        texture_file_path: str,
    ) -> UsdShade.Shader:
        """
        Create a UsdUVTexture shader node that references a file texture.

        Args:
            stage: USD stage
            mat_path: Parent material prim path
            tex_name: Name for the texture shader prim
            texture_file_path: Absolute path to the PNG texture file

        Returns:
            The UsdShade.Shader for the texture node
        """
        tex_shader_path = f"{mat_path}/{self._sanitize_prim_name(tex_name)}"
        tex_shader = UsdShade.Shader.Define(stage, tex_shader_path)
        tex_shader.CreateIdAttr("UsdUVTexture")

        # Set file path
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_file_path)
        )

        # Set default wrap modes
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

        # Create output
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tex_shader.CreateOutput("a", Sdf.ValueTypeNames.Float)

        # Create a primvar reader for UVs (st)
        st_reader_path = f"{mat_path}/PrimvarReader_st"
        if not stage.GetPrimAtPath(st_reader_path).IsValid():
            st_reader = UsdShade.Shader.Define(stage, st_reader_path)
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        else:
            st_reader = UsdShade.Shader(stage.GetPrimAtPath(st_reader_path))

        # Connect st reader -> texture st input
        tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )

        return tex_shader

    # ------------------------------------------------------------------
    # Build from parts
    # ------------------------------------------------------------------

    def build_from_parts(
        self,
        filename: str,
        model_name: str,
        mesh_data: Dict[str, Dict[str, Any]],
        part_info: List[Dict[str, Any]],
    ) -> Tuple[Usd.Stage, Dict[str, str]]:
        """
        Build a complete USD stage from parsed mesh data.

        Creates the stage, adds all meshes (with materials) as prims,
        and returns a mapping from part IDs to their USD prim paths.

        Args:
            filename: Output USD filename
            model_name: Name for the root prim
            mesh_data: Dict mapping part_id to
                       {vertices, faces, normals, material}
            part_info: List of part dicts with id and type info

        Returns:
            Tuple of (stage, part_id_to_path mapping)
        """
        stage = self.create_stage(filename, model_name)
        root_prim = stage.GetDefaultPrim()
        root_path = str(root_prim.GetPath())

        # Add physics scene
        self.add_physics_scene(stage)

        part_paths: Dict[str, str] = {}
        part_info_map = {p["id"]: p for p in part_info}

        for part_id, data in mesh_data.items():
            info = part_info_map.get(part_id, {"type": "link"})
            part_name = info.get("name", part_id)

            # Extract material (ParsedMaterial or None)
            material = data.get("material")

            prim_path = self.add_mesh_prim(
                stage=stage,
                parent_path=root_path,
                name=part_name,
                vertices=data["vertices"],
                faces=data["faces"],
                normals=data.get("normals"),
                material=material,
            )

            part_paths[part_id] = prim_path

        logger.info(f"Built USD stage with {len(part_paths)} parts")
        return stage, part_paths

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_stage(self, stage: Usd.Stage) -> str:
        """
        Save the USD stage to disk.

        Returns:
            Path to saved file
        """
        stage.GetRootLayer().Save()
        filepath = stage.GetRootLayer().realPath
        logger.info(f"Saved USD stage: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sanitize_prim_name(self, name: str) -> str:
        """
        Sanitize a name for use as a USD prim name.

        USD prim names must:
        - Start with a letter or underscore
        - Contain only letters, numbers, and underscores
        """
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)

        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized

        if not sanitized:
            sanitized = "_unnamed"

        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")

        return sanitized


# Singleton instance
usd_builder = USDBuilder()
