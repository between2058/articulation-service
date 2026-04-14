"""
USD Builder Service

Builds USD (Universal Scene Description) stages from mesh data.
Creates the scene hierarchy with Xform and Mesh prims.

This module handles:
- Creating USD stage with proper metadata
- Adding mesh geometry as Mesh prims
- Setting up Xform hierarchy for parts
- Basic geometry conversion from numpy arrays

The physics schemas are added separately by physics_injector.py
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# USD imports
from pxr import Usd, UsdGeom, Gf, Vt, Sdf, Tf

logger = logging.getLogger(__name__)


class USDBuilder:
    """
    Builds USD stages from mesh data.
    
    Creates a well-structured USD file with proper hierarchy
    suitable for physics simulation in Isaac Sim.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the USD builder.
        
        Args:
            output_dir: Directory to save generated USD files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_stage(
        self,
        filename: str,
        model_name: str = "Robot",
        up_axis: str = "Z",
        meters_per_unit: float = 1.0
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
        
        # Create the stage
        stage = Usd.Stage.CreateNew(str(filepath))
        
        # Set stage metadata
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        
        # Create root prim
        root_path = f"/{self._sanitize_prim_name(model_name)}"
        root_xform = UsdGeom.Xform.Define(stage, root_path)
        
        # Set as default prim
        stage.SetDefaultPrim(root_xform.GetPrim())
        
        logger.info(f"Created USD stage: {filepath}")
        
        return stage
    
    def add_physics_scene(self, stage: Usd.Stage) -> str:
        """
        Add a PhysicsScene prim to the stage.
        
        The PhysicsScene is required for simulation and defines
        global physics parameters like gravity.
        
        Args:
            stage: USD stage
            
        Returns:
            Path to the PhysicsScene prim
        """
        from pxr import UsdPhysics
        
        root_prim = stage.GetDefaultPrim()
        scene_path = root_prim.GetPath().AppendChild("PhysicsScene")
        
        # Define physics scene
        physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        
        # Set gravity for Z-up axis (Isaac Sim convention)
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
        
        logger.info(f"Added PhysicsScene at {scene_path}")

        return str(scene_path)

    def create_material(
        self,
        stage: Usd.Stage,
        parent_path: str,
        name: str,
        material_info: Dict[str, Any],
        model_name: str = None,
        embed_textures: bool = False,
        texture_files_dir: str = None
    ) -> str:
        """
        Create a USD Preview Surface material with proper relative texture paths for USDZ packaging.

        Creates materials under Looks path using UsdPreviewSurface for better compatibility.
        Textures are referenced with relative paths for proper USDZ packaging.

        Args:
            stage: USD stage
            parent_path: Parent path (e.g., "/Robot")
            name: Material name
            material_info: Dict or MaterialInfo with material properties
            model_name: Name of the model (used for texture path resolution)
            embed_textures: If True, embed texture data directly in USD file
            texture_files_dir: Directory where texture files are located (for USDZ)

        Returns:
            Path to the created Material prim
        """
        from pxr import UsdShade

        # Sanitize name for USD
        safe_name = self._sanitize_prim_name(name)

        # Create Looks path following USD conventions
        looks_path = f"{parent_path}/Looks"
        if not stage.GetPrimAtPath(looks_path):
            UsdGeom.Xform.Define(stage, looks_path)

        # Create Material prim under Looks
        material_path = f"{looks_path}/{safe_name}"
        material = UsdShade.Material.Define(stage, material_path)

        # Extract fields from MaterialInfo object or dict
        def _mi_get(key, default=None):
            if hasattr(material_info, key):
                return getattr(material_info, key)
            if isinstance(material_info, dict):
                return material_info.get(key, default)
            return default

        def _tex_filename(texture):
            """MaterialInfo textures can be a TextureInfo object or a plain string."""
            if texture is None:
                return None
            if isinstance(texture, str):
                return texture
            return getattr(texture, 'filename', None)

        diffuse_color = _mi_get('diffuse_color', [0.8, 0.8, 0.8])
        metallic_factor = float(_mi_get('metallic', 0.0) or 0.0)
        roughness_factor = float(_mi_get('roughness', 0.5) or 0.5)
        emissive_factor = _mi_get('emissive_factor', (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)

        base_color_tex = _tex_filename(_mi_get('base_color_texture')) if _mi_get('has_base_color_texture') else None
        mr_tex         = _tex_filename(_mi_get('metallic_roughness_texture')) if _mi_get('has_metallic_roughness_texture') else None
        normal_tex     = _tex_filename(_mi_get('normal_texture')) if _mi_get('has_normal_texture') else None
        occlusion_tex  = _tex_filename(_mi_get('occlusion_texture')) if _mi_get('has_occlusion_texture') else None
        emissive_tex   = _tex_filename(_mi_get('emissive_texture')) if _mi_get('has_emissive_texture') else None

        # Create preview surface shader (simpler and more compatible than MDL)
        shader_path = f"{material_path}/shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # Scalar factor fallbacks (overridden by connections below if textures exist)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(diffuse_color[0]), float(diffuse_color[1]), float(diffuse_color[2]))
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic_factor)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness_factor)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(emissive_factor[0]), float(emissive_factor[1]), float(emissive_factor[2]))
        )

        # Lazy-create a single shared UV reader for st primvar (all textures sample the same UVs)
        st_reader_cache = {}

        def _ensure_st_reader():
            if 'reader' in st_reader_cache:
                return st_reader_cache['output']
            st_reader_path = f"{material_path}/stReader"
            st_reader = UsdShade.Shader.Define(stage, st_reader_path)
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            st_reader_cache['reader'] = st_reader
            st_reader_cache['output'] = st_output
            return st_output

        def _make_uv_texture(prim_name, filename, color_space="auto",
                             scale=None, bias=None):
            """Create a UsdUVTexture sampling the shared st reader.

            color_space: "sRGB" for base color / emissive, "raw" for data maps
                         (metallic-roughness, normal, occlusion).
            scale/bias: optional 4-tuples for normal maps ([0,1] -> [-1,1]).
            """
            tex_path = f"{material_path}/{prim_name}"
            tex = UsdShade.Shader.Define(stage, tex_path)
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(f"./{filename}")
            )
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(_ensure_st_reader())
            if scale is not None:
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            if bias is not None:
                tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
            return tex

        any_texture = False

        # Base color texture → diffuseColor (sRGB)
        if base_color_tex:
            print(f"  🖼️  base color texture: ./{base_color_tex}")
            tex = _make_uv_texture("baseColorTexture", base_color_tex, color_space="sRGB")
            rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
            any_texture = True

        # Metallic-roughness texture (GLTF packs: B=metallic, G=roughness, R=occlusion)
        if mr_tex:
            print(f"  🖼️  metallic/roughness texture: ./{mr_tex}")
            tex = _make_uv_texture("metallicRoughnessTexture", mr_tex, color_space="raw")
            g_out = tex.CreateOutput("g", Sdf.ValueTypeNames.Float)
            b_out = tex.CreateOutput("b", Sdf.ValueTypeNames.Float)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(g_out)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(b_out)
            any_texture = True

        # Normal map → normal (tangent-space, scale [0,1] → [-1,1])
        if normal_tex:
            print(f"  🖼️  normal texture: ./{normal_tex}")
            tex = _make_uv_texture(
                "normalTexture", normal_tex,
                color_space="raw",
                scale=(2.0, 2.0, 2.0, 1.0),
                bias=(-1.0, -1.0, -1.0, 0.0),
            )
            rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(rgb_out)
            any_texture = True

        # Occlusion texture (GLTF stores AO in R channel; emissive and occlusion can
        # share a texture with MR but per-channel connection still works)
        if occlusion_tex:
            print(f"  🖼️  occlusion texture: ./{occlusion_tex}")
            # Reuse MR texture node if occlusion is the same file (GLTF ORM packing)
            if occlusion_tex == mr_tex:
                # Shader node for occlusion share was created above; just wire R channel
                mr_tex_prim = stage.GetPrimAtPath(f"{material_path}/metallicRoughnessTexture")
                tex = UsdShade.Shader(mr_tex_prim)
            else:
                tex = _make_uv_texture("occlusionTexture", occlusion_tex, color_space="raw")
            r_out = tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
            shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(r_out)
            any_texture = True

        # Emissive texture → emissiveColor (sRGB)
        if emissive_tex:
            print(f"  🖼️  emissive texture: ./{emissive_tex}")
            tex = _make_uv_texture("emissiveTexture", emissive_tex, color_space="sRGB")
            rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
            any_texture = True

        if not any_texture:
            print(f"  ⚪ No textures available, using solid color + scalar factors")

        # Connect shader to material
        material_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(material_output)

        logger.info(f"Created USD Preview Surface material '{safe_name}'")

        return material_path

    def bind_material_to_prim(self, stage: Usd.Stage, prim_path: str, material_path: str):
        """
        Bind a material to a prim.

        Args:
            stage: USD stage
            prim_path: Path to the prim to bind material to
            material_path: Path to the material prim
        """
        from pxr import UsdShade

        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            logger.warning(f"Prim not found at {prim_path}, cannot bind material")
            return

        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            logger.warning(f"Material not found at {material_path}")
            return

        # Bind the material to the prim
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        logger.debug(f"Bound material {material_path} to {prim_path}")

    def add_mesh_prim(
        self,
        stage: Usd.Stage,
        parent_path: str,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        uv_coords: Optional[np.ndarray] = None,
        transform: Optional[Tuple[Tuple[float, ...], ...]] = None,
        material_info: Optional[Dict[str, Any]] = None,
        model_name: str = None,
        embed_textures: bool = False,
        texture_files_dir: str = None
    ) -> Tuple[str, Optional[str]]:
        """
        Add a mesh geometry prim to the stage.

        Creates an Xform parent with a Mesh child containing the geometry.
        Optionally creates and binds a material.

        Args:
            stage: USD stage
            parent_path: Parent prim path
            name: Name for the mesh prim
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices (triangles)
            normals: Optional Nx3 array of vertex normals
            uv_coords: Optional Nx2 array of UV texture coordinates
            transform: Optional 4x4 transform matrix
            material_info: Optional material information dict
            model_name: Name of the model (for texture path resolution)
            embed_textures: If True, embed texture data directly in USD
            texture_files_dir: Directory where texture files are located

        Returns:
            Tuple of (path to Xform prim, path to material prim or None)
        """
        # Sanitize name for USD
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

        material_path = None

        # Create material if provided
        if material_info:
            # Generate material name from mesh name if not provided
            if hasattr(material_info, 'name'):
                material_name = material_info.name
            elif isinstance(material_info, dict) and 'name' in material_info:
                material_name = material_info['name']
            else:
                material_name = f"{safe_name}_Material"
            material_path = self.create_material(stage, parent_path, material_name, material_info, model_name, embed_textures, texture_files_dir)

            # Bind material to Mesh prim with MaterialBindingAPI
            from pxr import UsdShade
            mesh_prim = mesh.GetPrim()

            # Add MaterialBindingAPI schema to the Mesh
            UsdShade.MaterialBindingAPI.Apply(mesh_prim)

            # Get the material and bind it
            material = UsdShade.Material.Get(stage, material_path)
            if material:
                UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
                print(f"✅ Bound material {material_name} to Mesh {mesh_path}")

        # Set vertex positions - convert numpy to native Python floats
        points = Vt.Vec3fArray([
            Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
            for v in vertices
        ])
        mesh.CreatePointsAttr(points)

        # Set face vertex counts (all triangles = 3)
        face_vertex_counts = Vt.IntArray([3] * len(faces))
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)

        # Set face vertex indices (flattened)
        face_vertex_indices = Vt.IntArray([int(i) for i in faces.flatten()])
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)

        # Set normals if provided - convert numpy to native Python floats
        if normals is not None:
            normal_array = Vt.Vec3fArray([
                Gf.Vec3f(float(n[0]), float(n[1]), float(n[2]))
                for n in normals
            ])
            mesh.CreateNormalsAttr(normal_array)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UV coordinates if provided - convert numpy to native Python floats
        if uv_coords is not None and len(uv_coords) > 0:
            print(f"[DEBUG] Setting UV coordinates for mesh '{safe_name}': {len(uv_coords)} coords")
            print(f"  Sample UVs: {uv_coords[:3]}")
            print(f"  UV range - U: [{uv_coords[:, 0].min():.4f}, {uv_coords[:, 0].max():.4f}]")
            print(f"  UV range - V: [{uv_coords[:, 1].min():.4f}, {uv_coords[:, 1].max():.4f}]")

            uv_array = Vt.Vec2fArray([
                Gf.Vec2f(float(uv[0]), float(uv[1]))
                for uv in uv_coords
            ])
            # Create primvar for UV coordinates (st) - required for texture mapping
            prim = mesh.GetPrim()

            # In USD, primvars are special attributes with specific naming conventions
            # We create "primvars:st" which is the standard way to store texture coordinates
            try:
                st_attr = prim.CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, False)
                st_attr.Set(uv_array)
                # Set primvar-specific metadata for proper interpolation
                st_attr.SetMetadata("interpolation", UsdGeom.Tokens.vertex)
                # Note: role metadata is optional, the primvar name "primvars:st" is sufficient
                print(f"✅ Added UV primvar (primvars:st) to mesh '{safe_name}'")
            except Exception as e:
                logger.warning(f"Could not create UV primvar for {safe_name}: {e}")
                print(f"[WARNING] Failed to create UV primvar: {e}")

        # Add DisplayColor primvar for proper rendering in Isaac Sim
        # DisplayColor is a fallback for viewport when materials fail
        # According to USD docs, DisplayColor should be used when NO MATERIAL is bound
        # Since we're binding a material, DisplayColor might interfere
        if not material_info:
            # Only add DisplayColor if there's NO material
            print(f"[DEBUG] No material, adding DisplayColor as fallback")
            color = [0.5, 0.5, 0.5]
            try:
                color_array = Vt.Vec3fArray([
                    Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
                    for _ in range(len(vertices))
                ])
                prim = mesh.GetPrim()
                display_color_attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, False)
                display_color_attr.Set(color_array)
                print(f"✅ Added DisplayColor fallback to mesh '{safe_name}'")
            except Exception as e:
                print(f"[WARNING] Could not create DisplayColor: {e}")
        else:
            print(f"[DEBUG] Material exists, not adding DisplayColor (material will provide color)")

        # Set subdivision scheme to none (we want to render actual triangles)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        logger.info(f"Added mesh '{safe_name}' with {len(vertices)} vertices, {len(faces)} faces")

        return xform_path, material_path
    
    def build_from_parts(
        self,
        filename: str,
        model_name: str,
        mesh_data: Dict[str, Dict[str, Any]],
        part_info: List[Dict[str, Any]],
        embed_textures: bool = False,
        texture_files_dir: str = None
    ) -> Tuple[Usd.Stage, Dict[str, str]]:
        """
        Build a complete USD stage from parsed mesh data.

        Creates the stage, adds all meshes as prims, and returns
        a mapping from part IDs to their USD prim paths.

        Args:
            filename: Output USD filename
            model_name: Name for the root prim
            mesh_data: Dict mapping part_id to {vertices, faces, normals}
            part_info: List of part dicts with id and type info
            embed_textures: If True, embed texture data directly in USD file
            texture_files_dir: Directory where texture files are located

        Returns:
            Tuple of (stage, part_id_to_path mapping)
        """
        # Create stage with Z-up axis (Isaac Sim convention)
        stage = self.create_stage(filename, model_name, up_axis="Z")
        root_prim = stage.GetDefaultPrim()
        root_path = str(root_prim.GetPath())

        # Add physics scene
        self.add_physics_scene(stage)

        # Track part paths
        part_paths = {}

        # Create part info lookup
        part_info_map = {p['id']: p for p in part_info}

        # Add each mesh as a prim
        for part_id, data in mesh_data.items():
            # Get part metadata if available
            info = part_info_map.get(part_id, {'type': 'link'})
            part_name = info.get('name', part_id)

            # Extract material info from mesh data
            material_info = data.get('material')

            # Transform from Y-up (GLB) to Z-up (USD/Isaac Sim)
            # Rotation 90 deg around X: x' = x, y' = -z, z' = y
            import numpy as np
            verts = data['vertices']
            verts_zup = np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])

            normals_data = data.get('normals')
            if normals_data is not None:
                normals_data = np.column_stack(
                    [normals_data[:, 0], -normals_data[:, 2], normals_data[:, 1]]
                )

            # Add the mesh
            prim_path, material_path = self.add_mesh_prim(
                stage=stage,
                parent_path=root_path,
                name=part_name,
                vertices=verts_zup,
                faces=data['faces'],
                normals=normals_data,
                uv_coords=data.get('uv_coords'),
                material_info=material_info,
                model_name=model_name,
                embed_textures=embed_textures,
                texture_files_dir=texture_files_dir
            )

            part_paths[part_id] = prim_path

        logger.info(f"Built USD stage with {len(part_paths)} parts")

        return stage, part_paths
    
    def save_stage(self, stage: Usd.Stage) -> str:
        """
        Save the USD stage to disk.

        Args:
            stage: USD stage to save

        Returns:
            Path to saved file
        """
        # Check if DisplayColor exists before saving
        print(f"\n[SAVE-CHECK] Root layer: {stage.GetRootLayer().identifier}")
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                dc_attr = prim.GetAttribute("primvars:displayColor")
                if dc_attr.IsValid():
                    dc_data = dc_attr.Get()
                    print(f"[SAVE-CHECK] Mesh {prim.GetPath()}: DisplayColor has {len(dc_data) if dc_data else 0} values")
                else:
                    print(f"[SAVE-CHECK] Mesh {prim.GetPath()}: NO DisplayColor")

        # Export to string to verify content
        print("\n[SAVE-CHECK] Exporting stage to string to verify content...")
        exported = stage.GetRootLayer().ExportToString()
        # Check for displayColor (lowercase d as USD exports it)
        if 'displayColor' in exported or 'primvars:displayColor' in exported:
            print("✅ DisplayColor found in exported content!")
            # Show the line
            for line in exported.split('\n'):
                if 'displayColor' in line.lower():
                    print(f"  Found: {line.strip()}")
        else:
            print("❌ DisplayColor NOT found in exported content")

        stage.GetRootLayer().Save()
        filepath = stage.GetRootLayer().realPath
        logger.info(f"Saved USD stage: {filepath}")
        return filepath
    
    def _sanitize_prim_name(self, name: str) -> str:
        """
        Sanitize a name for use as a USD prim name.
        
        USD prim names must:
        - Start with a letter or underscore
        - Contain only letters, numbers, and underscores
        
        Args:
            name: Raw name
            
        Returns:
            Sanitized prim name
        """
        # Replace invalid characters with underscore
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        
        # Ensure starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        
        # Handle empty string
        if not sanitized:
            sanitized = '_unnamed'
        
        # Remove consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        return sanitized


# Singleton instance
usd_builder = USDBuilder()
