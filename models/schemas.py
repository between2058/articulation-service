"""
Pydantic schemas for Phidias Articulation Service.

Combines editor bugfix-branch material schemas (TextureInfo, MaterialInfo)
with the service's physics schemas (ArticulationPart/Joint/Data).
"""

from typing import List, Literal, Tuple, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Material schemas (from articulation-editor bugfix--usdz-texture)
# ---------------------------------------------------------------------------

class TextureInfo(BaseModel):
    """Information about a texture used in a material."""
    index: int
    filename: Optional[str] = None
    source: Optional[str] = None  # MIME type or source info
    has_data: bool = False  # Whether raw texture data is available


class MaterialInfo(BaseModel):
    """
    Enhanced material information extracted from GLB materials.
    Supports PBR properties, texture channels, and advanced material types.
    """
    # Basic PBR properties
    diffuse_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5

    # Texture information
    has_base_color_texture: bool = False
    base_color_texture: Optional[TextureInfo] = None

    # Advanced texture channels (for separated channels)
    has_metallic_roughness_texture: bool = False
    metallic_roughness_texture: Optional[TextureInfo] = None

    has_normal_texture: bool = False
    normal_texture: Optional[TextureInfo] = None

    has_occlusion_texture: bool = False
    occlusion_texture: Optional[TextureInfo] = None

    has_emissive_texture: bool = False
    emissive_texture: Optional[TextureInfo] = None

    # Emissive properties
    emissive_factor: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Material flags
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"  # OPAQUE, MASK, BLEND

    # Texture transform (KHR_texture_transform extension)
    tex_coord_offset: Tuple[float, float] = (0.0, 0.0)
    tex_coord_scale: Tuple[float, float] = (1.0, 1.0)
    tex_coord_rotation: float = 0.0  # in radians

    # Material type
    is_unlit: bool = False  # KHR_materials_unlit extension


# ---------------------------------------------------------------------------
# Parse response
# ---------------------------------------------------------------------------

class ParsedPart(BaseModel):
    """
    Part info returned after parsing a GLB file.
    Includes geometric info for display purposes.
    """
    id: str
    name: str
    vertex_count: int = 0
    face_count: int = 0
    bounds_min: Tuple[float, float, float] = (0, 0, 0)
    bounds_max: Tuple[float, float, float] = (0, 0, 0)
    is_watertight: bool = True
    material: Optional[MaterialInfo] = None


# ---------------------------------------------------------------------------
# Physics / articulation schemas (service-specific, used by physics_injector)
# ---------------------------------------------------------------------------

class ArticulationPart(BaseModel):
    """Represents a single mesh part with physics semantics."""
    id: str = Field(..., description="Unique part identifier from mesh name")
    name: str = Field(..., description="Display name for the part")
    type: Literal["link", "joint", "base", "tool"] = Field(
        default="link",
        description="Semantic type of the part",
    )
    role: Literal["actuator", "support", "gripper", "sensor", "other"] = Field(
        default="other",
        description="Functional role of the part",
    )
    mobility: Literal["fixed", "revolute", "prismatic"] = Field(
        default="fixed",
        description="Movement capability of the part",
    )

    # Mass properties
    mass: Optional[float] = Field(
        default=None,
        description="Mass in kg (None = auto-compute from density)",
    )
    density: Optional[float] = Field(
        default=1000.0,
        description="Density in kg/m³ for mass computation",
    )
    center_of_mass: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Center of mass offset [x, y, z]",
    )

    # Collision properties
    collision_type: Literal["mesh", "convexHull", "convexDecomposition", "none"] = Field(
        default="convexHull",
        description="Collision approximation type",
    )

    # Physics material
    static_friction: float = Field(default=0.5, description="Static friction coefficient")
    dynamic_friction: float = Field(default=0.5, description="Dynamic friction coefficient")
    restitution: float = Field(default=0.0, description="Restitution (bounciness) coefficient")


class ArticulationJoint(BaseModel):
    """Represents an articulation joint connecting two parts."""
    name: str = Field(..., description="Unique joint name")
    parent: str = Field(..., description="Parent part ID")
    child: str = Field(..., description="Child part ID")
    type: Literal["fixed", "revolute", "prismatic"] = Field(
        default="revolute",
        description="Joint type",
    )
    axis: Tuple[float, float, float] = Field(
        default=(0, 0, 1),
        description="Joint axis direction [x, y, z]",
    )
    anchor: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Joint pivot point relative to child part [x, y, z]",
    )

    # Joint limits
    lower_limit: Optional[float] = Field(default=-180.0)
    upper_limit: Optional[float] = Field(default=180.0)

    # Drive parameters
    drive_stiffness: Optional[float] = Field(default=1000.0)
    drive_damping: Optional[float] = Field(default=100.0)
    drive_max_force: Optional[float] = Field(default=1000.0)
    drive_type: Literal["position", "velocity", "none"] = Field(default="position")

    # Collision filtering
    disable_collision: bool = Field(default=True)


class ArticulationData(BaseModel):
    """Complete articulation data for USD export."""
    model_name: str = Field(default="robot")
    parts: List[ArticulationPart] = Field(default_factory=list)
    joints: List[ArticulationJoint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API response schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response after successfully uploading and parsing a GLB file."""
    success: bool
    message: str
    filename: str
    model_url: str
    parts: List[ParsedPart]


class ExportResponse(BaseModel):
    """Response after USD/USDZ export."""
    success: bool
    message: str
    download_url: Optional[str] = None
    filename: Optional[str] = None
