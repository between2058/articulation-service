"""
Pydantic schemas for Phidias Articulation Service.

Defines data models for:
- ParsedMaterial: PBR material data extracted from GLB meshes
- ParsedPart: Individual mesh parts with geometric info
- ArticulationPart: Part with semantic labels for physics
- ArticulationJoint: Joint connecting two parts
- ArticulationData: Complete model data for USD export
- UploadResponse: API response after GLB upload/parse
- ExportResponse: API response after USD export
"""

from typing import List, Literal, Tuple, Optional
from pydantic import BaseModel, Field


class ParsedMaterial(BaseModel):
    """
    PBR material data extracted from a GLB mesh.

    Stores the metallic-roughness PBR parameters needed
    to reconstruct UsdPreviewSurface materials in USD.
    """
    name: str = Field(default="default", description="Material name from GLB")
    base_color_factor: Tuple[float, float, float, float] = Field(
        default=(0.8, 0.8, 0.8, 1.0),
        description="Base color RGBA (linear, 0-1)",
    )
    base_color_texture_path: Optional[str] = Field(
        default=None,
        description="Path to saved base-color texture PNG (None if no texture)",
    )
    metallic_factor: float = Field(
        default=0.0,
        description="Metallic factor 0-1",
    )
    roughness_factor: float = Field(
        default=0.5,
        description="Roughness factor 0-1",
    )
    normal_texture_path: Optional[str] = Field(
        default=None,
        description="Path to saved normal-map texture PNG (None if no texture)",
    )


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
    material: Optional[ParsedMaterial] = Field(
        default=None,
        description="PBR material data for this part",
    )


class ArticulationPart(BaseModel):
    """
    Represents a single mesh part with physics semantics.

    Attributes:
        id: Unique identifier (mesh name from GLB)
        name: Human-readable name
        type: Semantic type (link / base / tool)
        role: Functional role
        mobility: How the part can move
        mass: Mass in kg (None = auto-compute)
        density: Density in kg/m³
        center_of_mass: CoM offset [x, y, z]
        collision_type: Collision approximation
        static_friction / dynamic_friction / restitution: Physics material
    """
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
    """
    Represents an articulation joint connecting two parts.

    Attributes:
        name: Unique joint name
        parent: Parent part ID
        child: Child part ID
        type: Joint type (fixed / revolute / prismatic)
        axis: Direction vector [x, y, z]
        anchor: Pivot point relative to child frame
        lower_limit / upper_limit: Motion limits
        drive_*: Motor control parameters
        disable_collision: Disable collision between parent and child
    """
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
    lower_limit: Optional[float] = Field(
        default=-180.0,
        description="Lower motion limit (degrees for revolute, meters for prismatic)",
    )
    upper_limit: Optional[float] = Field(
        default=180.0,
        description="Upper motion limit (degrees for revolute, meters for prismatic)",
    )

    # Drive parameters
    drive_stiffness: Optional[float] = Field(
        default=1000.0,
        description="Position drive stiffness (spring constant)",
    )
    drive_damping: Optional[float] = Field(
        default=100.0,
        description="Velocity drive damping",
    )
    drive_max_force: Optional[float] = Field(
        default=1000.0,
        description="Maximum drive force/torque",
    )
    drive_type: Literal["position", "velocity", "none"] = Field(
        default="position",
        description="Type of joint drive",
    )

    # Collision filtering
    disable_collision: bool = Field(
        default=True,
        description="Disable collision between parent and child bodies",
    )


class ArticulationData(BaseModel):
    """
    Complete articulation data for USD export.

    Contains all parts and joints needed to generate a physics-enabled
    USD file compatible with NVIDIA Isaac Sim.
    """
    model_name: str = Field(
        default="robot",
        description="Name for the USD root prim",
    )
    parts: List[ArticulationPart] = Field(
        default_factory=list,
        description="List of all parts in the model",
    )
    joints: List[ArticulationJoint] = Field(
        default_factory=list,
        description="List of all joints in the model",
    )


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
