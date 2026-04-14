"""
Pydantic schemas for Phidias Articulation MVP.

Defines data models for:
- Part: Individual mesh parts with semantic labels
- Joint: Articulation joints between parts  
- ArticulationData: Complete model data for USD export
- UploadResponse: API response after GLB upload
- ExportRequest/Response: USD export request/response
"""

from typing import List, Literal, Tuple, Optional
from pydantic import BaseModel, Field


class Part(BaseModel):
    """
    Represents a single mesh part in the model.
    
    Attributes:
        id: Unique identifier for the part (usually mesh name from GLB)
        name: Human-readable name for the part
        type: Semantic type of the part
            - link: A rigid body that moves
            - joint: A joint connector (rarely used as part type)
            - base: Fixed/root part of the assembly
            - tool: End effector or tool attachment
        role: Functional role of the part
            - actuator: Part that applies force/motion
            - support: Structural support element
            - gripper: Grasping mechanism
            - sensor: Sensor mount or housing
            - other: Uncategorized
        mobility: How the part can move
            - fixed: Cannot move relative to parent
            - revolute: Rotates around an axis
            - prismatic: Slides along an axis
        
        # Mass properties
        mass: Mass in kilograms (None = auto-compute from density)
        density: Density in kg/m³ for auto mass computation
        center_of_mass: Center of mass offset [x, y, z] (None = auto-compute)
    """
    id: str = Field(..., description="Unique part identifier from mesh name")
    name: str = Field(..., description="Display name for the part")
    type: Literal["link", "joint", "base", "tool"] = Field(
        default="link", 
        description="Semantic type of the part"
    )
    role: Literal["actuator", "support", "gripper", "sensor", "other"] = Field(
        default="other",
        description="Functional role of the part"
    )
    mobility: Literal["fixed", "revolute", "prismatic"] = Field(
        default="fixed",
        description="Movement capability of the part"
    )
    
    # Mass properties
    mass: Optional[float] = Field(
        default=None,
        description="Mass in kg (None = auto-compute from density)"
    )
    density: Optional[float] = Field(
        default=1000.0,
        description="Density in kg/m³ for mass computation"
    )
    center_of_mass: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Center of mass offset [x, y, z] relative to part origin"
    )
    
    # Collision properties
    collision_type: Literal["mesh", "convexHull", "convexDecomposition", "none"] = Field(
        default="convexHull",
        description="Collision approximation type"
    )
    
    # Physics Material
    static_friction: float = Field(default=0.5, description="Static friction coefficient")
    dynamic_friction: float = Field(default=0.5, description="Dynamic friction coefficient")
    restitution: float = Field(default=0.0, description="Restitution (bounciness) coefficient")


class Joint(BaseModel):
    """
    Represents an articulation joint connecting two parts.
    
    Joints define the kinematic relationship between parent and child parts.
    The joint axis defines the direction of rotation (revolute) or 
    translation (prismatic).
    
    Attributes:
        name: Unique name for the joint
        parent: ID of the parent part
        child: ID of the child part
        type: Joint type
            - fixed: No relative motion allowed
            - revolute: Rotation around axis
            - prismatic: Translation along axis
        axis: Direction vector for joint motion [x, y, z]
        lower_limit: Lower limit in radians (revolute) or meters (prismatic)
        upper_limit: Upper limit in radians (revolute) or meters (prismatic)
        
        # Drive parameters (for motor/controller)
        drive_stiffness: Position gain (spring constant) for PD control
        drive_damping: Velocity gain (damping) for PD control
        drive_max_force: Maximum force/torque the drive can apply
        drive_type: Type of drive control (position, velocity, or none)
    """
    name: str = Field(..., description="Unique joint name")
    parent: str = Field(..., description="Parent part ID")
    child: str = Field(..., description="Child part ID")
    type: Literal["fixed", "revolute", "prismatic"] = Field(
        default="revolute",
        description="Joint type"
    )
    axis: Tuple[float, float, float] = Field(
        default=(0, 0, 1),
        description="Joint axis direction [x, y, z]"
    )
    anchor: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Joint pivot point relative to child part [x, y, z]"
    )
    
    # Joint limits
    lower_limit: Optional[float] = Field(
        default=-180.0,
        description="Lower motion limit (degrees for revolute, meters for prismatic)"
    )
    upper_limit: Optional[float] = Field(
        default=180.0,
        description="Upper motion limit (degrees for revolute, meters for prismatic)"
    )
    
    # Drive parameters
    drive_stiffness: Optional[float] = Field(
        default=1000.0,
        description="Position drive stiffness (spring constant)"
    )
    drive_damping: Optional[float] = Field(
        default=100.0,
        description="Velocity drive damping"
    )
    drive_max_force: Optional[float] = Field(
        default=1000.0,
        description="Maximum drive force/torque"
    )
    drive_type: Literal["position", "velocity", "none"] = Field(
        default="position",
        description="Type of joint drive"
    )
    
    # Collision filtering
    disable_collision: bool = Field(
        default=True,
        description="Disable collision between parent and child bodies"
    )


class ArticulationData(BaseModel):
    """
    Complete articulation data for USD export.
    
    Contains all parts and joints needed to generate a physics-enabled
    USD file compatible with NVIDIA Isaac Sim.
    """
    model_name: str = Field(
        default="robot",
        description="Name for the USD root prim"
    )
    parts: List[Part] = Field(
        default_factory=list,
        description="List of all parts in the model"
    )
    joints: List[Joint] = Field(
        default_factory=list,
        description="List of all joints in the model"
    )


class TextureInfo(BaseModel):
    """
    Information about a texture used in a material.
    """
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
    is_watertight: bool = True  # Whether the mesh is closed/manifold
    material: Optional[MaterialInfo] = None  # Enhanced material information


class UploadResponse(BaseModel):
    """Response after successfully uploading a GLB file."""
    success: bool
    message: str
    filename: str
    model_url: str  # URL to access the uploaded GLB
    parts: List[ParsedPart]  # Parsed parts from the model


class ExportRequest(BaseModel):
    """Request to export articulation data to USD."""
    glb_filename: str = Field(..., description="Original GLB filename")
    articulation: ArticulationData
    output_format: Literal["usda", "usdz"] = Field(
        default="usda",
        description="Output format: usda (text) or usdz (zip archive)"
    )


class ExportResponse(BaseModel):
    """Response after USD/USDZ export."""
    success: bool
    message: str
    download_url: Optional[str] = None
    filename: Optional[str] = None
    format: Optional[str] = Field(
        default=None,
        description="Output format used (usda or usdz)"
    )
