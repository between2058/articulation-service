"""
Physics Schema Injector

Injects NVIDIA USD physics schemas into USD stages for simulation.

Adds:
- UsdPhysics.ArticulationRootAPI - Marks root of articulated assembly
- UsdPhysics.RigidBodyAPI - Makes parts simulate as rigid bodies
- UsdPhysics.CollisionAPI - Enables collision detection
- UsdPhysics.MassAPI - Mass properties
- Joint prims (RevoluteJoint, PrismaticJoint, FixedJoint)

These schemas are compatible with NVIDIA Isaac Sim and PhysX.
"""

import logging
from typing import Dict, List, Optional, Tuple

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

from app.models.schemas import Part, Joint, ArticulationData

logger = logging.getLogger(__name__)


class PhysicsInjector:
    """
    Injects physics schemas into USD stages.
    
    Takes a USD stage with mesh geometry and adds the necessary
    physics APIs and joint prims for articulated simulation.
    """
    
    def __init__(self, default_density: float = 1000.0):
        """
        Initialize the physics injector.
        
        Args:
            default_density: Default density for rigid bodies in kg/m³
        """
        self.default_density = default_density
    
    def apply_articulation_root(
        self, 
        stage: Usd.Stage, 
        prim_path: str
    ) -> None:
        """
        Apply ArticulationRootAPI to a prim.
        
        This marks the prim as the root of an articulated assembly,
        which enables reduced-coordinate simulation in PhysX.
        
        Args:
            stage: USD stage
            prim_path: Path to the prim to mark as articulation root
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        
        # Apply ArticulationRootAPI
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        
        logger.info(f"Applied ArticulationRootAPI to {prim_path}")
    
    def apply_rigid_body(
        self,
        stage: Usd.Stage,
        prim_path: str,
        is_kinematic: bool = False,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        center_of_mass: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Apply RigidBodyAPI to make a prim a dynamic rigid body.
        
        Args:
            stage: USD stage
            prim_path: Path to the prim
            is_kinematic: If True, body is kinematic (animated, not simulated)
            mass: Explicit mass in kg (overrides density if provided)
            density: Body density in kg/m³ (used if mass not provided)
            center_of_mass: Center of mass offset [x, y, z]
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        
        # Apply RigidBodyAPI
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        
        if is_kinematic:
            rigid_body.CreateKinematicEnabledAttr(True)
        
        # Apply MassAPI for mass properties
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        
        if mass is not None:
            # Use explicit mass value
            mass_api.CreateMassAttr(float(mass))
            logger.info(f"Applied mass={mass}kg to {prim_path}")
        else:
            # Use density for auto mass computation
            mass_api.CreateDensityAttr(float(density or self.default_density))
        
        # Set center of mass if provided
        if center_of_mass is not None:
            mass_api.CreateCenterOfMassAttr(
                Gf.Vec3f(float(center_of_mass[0]), float(center_of_mass[1]), float(center_of_mass[2]))
            )
            logger.info(f"Set center of mass to {center_of_mass} for {prim_path}")
        
        logger.info(f"Applied RigidBodyAPI to {prim_path} (kinematic={is_kinematic})")
    
    def apply_collision(
        self,
        stage: Usd.Stage,
        prim_path: str,
        collision_type: str = "mesh"
    ) -> None:
        """
        Apply CollisionAPI to enable collision detection.
        
        For mesh collision, also applies MeshCollisionAPI.
        
        Args:
            stage: USD stage
            prim_path: Path to the mesh prim
            collision_type: Type of collision shape ("mesh", "convexHull", etc.)
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        
        # Apply CollisionAPI
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        
        # For mesh prims, we need to specify the collision approximation
        # Check if this is a Mesh prim
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            
            # Set approximation type
            if collision_type == "convexHull":
                mesh_collision.CreateApproximationAttr("convexHull")
            elif collision_type == "convexDecomposition":
                mesh_collision.CreateApproximationAttr("convexDecomposition")
            else:
                # Default to triangle mesh (most accurate but slower)
                mesh_collision.CreateApproximationAttr("none")
        
        logger.info(f"Applied CollisionAPI to {prim_path} (type={collision_type})")
    
    def apply_collision_filtering(
        self,
        stage: Usd.Stage,
        child_path: str,
        parent_path: str
    ) -> None:
        """
        Apply FilteredPairsAPI to disable collision between two bodies.
        
        Args:
            stage: USD stage
            child_path: Path to child body
            parent_path: Path to parent body
        """
        child_prim = stage.GetPrimAtPath(child_path)
        if not child_prim.IsValid():
            return
            
        # Apply FilteredPairsAPI to the child prim
        filtered_api = UsdPhysics.FilteredPairsAPI.Apply(child_prim)
        
        # Add parent to the filtered pairs relationship
        filtered_api.GetFilteredPairsRel().AddTarget(Sdf.Path(parent_path))
        
        logger.info(f"Collision disabled between {child_path} and {parent_path}")
    
    def apply_physics_material(
        self,
        stage: Usd.Stage,
        prim_path: str,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0
    ) -> None:
        """
        Apply physics material properties to a prim.

        If the prim already has a material bound (e.g., a PBR material with texture),
        this will add PhysicsMaterialAPI to that existing material instead of
        creating a new material. This prevents overriding visual materials.

        Args:
            stage: USD stage
            prim_path: Path to the prim (usually collision mesh)
            static_friction: 0.0 - 1.0+
            dynamic_friction: 0.0 - 1.0+
            restitution: 0.0 - 1.0
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        # Check if there's already a material bound
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material_binding = binding_api.GetDirectBinding()

        # Get the material path
        material_path = material_binding.GetMaterialPath()

        if material_path and len(str(material_path)) > 0:
            # Material already exists - add PhysicsMaterialAPI to it
            material_prim = stage.GetPrimAtPath(material_path)
            if material_prim and material_prim.IsValid():
                physics_mat = UsdPhysics.MaterialAPI.Apply(material_prim)
                physics_mat.CreateStaticFrictionAttr(static_friction)
                physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
                physics_mat.CreateRestitutionAttr(restitution)
                print(f"[INFO] Added PhysicsMaterialAPI to existing material: {material_path}")
            else:
                # No material exists - create a new physics-only material
                materials_path = "/World/Materials"
                if not stage.GetPrimAtPath(materials_path).IsValid():
                    UsdGeom.Scope.Define(stage, materials_path)

                # Create a unique material name based on properties
                mat_name = f"Mat_F{int(static_friction*100)}_R{int(restitution*100)}"
                mat_path = f"{materials_path}/{mat_name}"

                # Create Material if it doesn't exist
                material_prim_check = stage.GetPrimAtPath(mat_path)
                if not material_prim_check.IsValid():
                    material = UsdShade.Material.Define(stage, mat_path)

                    # Add PhysicsMaterialAPI
                    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                    physics_mat.CreateStaticFrictionAttr(static_friction)
                    physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
                    physics_mat.CreateRestitutionAttr(restitution)
                else:
                    material = UsdShade.Material(material_prim_check)

                # Bind material to the prim
                binding_api.Bind(material)
                print(f"[INFO] Created and bound physics material: {mat_path}")
        else:
            # No material bound at all - create a new physics-only material
            materials_path = "/World/Materials"
            if not stage.GetPrimAtPath(materials_path).IsValid():
                UsdGeom.Scope.Define(stage, materials_path)

            # Create a unique material name based on properties
            mat_name = f"Mat_F{int(static_friction*100)}_R{int(restitution*100)}"
            mat_path = f"{materials_path}/{mat_name}"

            # Create Material if it doesn't exist
            material_prim_check = stage.GetPrimAtPath(mat_path)
            if not material_prim_check.IsValid():
                material = UsdShade.Material.Define(stage, mat_path)

                # Add PhysicsMaterialAPI
                physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                physics_mat.CreateStaticFrictionAttr(static_friction)
                physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
                physics_mat.CreateRestitutionAttr(restitution)
            else:
                material = UsdShade.Material(material_prim_check)

            # Bind material to the prim
            binding_api.Bind(material)
            print(f"[INFO] Created and bound physics material: {mat_path}")
        
    def _calculate_parent_local_pos(
        self,
        stage: Usd.Stage,
        parent_path: str,
        child_path: str,
        child_anchor: Tuple[float, float, float]
    ) -> Gf.Vec3f:
        """
        Calculate the local position in parent frame that matches 
        the child anchor point in world space.
        """
        parent_prim = stage.GetPrimAtPath(parent_path)
        child_prim = stage.GetPrimAtPath(child_path)
        
        if not parent_prim.IsValid() or not child_prim.IsValid():
            return Gf.Vec3f(0, 0, 0)
            
        # Get World Transforms
        parent_xform = UsdGeom.Xformable(parent_prim)
        child_xform = UsdGeom.Xformable(child_prim)
        
        time = Usd.TimeCode.Default()
        parent_msg = parent_xform.ComputeLocalToWorldTransform(time)
        child_msg = child_xform.ComputeLocalToWorldTransform(time)
        
        # Calculate Joint World Position (Child World * Anchor)
        anchor_vec = Gf.Vec3d(child_anchor[0], child_anchor[1], child_anchor[2])
        joint_world_pos = child_msg.Transform(anchor_vec)
        
        # Calculate Parent Local Position (Inv(Parent World) * Joint World)
        parent_inv = parent_msg.GetInverse()
        parent_local_pos = parent_inv.Transform(joint_world_pos)
        
        return Gf.Vec3f(parent_local_pos)

    def create_revolute_joint(
        self,
        stage: Usd.Stage,
        joint_name: str,
        parent_path: str,
        child_path: str,
        axis: Tuple[float, float, float] = (0, 0, 1),
        anchor: Tuple[float, float, float] = (0, 0, 0),
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        drive_stiffness: Optional[float] = None,
        drive_damping: Optional[float] = None,
        drive_max_force: Optional[float] = None,
        drive_type: str = "position",
        joints_parent_path: Optional[str] = None
    ) -> str:
        """
        Create a revolute (rotation) joint between two bodies.
        
        Args:
            stage: USD stage
            joint_name: Name for the joint prim
            parent_path: Path to parent body prim
            child_path: Path to child body prim
            axis: Rotation axis [x, y, z] (normalized)
            anchor: Pivot point relative to child frame [x, y, z]
            lower_limit: Lower rotation limit in degrees
            upper_limit: Upper rotation limit in degrees
            drive_stiffness: Position drive stiffness (spring constant)
            drive_damping: Velocity drive damping
            drive_max_force: Maximum drive force
            drive_type: "position", "velocity", or "none"
            joints_parent_path: Path to create joint under (default: root/Joints)
            
        Returns:
            Path to the created joint prim
        """
        # Determine where to create the joint
        if joints_parent_path is None:
            root_prim = stage.GetDefaultPrim()
            joints_parent_path = str(root_prim.GetPath().AppendChild("Joints"))
            
            # Create Joints xform if it doesn't exist
            if not stage.GetPrimAtPath(joints_parent_path).IsValid():
                UsdGeom.Xform.Define(stage, joints_parent_path)
        
        joint_path = f"{joints_parent_path}/{self._sanitize_name(joint_name)}"
        
        # Create the revolute joint
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        
        # Set body relationships
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
        
        # Set joint axis
        joint.CreateAxisAttr(self._axis_to_token(axis))
        
        # Set joint limits if provided
        if lower_limit is not None:
            joint.CreateLowerLimitAttr(float(lower_limit))
        if upper_limit is not None:
            joint.CreateUpperLimitAttr(float(upper_limit))
        
        # Calculate and set local poses
        # pos1 is the anchor relative to child (user provided)
        # pos0 must be calculated to match pos1 in world space
        parent_local_pos = self._calculate_parent_local_pos(
            stage, parent_path, child_path, anchor
        )
        
        joint.CreateLocalPos0Attr(parent_local_pos)
        joint.CreateLocalPos1Attr(Gf.Vec3f(anchor[0], anchor[1], anchor[2]))
        
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        
        # Apply joint drive if parameters provided
        if drive_type != "none" and drive_stiffness is not None:
            self._apply_joint_drive(
                stage=stage,
                joint_prim=joint.GetPrim(),
                drive_type=drive_type,
                stiffness=drive_stiffness,
                damping=drive_damping,
                max_force=drive_max_force
            )
        
        logger.info(f"Created RevoluteJoint: {joint_path} ({parent_path} -> {child_path})")
        
        return joint_path
    
    def create_prismatic_joint(
        self,
        stage: Usd.Stage,
        joint_name: str,
        parent_path: str,
        child_path: str,
        axis: Tuple[float, float, float] = (0, 0, 1),
        anchor: Tuple[float, float, float] = (0, 0, 0),
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        drive_stiffness: Optional[float] = None,
        drive_damping: Optional[float] = None,
        drive_max_force: Optional[float] = None,
        drive_type: str = "position",
        joints_parent_path: Optional[str] = None
    ) -> str:
        """
        Create a prismatic (sliding) joint between two bodies.
        
        Args:
            stage: USD stage
            joint_name: Name for the joint prim
            parent_path: Path to parent body prim
            child_path: Path to child body prim
            axis: Translation axis [x, y, z] (normalized)
            anchor: Pivot point relative to child frame [x, y, z]
            lower_limit: Lower translation limit in stage units (meters)
            upper_limit: Upper translation limit in stage units (meters)
            drive_stiffness: Position drive stiffness
            drive_damping: Velocity drive damping
            drive_max_force: Maximum drive force
            drive_type: "position", "velocity", or "none"
            joints_parent_path: Path to create joint under
            
        Returns:
            Path to the created joint prim
        """
        # Determine where to create the joint
        if joints_parent_path is None:
            root_prim = stage.GetDefaultPrim()
            joints_parent_path = str(root_prim.GetPath().AppendChild("Joints"))
            
            if not stage.GetPrimAtPath(joints_parent_path).IsValid():
                UsdGeom.Xform.Define(stage, joints_parent_path)
        
        joint_path = f"{joints_parent_path}/{self._sanitize_name(joint_name)}"
        
        # Create the prismatic joint
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        
        # Set body relationships
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
        
        # Set joint axis
        joint.CreateAxisAttr(self._axis_to_token(axis))
        
        # Set joint limits if provided
        if lower_limit is not None:
            joint.CreateLowerLimitAttr(float(lower_limit))
        if upper_limit is not None:
            joint.CreateUpperLimitAttr(float(upper_limit))
        
        # Calculate and set local poses
        parent_local_pos = self._calculate_parent_local_pos(
            stage, parent_path, child_path, anchor
        )
        
        joint.CreateLocalPos0Attr(parent_local_pos)
        joint.CreateLocalPos1Attr(Gf.Vec3f(anchor[0], anchor[1], anchor[2]))
        
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        
        # Apply joint drive if parameters provided
        if drive_type != "none" and drive_stiffness is not None:
            self._apply_joint_drive(
                stage=stage,
                joint_prim=joint.GetPrim(),
                drive_type=drive_type,
                stiffness=drive_stiffness,
                damping=drive_damping,
                max_force=drive_max_force,
                is_angular=False  # Prismatic uses linear drive
            )
        
        logger.info(f"Created PrismaticJoint: {joint_path} ({parent_path} -> {child_path})")
        
        return joint_path
    
    def create_fixed_joint(
        self,
        stage: Usd.Stage,
        joint_name: str,
        parent_path: str,
        child_path: str,
        joints_parent_path: Optional[str] = None
    ) -> str:
        """
        Create a fixed (welded) joint between two bodies.
        
        Fixed joints lock all relative motion between bodies.
        
        Args:
            stage: USD stage
            joint_name: Name for the joint prim
            parent_path: Path to parent body prim
            child_path: Path to child body prim
            joints_parent_path: Path to create joint under
            
        Returns:
            Path to the created joint prim
        """
        if joints_parent_path is None:
            root_prim = stage.GetDefaultPrim()
            joints_parent_path = str(root_prim.GetPath().AppendChild("Joints"))
            
            if not stage.GetPrimAtPath(joints_parent_path).IsValid():
                UsdGeom.Xform.Define(stage, joints_parent_path)
        
        joint_path = f"{joints_parent_path}/{self._sanitize_name(joint_name)}"
        
        # Create the fixed joint
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        
        # Set body relationships
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
        
        # Set default local poses
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        
        logger.info(f"Created FixedJoint: {joint_path} ({parent_path} -> {child_path})")
        
        return joint_path
    
    def inject_physics(
        self,
        stage: Usd.Stage,
        articulation_data: ArticulationData,
        part_paths: Dict[str, str]
    ) -> None:
        """
        Inject all physics schemas based on articulation data.
        
        This is the main entry point for adding physics to a stage.
        
        Args:
            stage: USD stage with mesh geometry
            articulation_data: Parts and joints data
            part_paths: Mapping from part IDs to USD prim paths
        """
        # Find the root/base part
        base_part = self._find_base_part(articulation_data.parts)
        
        if base_part and base_part.id in part_paths:
            # Apply ArticulationRootAPI to base
            self.apply_articulation_root(stage, part_paths[base_part.id])
        else:
            # Apply to first part if no base defined
            if part_paths:
                first_path = list(part_paths.values())[0]
                self.apply_articulation_root(stage, first_path)
        
        # Apply RigidBodyAPI and CollisionAPI to all parts
        for part in articulation_data.parts:
            if part.id not in part_paths:
                logger.warning(f"Part {part.id} not found in USD stage")
                continue
            
            prim_path = part_paths[part.id]
            
            # Base parts are kinematic (fixed in world)
            is_kinematic = (part.type == "base")
            
            # Apply rigid body with mass properties
            self.apply_rigid_body(
                stage, 
                prim_path, 
                is_kinematic=is_kinematic,
                mass=part.mass,
                density=part.density,
                center_of_mass=part.center_of_mass
            )
            
            # Apply collision to the mesh child
            mesh_path = f"{prim_path}/mesh"
            if stage.GetPrimAtPath(mesh_path).IsValid():
                self.apply_collision(stage, mesh_path, collision_type=part.collision_type)
                
                # Apply physics material (friction/restitution) to the collision mesh
                self.apply_physics_material(
                    stage=stage,
                    prim_path=mesh_path,
                    static_friction=part.static_friction,
                    dynamic_friction=part.dynamic_friction,
                    restitution=part.restitution
                )
        
        # Create joints
        for joint in articulation_data.joints:
            if joint.parent not in part_paths or joint.child not in part_paths:
                logger.warning(f"Joint {joint.name}: parent or child not found")
                continue
            
            parent_path = part_paths[joint.parent]
            child_path = part_paths[joint.child]
            
            if joint.type == "revolute":
                self.create_revolute_joint(
                    stage=stage,
                    joint_name=joint.name,
                    parent_path=parent_path,
                    child_path=child_path,
                    axis=joint.axis,
                    anchor=joint.anchor,
                    lower_limit=joint.lower_limit,
                    upper_limit=joint.upper_limit,
                    drive_stiffness=joint.drive_stiffness,
                    drive_damping=joint.drive_damping,
                    drive_max_force=joint.drive_max_force,
                    drive_type=joint.drive_type
                )
            elif joint.type == "prismatic":
                self.create_prismatic_joint(
                    stage=stage,
                    joint_name=joint.name,
                    parent_path=parent_path,
                    child_path=child_path,
                    axis=joint.axis,
                    anchor=joint.anchor,
                    lower_limit=joint.lower_limit,
                    upper_limit=joint.upper_limit,
                    drive_stiffness=joint.drive_stiffness,
                    drive_damping=joint.drive_damping,
                    drive_max_force=joint.drive_max_force,
                    drive_type=joint.drive_type
                )
            elif joint.type == "fixed":
                self.create_fixed_joint(
                    stage=stage,
                    joint_name=joint.name,
                    parent_path=parent_path,
                    child_path=child_path
                )

            # Apply collision filtering if requested (default=True)
            if joint.disable_collision:
                self.apply_collision_filtering(stage, child_path, parent_path)
        
        logger.info(f"Injected physics for {len(articulation_data.parts)} parts and {len(articulation_data.joints)} joints")
    
    def _find_base_part(self, parts: List[Part]) -> Optional[Part]:
        """Find the part marked as 'base' type."""
        for part in parts:
            if part.type == "base":
                return part
        return None
    
    def _axis_to_token(self, axis: Tuple[float, float, float]) -> str:
        """
        Convert axis vector to USD axis token.
        
        USD joints use "X", "Y", or "Z" tokens for axis specification.
        
        Args:
            axis: [x, y, z] direction vector
            
        Returns:
            "X", "Y", or "Z" based on dominant axis
        """
        x, y, z = abs(axis[0]), abs(axis[1]), abs(axis[2])
        
        if x >= y and x >= z:
            return "X"
        elif y >= x and y >= z:
            return "Y"
        else:
            return "Z"
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as a USD prim name."""
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        if not sanitized:
            sanitized = '_unnamed'
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        return sanitized
    
    def _apply_joint_drive(
        self,
        stage: Usd.Stage,
        joint_prim: Usd.Prim,
        drive_type: str,
        stiffness: Optional[float],
        damping: Optional[float],
        max_force: Optional[float],
        is_angular: bool = True
    ) -> None:
        """
        Apply drive API to a joint for motor control.
        
        UsdPhysics.DriveAPI allows setting PD control parameters
        for joint actuation in simulation.
        
        Args:
            stage: USD stage
            joint_prim: The joint prim to add drive to
            drive_type: "position" or "velocity"
            stiffness: Position gain / spring constant
            damping: Velocity gain / damping coefficient
            max_force: Maximum force/torque the drive can apply
            is_angular: True for revolute joints, False for prismatic
        """
        # Determine the drive name based on joint type
        # For revolute joints, we use "angular" drive
        # For prismatic joints, we use "linear" drive
        drive_name = "angular" if is_angular else "linear"
        
        # Apply DriveAPI with the appropriate name
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)
        
        # Set drive type
        if drive_type == "position":
            drive.CreateTypeAttr("force")  # Position control uses force mode
        else:
            drive.CreateTypeAttr("force")  # Velocity control also uses force mode
        
        # Set stiffness (position gain)
        if stiffness is not None:
            drive.CreateStiffnessAttr(float(stiffness))
        
        # Set damping (velocity gain)
        if damping is not None:
            drive.CreateDampingAttr(float(damping))
        
        # Set max force
        if max_force is not None:
            drive.CreateMaxForceAttr(float(max_force))
        
        # For position drive, set a target position of 0 (can be changed in simulation)
        if drive_type == "position":
            drive.CreateTargetPositionAttr(0.0)
        elif drive_type == "velocity":
            drive.CreateTargetVelocityAttr(0.0)
        
        logger.info(f"Applied DriveAPI ({drive_name}) to {joint_prim.GetPath()}: "
                    f"stiffness={stiffness}, damping={damping}, max_force={max_force}")


# Singleton instance
physics_injector = PhysicsInjector()
