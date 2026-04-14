"""
USDZ Packager Service

Packages USD files and their assets (textures) into USDZ format.
USDZ is a zip archive containing USD files and referenced assets.

This module handles:
- Creating USDZ archives from USD files using USD's built-in packaging utilities
- Properly embedding texture images into the archive
- Ensuring asset references work correctly within the USDZ container
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# USD imports
from pxr import UsdUtils, Sdf

logger = logging.getLogger(__name__)


class USDZPackager:
    """
    Packages USD files and assets into USDZ format using USD's built-in utilities.

    USDZ is the preferred format for iOS/macOS and is widely supported
    across 3D applications. This implementation uses UsdUtilsCreateNewARKitUsdzPackage
    to ensure proper packaging according to Apple's ARKit specifications.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the USDZ packager.

        Args:
            output_dir: Directory to save generated USDZ files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_usdz(
        self,
        usd_file_path: str,
        texture_files: Optional[List[str]] = None,
        texture_mapping: Optional[Dict[str, str]] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Create a USDZ archive from a USD file and its texture assets using USD's built-in packaging utilities.

        This method uses UsdUtilsCreateNewARKitUsdzPackage which properly handles asset references
        and ensures compatibility with Apple's ARKit specifications.

        Args:
            usd_file_path: Path to the source USD file
            texture_files: List of texture file paths to include (for logging purposes)
            texture_mapping: Dict mapping original texture names to new names (for logging purposes)
            output_filename: Optional output filename (without .usdz extension)

        Returns:
            Path to the created USDZ file
        """
        usd_path = Path(usd_file_path)

        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        # Determine output filename
        if output_filename is None:
            output_filename = usd_path.stem + ".usdz"

        usdz_path = self.output_dir / output_filename

        logger.info(f"Creating USDZ using ARKit package utility: {usdz_path}")

        # Use USD's built-in packaging utility which properly handles asset references
        # This ensures textures are correctly embedded and referenced within the USDZ
        try:
            # The UsdUtilsCreateNewARKitUsdzPackage function expects:
            # 1. An SdfAssetPath to the input USD file
            # 2. The output USDZ file path as a string
            input_asset_path = Sdf.AssetPath(str(usd_path))
            success = UsdUtils.CreateNewARKitUsdzPackage(input_asset_path, str(usdz_path))

            if not success:
                raise RuntimeError("Failed to create USDZ package using USD utilities")

            logger.info(f"Created USDZ: {usdz_path}")
            return str(usdz_path)

        except Exception as e:
            logger.error(f"Failed to create USDZ using USD utilities: {e}")
            raise

    def package_from_usda(
        self,
        usda_file_path: str,
        base_dir: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Package a USD file and automatically package its textures using USD's built-in utilities.

        This method leverages USD's UsdUtilsCreateNewARKitUsdzPackage which automatically
        discovers and packages all referenced assets including textures.

        Args:
            usda_file_path: Path to the USD file
            base_dir: Base directory (kept for API compatibility but not used in new implementation)
            output_filename: Optional output filename

        Returns:
            Path to the created USDZ file
        """
        usda_path = Path(usda_file_path)

        # Create the USDZ using USD's built-in packaging which automatically
        # discovers and packages all referenced assets
        return self.create_usdz(
            usd_file_path=str(usda_path),
            output_filename=output_filename
        )

    def _find_texture_references(self, usda_path: Path, base_dir: Path) -> List[str]:
        """
        Find all texture files referenced in a USD file.

        DEPRECATED: This method is kept for backward compatibility but is no longer
        used in the new implementation which relies on USD's built-in asset discovery.

        Args:
            usda_path: Path to USD file
            base_dir: Base directory for resolving paths

        Returns:
            List of texture file paths found
        """
        logger.warning("Using deprecated texture reference finder - new implementation uses USD's built-in asset discovery")
        texture_files = []

        try:
            with open(usda_path, 'r') as f:
                content = f.read()

            # Look for asset references (@filename@ pattern)
            import re
            asset_pattern = r'@([^@]+)@'
            matches = re.findall(asset_pattern, content)

            for match in matches:
                # Try to resolve the texture path
                texture_path = Path(match)

                # If it's already an absolute path, use it
                if texture_path.is_absolute():
                    full_path = texture_path
                else:
                    # Resolve relative to base_dir
                    full_path = (base_dir / texture_path).resolve()

                # Check if file exists
                if full_path.exists():
                    texture_files.append(str(full_path))
                    logger.debug(f"Found texture reference: {full_path}")
                else:
                    logger.warning(f"Texture not found: {full_path}")

        except Exception as e:
            logger.warning(f"Failed to read USD file for texture references: {e}")

        return texture_files


# Singleton instance
usdz_packager = USDZPackager()