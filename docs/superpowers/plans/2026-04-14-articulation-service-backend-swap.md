# Articulation Service — Backend Swap to Editor Pipeline: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `articulation-service`'s ported texture/material pipeline with
`articulation-editor`'s validated `bugfix--usdz-texture` backend, adapted to the
API paths that `phidias-standalone` frontend already calls, with a three-line Z-up
patch for Isaac Sim.

**Architecture:** Copy editor's `backend/app/` tree wholesale into
`articulation-service/app/`. Rename route paths in a single FastAPI router so
existing frontend calls resolve unchanged. Apply three small edits in
`usd_builder.py` to flip stage up-axis from Y to Z and rotate vertices/normals
accordingly. Delete the old flat-layout entrypoint and service/model modules.
Dockerfile CMD and file copies updated for the new layout; port 52071 unchanged.

**Tech Stack:** Python 3.10, FastAPI 0.115, Pydantic 2.5, USD `usd-core` ≥24.8,
trimesh 4.0, NumPy 1.26, Pillow ≥10, Docker (Debian Bookworm base).

**Reference spec:** `docs/superpowers/specs/2026-04-14-articulation-service-backend-swap-design.md`

**Editor source:** `articulation-editor` branch `origin/bugfix--usdz-texture`,
commit `cc91f91`. Accessed via `git show origin/bugfix--usdz-texture:<path>`
while cwd is `/Users/between2058/Documents/code/articulation-editor`.

**Sample GLB for acceptance tests:** verify with the user what sample to use.
If none is at hand, a textured GLB such as
[KhronosGroup/glTF-Sample-Models `BoomBox.glb`](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/BoomBox/glTF-Binary)
works for material verification.

---

## Task 1: Prep — commit diagnostic script, create feature branch

The diagnostic script `scripts/diagnose_usdz.py` was written during brainstorming
but is not yet tracked. It is used in Task 12 for verifying material bindings in
the produced USDZ. Commit it first so subsequent commits stay focused on the
swap.

**Files:**
- Commit: `scripts/diagnose_usdz.py` (already exists, untracked)
- Branch: new `feat/backend-swap-to-editor` branched from current `main`

- [ ] **Step 1: Confirm current state**

Run: `git status`
Expected: `main` branch clean, with `scripts/diagnose_usdz.py` listed as untracked.

- [ ] **Step 2: Create feature branch**

```bash
git checkout -b feat/backend-swap-to-editor
```

- [ ] **Step 3: Stage and commit the diagnostic script**

```bash
git add scripts/diagnose_usdz.py
git commit -m "chore: add diagnose_usdz.py for material binding verification

Dumps primvars:st, material bindings, and UsdPreviewSurface shader graph
from a USDZ/USDA file. Used to verify end-to-end texture pipeline.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify editor source is reachable**

Run:
```bash
cd /Users/between2058/Documents/code/articulation-editor && \
  git log --oneline origin/bugfix--usdz-texture -1 && \
  cd -
```
Expected: the first line prints `cc91f91 Add USDZ packager with proper texture embedding support and validation scripts`.

---

## Task 2: Copy editor's `backend/app/` into `articulation-service/app/`

Copy the entire `app/` tree verbatim from editor's `bugfix--usdz-texture` branch
into `articulation-service/`. This is a mechanical copy — no edits in this task.
Subsequent tasks modify specific files. Copying first (and committing) gives us a
clean "before" state to diff against.

**Files created:**
- `app/__init__.py`
- `app/main.py`
- `app/api/__init__.py`
- `app/api/routes.py`
- `app/models/__init__.py`
- `app/models/schemas.py`
- `app/services/__init__.py`
- `app/services/glb_parser.py`
- `app/services/usd_builder.py`
- `app/services/physics_injector.py`
- `app/services/usdz_packager.py`

- [ ] **Step 1: Create directory skeleton**

```bash
mkdir -p app/api app/models app/services
```

- [ ] **Step 2: Extract each file from editor's branch**

From `/Users/between2058/Documents/code/articulation-editor`:

```bash
EDITOR_REPO="/Users/between2058/Documents/code/articulation-editor"
SERVICE_REPO="/Users/between2058/Documents/code/articulation-service"
REF="origin/bugfix--usdz-texture"

cd "$EDITOR_REPO"
for f in \
  backend/app/__init__.py \
  backend/app/main.py \
  backend/app/api/__init__.py \
  backend/app/api/routes.py \
  backend/app/models/__init__.py \
  backend/app/models/schemas.py \
  backend/app/services/__init__.py \
  backend/app/services/glb_parser.py \
  backend/app/services/usd_builder.py \
  backend/app/services/physics_injector.py \
  backend/app/services/usdz_packager.py
do
  dest="${SERVICE_REPO}/${f#backend/}"
  git show "${REF}:${f}" > "$dest"
done
cd "$SERVICE_REPO"
```

- [ ] **Step 3: Verify no stray path references**

Run:
```bash
grep -rn "from app\." app/ | wc -l
grep -rn "from backend\." app/ | wc -l
```
Expected: first grep non-zero (legit `from app.models.schemas import ...` references); second grep returns 0 (no leftover `backend.` prefixes).

- [ ] **Step 4: Commit the mechanical copy**

```bash
git add app/
git commit -m "feat: copy editor bugfix--usdz-texture app/ tree verbatim

Mechanical copy of articulation-editor backend/app/ at commit cc91f91.
No edits yet — subsequent commits add Z-up patch, route renames, and
debug-glb handler.

Source: articulation-editor origin/bugfix--usdz-texture:backend/app/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Z-up patch — three edits in `app/services/usd_builder.py`

Editor ships with Y-up stages (GLB-native). Isaac Sim expects Z-up. Apply the
same three edits that currently exist in the service's `services/usd_builder.py`
(commit `65eeabe`'s intent, minus the material regression that came with it):

1. Default `up_axis` argument of `create_stage` flips Y → Z.
2. Gravity direction in `add_physics_scene` flips `(0, -1, 0)` → `(0, 0, -1)`.
3. Vertices and normals in `build_from_parts` rotate via `(x, y, z) → (x, -z, y)`
   before being passed to `add_mesh_prim`.

**Files:**
- Modify: `app/services/usd_builder.py`

- [ ] **Step 1: Flip `create_stage` default `up_axis` from "Y" to "Z"**

Edit the function signature at `create_stage` (around line 52 in editor's file):

Old:
```python
    def create_stage(
        self,
        filename: str,
        model_name: str = "Robot",
        up_axis: str = "Y",
        meters_per_unit: float = 1.0
    ) -> Usd.Stage:
```

New:
```python
    def create_stage(
        self,
        filename: str,
        model_name: str = "Robot",
        up_axis: str = "Z",
        meters_per_unit: float = 1.0
    ) -> Usd.Stage:
```

- [ ] **Step 2: Flip gravity in `add_physics_scene`**

Find in `add_physics_scene`:

Old:
```python
        # Set gravity for Y-up axis (gravity points down in Y direction)
        # So gravity should be (0, -1, 0) for Y-up stages with magnitude 9.81
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
```

New:
```python
        # Set gravity for Z-up axis (Isaac Sim convention)
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
```

- [ ] **Step 3: Rotate vertices/normals in `build_from_parts`**

In `build_from_parts`, find the call to `create_stage` and the per-part mesh
assembly. Replace the block that currently looks like:

Old:
```python
        # Create stage with Y-up axis as requested
        stage = self.create_stage(filename, model_name, up_axis="Y")
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

            # Add the mesh
            prim_path, material_path = self.add_mesh_prim(
                stage=stage,
                parent_path=root_path,
                name=part_name,
                vertices=data['vertices'],
                faces=data['faces'],
                normals=data.get('normals'),
                uv_coords=data.get('uv_coords'),
                material_info=material_info,
                model_name=model_name,
                embed_textures=embed_textures,
                texture_files_dir=texture_files_dir
            )

            part_paths[part_id] = prim_path
```

New:
```python
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
```

- [ ] **Step 4: Syntax check**

Run: `python -m py_compile app/services/usd_builder.py`
Expected: no output (no SyntaxError).

- [ ] **Step 5: Commit**

```bash
git add app/services/usd_builder.py
git commit -m "fix: flip usd_builder to Z-up for Isaac Sim

Three edits: default up_axis in create_stage, gravity direction in
add_physics_scene, and Y→Z vertex/normal rotation in build_from_parts.
Material/shader graph and USDZ packaging logic unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Rename `/upload` → `/parse-glb` in router

Editor's `/upload` endpoint matches the frontend's `/parse-glb` in purpose (file
upload + parsed parts return). Rename the route string; leave the handler body
unchanged.

**Files:**
- Modify: `app/api/routes.py`

- [ ] **Step 1: Rename the route decorator**

Find (around line 45):
```python
@router.post("/upload", response_model=UploadResponse)
async def upload_glb(file: UploadFile = File(...)):
```

Replace with:
```python
@router.post("/parse-glb", response_model=UploadResponse)
async def parse_glb(file: UploadFile = File(...)):
```

- [ ] **Step 2: Update the file-serve URL inside the handler**

In the same handler, the `model_url` construction references `/api/models/...`,
which stays correct because the router is still mounted at prefix `/api`. No
change needed here. Verify that editor's original line is:

```python
model_url = f"/api/models/{safe_filename}"
```

and leave it as is.

- [ ] **Step 3: Update docstring header comment at top of file**

At the top of `app/api/routes.py`, find:

```python
"""
API Routes for Phidias Articulation MVP

Endpoints:
- POST /upload: Upload GLB file, parse and return parts list
- POST /export: Generate USD from articulation data
- GET /download/{filename}: Download generated USD files
- GET /models/{filename}: Serve uploaded GLB files for frontend viewer
"""
```

Replace with:

```python
"""
API Routes for Phidias Articulation MVP

Endpoints:
- POST /parse-glb: Upload GLB file, parse and return parts list
- POST /export-usda: Generate USDA from articulation data
- POST /export-usdz: Generate USDZ from articulation data
- POST /debug-glb: Diagnostic report of pipeline stages
- GET /download/{filename}: Download generated USD files
- GET /models/{filename}: Serve uploaded GLB files for frontend viewer
- GET /health: Health check
"""
```

- [ ] **Step 4: Syntax check**

Run: `python -m py_compile app/api/routes.py`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add app/api/routes.py
git commit -m "refactor: rename /upload to /parse-glb to match frontend

Phidias frontend already calls /phidias/articulation/parse-glb, which the
upstream gateway forwards to /api/parse-glb. Handler body unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Split `/export` into `/export-usda` + `/export-usdz`

Editor's single `/export` endpoint takes a JSON `ExportRequest` with a
`glb_filename` that references a previously-uploaded file on disk. Phidias
frontend instead re-uploads the file in the same request to
`/export-usda` or `/export-usdz` (multipart). The handler body is the same after
the file is loaded — refactor it into an internal helper that both route
variants call.

**Files:**
- Modify: `app/api/routes.py`

- [ ] **Step 1: Add required imports**

At the top of `app/api/routes.py`, the imports currently include:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
```

Add `Form` and `json`:

```python
import json
...
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
```

- [ ] **Step 2: Locate the existing `/export` handler**

The existing handler starts at:

```python
@router.post("/export", response_model=ExportResponse)
async def export_usd(request: ExportRequest):
```

and runs until the function returns. Identify its full range (it is the longest
handler in the file). Copy the body (everything inside `async def export_usd` —
not the decorator or signature) into a clipboard/scratch for the next step.

- [ ] **Step 3: Replace the `/export` handler with the full new block**

Delete the old `@router.post("/export", ...)` + `async def export_usd(request)`
handler entirely (the full range located in Step 2). In its place, insert this
complete block — it contains `_do_export` (editor's original body with
substitutions applied), `_save_and_export` (multipart file-saving wrapper), and
the two thin route handlers:

```python
async def _do_export(
    glb_path: Path,
    glb_filename: str,
    articulation: ArticulationData,
    output_format: str,
) -> ExportResponse:
    """
    Shared export pipeline: mesh extraction → material/texture → USD stage
    → physics schemas → USDA on disk → optional USDZ packaging.

    Body lifted from editor's original /export handler. Substitutions:
    request.articulation.* → articulation.* ; request.output_format →
    output_format ; request.glb_filename → glb_filename. The caller has
    already written the GLB to glb_path.
    """
    try:
        # Generate output filename based on format
        base_name = articulation.model_name or "robot"

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
            for part in articulation.parts
        ]

        # Build USD stage with meshes
        usda_filename = output_filename.replace('.usdz', '.usda')

        # Extract textures to separate files (not embedded)
        glb_name = Path(glb_filename).stem
        texture_dir = OUTPUT_DIR / glb_name
        texture_files = []
        texture_mapping = {}

        # Extract textures from GLB
        try:
            extracted_textures, extracted_data = glb_parser.extract_textures(str(glb_path), glb_name)
            if extracted_textures:
                texture_files = [str(texture_dir / fname) for fname in extracted_textures.values()]

                for idx, fname in enumerate(extracted_textures.values()):
                    old_path = str(texture_dir / fname)
                    new_name = f"texture_{idx}{Path(fname).suffix}"
                    texture_mapping[old_path] = new_name
                    print(f"[INFO] Texture mapping: {fname} -> {new_name}")

                print(f"[INFO] Extracted {len(texture_files)} textures to {texture_dir}")
        except Exception as e:
            print(f"[WARNING] Could not extract textures: {e}")

        # Build USD stage (textures will be referenced externally, not embedded)
        embed_textures = False

        # Update mesh data to use mapped texture names if USDZ
        if output_format == "usdz" and texture_mapping:
            for part_id, data in mesh_data.items():
                if 'material' in data:
                    material = data['material']
                    if hasattr(material, 'has_base_color_texture') and material.has_base_color_texture and material.base_color_texture:
                        base_color_texture = material.base_color_texture
                        if hasattr(base_color_texture, 'filename') and base_color_texture.filename:
                            old_texture_name = base_color_texture.filename
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name and old_texture_name in old_path:
                                    base_color_texture.filename = new_name
                                    print(f"[DEBUG] Updated base color texture reference for USDZ: {old_texture_name} -> {new_name}")
                                    break
                    elif isinstance(material, dict) and material.get('has_base_color_texture') and material.get('base_color_texture'):
                        base_color_texture = material['base_color_texture']
                        if isinstance(base_color_texture, dict) and 'filename' in base_color_texture:
                            old_texture_name = base_color_texture['filename']
                            for old_path, new_name in texture_mapping.items():
                                if old_texture_name and old_texture_name in old_path:
                                    material['base_color_texture']['filename'] = new_name
                                    print(f"[DEBUG] Updated base color texture reference for USDZ: {old_texture_name} -> {new_name}")
                                    break

                    texture_channels = ['metallic_roughness_texture', 'normal_texture', 'occlusion_texture', 'emissive_texture']
                    for channel in texture_channels:
                        attr_name = f'has_{channel}'
                        texture_attr = channel
                        if hasattr(material, attr_name) and getattr(material, attr_name) and hasattr(material, texture_attr):
                            texture_info = getattr(material, texture_attr)
                            if texture_info and hasattr(texture_info, 'filename') and texture_info.filename:
                                old_texture_name = texture_info.filename
                                if old_texture_name:
                                    for old_path, new_name in texture_mapping.items():
                                        if old_texture_name in old_path:
                                            texture_info.filename = new_name
                                            print(f"[DEBUG] Updated {channel} reference for USDZ: {old_texture_name} -> {new_name}")
                                            break
                        elif isinstance(material, dict) and material.get(f'has_{channel}') and material.get(channel):
                            texture_info = material[channel]
                            if isinstance(texture_info, dict) and 'filename' in texture_info:
                                old_texture_name = texture_info['filename']
                                if old_texture_name:
                                    for old_path, new_name in texture_mapping.items():
                                        if old_texture_name in old_path:
                                            material[channel]['filename'] = new_name
                                            print(f"[DEBUG] Updated {channel} reference for USDZ: {old_texture_name} -> {new_name}")
                                            break

        stage, part_paths = usd_builder.build_from_parts(
            filename=usda_filename,
            model_name=articulation.model_name,
            mesh_data=mesh_data,
            part_info=part_info,
            embed_textures=embed_textures,
            texture_files_dir=str(texture_dir) if texture_dir.exists() else None
        )

        # Inject physics schemas
        physics_injector.inject_physics(
            stage=stage,
            articulation_data=articulation,
            part_paths=part_paths
        )

        # Save the USD stage
        usd_path = usd_builder.save_stage(stage)

        # Enhanced validation (optional — script lives only in editor repo;
        # import will fail silently in this service, which is fine)
        try:
            from scripts.validate_usd import validate_physics_usd
            validation_results = validate_physics_usd(usd_path)
            if not validation_results["valid"]:
                logger.warning(f"Validation failed for {usd_path}: {validation_results['errors']}")
            else:
                logger.info(f"Validation passed for {usd_path}")
        except Exception as e:
            logger.warning(f"Validation skipped due to error: {e}")

        # Copy texture files to USD directory (same behavior as editor)
        copied_textures = []
        if texture_files and texture_mapping:
            from shutil import copy2
            usd_dir = Path(usd_path).parent
            print(f"[INFO] Copying {len(texture_files)} textures to USD directory: {usd_dir}")

            for old_path, new_name in texture_mapping.items():
                dest_path = usd_dir / new_name
                try:
                    copy2(old_path, dest_path)
                    copied_textures.append(dest_path)
                    print(f"[INFO] Copied texture: {old_path} -> {dest_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to copy texture {old_path} -> {dest_path}: {e}")

        # If USDZ requested, package it
        final_output_path = usd_path
        if output_format == "usdz":
            usdz_path = usdz_packager.create_usdz(
                usd_file_path=usd_path,
                texture_files=texture_files if texture_files else None,
                texture_mapping=texture_mapping if texture_mapping else None,
                output_filename=output_filename
            )

            try:
                Path(usd_path).unlink()
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

        download_url = f"/api/download/{Path(final_output_path).name}"

        return ExportResponse(
            success=True,
            message=f"Successfully exported to {Path(final_output_path).name}",
            download_url=download_url,
            filename=Path(final_output_path).name,
            format=output_format
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


async def _save_and_export(
    file: UploadFile,
    articulation: str,
    output_format: str,
) -> ExportResponse:
    """
    Save the uploaded GLB to UPLOAD_DIR, parse the articulation JSON, then
    delegate to _do_export.
    """
    if not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Only .glb files are supported")

    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    glb_path = UPLOAD_DIR / safe_filename

    content = await file.read()
    with open(glb_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved uploaded file for export: {glb_path} ({len(content)} bytes)")

    try:
        art_payload = json.loads(articulation)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid articulation JSON: {e}")

    try:
        art_data = ArticulationData(**art_payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Articulation schema error: {e}")

    return await _do_export(glb_path, safe_filename, art_data, output_format)


@router.post("/export-usda", response_model=ExportResponse)
async def export_usda(
    file: UploadFile = File(...),
    articulation: str = Form(...),
):
    """Export articulation data to a USDA file. Multipart form-data."""
    return await _save_and_export(file, articulation, "usda")


@router.post("/export-usdz", response_model=ExportResponse)
async def export_usdz(
    file: UploadFile = File(...),
    articulation: str = Form(...),
):
    """Export articulation data to a USDZ file. Multipart form-data."""
    return await _save_and_export(file, articulation, "usdz")
```

- [ ] **Step 4: Sanity check the helper bodies**

Confirm the new file has exactly one `_do_export`, one `_save_and_export`, one
`export_usda`, one `export_usdz`, and no residual `export_usd` or
`/export` references:

```bash
grep -nE "def (_do_export|_save_and_export|export_usda|export_usdz)|@router\.post\(" app/api/routes.py
```
Expected: lines showing each helper declared exactly once, and `@router.post`
entries for `/parse-glb`, `/export-usda`, `/export-usdz` (and later, after
Task 6, `/debug-glb`).

Run:
```bash
grep -nE "/export[^-]|ExportRequest|export_usd\(" app/api/routes.py
```
Expected: no matches (or only matches in comments/docstrings, which is fine).

- [ ] **Step 5: Remove now-unused `ExportRequest` import**

Editor's routes.py imports `ExportRequest` from `app.models.schemas`. Since
`_do_export` takes `articulation: ArticulationData` directly, and the two new
route handlers take multipart form data, `ExportRequest` is no longer used.
Remove it from the import list:

Old:
```python
from app.models.schemas import (
    UploadResponse,
    ExportRequest,
    ExportResponse,
    Part,
    ArticulationData
)
```

New:
```python
from app.models.schemas import (
    UploadResponse,
    ExportResponse,
    ArticulationData,
)
```

(`Part` was already unused in editor's routes.py — removing it too is a tidy-up.)

- [ ] **Step 6: Syntax check**

Run: `python -m py_compile app/api/routes.py`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add app/api/routes.py
git commit -m "refactor: split /export into /export-usda and /export-usdz

Phidias frontend POSTs multipart (file + articulation JSON) to
/export-usda and /export-usdz, not a JSON body referencing a previously
uploaded file. Factor editor's /export body into _do_export; two thin
route handlers save the upload and delegate with the right format.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Port `/debug-glb` handler into router

The `/debug-glb` endpoint is specific to the service (added in commit
`6331796`) and does not exist in editor. Port the handler body from the old
`articulation_api.py` into the new router, adjusting imports.

**Files:**
- Modify: `app/api/routes.py`
- Read-only reference: `articulation_api.py` (lines 437-583)

- [ ] **Step 1: Add `JSONResponse` import**

Near the other fastapi imports at the top of `app/api/routes.py`:

Old:
```python
from fastapi.responses import FileResponse
```

New:
```python
from fastapi.responses import FileResponse, JSONResponse
```

- [ ] **Step 2: Append the `/debug-glb` handler at the end of routes.py**

Add this handler just before the final blank line of `app/api/routes.py`:

```python
@router.post("/debug-glb")
async def debug_glb(file: UploadFile = File(...)):
    """
    Upload a GLB and return a diagnostic report showing what the
    pipeline sees at each step. Use this to find why materials are missing.
    """
    import json as _json
    import struct

    if not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Only .glb files are supported")

    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    glb_path = UPLOAD_DIR / safe_filename

    report = {"filename": file.filename, "steps": {}}

    try:
        content = await file.read()
        with open(glb_path, "wb") as f:
            f.write(content)

        # ---- Step 1: Read GLTF JSON from GLB ----
        step1 = {}
        try:
            with open(glb_path, 'rb') as f:
                f.read(12)
                jlen = struct.unpack('<I', f.read(4))[0]
                f.read(4)
                gltf = _json.loads(f.read(jlen))

            step1["materials_count"] = len(gltf.get('materials', []))
            step1["images_count"] = len(gltf.get('images', []))
            step1["textures_count"] = len(gltf.get('textures', []))
            step1["meshes_count"] = len(gltf.get('meshes', []))

            materials_detail = []
            for i, mat in enumerate(gltf.get('materials', [])):
                pbr = mat.get('pbrMetallicRoughness', {})
                materials_detail.append({
                    "index": i,
                    "name": mat.get("name", "unnamed"),
                    "baseColorFactor": pbr.get('baseColorFactor', 'N/A'),
                    "hasBaseColorTexture": 'baseColorTexture' in pbr,
                    "baseColorTextureIndex": pbr.get('baseColorTexture', {}).get('index', 'N/A'),
                    "metallicFactor": pbr.get('metallicFactor', 'N/A'),
                    "roughnessFactor": pbr.get('roughnessFactor', 'N/A'),
                })
            step1["materials"] = materials_detail

            meshes_detail = []
            for i, mesh_def in enumerate(gltf.get('meshes', [])):
                for j, prim in enumerate(mesh_def.get('primitives', [])):
                    meshes_detail.append({
                        "mesh": i,
                        "primitive": j,
                        "material_index": prim.get('material', 'NONE'),
                    })
            step1["mesh_primitives"] = meshes_detail
        except Exception as e:
            step1["error"] = str(e)
        report["steps"]["1_gltf_json"] = step1

        # ---- Step 2: Extract textures ----
        step2 = {}
        try:
            extracted_textures, extracted_data = glb_parser.extract_textures(
                str(glb_path), glb_path.stem
            )
            texture_dir = OUTPUT_DIR / glb_path.stem
            step2["extracted_count"] = len(extracted_textures)
            tex_details = []
            for idx, fname in extracted_textures.items():
                fpath = texture_dir / fname
                tex_details.append({
                    "image_index": idx,
                    "filename": fname,
                    "exists_on_disk": fpath.exists(),
                    "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
                })
            step2["textures"] = tex_details
        except Exception as e:
            step2["error"] = str(e)
        report["steps"]["2_texture_extraction"] = step2

        # ---- Step 3: get_all_mesh_data ----
        step3 = {}
        try:
            mesh_data = glb_parser.get_all_mesh_data(str(glb_path))
            step3["parts_count"] = len(mesh_data)
            parts_detail = []
            for part_id, data in mesh_data.items():
                mat = data.get('material')
                has_uv = data.get('uv_coords') is not None
                detail = {
                    "part_id": part_id,
                    "vertex_count": len(data['vertices']),
                    "has_uv_coords": has_uv,
                    "uv_count": len(data['uv_coords']) if has_uv else 0,
                    "has_material": mat is not None,
                }
                if mat is not None:
                    detail["material_type"] = type(mat).__name__
                    detail["diffuse_color"] = list(getattr(mat, 'diffuse_color', []))
                    detail["has_base_color_texture"] = getattr(mat, 'has_base_color_texture', False)
                    tex = getattr(mat, 'base_color_texture', None)
                    detail["texture_filename"] = getattr(tex, 'filename', None) if tex else None
                else:
                    detail["material_type"] = "None"
                    detail["PROBLEM"] = "No material extracted! Model will be white."
                parts_detail.append(detail)
            step3["parts"] = parts_detail
        except Exception as e:
            step3["error"] = str(e)
        report["steps"]["3_mesh_data"] = step3

        # ---- Summary ----
        problems = []
        if step1.get("images_count", 0) == 0:
            problems.append("GLB has NO embedded images — no textures possible")
        if step1.get("materials_count", 0) == 0:
            problems.append("GLB has NO materials defined")
        for mat in step1.get("materials", []):
            if not mat.get("hasBaseColorTexture") and mat.get("baseColorFactor") in ([1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0], 'N/A'):
                problems.append(f"Material[{mat['index']}] '{mat['name']}': white color + no texture = WHITE")
        if step2.get("extracted_count", 0) == 0 and step1.get("images_count", 0) > 0:
            problems.append("Images exist in GLTF but extraction FAILED")
        for p in step3.get("parts", []):
            if not p.get("has_material"):
                problems.append(f"Part '{p['part_id']}': no material extracted")
            elif p.get("has_base_color_texture") and not p.get("texture_filename"):
                problems.append(f"Part '{p['part_id']}': has texture flag but no filename")
            elif not p.get("has_base_color_texture") and p.get("diffuse_color") == [1.0, 1.0, 1.0]:
                problems.append(f"Part '{p['part_id']}': white diffuse + no texture = WHITE")

        report["problems"] = problems if problems else ["No obvious problems detected"]

        return JSONResponse(content=report)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if glb_path.exists():
            glb_path.unlink(missing_ok=True)
```

Note: this handler imports `glb_parser` from the module scope (editor's routes.py
already does `from app.services.glb_parser import glb_parser`), so no additional
imports are needed. `UPLOAD_DIR` and `OUTPUT_DIR` are already defined at module
scope.

- [ ] **Step 3: Syntax check**

Run: `python -m py_compile app/api/routes.py`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add app/api/routes.py
git commit -m "feat: port /debug-glb diagnostic handler from articulation_api.py

Returns a three-step report (GLTF JSON parse, texture extraction,
get_all_mesh_data) with detected problems. Same behavior as the
standalone article_api.py version, just inside the router.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Merge `requirements.txt`

Editor and service pin different package versions. Take the pins the current
service uses for packages both have (Dockerfile is built against them), and add
editor's extra pins where they don't exist in service. Drop editor's vestigial
dependencies (`scipy`, `aiofiles`, `starlette`) that are not actually imported
anywhere in editor's source (verified by grep).

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Replace `requirements.txt` with the merged set**

Overwrite the entire `requirements.txt` with:

```
# Web Framework
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.17

# Data validation
pydantic>=2.5.0

# 3D mesh processing
trimesh==4.0.8
numpy==1.26.3
Pillow>=10.0.0

# USD (Universal Scene Description)
usd-core>=24.8
```

Rationale per package:
- `fastapi/uvicorn/python-multipart`: keep service's newer pins — the container has been building against them.
- `pydantic`: loosen to `>=2.5.0` (editor validated with 2.5.3; fastapi 0.115.5 works with 2.5+).
- `trimesh` / `numpy`: adopt editor's pinned versions (validated against the material pipeline).
- `Pillow`: service-only, retained.
- `usd-core`: tighten to editor's `>=24.8`.
- `starlette` / `scipy` / `aiofiles`: editor had these but does not import them; dropping.

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: merge requirements.txt with editor's validated pins

Adopt editor's pinned trimesh 4.0.8 / numpy 1.26.3 and tighten usd-core
to >=24.8. Keep service's newer fastapi/uvicorn/multipart. Drop editor's
vestigial scipy, aiofiles, and starlette pins (none are imported).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Update `Dockerfile` (CMD and COPY layout)

The Dockerfile currently `COPY`s flat-layout `models/`, `services/`, and
`articulation_api.py` into `/app/`. After the swap, the code lives under `app/`
instead. CMD also needs to point at the new module.

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Replace the COPY block**

Old (lines 32-35):
```dockerfile
# Copy application source
COPY models/        /app/models/
COPY services/      /app/services/
COPY articulation_api.py /app/articulation_api.py
```

New:
```dockerfile
# Copy application source
COPY app/ /app/app/
```

- [ ] **Step 2: Replace the CMD**

Old (lines 49-53):
```dockerfile
CMD ["python", "-m", "uvicorn", "articulation_api:app", \
     "--host", "0.0.0.0", \
     "--port", "52071", \
     "--workers", "1", \
     "--log-level", "info"]
```

New:
```dockerfile
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "52071", \
     "--workers", "1", \
     "--log-level", "info"]
```

Port 52071, `EXPOSE`, and HEALTHCHECK are unchanged.

- [ ] **Step 3: Verify healthcheck URL is `/health` (unprefixed)**

Line 47 currently has:
```dockerfile
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:52071/health')" || exit 1
```

Editor's router mounts at prefix `/api`, so the health endpoint is actually at
`/api/health`, not `/health`. Update:

Old:
```dockerfile
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:52071/health')" || exit 1
```

New:
```dockerfile
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:52071/api/health')" || exit 1
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "chore: point Dockerfile at new app.main:app entrypoint

COPY app/ instead of flat-layout services/models/articulation_api.py.
CMD uvicorn app.main:app. Healthcheck URL corrected to /api/health
(router is mounted at prefix /api in editor's main.py).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Delete obsolete files

With the new `app/` layer in place, the flat-layout entrypoint and the ported
service modules are obsolete. Remove them.

**Files deleted:**
- `articulation_api.py`
- `services/` (entire directory)
- `models/` (entire directory)

- [ ] **Step 1: Confirm no external imports reference them**

Run:
```bash
grep -rn "from services\." --include="*.py" . || true
grep -rn "from models\." --include="*.py" . || true
grep -rn "import articulation_api" --include="*.py" . || true
```
Expected: no matches anywhere outside the files about to be deleted. If any
match exists inside `app/`, pause and investigate — a leftover import would
indicate the copy in Task 2 brought in something unexpected. In that case, fix
the import to `from app.services.X` or `from app.models.schemas`.

- [ ] **Step 2: Remove old files**

```bash
git rm articulation_api.py
git rm -r services/
git rm -r models/
```

- [ ] **Step 3: Confirm tree layout**

Run:
```bash
ls -la
```

Expected: no `services/`, `models/`, `articulation_api.py` at top level. `app/`
present.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove flat-layout legacy modules

articulation_api.py, services/, and models/ are superseded by app/.
All consumers in this repo now import from app.services.* / app.models.*
(verified by grep before deletion).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Build image and smoke-test `/api/health`

First verification: the container starts and health endpoint responds. Failure
here means CMD, imports, or requirements are wrong.

- [ ] **Step 1: Rebuild the image**

Run: `docker compose build`
Expected: build succeeds. If a pip install step fails, the most likely culprit is
`usd-core>=24.8` on the slim Bookworm image (pre-built wheels only exist for
certain Python / architecture combinations). Confirm the base image is Python
3.10-slim and retry. If it still fails, raise to the user before proceeding.

- [ ] **Step 2: Start the container**

Run: `docker compose up -d`
Expected: container reports `Up` in `docker compose ps`. Wait ~5 seconds for
startup.

- [ ] **Step 3: Check container logs for startup messages**

Run: `docker compose logs --tail 30 articulation`
Expected: log lines include "Starting Phidias Articulation API...", "USD
libraries loaded successfully", "Trimesh version: 4.0.8", "API startup complete",
and an uvicorn listening line on `0.0.0.0:52071`.

- [ ] **Step 4: Hit `/api/health`**

Run: `curl -sSf http://localhost:52071/api/health | head -c 200`
Expected: HTTP 200 with JSON body (editor's `/health` handler returns a health
dict; exact shape is defined inside editor's routes.py around the `/health`
handler). Print the body for the reviewer.

- [ ] **Step 5: Stop the container**

Run: `docker compose down`

- [ ] **Step 6: (No code changes in this task; nothing to commit.)**

If any step failed, fix the root cause in a follow-up commit on this branch and
rerun this task.

---

## Task 11: Smoke-test `/api/parse-glb` with a sample GLB

Verify the file upload path and response shape. If the user has a specific
sample GLB, use it. Otherwise, use BoomBox from the Khronos sample repo as a
known-textured test case.

- [ ] **Step 1: Obtain a sample GLB**

If a sample is provided by the user, save it as `/tmp/sample.glb`. Otherwise:

```bash
curl -sSL -o /tmp/sample.glb \
  "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BoomBox/glTF-Binary/BoomBox.glb"
ls -la /tmp/sample.glb
```
Expected: file size > 100 KB.

- [ ] **Step 2: Start the container**

Run: `docker compose up -d`
Wait ~5 seconds.

- [ ] **Step 3: POST the GLB to `/api/parse-glb`**

Run:
```bash
curl -sS -X POST -F "file=@/tmp/sample.glb" \
  http://localhost:52071/api/parse-glb | python -m json.tool | head -80
```

Expected: JSON response with keys `success: true`, `message`, `filename`,
`model_url`, `parts: [...]`. Each part should have `id`, `name`, `vertex_count`,
`face_count`, `bounds_min`, `bounds_max`, `is_watertight`, `material` (may be an
object with `diffuse_color`, `metallic`, etc., or `null`).

- [ ] **Step 4: Note the returned filename for Task 12**

The `filename` field in the response is the saved GLB under `uploads/`
(e.g. `a1b2c3d4_BoomBox.glb`). Task 12 uses this filename. Save it:

```bash
PARSED=$(curl -sS -X POST -F "file=@/tmp/sample.glb" \
  http://localhost:52071/api/parse-glb)
echo "$PARSED" | python -c "import sys, json; print(json.load(sys.stdin)['filename'])"
```
Record the value for Task 12.

- [ ] **Step 5: (No code changes; no commit.)**

If the response shape is wrong or a 500 occurred, fix on this branch and rerun.

---

## Task 12: End-to-end test `/api/export-usdz` with material verification

The acceptance gate: produce a USDZ from a sample GLB + a minimal articulation
payload, unzip it, run `diagnose_usdz.py` against the inner USDC/USDA, and
verify each mesh has a bound Material, non-empty `primvars:st`, and a
`UsdPreviewSurface` with either a `diffuseColor` connection (texture case) or a
non-default RGB value (solid-color case).

- [ ] **Step 1: Build a minimal articulation JSON**

Using the `parts` returned from Task 11's `/api/parse-glb` call, synthesize a
minimal articulation JSON with the first part marked `base` and no joints. Save
to `/tmp/articulation.json`:

```bash
PARTS_JSON=$(curl -sS -X POST -F "file=@/tmp/sample.glb" \
  http://localhost:52071/api/parse-glb)

python - <<'PY' > /tmp/articulation.json
import json
parts = json.loads(open("/dev/stdin").read())["parts"]
out = {
    "model_name": "smoke_test",
    "parts": [
        {
            "id": p["id"],
            "name": p["name"],
            "type": "base" if i == 0 else "link",
            "role": "other",
            "mobility": "fixed",
            "mass": None,
            "density": 1000.0,
            "collision_type": "convexHull",
            "static_friction": 0.5,
            "dynamic_friction": 0.5,
            "restitution": 0.0,
        }
        for i, p in enumerate(parts)
    ],
    "joints": [],
}
print(json.dumps(out))
PY
```

Note: the heredoc reads stdin — pipe `PARTS_JSON` in:

```bash
echo "$PARTS_JSON" | python - > /tmp/articulation.json <<'PY'
import sys, json
parts = json.loads(sys.stdin.read())["parts"]
out = {
    "model_name": "smoke_test",
    "parts": [
        {
            "id": p["id"],
            "name": p["name"],
            "type": "base" if i == 0 else "link",
            "role": "other",
            "mobility": "fixed",
            "mass": None,
            "density": 1000.0,
            "collision_type": "convexHull",
            "static_friction": 0.5,
            "dynamic_friction": 0.5,
            "restitution": 0.0,
        }
        for i, p in enumerate(parts)
    ],
    "joints": [],
}
print(json.dumps(out))
PY
```
Expected: `/tmp/articulation.json` contains valid JSON with at least one part
and an empty joint list.

- [ ] **Step 2: POST to `/api/export-usdz`**

Run:
```bash
RESP=$(curl -sS -X POST \
  -F "file=@/tmp/sample.glb" \
  -F "articulation=$(cat /tmp/articulation.json)" \
  http://localhost:52071/api/export-usdz)
echo "$RESP"
FILENAME=$(echo "$RESP" | python -c "import sys, json; print(json.load(sys.stdin)['filename'])")
echo "Generated USDZ: $FILENAME"
```

Expected: JSON response with `success: true`, `download_url`,
`filename` ending in `.usdz`.

- [ ] **Step 3: Download the USDZ**

```bash
curl -sSf -o /tmp/out.usdz \
  "http://localhost:52071/api/download/${FILENAME}"
ls -la /tmp/out.usdz
```

Expected: file exists, size > 10 KB.

- [ ] **Step 4: Run `diagnose_usdz.py`**

Run:
```bash
python scripts/diagnose_usdz.py /tmp/out.usdz > /tmp/diag.txt
cat /tmp/diag.txt
```

Expected output should include:
- `upAxis       : Z`
- At least one Mesh with:
  - `primvars:st           : <N> values, interpolation=vertex` (non-zero N)
  - `bound material        : /<model>/Looks/<name>` (not `NONE`)
- At least one Material with:
  - `surface -> <shader>.surface`
  - Shader `id=UsdPreviewSurface` with `input diffuseColor` showing either a
    non-white RGB value or a connection to `texture.rgb`
- USDZ archive listing showing both `.usdc` (or `.usda`) file(s) AND at least
  one `.png` texture (if the source GLB has textures).

- [ ] **Step 5: Stop the container**

Run: `docker compose down`

- [ ] **Step 6: Open the USDZ in Blender (or usdview) for visual spot check**

Manual: open `/tmp/out.usdz` in Blender or `usdview` (if installed). Confirm
meshes show color/texture, not pure white/gray. This matches the original user
complaint — if materials are still missing, the swap did not succeed and we
need to investigate before declaring done.

- [ ] **Step 7: (No code changes; commit diag output to /tmp only.)**

If all checks pass, this branch is ready for merge. Fast-forward `main` to the
current branch (or open a PR if the team prefers PRs):

```bash
git checkout main
git merge --ff-only feat/backend-swap-to-editor
```

Do not delete the feature branch until a visual QA in Isaac Sim has also
confirmed materials render — that is an out-of-loop check not representable in
this automated plan.

---

## Self-review notes

- **Spec coverage**: every item in the spec's Scope (delete, copy, modify) maps
  to a task: delete → Task 9, copy → Task 2, modify usd_builder Z-up → Task 3,
  route renames → Tasks 4+5, `/debug-glb` port → Task 6, Dockerfile → Task 8,
  requirements merge → Task 7, docker-compose verify (spec said "no change
  expected") covered implicitly by Task 10 smoke test; acceptance criteria →
  Tasks 10, 11, 12. Not-in-scope items stay out.
- **Types and names used consistently**: `_do_export`, `_save_and_export`,
  `export_usda`, `export_usdz`, `parse_glb`, `debug_glb` defined in Tasks 4–6;
  Task 10+ refers to the same endpoint paths `/api/parse-glb`, `/api/export-usda`,
  `/api/export-usdz`, `/api/debug-glb`, `/api/health`, `/api/download/*`.
- **Gateway prefix risk**: editor's `main.py` already mounts the router at
  `/api`. Frontend's `/phidias/articulation/*` → gateway → backend `/api/*`.
  Task 10 verifies health via `/api/health`, validating the assumption.
- **Joint-axis Y→Z risk** (spec Risk 3): acknowledged but not addressed by this
  plan. Joint axes come through the frontend unchanged; if Isaac Sim shows
  rotations around the wrong axis, a follow-up to rotate joint `axis` and
  `anchor` in `_do_export` before passing to `physics_injector` will be
  needed. Not in scope for this swap.
