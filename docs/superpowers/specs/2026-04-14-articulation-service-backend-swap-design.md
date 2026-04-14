# Articulation Service — Backend Swap to Editor Pipeline

**Date**: 2026-04-14
**Status**: Approved for implementation planning

## Problem

The ported texture/material pipeline in `articulation-service` produces USDZ files where
geometry is visible in Isaac Sim and Blender but materials and textures do not render
(no color, no base-color texture). The pipeline logic was ported piece-by-piece from
`articulation-editor`'s `bugfix--usdz-texture` branch, but the port has accumulated
divergences (import layout flattening, Z-up rotation, schema renames, refactored
`physics_injector`) and several regression fixes.

The upstream `articulation-editor` branch `bugfix--usdz-texture` has a working,
validated pipeline that produces textured USDZ files. Rather than continue debugging
the port, we replace the service's pipeline wholesale with editor's code plus a
minimal Z-up delta needed for Isaac Sim.

## Decision

Copy `articulation-editor`'s `backend/app/` (at commit `cc91f91`, branch
`bugfix--usdz-texture`) into `articulation-service/app/`. Rename backend route paths
to match the names Phidias frontend already calls. Apply a three-line Z-up patch in
`usd_builder.py`. Delete the old `services/`, `models/`, and `articulation_api.py`.

Rejected alternatives:

- **Align frontend to editor's paths** (`/upload` + single `/export`): requires
  changing `PhysicsExportButtons.tsx`, `lib/api/phidias.ts`, `lib/api/types.ts`, and
  `lib/api/mock.ts`. Frontend is stable; backend is the problem surface — change
  should go there.
- **Dual-name router (both old and new paths)**: unnecessary since Phidias is the
  only consumer. Adds maintenance without benefit.

## Scope

### Delete

- `services/` (entire directory — old `glb_parser.py`, `usd_builder.py`,
  `physics_injector.py`, `usdz_packager.py`)
- `models/` (entire directory — old `schemas.py` with `ArticulationPart`,
  `ArticulationJoint` renames)
- `articulation_api.py` (FastAPI entrypoint; `/api/debug-glb` handler is preserved
  by porting it into the new router, see below)

### Copy from `articulation-editor` `origin/bugfix--usdz-texture`

Source: commit `cc91f91`, path `backend/`.

- `backend/app/` → `articulation-service/app/` (entire tree)
  - `app/main.py`
  - `app/api/routes.py`
  - `app/models/schemas.py`
  - `app/services/glb_parser.py`
  - `app/services/usd_builder.py`
  - `app/services/physics_injector.py`
  - `app/services/usdz_packager.py`
- `backend/requirements.txt` → merge into existing `requirements.txt` (take union;
  prefer newer pins on conflict — verify each conflict during implementation)

### Modify after copy

**`app/api/routes.py`** — rename route paths to match what Phidias frontend calls
(frontend's `/phidias/articulation/*` prefix is stripped by the upstream gateway;
backend sees the suffix):

| Editor original | After rename |
|---|---|
| `POST /upload` | `POST /parse-glb` |
| `POST /export` (single, `output_format` field) | split into `POST /export-usda` and `POST /export-usdz`; both handlers delegate to the same internal function with a different `output_format` argument |
| `GET /download/{filename}` | unchanged |
| `GET /models/{filename}` | unchanged |
| `GET /health` | unchanged |
| *(new)* | `POST /debug-glb` — port handler body from old `articulation_api.py` |

The `/debug-glb` endpoint inspects a GLB file and reports what the pipeline sees at
each stage (GLTF materials, extracted textures, `MaterialInfo` per part, detected
problems). It is useful for debugging and was the most recent addition to the
service before this swap; we keep it.

**`app/services/usd_builder.py`** — apply three changes for Isaac Sim Z-up:

1. `create_stage` default `up_axis="Y"` → `up_axis="Z"`.
2. `add_physics_scene` gravity `Gf.Vec3f(0, -1, 0)` → `Gf.Vec3f(0, 0, -1)`.
3. In `build_from_parts`, rotate vertices and normals before passing to
   `add_mesh_prim`:
   ```python
   verts_zup = np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])
   # same transform for normals if present
   ```

Nothing else in editor's `usd_builder.py` changes — the material/shader graph,
UV primvar, `MaterialBindingAPI.Apply`, and USDZ packaging logic are kept exactly.

**`Dockerfile`** — update CMD to new entrypoint (port stays 52071):

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "52071"]
```

`EXPOSE 52071` and the healthcheck URL remain unchanged.

**`docker-compose.yml`** — no change expected (`${API_PORT:-52071}:52071` already
matches). Verify volume mounts do not reference deleted paths (`services/`,
`articulation_api.py`).

### Do not change

- **Frontend**: `src/lib/api/phidias.ts` and `src/lib/api/types.ts` in
  `phidias-standalone` stay as-is. Backend route names are chosen to match.
- **Schemas**: editor's `Part` / `Joint` / `ArticulationData` are used verbatim.
  Frontend's `ArticulationExportPart` / `ArticulationExportJoint` already have
  field-name parity with editor's schemas
  (`id`, `name`, `type`, `mass`, `density`, `collision_type`, `static_friction`,
  `dynamic_friction`, `restitution`, `parent`, `child`, `axis`, `anchor`,
  `lower_limit`, `upper_limit`, `drive_stiffness`, `drive_damping`,
  `drive_max_force`, `drive_type`, `disable_collision`). Editor's `role`,
  `mobility`, and `center_of_mass` have defaults or are `Optional`, so
  frontend omitting them is fine.
- **Editor's `physics_injector.py`, `glb_parser.py`, `usdz_packager.py`**: no
  modifications. These are the files whose divergence caused the texture
  regression; they get reset to the known-working editor version.

## Data flow

Unchanged semantics, same modules:

```
Frontend ArticulationExportData (FormData)
  ├─ glb_file: File
  └─ articulation: JSON → pydantic ArticulationData
                           ├─ model_name
                           ├─ parts:  List[Part]
                           └─ joints: List[Joint]
       ↓
  glb_parser.get_all_mesh_data(glb)
       ↓  mesh_data + MaterialInfo + extracted textures
  usd_builder.build_from_parts(..., up_axis="Z", Y→Z rotation applied)
       ↓  USD stage with UsdPreviewSurface shader graph
  physics_injector.inject_physics(stage, articulation, part_paths)
       ↓
  usd_builder.save_stage → USDA on disk
       ↓
  usdz_packager.create_usdz (UsdUtils.CreateNewARKitUsdzPackage)
       ↓
  Response: { success, filename, download_url }
```

## Risks

1. **Gateway prefix assumption**: the design assumes
   `/phidias/articulation/parse-glb` on the frontend reaches the backend as
   `/parse-glb`. If the gateway instead forwards the full path, the backend needs
   `app.include_router(router, prefix="/phidias/articulation")`. Verify in the
   first implementation step by running the container and hitting one endpoint.

2. **`requirements.txt` merge conflicts**: editor and service may pin different
   versions of `fastapi`, `pydantic`, `usd-core`, `trimesh`, `Pillow`, `numpy`.
   Resolve during implementation by taking the union and, on conflict, the pin
   that matches the Python 3.10 / Debian Bookworm base image. If a conflict is
   unclear, raise it before proceeding.

3. **Z-up regression in physics**: editor runs Y-up; service runs Z-up with
   geometry rotation. Physics joint axes and anchors in the exported USDA use the
   same coordinates the frontend sends. If the frontend sends joint axes in GLB
   (Y-up) frame, they also need to be rotated. Verify joint rendering in Isaac
   Sim after the swap — this is not a new risk (current service already has this
   patch), but noted for the test plan.

4. **`/debug-glb` payload drift**: the old handler imports from `models.schemas`
   and `services.*`. After the swap, imports become `app.models.schemas` and
   `app.services.*`. Mechanical rename during port.

## Acceptance criteria

- `docker compose up` starts the service on port 52071; `GET /health` returns 200.
- `POST /parse-glb` with a sample GLB returns the same shape as Phidias frontend
  currently expects (check against `ParsedPhysicsResult` type).
- `POST /export-usdz` with the sky_car articulation data produces a USDZ whose
  meshes show materials/textures when opened in Blender and Isaac Sim.
- `diagnose_usdz.py scripts/diagnose_usdz.py <output>.usdz` reports, for each
  Mesh: a bound Material, a non-empty `primvars:st`, and a `UsdPreviewSurface`
  shader with `diffuseColor` connected (if texture) or with a non-default RGB
  value (if solid color).
- Phidias frontend's Export USDA and Export USDZ buttons work end-to-end without
  any frontend code change.
- `/debug-glb` returns the same diagnostic payload as before the swap.

## Not in scope

- Refactoring editor's route handlers to merge `export-usda` and `export-usdz`
  into a single endpoint (the two-handler shape is chosen to match frontend's
  current API surface; revisit after this lands).
- Making the Z-up axis configurable via query param or env var.
- Porting other changes from service into editor (this is a one-way replacement).
- Cleaning up the `services/`, `models/`, and `articulation_api.py` imports
  elsewhere — there are none outside those files; a grep will confirm during
  implementation.
