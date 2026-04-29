# xBRDF

xBRDF is an explicit BRDF baker for microgeometry tiles. The current baker loads OBJ or binary FBX tiles, treats X/Z as a periodic domain with Y up, renders one or more upper-hemisphere response tiles, and writes OpenEXR data plus a reproducibility manifest.

## Build

```powershell
cargo test --workspace
```

## Bake A Fixture

```powershell
cargo bake --config assets/fixtures/flat.toml --out out/flat
cargo bake --config assets/fixtures/colors.toml --out out/colors
cargo bake --config assets/fixtures/specular.toml --out out/specular
```

For optimized bakes, run in release mode:

```powershell
cargo bake-release --config assets/fixtures/specular.toml --out out/specular
```

Equivalent without the Cargo alias:

```powershell
cargo run -- --config assets/fixtures/flat.toml --out out/flat
cargo run --release -- --config assets/fixtures/specular.toml --out out/specular
```

The bake writes:

- `xbrdf_view.exr`: RGB float response. In `single` mode this is one hemisphere latlong pano. In atlas modes this is one large texture containing a grid of response tiles.
- `manifest.toml`: resolved bake settings and conventions.

After each bake, the CLI prints basic timing and workload stats: geometry size, BVH node count, image size, samples, periodic repeat cap, dispatch chunks, estimated ray work, and phase timings. GPU work is chunked by a trace budget rounded up to full 8-row compute workgroups. High-sample CLI bakes use a sample-parallel path that dispatches multiple sample lanes per pixel, improving occupancy for very specular, high-sample renders. Very large CLI bakes switch to larger 512-sample lanes to reduce wave count. The GPU BVH uses binned SAH splitting, which costs more to build than median splitting but improves traversal for large sample counts.

## GUI

Launch the Dear ImGui bake-control app from the repo root:

```powershell
cargo gui
```

The GUI exposes the same basic bake settings as the CLI: config path, source geometry, output folder, resolution, bake mode, light-grid dimensions, samples, repeat radius, light, material, color, and roughness. Bakes run on a background thread. `single` mode progressively updates the full image as new sample batches are integrated; atlas modes update the viewport and progress bar as each light tile completes. The GUI shows the calculated output texture extent from the camera tile and light grid. Finished renders are retained in session history; use the ticked slider under the viewport to scrub back through completed bakes. `Save Current Atlas` writes the selected live/history atlas as `xbrdf_current_atlas.exr` in the output folder, and `Load Current Atlas` reads that file back using the current GUI mode and atlas-dimension settings. The 3D preview window shades the selected preview model from the currently selected 2D preview or history entry and uses the matching `single`, `full`, or `isotropic` atlas lookup path. Preview models include the synthetic torus plus FBX files in `assets/preview`; FBX models are centered and uniformly scaled to fit the preview, use smoothed preview normals, and use UVs for anisotropic `du`/`dv` tangent directions when present. Drag inside the 3D image to rotate freely, right-drag to roll, and double-click it to reset rotation. The GUI sample input currently allows up to 100,000,000 samples. The GUI still writes `xbrdf_view.exr` and `manifest.toml` to the selected output folder.

`width` and `height` are the camera tile resolution. They are not read from the source mesh. OBJ/FBX files are used for triangle geometry and the periodic XZ tile bounds.

Put `mode`, `light_width`, and `light_height` in the TOML config for reproducible atlas bakes. CLI overrides are useful for quick tests, but config values should be the source of truth for planned bakes.

Bake modes:

- `mode = "single"` keeps the original fixed-light bake. Output size is `width x height`.
- `mode = "full"` bakes an anisotropic 4D table. Each light direction gets one `width x height` camera pano, and the panos are stored in a grid. Output size is `(width * light_width) x (height * light_height)`.
- `mode = "isotropic"` bakes a tangent-space-free table. Camera directions keep only elevation samples, so each light tile is `1 x height`. Output size is `light_width x (height * light_height)`.

Light-grid directions use the same upper-hemisphere latlong convention as camera directions. In `single` mode, the explicit `light = [x, y, z]` direction is used.

Input mesh support:

- OBJ: geometry, faceted normals, vertex colors, and MTL diffuse colors.
- FBX: binary FBX mesh geometry and common `LayerElementColor` color layers.

`max_repeat_radius` caps how many periodic tile copies are searched in each X/Z direction. The default is `2`, meaning up to a `5x5` neighborhood. Larger values improve grazing-angle periodic coverage but increase cost quickly.

Loaded geometry is shifted in Y so its highest point is at `0`. The original and shifted bounds, plus the applied Y offset, are recorded in the manifest.

For each output pixel, the camera direction is fixed and the ray footprint is the full periodic XZ tile. `samples` controls how many point rays estimate that tile-area integral. Low values such as `6` or `8` are useful for debugging but will look point-sampled on detailed geometry; increase `samples` to integrate the full microgeometry response. The GPU sampler uses a progressive low-discrepancy sequence, so an in-progress preview at a given sample count uses the same sample prefix as a render whose final target is that count.

The baked value is an effective BRDF-like response normalized by macro incident irradiance. For direct lighting, the consuming shader still multiplies the lookup by the local macro cosine `max(dot(N, L), 0)`. This keeps a flat Lambertian bake near `1/pi` and leaves the smooth surface terminator in the object shader rather than baking it into every light direction.

Shadow rays are enabled by default. Set `enable_shadows = false` in TOML, pass `--enable-shadows false` on the CLI, or clear the GUI `Shadows` checkbox to bake direct lighting without occlusion toward the light.

Sampler modes:

- `sampler = "halton"` is the default progressive low-discrepancy sequence.
- `sampler = "random"` uses hashed independent tile samples. It is mainly a diagnostic mode for checking whether persistent image structure comes from deterministic sampler bias or from the actual microgeometry response. CLI override: `--sampler random`.

## Material Config

The default material is white Lambertian:

```toml
[material]
kind = "lambertian"
color = [1.0, 1.0, 1.0]
```

For a mirror-like finite lobe, use normalized Phong specular:

```toml
[material]
kind = "specular_phong"
color = [1.0, 1.0, 1.0]
roughness = 0.02
```

`roughness` is clamped to `0..=1`; `0` is treated as a very sharp finite lobe rather than a mathematical delta, because a perfect mirror delta cannot be represented directly in a finite latlong image.

## Coordinate Convention

- Houdini coordinates: Y up, sample tile lies in XZ.
- Macro normal: `+Y`.
- Pano horizontal center: `+Z`.
- Azimuth increases toward `+X`.
- Pano rows run from zenith at the top to horizon at the bottom.
- Shading uses faceted geometric triangle normals. OBJ vertex normals and smoothing groups are ignored.

## MVP Limits

- One fixed light direction.
- White Lambertian and normalized Phong specular materials.
- Direct lighting and visibility only.
- GPU BVH triangle intersection with chunked row dispatches.
- Texture maps, multiple scattering, multi-light BRDF sweeps, and Houdini shader lookup are post-MVP work.
