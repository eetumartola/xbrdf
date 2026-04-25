# xBRDF

xBRDF is an explicit BRDF baker for microgeometry tiles. The current baker loads OBJ or binary FBX tiles, treats X/Z as a periodic domain with Y up, renders one upper-hemisphere latlong response for a fixed directional light, and writes OpenEXR data plus a reproducibility manifest.

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

- `xbrdf_view.exr`: RGB float hemisphere latlong response.
- `manifest.toml`: resolved bake settings and conventions.

After each bake, the CLI prints basic timing and workload stats: geometry size, BVH node count, image size, samples, periodic repeat cap, dispatch chunks, estimated ray work, and phase timings. GPU work is chunked by a trace budget rounded up to full 8-row compute workgroups, so high-sample renders avoid one-row dispatch underutilization while still splitting very large bakes.

## GUI

Launch the Dear ImGui bake-control app from the repo root:

```powershell
cargo gui
```

The GUI exposes the same basic bake settings as the CLI: config path, source geometry, output folder, resolution, samples, repeat radius, light, tile overrides, material, color, and roughness. Bakes run on a background thread and the viewport progressively updates the full image as new sample batches are integrated. The update interval is configurable in seconds, and the progress bar shows completed samples against the target sample count. Finished renders are retained in session history; use the ticked slider under the viewport to scrub back through completed bakes. The GUI still writes `xbrdf_view.exr` and `manifest.toml` to the selected output folder.

`width` and `height` are the output pano resolution. They are not read from the source mesh. OBJ/FBX files are used for triangle geometry and, unless overridden by `tile_width` / `tile_depth`, the periodic XZ tile bounds.

Input mesh support:

- OBJ: geometry, faceted normals, vertex colors, and MTL diffuse colors.
- FBX: binary FBX mesh geometry and common `LayerElementColor` color layers.

`max_repeat_radius` caps how many periodic tile copies are searched in each X/Z direction. The default is `2`, meaning up to a `5x5` neighborhood. Larger values improve grazing-angle periodic coverage but increase cost quickly.

Loaded geometry is shifted in Y so its highest point is at `0`. The original and shifted bounds, plus the applied Y offset, are recorded in the manifest.

For each output pixel, the camera direction is fixed and the ray footprint is the full periodic XZ tile. `samples` controls how many point rays estimate that tile-area integral. Low values such as `6` or `8` are useful for debugging but will look point-sampled on detailed geometry; increase `samples` to integrate the full microgeometry response.

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
