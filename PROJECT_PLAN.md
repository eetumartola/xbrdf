# xBRDF Project Plan

## Summary

Build xBRDF as a Rust GPU baker plus later preview/shader tooling. The MVP is a headless `wgpu` baker that loads one periodic OBJ microgeometry tile, uses one fixed directional light, samples camera directions over the upper hemisphere, and writes one hemisphere latlong OpenEXR image plus a manifest.

## Task List

- [x] Inspect workspace and Rust toolchain.
- [x] Lock MVP scope: one fixed light, one hemisphere pano, headless first, `wgpu`, EXR output.
- [x] Initialize git repository.
- [x] Create `PROJECT_PLAN.md` and record the initial decisions above.
- [x] Scaffold Rust workspace and `xbrdf-bake` CLI.
- [x] Implement TOML config, CLI overrides, resolved settings, and manifest writing.
- [x] Implement OBJ loading, triangulation, bounds, tile period, and validation errors.
- [x] Implement hemisphere latlong mapping and deterministic sample generation.
- [x] Implement CPU reference math for ray-triangle hits and normalization tests.
- [x] Implement `wgpu` compute buffers, pipeline, and shader dispatch.
- [x] Implement brute-force triangle intersection, periodic wrapping, direct lighting, and shadow rays in WGSL.
- [x] Replace brute-force triangle traversal with a GPU BVH and chunked row dispatches for larger meshes.
- [x] Implement EXR writing and smoke-bake output folder creation.
- [x] Add fixture OBJs: flat plane, seam-crossing tile, and shadow ridge.
- [x] Add unit tests and one end-to-end smoke test.
- [x] Document MVP usage in `README.md`.
- [x] Add configurable Lambertian/specular material support with sharp finite mirror-like lobes.
- [x] Add per-triangle color support from OBJ vertex colors, OBJ material diffuse colors, and binary FBX color layers.
- [ ] Post-MVP: add light-direction sweep producing one pano per light direction.
- [ ] Post-MVP: add interpolation metadata and lookup conventions for shader consumption.
- [x] Post-MVP: add BVH acceleration.
- [x] Post-MVP: add Dear ImGui preview and bake-control app.
- [ ] Post-MVP: prototype Houdini shader/import workflow.

## Decisions

- 2026-04-25: MVP writes one pano for one fixed light direction so the renderer, normalization, and artifact shape can be validated before adding light sweeps.
- 2026-04-25: Use `wgpu` compute for the bake path to keep the renderer GPU accelerated while preserving portability.
- 2026-04-25: Use arbitrary OBJ triangle meshes, not a heightfield-only representation, because the microfacet sample should be explicitly modeled geometry.
- 2026-04-25: Use brute-force GPU triangle intersection first. BVH acceleration is a later scalability task after the data and math are validated.
- 2026-04-25: Use OpenEXR RGB float output plus a TOML manifest for inspectable, reproducible bake artifacts.
- 2026-04-25: Use Houdini coordinates: Y up, sample plane in XZ, macro normal +Y, +Z at the horizontal center of the pano, azimuth increasing toward +X.
- 2026-04-25: Keep the GPU smoke test ignored by default because `cargo test` should pass on machines without a compatible `wgpu` adapter; run it explicitly when validating the renderer.
- 2026-04-25: On this Dropbox-backed Windows workspace, use a temp Cargo target directory for reliable builds because the default `target` directory can hit file-locking errors.
- 2026-04-25: Add `specular_phong` as a finite normalized reflection lobe rather than a perfect mirror delta. A mathematical mirror is not representable as ordinary finite latlong pixels, so roughness `0` maps to a very high clamped exponent.
- 2026-04-25: Keep output `width` and `height` as config/CLI settings. The OBJ determines geometry bounds and the default XZ tile period, not pano resolution.
- 2026-04-25: Replaced brute-force GPU traversal with a CPU-built/GPU-traversed BVH after 32k+ triangle tests exposed Windows GPU watchdog risk and black outputs.
- 2026-04-25: Added `max_repeat_radius` with default `2`. Periodic copies are still searched, but the cap is explicit because near-horizon rays can otherwise multiply large meshes into unsafe workloads.
- 2026-04-25: Split GPU work into row chunks and wait between dispatches so long bakes are less likely to freeze the desktop or trigger a driver reset.
- 2026-04-25: Normalize loaded OBJ geometry so the highest point is at Y=0. This keeps camera origins and grazing behavior consistent across assets and records the Y offset in the manifest.
- 2026-04-25: Define each pixel's camera-ray footprint as the full periodic XZ tile. Individual rays are point samples of that footprint, using per-pixel low-discrepancy tile samples to reduce coherent point/grid artifacts.
- 2026-04-25: Keep shading faceted by using per-triangle geometric normals only. OBJ vertex normals and smoothing groups are ignored because the explicit microgeometry should define the BRDF response.
- 2026-04-25: Simplify local CLI use with `default-members`, optional `bake` subcommand syntax, and a `cargo bake` alias. Preferred repo-root command is now `cargo bake --config ... --out ...`.
- 2026-04-25: Add the Dear ImGui bake-control app as a separate `xbrdf-gui` crate using a Glium ImGui renderer. The bake path remains `wgpu`; the GUI only owns controls, the preview texture, and the event loop.
- 2026-04-25: Add a progress-capable GPU bake entry point that reads back completed row chunks. This is slower than the CLI's final-only readback, but it lets long renders show real viewport progress.

## Verification Log

- 2026-04-25: `cargo test` passed with `CARGO_INCREMENTAL=0` and `CARGO_TARGET_DIR=%TEMP%\xbrdf-target`; 8 unit tests passed, 1 GPU smoke test ignored by default.
- 2026-04-25: `cargo test -p xbrdf-cli --test smoke_bake -- --ignored` passed; it runs the GPU baker, writes EXR/manifest, reads the EXR, and verifies the flat-plane response is approximately `1/pi`.
- 2026-04-25: `cargo run -p xbrdf-cli -- bake --config assets/fixtures/flat.toml --out out/flat` passed and wrote `out/flat/xbrdf_view.exr` plus `out/flat/manifest.toml`.
- 2026-04-25: After material support, `cargo test` passed with 10 unit tests, the ignored GPU smoke test passed explicitly, and `cargo run -p xbrdf-cli -- bake --config assets/fixtures/specular.toml --out out/specular` wrote a specular EXR/manifest.
- 2026-04-25: After BVH/chunking support, `cargo test` passed, the ignored GPU smoke test passed explicitly, and the current `assets/fixtures/specular.toml` using `rough_a.obj` baked successfully with `triangle_count = 32258`, `width = 128`, `height = 64`, `samples = 8`, and `max_repeat_radius = 2`.
- 2026-04-25: After tile-footprint sampling and height normalization changes, `cargo test` passed with 11 unit tests, the ignored GPU smoke test passed explicitly, and the current specular fixture baked with shifted bounds where `bounds_max.y = 0`.
- 2026-04-25: After CLI simplification, `cargo test --workspace` passed, `cargo bake --config assets/fixtures/flat.toml --out out/flat_alias` passed, and the compatibility form `cargo run -- bake --config assets/fixtures/flat.toml --out out/flat_compat` passed.
- 2026-04-25: Add CLI bake stats after output writes: geometry/BVH size, image and sample counts, repeat cap, dispatch chunks, estimated ray work, phase timings, and GPU dispatch throughput.
- 2026-04-25: Add `cargo bake-release` alias because Cargo forwards arguments after alias expansion; `cargo bake --release ...` would pass `--release` to `xbrdf-bake` instead of Cargo.
- 2026-04-25: Add per-triangle color as a mesh attribute multiplied by material color in the shader. OBJ uses vertex colors first, MTL diffuse second, and white fallback. FBX support starts with binary FBX mesh geometry plus common `LayerElementColor` mappings.
- 2026-04-25: Added `assets/fixtures/colors.obj` and `colors.toml` as a small vertex-color bake fixture.
- 2026-04-25: `cargo test --workspace` passed with 12 unit tests. `cargo bake --config assets/fixtures/colors.toml --out out/colors` passed and reported `colors: obj_vertex_color`. Current large specular fixture also baked with the color-capable triangle format.
- 2026-04-25: Tuned GPU BVH traversal for large meshes. Shadow rays now use an any-hit early-out traversal, and camera closest-hit traversal visits nearer child AABBs first. On the 129032-triangle FBX fixture at 128x64, 1200 samples, release GPU dispatch improved from about 12.8s / 0.77M camera rays/s to about 8.5s / 1.16M camera rays/s. A tested 8-triangle leaf size and near-sorted shadow traversal were slower, so the leaf size remains 4 and shadow traversal stays unsorted.
- 2026-04-25: Added `xbrdf-gui` with ImGui controls for config/source/output paths, resolution, samples, repeat radius, light, tile overrides, material, color, and roughness. `cargo test --workspace -j 1` passed with 12 unit tests and the default-ignored GPU smoke test; `cargo test -p xbrdf-cli --test smoke_bake -- --ignored` also passed explicitly.
- 2026-04-25: Fixed GUI startup/config ergonomics. The app now initializes controls from the default config path, preserves viewport aspect ratio when fitting to panel height, and resolves editable geometry-path overrides from the working directory so loaded config-relative paths are not joined twice.
- 2026-04-25: Replaced GUI row-progress preview with progressive full-frame sample integration. The GUI now accumulates whole-image sample batches, updates at a configurable interval, and reports completed samples against the target sample count. `cargo test --workspace -j 1` and the explicit GPU smoke test passed.
- 2026-04-25: Optimized release bake dispatch sizing and specular shadow work. Row chunks now use a trace-budget rounded to the 8-row workgroup height instead of one-row dispatches, shadow rays use a tighter periodic repeat radius, and zero-lobe specular samples skip shadow traversal. On `assets/fixtures/specular.toml`, `cargo bake-release --config assets/fixtures/specular.toml --out out/specular` improved GPU dispatch from about 4.61s / 1.14M rays/s to about 0.49s / 10.65M rays/s.
- 2026-04-25: Fixed high-sample dispatch underutilization. The row-chunk heuristic now rounds up to at least one full 8-row workgroup and the GUI progressive path sizes row chunks from the active sample batch instead of the final sample target. On the 12200-sample specular fixture, `cargo bake-release --config assets/fixtures/specular.toml --out out/specular` changed from 64 one-row dispatches at about 73.6s / 1.36M rays/s to 4 sixteen-row dispatches at about 10.9s GPU dispatch / 9.16M rays/s, with total time about 16.1s.
- 2026-04-26: Added GUI render history. Progressive preview frames are retained for the session, the viewport keeps showing previous output while a new bake starts, and a ticked slider under the image can scrub through retained renders. `cargo check -p xbrdf-gui` and `cargo test --workspace -j 1` passed.
- 2026-04-26: Changed GUI render history to retain only finished bakes. Progressive updates still drive the live viewport, but the history slider now scrubs completed renders only. `cargo check -p xbrdf-gui` and `cargo test --workspace -j 1` passed.

## Acceptance Criteria

- A fresh checkout can run `cargo test`.
- A debug bake produces `xbrdf_view.exr` and `manifest.toml`.
- The manifest contains every resolved setting required to reproduce the bake.
- The flat-plane fixture matches the expected Lambertian result within a small tolerance.
- The code path is GPU accelerated for the actual bake computation.
- This task list and decision log are updated as implementation progresses.
