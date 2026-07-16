use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .to_path_buf()
}

fn run_bake(out: &Path, extra_args: &[String]) {
    let binary = env!("CARGO_BIN_EXE_xbrdf-bake");
    let config = workspace().join("assets/fixtures/flat.toml");
    let mut command = Command::new(binary);
    command
        .args(["bake", "--config"])
        .arg(config)
        .arg("--out")
        .arg(out)
        .args(extra_args);
    assert!(command.status().unwrap().success());
}

fn read_rgb(path: &Path) -> Vec<Vec<[f32; 3]>> {
    exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| vec![vec![[0.0_f32; 3]; resolution.width()]; resolution.height()],
        |pixels, position, (r, g, b, _a): (f32, f32, f32, f32)| {
            pixels[position.y()][position.x()] = [r, g, b];
        },
    )
    .unwrap()
    .layer_data
    .channel_data
    .pixels
}

#[test]
#[ignore = "requires a compatible wgpu adapter"]
fn fixture_bake_writes_exr_and_manifest() {
    let out_dir = tempfile::tempdir().unwrap();
    run_bake(out_dir.path(), &[]);

    let exr_path = out_dir.path().join("xbrdf_view.exr");
    assert!(exr_path.exists());
    assert!(out_dir.path().join("manifest.toml").exists());

    let expected = 1.0 / std::f32::consts::PI;
    for row in read_rgb(&exr_path) {
        for pixel in row {
            assert!((pixel[0] - expected).abs() < 0.01, "{pixel:?}");
            assert!((pixel[1] - expected).abs() < 0.01, "{pixel:?}");
            assert!((pixel[2] - expected).abs() < 0.01, "{pixel:?}");
        }
    }
}

#[test]
#[ignore = "requires a compatible wgpu adapter"]
fn full_atlas_tiles_match_single_light_bakes() {
    let root = tempfile::tempdir().unwrap();
    let atlas_dir = root.path().join("atlas");
    let fixture = root.path().join("ridge.obj");
    std::fs::write(
        &fixture,
        "v -0.5 0.0 -0.5\n\
         v 0.0 0.35 -0.5\n\
         v 0.5 0.0 -0.5\n\
         v -0.5 0.0 0.5\n\
         v 0.0 0.35 0.5\n\
         v 0.5 0.0 0.5\n\
         f 1 4 5\n\
         f 1 5 2\n\
         f 2 5 6\n\
         f 2 6 3\n",
    )
    .unwrap();
    let common = vec![
        "--obj".to_string(),
        fixture.display().to_string(),
        "--width".to_string(),
        "8".to_string(),
        "--height".to_string(),
        "4".to_string(),
        "--samples".to_string(),
        "64".to_string(),
    ];
    let mut atlas_args = common.clone();
    atlas_args.extend([
        "--mode".to_string(),
        "full".to_string(),
        "--light-width".to_string(),
        "2".to_string(),
        "--light-height".to_string(),
        "2".to_string(),
    ]);
    run_bake(&atlas_dir, &atlas_args);
    let atlas = read_rgb(&atlas_dir.join("xbrdf_view.exr"));

    for light_y in 0..2 {
        for light_x in 0..2 {
            let direction =
                xbrdf_core::sampling::hemisphere_latlong_direction(light_x, light_y, 2, 2);
            let single_dir = root.path().join(format!("single-{light_x}-{light_y}"));
            let mut single_args = common.clone();
            single_args.extend([
                "--mode".to_string(),
                "single".to_string(),
                format!("--light={},{},{}", direction.x, direction.y, direction.z),
            ]);
            run_bake(&single_dir, &single_args);
            let single = read_rgb(&single_dir.join("xbrdf_view.exr"));

            for y in 0..4usize {
                for x in 0..8usize {
                    let actual = atlas[light_y as usize * 4 + y][light_x as usize * 8 + x];
                    let expected = single[y][x];
                    for channel in 0..3 {
                        assert!(
                            (actual[channel] - expected[channel]).abs() < 1.0e-5,
                            "tile=({light_x},{light_y}) pixel=({x},{y}) actual={actual:?} expected={expected:?}"
                        );
                    }
                }
            }
        }
    }
}
