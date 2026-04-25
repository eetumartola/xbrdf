use std::process::Command;

#[test]
#[ignore = "requires a compatible wgpu adapter"]
fn fixture_bake_writes_exr_and_manifest() {
    let out_dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_xbrdf-bake");
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap();
    let config = workspace.join("assets/fixtures/flat.toml");

    let status = Command::new(binary)
        .args([
            "bake",
            "--config",
            config.to_str().unwrap(),
            "--out",
            out_dir.path().to_str().unwrap(),
        ])
        .status()
        .unwrap();

    assert!(status.success());
    let exr_path = out_dir.path().join("xbrdf_view.exr");
    assert!(exr_path.exists());
    assert!(out_dir.path().join("manifest.toml").exists());

    let image = exr::prelude::read_first_rgba_layer_from_file(
        &exr_path,
        |resolution, _channels| vec![vec![[0.0_f32; 4]; resolution.width()]; resolution.height()],
        |pixels, position, (r, g, b, a): (f32, f32, f32, f32)| {
            pixels[position.y()][position.x()] = [r, g, b, a];
        },
    )
    .unwrap();

    let expected = 1.0 / std::f32::consts::PI;
    for row in image.layer_data.channel_data.pixels {
        for pixel in row {
            assert!((pixel[0] - expected).abs() < 0.01, "{pixel:?}");
            assert!((pixel[1] - expected).abs() < 0.01, "{pixel:?}");
            assert!((pixel[2] - expected).abs() < 0.01, "{pixel:?}");
        }
    }
}
