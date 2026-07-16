use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use exr::prelude::{read_first_rgba_layer_from_file, write_rgb_file};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use xbrdf_core::{
    BakeConfigFile, BakeMode, BakeOverrides, Manifest, MaterialKind, Mesh, SamplerKind,
};

#[derive(Debug, Parser)]
#[command(name = "xbrdf-bake")]
#[command(about = "Bake explicit BRDF data from periodic microgeometry")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    bake_args: BakeArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    Bake(BakeArgs),
    Validate(BakeArgs),
    Compare(CompareArgs),
}

#[derive(Debug, Args, Default)]
struct BakeArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    obj: Option<PathBuf>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long, value_parser = parse_bake_mode)]
    mode: Option<BakeMode>,
    #[arg(long)]
    light_width: Option<u32>,
    #[arg(long)]
    light_height: Option<u32>,
    #[arg(long)]
    samples: Option<u32>,
    #[arg(long, value_parser = parse_vec3)]
    light: Option<[f32; 3]>,
    #[arg(long)]
    max_repeat_radius: Option<u32>,
    #[arg(long, value_parser = parse_sampler_kind)]
    sampler: Option<SamplerKind>,
    #[arg(long)]
    enable_shadows: Option<bool>,
    #[arg(long, value_parser = parse_material_kind)]
    material: Option<MaterialKind>,
    #[arg(long, value_parser = parse_vec3)]
    material_color: Option<[f32; 3]>,
    #[arg(long)]
    roughness: Option<f32>,
}

#[derive(Debug, Args)]
struct CompareArgs {
    first: PathBuf,
    second: PathBuf,
    #[arg(long)]
    tolerance: Option<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Bake(args)) => bake(args),
        Some(Command::Validate(args)) => validate(args),
        Some(Command::Compare(args)) => compare(args),
        None => bake(cli.bake_args),
    }
}

fn bake(args: BakeArgs) -> Result<()> {
    let total_start = Instant::now();
    let LoadedInputs {
        resolved,
        mesh,
        config_time,
        load_time,
    } = load_inputs(&args)?;

    let out = args
        .out
        .as_ref()
        .context("missing required output directory; pass --out <path>")?;

    std::fs::create_dir_all(out)
        .with_context(|| format!("failed to create output directory {}", out.display()))?;

    let gpu_result =
        pollster::block_on(xbrdf_gpu::bake_atlas(&resolved, &mesh)).context("GPU bake failed")?;

    let write_start = Instant::now();
    let image_path = out.join("xbrdf_view.exr");
    write_rgb_file(
        &image_path,
        resolved.atlas_width() as usize,
        resolved.atlas_height() as usize,
        |x, y| {
            let pixel = gpu_result.pixels[y * resolved.atlas_width() as usize + x];
            (pixel[0], pixel[1], pixel[2])
        },
    )
    .with_context(|| format!("failed to write EXR {}", image_path.display()))?;

    let manifest = Manifest::new(&resolved, &mesh).context("failed to hash input geometry")?;
    let manifest_path = out.join("manifest.toml");
    let manifest_text = toml::to_string_pretty(&manifest).context("failed to encode manifest")?;
    std::fs::write(&manifest_path, manifest_text)
        .with_context(|| format!("failed to write manifest {}", manifest_path.display()))?;
    let write_time = write_start.elapsed();

    println!("wrote {}", image_path.display());
    println!("wrote {}", manifest_path.display());
    print_stats(
        &resolved,
        &mesh,
        &gpu_result.stats,
        PhaseTimes {
            config: config_time,
            load: load_time,
            write: write_time,
            total: total_start.elapsed(),
        },
    );

    Ok(())
}

struct LoadedInputs {
    resolved: xbrdf_core::ResolvedBakeConfig,
    mesh: Mesh,
    config_time: Duration,
    load_time: Duration,
}

fn load_inputs(args: &BakeArgs) -> Result<LoadedInputs> {
    let config_start = Instant::now();
    let file_config = if let Some(config_path) = &args.config {
        BakeConfigFile::read(config_path)
            .with_context(|| format!("failed to load bake config {}", config_path.display()))?
    } else {
        BakeConfigFile::default()
    };
    let resolved = file_config
        .resolve(
            args.config.as_deref(),
            BakeOverrides {
                obj: args.obj.clone(),
                width: args.width,
                height: args.height,
                mode: args.mode,
                light_width: args.light_width,
                light_height: args.light_height,
                samples: args.samples,
                light: args.light,
                max_repeat_radius: args.max_repeat_radius,
                sampler: args.sampler,
                enable_shadows: args.enable_shadows,
                material_kind: args.material,
                material_color: args.material_color,
                material_roughness: args.roughness,
            },
        )
        .context("failed to resolve bake settings")?;
    let config_time = config_start.elapsed();
    let load_start = Instant::now();
    let mesh = Mesh::load(&resolved.obj)
        .with_context(|| format!("failed to load geometry {}", resolved.obj.display()))?;
    Ok(LoadedInputs {
        resolved,
        mesh,
        config_time,
        load_time: load_start.elapsed(),
    })
}

fn validate(args: BakeArgs) -> Result<()> {
    let loaded = load_inputs(&args)?;
    let config = loaded.resolved;
    println!("configuration valid");
    println!("  geometry: {} triangles", loaded.mesh.triangles.len());
    println!(
        "  output: {}x{}; tile {}x{}; {} light directions",
        config.atlas_width(),
        config.atlas_height(),
        config.camera_tile_width(),
        config.camera_tile_height(),
        config.light_count()
    );
    println!(
        "  camera rays: {}",
        config.atlas_width() as u64 * config.atlas_height() as u64 * config.samples as u64
    );
    Ok(())
}

fn compare(args: CompareArgs) -> Result<()> {
    let (first_width, first_height, first) = read_exr(&args.first)?;
    let (second_width, second_height, second) = read_exr(&args.second)?;
    if (first_width, first_height) != (second_width, second_height) {
        anyhow::bail!(
            "image dimensions differ: {first_width}x{first_height} versus {second_width}x{second_height}"
        );
    }

    let mut absolute_sum = 0.0f64;
    let mut squared_sum = 0.0f64;
    let mut maximum = 0.0f32;
    for (a, b) in first.iter().zip(&second) {
        for channel in 0..3 {
            let difference = (a[channel] - b[channel]).abs();
            absolute_sum += difference as f64;
            squared_sum += (difference * difference) as f64;
            maximum = maximum.max(difference);
        }
    }
    let value_count = (first.len() * 3) as f64;
    println!("pixels: {}", first.len());
    println!("mean absolute error: {:.8}", absolute_sum / value_count);
    println!(
        "root mean square error: {:.8}",
        (squared_sum / value_count).sqrt()
    );
    println!("maximum absolute error: {maximum:.8}");
    if args.tolerance.is_some_and(|tolerance| maximum > tolerance) {
        anyhow::bail!("maximum error {maximum} exceeds tolerance");
    }
    Ok(())
}

fn read_exr(path: &std::path::Path) -> Result<(u32, u32, Vec<[f32; 3]>)> {
    let image = read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            (
                resolution.width() as u32,
                resolution.height() as u32,
                vec![[0.0f32; 3]; resolution.area()],
            )
        },
        |pixels, position, (r, g, b, _a): (f32, f32, f32, f32)| {
            pixels.2[position.y() * pixels.0 as usize + position.x()] = [r, g, b];
        },
    )
    .with_context(|| format!("failed to read EXR {}", path.display()))?;
    Ok(image.layer_data.channel_data.pixels)
}

struct PhaseTimes {
    config: Duration,
    load: Duration,
    write: Duration,
    total: Duration,
}

fn print_stats(
    resolved: &xbrdf_core::ResolvedBakeConfig,
    mesh: &Mesh,
    stats: &xbrdf_gpu::GpuBakeStats,
    times: PhaseTimes,
) {
    let pixel_count = stats.width as u64 * stats.height as u64;
    let throughput_time = stats.gpu_timestamp_time.unwrap_or(stats.gpu_dispatch_time);
    let rays_per_second = if throughput_time.as_secs_f64() > 0.0 {
        stats.camera_ray_count as f64 / throughput_time.as_secs_f64()
    } else {
        0.0
    };

    println!();
    println!("Bake stats");
    println!(
        "  image: {}x{} ({} pixels)",
        stats.width, stats.height, pixel_count
    );
    println!(
        "  mode: {}, camera tile {}x{}, light grid {}x{} ({} light directions)",
        resolved.mode,
        resolved.camera_tile_width(),
        resolved.camera_tile_height(),
        resolved.effective_light_width(),
        resolved.effective_light_height(),
        resolved.light_count()
    );
    println!(
        "  geometry: {} triangles, {} BVH nodes",
        stats.triangle_count, stats.bvh_node_count
    );
    println!("  colors: {}", mesh.color_source.as_str());
    println!(
        "  tile: min=({:.6}, {:.6}) size=({:.6}, {:.6}) y=[{:.6}, {:.6}] offset={:.6}",
        mesh.tile_min_x,
        mesh.tile_min_z,
        mesh.tile_width,
        mesh.tile_depth,
        mesh.bounds.min.y,
        mesh.bounds.max.y,
        mesh.y_offset_to_zero
    );
    println!(
        "  sampling: {} samples/pixel, {}, {} camera rays, max repeat radius {}, max {} periodic copies/ray",
        stats.samples,
        resolved.sampler,
        stats.camera_ray_count,
        stats.max_repeat_radius,
        stats.max_periodic_copies_per_ray
    );
    println!(
        "  shadows: {}",
        if resolved.enable_shadows {
            "enabled"
        } else {
            "disabled"
        }
    );
    let trace_label = if resolved.enable_shadows {
        "including shadows"
    } else {
        "camera visibility only"
    };
    if stats.sample_lanes > 1 {
        println!(
            "  dispatch: {} sample waves, {} rows/wave, up to {} sample lanes x {} samples/lane, max {} BVH traces ({})",
            stats.dispatch_count,
            stats.rows_per_dispatch,
            stats.sample_lanes,
            stats.samples_per_lane,
            stats.max_bvh_traces,
            trace_label
        );
    } else {
        println!(
            "  dispatch: {} row chunks, {} rows/chunk, max {} BVH traces ({})",
            stats.dispatch_count, stats.rows_per_dispatch, stats.max_bvh_traces, trace_label
        );
    }
    println!(
        "  material: {} color=({:.3}, {:.3}, {:.3}) roughness={}",
        resolved.material.kind,
        resolved.material.color[0],
        resolved.material.color[1],
        resolved.material.color[2],
        resolved
            .material
            .roughness
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  timing: config {}, load {}, BVH {}, GPU setup {}, GPU dispatch {}, readback {}, write {}, total {}",
        format_duration(times.config),
        format_duration(times.load),
        format_duration(stats.bvh_build_time),
        format_duration(stats.gpu_setup_time),
        format_duration(stats.gpu_dispatch_time),
        format_duration(stats.readback_time),
        format_duration(times.write),
        format_duration(times.total)
    );
    if let Some(gpu_time) = stats.gpu_timestamp_time {
        println!(
            "  GPU timestamp: {} compute time",
            format_duration(gpu_time)
        );
    } else {
        println!("  GPU timestamp: unsupported by the selected adapter");
    }
    println!("  throughput: {rays_per_second:.0} camera rays/s");
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.3}s")
    } else {
        format!("{:.2}ms", seconds * 1000.0)
    }
}

fn parse_vec3(value: &str) -> Result<[f32; 3], String> {
    let parts: Vec<_> = value.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err("expected x,y,z".to_string());
    }

    let mut parsed = [0.0; 3];
    for (index, part) in parts.iter().enumerate() {
        parsed[index] = part
            .parse::<f32>()
            .map_err(|_| format!("invalid float component `{part}`"))?;
    }

    Ok(parsed)
}

fn parse_material_kind(value: &str) -> Result<MaterialKind, String> {
    value.parse()
}

fn parse_bake_mode(value: &str) -> Result<BakeMode, String> {
    value.parse()
}

fn parse_sampler_kind(value: &str) -> Result<SamplerKind, String> {
    value.parse()
}
