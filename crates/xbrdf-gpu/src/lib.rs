use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use xbrdf_core::{MaterialKind, Mesh, ResolvedBakeConfig, Triangle, Vec3};

const SHADER: &str = include_str!("bake.wgsl");
const BVH_LEAF_SIZE: usize = 4;
const TARGET_RAY_TRACES_PER_DISPATCH: u64 = 250_000;

#[derive(Debug, thiserror::Error)]
pub enum GpuBakeError {
    #[error("no compatible GPU adapter was found")]
    NoAdapter,
    #[error("failed to request GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("GPU buffer map failed")]
    BufferMapFailed,
    #[error("GPU output channel closed before mapping completed")]
    MapChannelClosed,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriangle {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    normal: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBvhNode {
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    child_or_first: u32,
    child_b: u32,
    triangle_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    samples: u32,
    triangle_count: u32,
    node_count: u32,
    max_repeat_radius: u32,
    y_offset: u32,
    active_height: u32,
    sample_offset: u32,
    target_samples: u32,
    tile_min: [f32; 2],
    tile_size: [f32; 2],
    _pad0: [u32; 2],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    light_dir: [f32; 4],
    material_color: [f32; 4],
    material_kind: [u32; 4],
    material_params: [f32; 4],
}

#[derive(Clone, Copy)]
struct BuildTriangle {
    triangle: Triangle,
    bounds_min: Vec3,
    bounds_max: Vec3,
    centroid: Vec3,
}

#[derive(Debug, Clone)]
pub struct GpuBakeResult {
    pub pixels: Vec<[f32; 3]>,
    pub stats: GpuBakeStats,
}

#[derive(Debug, Clone)]
pub struct ProgressChunk {
    pub y_offset: u32,
    pub height: u32,
    pub width: u32,
    pub completed_rows: u32,
    pub total_rows: u32,
    pub pixels: Vec<[f32; 3]>,
}

#[derive(Debug, Clone)]
pub struct ProgressiveFrame {
    pub completed_samples: u32,
    pub total_samples: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

#[derive(Debug, Clone)]
pub struct GpuBakeStats {
    pub triangle_count: usize,
    pub bvh_node_count: usize,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub max_repeat_radius: u32,
    pub rows_per_dispatch: u32,
    pub dispatch_count: u32,
    pub camera_ray_count: u64,
    pub max_periodic_copies_per_ray: u64,
    pub max_bvh_traces: u64,
    pub bvh_build_time: Duration,
    pub gpu_setup_time: Duration,
    pub gpu_dispatch_time: Duration,
    pub readback_time: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct ProgressiveBakeOptions {
    pub update_interval: Duration,
}

impl Default for ProgressiveBakeOptions {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(500),
        }
    }
}

pub async fn bake(config: &ResolvedBakeConfig, mesh: &Mesh) -> Result<GpuBakeResult, GpuBakeError> {
    bake_inner(config, mesh, None).await
}

pub async fn bake_progressive<F>(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    options: ProgressiveBakeOptions,
    mut progress: F,
) -> Result<GpuBakeResult, GpuBakeError>
where
    F: FnMut(ProgressiveFrame),
{
    bake_progressive_inner(config, mesh, options, &mut progress).await
}

pub async fn bake_with_progress<F>(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    mut progress: F,
) -> Result<GpuBakeResult, GpuBakeError>
where
    F: FnMut(ProgressChunk),
{
    bake_inner(config, mesh, Some(&mut progress)).await
}

async fn bake_progressive_inner(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    options: ProgressiveBakeOptions,
    progress: &mut dyn FnMut(ProgressiveFrame),
) -> Result<GpuBakeResult, GpuBakeError> {
    let total_start = Instant::now();
    let (triangles, bvh_nodes) = build_bvh(&mesh.triangles);
    let bvh_build_time = total_start.elapsed();

    let gpu_setup_start = Instant::now();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or(GpuBakeError::NoAdapter)?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xbrdf progressive device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    let light = Vec3::from_array(config.light);
    let base_params = GpuParams {
        width: config.width,
        height: config.height,
        samples: config.samples,
        triangle_count: triangles.len() as u32,
        node_count: bvh_nodes.len() as u32,
        max_repeat_radius: config.max_repeat_radius,
        y_offset: 0,
        active_height: config.height,
        sample_offset: 0,
        target_samples: config.samples,
        tile_min: [mesh.tile_min_x, mesh.tile_min_z],
        tile_size: [mesh.tile_width, mesh.tile_depth],
        _pad0: [0; 2],
        bounds_min: vec4(mesh.bounds.min),
        bounds_max: vec4(mesh.bounds.max),
        light_dir: vec4(light),
        material_color: [
            config.material.color[0],
            config.material.color[1],
            config.material.color[2],
            0.0,
        ],
        material_kind: [
            match config.material.kind {
                MaterialKind::Lambertian => 0,
                MaterialKind::SpecularPhong => 1,
            },
            0,
            0,
            0,
        ],
        material_params: [
            1.0 / light.y.max(1.0e-4),
            ((mesh.bounds.max.y - mesh.bounds.min.y).abs() * 0.01).max(1.0e-3),
            config.material.roughness.unwrap_or(0.0),
            config.material.phong_exponent().unwrap_or(1.0),
        ],
    };

    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf progressive triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf progressive bvh"),
        contents: bytemuck::cast_slice(&bvh_nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf progressive params"),
        contents: bytemuck::bytes_of(&base_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let pixel_count = config.width as u64 * config.height as u64;
    let output_size = pixel_count * std::mem::size_of::<[f32; 4]>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xbrdf progressive output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xbrdf progressive bake shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xbrdf progressive bind group layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(3, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xbrdf progressive pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xbrdf progressive bake pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xbrdf progressive bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: triangle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bvh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let gpu_setup_time = gpu_setup_start.elapsed();

    let rows_per_dispatch = rows_per_dispatch(config);
    let mut completed_samples = 0u32;
    let mut batch_samples = 1u32;
    let mut dispatch_count = 0u32;
    let mut readback_time = Duration::ZERO;
    let mut accumulated = vec![[0.0; 3]; pixel_count as usize];
    let mut averaged = vec![[0.0; 3]; pixel_count as usize];
    let dispatch_start = Instant::now();
    let target_interval = options.update_interval.max(Duration::from_millis(1));

    while completed_samples < config.samples {
        let remaining = config.samples - completed_samples;
        let active_batch = batch_samples.min(remaining).max(1);
        let batch_start = Instant::now();
        let mut y_offset = 0;
        while y_offset < config.height {
            let active_height = rows_per_dispatch.min(config.height - y_offset);
            let params = GpuParams {
                samples: active_batch,
                sample_offset: completed_samples,
                target_samples: config.samples,
                y_offset,
                active_height,
                ..base_params
            };
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xbrdf progressive bake encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("xbrdf progressive bake pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(config.width.div_ceil(8), active_height.div_ceil(8), 1);
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            y_offset += active_height;
            dispatch_count += 1;
        }

        let readback_start = Instant::now();
        let batch_pixels = read_output_rows(
            &device,
            &queue,
            &output_buffer,
            config.width,
            0,
            config.height,
        )?;
        readback_time += readback_start.elapsed();
        completed_samples += active_batch;

        for ((sum, avg), batch) in accumulated
            .iter_mut()
            .zip(averaged.iter_mut())
            .zip(batch_pixels.iter())
        {
            sum[0] += batch[0] * active_batch as f32;
            sum[1] += batch[1] * active_batch as f32;
            sum[2] += batch[2] * active_batch as f32;
            avg[0] = sum[0] / completed_samples as f32;
            avg[1] = sum[1] / completed_samples as f32;
            avg[2] = sum[2] / completed_samples as f32;
        }

        progress(ProgressiveFrame {
            completed_samples,
            total_samples: config.samples,
            width: config.width,
            height: config.height,
            pixels: averaged.clone(),
        });

        let elapsed = batch_start.elapsed();
        if elapsed < target_interval / 2 && active_batch < remaining {
            batch_samples = (batch_samples.saturating_mul(2)).max(1);
        } else if elapsed > target_interval.saturating_mul(2) && batch_samples > 1 {
            batch_samples = (batch_samples / 2).max(1);
        }
    }
    let gpu_dispatch_time = dispatch_start.elapsed().saturating_sub(readback_time);

    let max_periodic_copies_per_axis = config.max_repeat_radius as u64 * 2 + 1;
    let max_periodic_copies_per_ray = max_periodic_copies_per_axis * max_periodic_copies_per_axis;
    let camera_ray_count = config.width as u64 * config.height as u64 * config.samples as u64;

    Ok(GpuBakeResult {
        pixels: averaged,
        stats: GpuBakeStats {
            triangle_count: triangles.len(),
            bvh_node_count: bvh_nodes.len(),
            width: config.width,
            height: config.height,
            samples: config.samples,
            max_repeat_radius: config.max_repeat_radius,
            rows_per_dispatch,
            dispatch_count,
            camera_ray_count,
            max_periodic_copies_per_ray,
            max_bvh_traces: camera_ray_count * max_periodic_copies_per_ray * 2,
            bvh_build_time,
            gpu_setup_time,
            gpu_dispatch_time,
            readback_time,
        },
    })
}

async fn bake_inner(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    mut progress: Option<&mut dyn FnMut(ProgressChunk)>,
) -> Result<GpuBakeResult, GpuBakeError> {
    let total_start = Instant::now();
    let (triangles, bvh_nodes) = build_bvh(&mesh.triangles);
    let bvh_build_time = total_start.elapsed();

    let gpu_setup_start = Instant::now();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or(GpuBakeError::NoAdapter)?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xbrdf device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    let light = Vec3::from_array(config.light);
    let base_params = GpuParams {
        width: config.width,
        height: config.height,
        samples: config.samples,
        triangle_count: triangles.len() as u32,
        node_count: bvh_nodes.len() as u32,
        max_repeat_radius: config.max_repeat_radius,
        y_offset: 0,
        active_height: config.height,
        sample_offset: 0,
        target_samples: config.samples,
        tile_min: [mesh.tile_min_x, mesh.tile_min_z],
        tile_size: [mesh.tile_width, mesh.tile_depth],
        _pad0: [0; 2],
        bounds_min: vec4(mesh.bounds.min),
        bounds_max: vec4(mesh.bounds.max),
        light_dir: vec4(light),
        material_color: [
            config.material.color[0],
            config.material.color[1],
            config.material.color[2],
            0.0,
        ],
        material_kind: [
            match config.material.kind {
                MaterialKind::Lambertian => 0,
                MaterialKind::SpecularPhong => 1,
            },
            0,
            0,
            0,
        ],
        material_params: [
            1.0 / light.y.max(1.0e-4),
            ((mesh.bounds.max.y - mesh.bounds.min.y).abs() * 0.01).max(1.0e-3),
            config.material.roughness.unwrap_or(0.0),
            config.material.phong_exponent().unwrap_or(1.0),
        ],
    };

    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf triangles"),
        contents: bytemuck::cast_slice(&triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf bvh"),
        contents: bytemuck::cast_slice(&bvh_nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf params"),
        contents: bytemuck::bytes_of(&base_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let pixel_count = config.width as u64 * config.height as u64;
    let output_size = pixel_count * std::mem::size_of::<[f32; 4]>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xbrdf output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xbrdf readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xbrdf bake shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xbrdf bind group layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(3, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xbrdf pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xbrdf bake pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xbrdf bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: triangle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bvh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let gpu_setup_time = gpu_setup_start.elapsed();

    let rows_per_dispatch = rows_per_dispatch(config);
    let mut y_offset = 0;
    let mut dispatch_count = 0;
    let dispatch_start = Instant::now();
    while y_offset < config.height {
        let active_height = rows_per_dispatch.min(config.height - y_offset);
        let params = GpuParams {
            y_offset,
            active_height,
            ..base_params
        };
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("xbrdf bake encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xbrdf bake pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(config.width.div_ceil(8), active_height.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        if let Some(progress) = progress.as_deref_mut() {
            let chunk_pixels = read_output_rows(
                &device,
                &queue,
                &output_buffer,
                config.width,
                y_offset,
                active_height,
            )?;
            progress(ProgressChunk {
                y_offset,
                height: active_height,
                width: config.width,
                completed_rows: y_offset + active_height,
                total_rows: config.height,
                pixels: chunk_pixels,
            });
        }

        y_offset += active_height;
        dispatch_count += 1;
    }
    let gpu_dispatch_time = dispatch_start.elapsed();

    let readback_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xbrdf readback encoder"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    receiver
        .recv()
        .map_err(|_| GpuBakeError::MapChannelClosed)?
        .map_err(|_| GpuBakeError::BufferMapFailed)?;

    let mapped = slice.get_mapped_range();
    let rgba: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    let rgb = rgba
        .iter()
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    drop(mapped);
    readback_buffer.unmap();
    let readback_time = readback_start.elapsed();

    let max_periodic_copies_per_axis = config.max_repeat_radius as u64 * 2 + 1;
    let max_periodic_copies_per_ray = max_periodic_copies_per_axis * max_periodic_copies_per_axis;
    let camera_ray_count = config.width as u64 * config.height as u64 * config.samples as u64;

    Ok(GpuBakeResult {
        pixels: rgb,
        stats: GpuBakeStats {
            triangle_count: triangles.len(),
            bvh_node_count: bvh_nodes.len(),
            width: config.width,
            height: config.height,
            samples: config.samples,
            max_repeat_radius: config.max_repeat_radius,
            rows_per_dispatch,
            dispatch_count,
            camera_ray_count,
            max_periodic_copies_per_ray,
            max_bvh_traces: camera_ray_count * max_periodic_copies_per_ray * 2,
            bvh_build_time,
            gpu_setup_time,
            gpu_dispatch_time,
            readback_time,
        },
    })
}

fn read_output_rows(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output_buffer: &wgpu::Buffer,
    width: u32,
    y_offset: u32,
    height: u32,
) -> Result<Vec<[f32; 3]>, GpuBakeError> {
    let pixel_size = std::mem::size_of::<[f32; 4]>() as u64;
    let row_offset = y_offset as u64 * width as u64 * pixel_size;
    let chunk_size = height as u64 * width as u64 * pixel_size;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xbrdf progress readback"),
        size: chunk_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xbrdf progress readback encoder"),
    });
    encoder.copy_buffer_to_buffer(output_buffer, row_offset, &readback_buffer, 0, chunk_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    receiver
        .recv()
        .map_err(|_| GpuBakeError::MapChannelClosed)?
        .map_err(|_| GpuBakeError::BufferMapFailed)?;

    let mapped = slice.get_mapped_range();
    let rgba: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    let rgb = rgba
        .iter()
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    drop(mapped);
    readback_buffer.unmap();

    Ok(rgb)
}

fn build_bvh(source: &[Triangle]) -> (Vec<GpuTriangle>, Vec<GpuBvhNode>) {
    let mut build_triangles: Vec<_> = source
        .iter()
        .copied()
        .map(|triangle| {
            let bounds_min = triangle.v0.min(triangle.v1).min(triangle.v2);
            let bounds_max = triangle.v0.max(triangle.v1).max(triangle.v2);
            let centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0;
            BuildTriangle {
                triangle,
                bounds_min,
                bounds_max,
                centroid,
            }
        })
        .collect();

    let mut ordered_triangles = Vec::with_capacity(source.len());
    let mut nodes = Vec::with_capacity(source.len().saturating_mul(2));
    build_bvh_node(&mut build_triangles, &mut ordered_triangles, &mut nodes);
    (ordered_triangles, nodes)
}

fn build_bvh_node(
    triangles: &mut [BuildTriangle],
    ordered_triangles: &mut Vec<GpuTriangle>,
    nodes: &mut Vec<GpuBvhNode>,
) -> u32 {
    let node_index = nodes.len() as u32;
    nodes.push(GpuBvhNode::zeroed());

    let (bounds_min, bounds_max) = triangle_bounds(triangles);
    if triangles.len() <= BVH_LEAF_SIZE {
        let first = ordered_triangles.len() as u32;
        for item in triangles {
            ordered_triangles.push(gpu_triangle(item.triangle));
        }
        nodes[node_index as usize] = GpuBvhNode {
            bounds_min: vec4(bounds_min),
            bounds_max: vec4(bounds_max),
            child_or_first: first,
            child_b: 0,
            triangle_count: ordered_triangles.len() as u32 - first,
            _pad: 0,
        };
        return node_index;
    }

    let (centroid_min, centroid_max) = centroid_bounds(triangles);
    let axis = longest_axis(centroid_max - centroid_min);
    triangles.sort_by(|a, b| compare_axis(a.centroid, b.centroid, axis));
    let mid = triangles.len() / 2;
    let (left_items, right_items) = triangles.split_at_mut(mid);
    let left = build_bvh_node(left_items, ordered_triangles, nodes);
    let right = build_bvh_node(right_items, ordered_triangles, nodes);

    nodes[node_index as usize] = GpuBvhNode {
        bounds_min: vec4(bounds_min),
        bounds_max: vec4(bounds_max),
        child_or_first: left,
        child_b: right,
        triangle_count: 0,
        _pad: 0,
    };
    node_index
}

fn triangle_bounds(triangles: &[BuildTriangle]) -> (Vec3, Vec3) {
    let mut bounds_min = triangles[0].bounds_min;
    let mut bounds_max = triangles[0].bounds_max;
    for triangle in &triangles[1..] {
        bounds_min = bounds_min.min(triangle.bounds_min);
        bounds_max = bounds_max.max(triangle.bounds_max);
    }
    (bounds_min, bounds_max)
}

fn centroid_bounds(triangles: &[BuildTriangle]) -> (Vec3, Vec3) {
    let mut bounds_min = triangles[0].centroid;
    let mut bounds_max = triangles[0].centroid;
    for triangle in &triangles[1..] {
        bounds_min = bounds_min.min(triangle.centroid);
        bounds_max = bounds_max.max(triangle.centroid);
    }
    (bounds_min, bounds_max)
}

fn longest_axis(value: Vec3) -> usize {
    if value.x >= value.y && value.x >= value.z {
        0
    } else if value.y >= value.z {
        1
    } else {
        2
    }
}

fn compare_axis(a: Vec3, b: Vec3, axis: usize) -> Ordering {
    let a = match axis {
        0 => a.x,
        1 => a.y,
        _ => a.z,
    };
    let b = match axis {
        0 => b.x,
        1 => b.y,
        _ => b.z,
    };
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn gpu_triangle(triangle: Triangle) -> GpuTriangle {
    GpuTriangle {
        v0: vec4(triangle.v0),
        v1: vec4(triangle.v1),
        v2: vec4(triangle.v2),
        normal: vec4(triangle.normal),
        color: vec4(triangle.color),
    }
}

fn rows_per_dispatch(config: &ResolvedBakeConfig) -> u32 {
    let repeat_diameter = config.max_repeat_radius as u64 * 2 + 1;
    let traces_per_row =
        config.width as u64 * config.samples as u64 * repeat_diameter * repeat_diameter * 2;
    let rows = (TARGET_RAY_TRACES_PER_DISPATCH / traces_per_row.max(1)).max(1);
    rows.min(config.height as u64) as u32
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn vec4(value: Vec3) -> [f32; 4] {
    [value.x, value.y, value.z, 0.0]
}

use wgpu::util::DeviceExt;
