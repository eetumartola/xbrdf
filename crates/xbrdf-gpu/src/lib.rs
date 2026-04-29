use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use xbrdf_core::{MaterialKind, Mesh, ResolvedBakeConfig, SamplerKind, Triangle, Vec3};

const SHADER: &str = include_str!("bake.wgsl");
const BVH_LEAF_SIZE: usize = 4;
const TARGET_RAY_TRACES_PER_DISPATCH: u64 = 1_024_000_000;
const WORKGROUP_WIDTH: u32 = 8;
const WORKGROUP_HEIGHT: u32 = 8;
const SAMPLE_PARALLEL_THRESHOLD: u32 = 2048;
const SAMPLE_PARALLEL_LANES: u32 = 128;
const SAMPLE_PARALLEL_SAMPLES_PER_LANE: u32 = 64;
const HIGH_SAMPLE_PARALLEL_SAMPLES_PER_LANE: u32 = 512;
const HIGH_SAMPLE_THRESHOLD: u32 = 65_536;
const MAX_SAMPLE_PARALLEL_BUFFER_BYTES: u64 = 512 * 1024 * 1024;
const SAH_BINS: usize = 12;

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
    sample_limit: u32,
    sample_lanes: u32,
    tile_min: [f32; 2],
    tile_size: [f32; 2],
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

#[derive(Clone, Copy)]
struct SahBin {
    count: usize,
    bounds_min: Vec3,
    bounds_max: Vec3,
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
pub struct AtlasProgressFrame {
    pub completed_tiles: u32,
    pub total_tiles: u32,
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
    pub sample_lanes: u32,
    pub samples_per_lane: u32,
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
    if config.samples >= SAMPLE_PARALLEL_THRESHOLD {
        return bake_sample_parallel(config, mesh).await;
    }

    bake_inner(config, mesh, None).await
}

pub async fn bake_atlas(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
) -> Result<GpuBakeResult, GpuBakeError> {
    bake_atlas_with_progress(config, mesh, |_| {}).await
}

pub async fn bake_atlas_with_progress<F>(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    mut progress: F,
) -> Result<GpuBakeResult, GpuBakeError>
where
    F: FnMut(AtlasProgressFrame),
{
    if config.light_count() == 1
        && config.camera_tile_width() == config.width
        && config.camera_tile_height() == config.height
    {
        return bake(config, mesh).await;
    }

    let atlas_width = config.atlas_width();
    let atlas_height = config.atlas_height();
    let tile_width = config.camera_tile_width();
    let tile_height = config.camera_tile_height();
    let light_width = config.effective_light_width();
    let light_height = config.effective_light_height();
    let mut atlas_pixels = vec![[0.0; 3]; atlas_width as usize * atlas_height as usize];
    let mut combined_stats = None;
    let mut completed_tiles = 0u32;

    let mut sample_context = SampleParallelContext::new(config, mesh).await?;

    for light_y in 0..light_height {
        for light_x in 0..light_width {
            let tile_config = config.config_for_tile(light_x, light_y);
            let tile = sample_context.bake(&tile_config)?;

            for y in 0..tile_height {
                let atlas_y = light_y * tile_height + y;
                let tile_row = y as usize * tile_width as usize;
                let atlas_row = atlas_y as usize * atlas_width as usize
                    + light_x as usize * tile_width as usize;
                let src = &tile.pixels[tile_row..tile_row + tile_width as usize];
                let dst = &mut atlas_pixels[atlas_row..atlas_row + tile_width as usize];
                dst.copy_from_slice(src);
            }

            if let Some(stats) = &mut combined_stats {
                merge_stats(stats, &tile.stats, atlas_width, atlas_height);
            } else {
                let mut stats = tile.stats;
                stats.width = atlas_width;
                stats.height = atlas_height;
                combined_stats = Some(stats);
            }

            completed_tiles += 1;
            progress(AtlasProgressFrame {
                completed_tiles,
                total_tiles: config.light_count(),
                width: atlas_width,
                height: atlas_height,
                pixels: atlas_pixels.clone(),
            });
        }
    }

    Ok(GpuBakeResult {
        pixels: atlas_pixels,
        stats: combined_stats.expect("atlas bake should produce at least one tile"),
    })
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

fn merge_stats(total: &mut GpuBakeStats, tile: &GpuBakeStats, atlas_width: u32, atlas_height: u32) {
    total.width = atlas_width;
    total.height = atlas_height;
    total.dispatch_count += tile.dispatch_count;
    total.camera_ray_count += tile.camera_ray_count;
    total.max_bvh_traces += tile.max_bvh_traces;
    total.bvh_build_time += tile.bvh_build_time;
    total.gpu_setup_time += tile.gpu_setup_time;
    total.gpu_dispatch_time += tile.gpu_dispatch_time;
    total.readback_time += tile.readback_time;
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

fn sample_parallel_params(
    config: &ResolvedBakeConfig,
    triangle_count: usize,
    node_count: usize,
    samples_per_lane: u32,
    sample_lanes: u32,
    tile_min: [f32; 2],
    tile_size: [f32; 2],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    ray_epsilon: f32,
) -> GpuParams {
    let light = Vec3::from_array(config.light);
    GpuParams {
        width: config.width,
        height: config.height,
        samples: samples_per_lane,
        triangle_count: triangle_count as u32,
        node_count: node_count as u32,
        max_repeat_radius: config.max_repeat_radius,
        y_offset: 0,
        active_height: config.height,
        sample_offset: 0,
        target_samples: config.samples,
        sample_limit: config.samples,
        sample_lanes,
        tile_min,
        tile_size,
        bounds_min,
        bounds_max,
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
            sampler_code(config.sampler),
            u32::from(config.enable_shadows),
            0,
        ],
        material_params: [
            1.0 / light.y.max(1.0e-4),
            ray_epsilon,
            config.material.roughness.unwrap_or(0.0),
            config.material.phong_exponent().unwrap_or(1.0),
        ],
    }
}

struct SampleParallelContext {
    triangles_len: usize,
    bvh_nodes_len: usize,
    bvh_build_time: Duration,
    gpu_setup_time: Duration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    sample_lanes: u32,
    samples_per_lane: u32,
    tile_min: [f32; 2],
    tile_size: [f32; 2],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    ray_epsilon: f32,
    setup_reported: bool,
}

impl SampleParallelContext {
    async fn new(config: &ResolvedBakeConfig, mesh: &Mesh) -> Result<Self, GpuBakeError> {
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
                    label: Some("xbrdf sample-parallel device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let pixel_count = config.width as u64 * config.height as u64;
        let sample_lanes = sample_parallel_lanes_for_pixel_count(pixel_count);
        let samples_per_lane = sample_parallel_samples_per_lane(config.samples);
        let tile_min = [mesh.tile_min_x, mesh.tile_min_z];
        let tile_size = [mesh.tile_width, mesh.tile_depth];
        let bounds_min = vec4(mesh.bounds.min);
        let bounds_max = vec4(mesh.bounds.max);
        let ray_epsilon = ((mesh.bounds.max.y - mesh.bounds.min.y).abs() * 0.01).max(1.0e-3);
        let base_params = sample_parallel_params(
            config,
            triangles.len(),
            bvh_nodes.len(),
            samples_per_lane,
            sample_lanes,
            tile_min,
            tile_size,
            bounds_min,
            bounds_max,
            ray_epsilon,
        );

        let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xbrdf sample-parallel triangles"),
            contents: bytemuck::cast_slice(&triangles),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xbrdf sample-parallel bvh"),
            contents: bytemuck::cast_slice(&bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xbrdf sample-parallel params"),
            contents: bytemuck::bytes_of(&base_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let partial_count = pixel_count * sample_lanes as u64;
        let output_size = partial_count * std::mem::size_of::<[f32; 4]>() as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xbrdf sample-parallel partials"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xbrdf sample-parallel bake shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xbrdf sample-parallel bind group layout"),
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
            label: Some("xbrdf sample-parallel pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("xbrdf sample-parallel bake pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xbrdf sample-parallel bind group"),
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

        Ok(Self {
            triangles_len: triangles.len(),
            bvh_nodes_len: bvh_nodes.len(),
            bvh_build_time,
            gpu_setup_time,
            device,
            queue,
            params_buffer,
            output_buffer,
            pipeline,
            bind_group,
            sample_lanes,
            samples_per_lane,
            tile_min,
            tile_size,
            bounds_min,
            bounds_max,
            ray_epsilon,
            setup_reported: false,
        })
    }

    fn bake(&mut self, config: &ResolvedBakeConfig) -> Result<GpuBakeResult, GpuBakeError> {
        let pixel_count = config.width as u64 * config.height as u64;
        let mut sums = vec![[0.0f32; 3]; pixel_count as usize];
        let mut completed_samples = 0u32;
        let mut dispatch_count = 0u32;
        let mut readback_time = Duration::ZERO;
        let dispatch_start = Instant::now();
        let samples_per_wave = self.sample_lanes * self.samples_per_lane;

        while completed_samples < config.samples {
            let remaining = config.samples - completed_samples;
            let active_lanes = remaining
                .div_ceil(self.samples_per_lane)
                .min(self.sample_lanes)
                .max(1);
            let params = GpuParams {
                sample_offset: completed_samples,
                sample_limit: (completed_samples + samples_per_wave).min(config.samples),
                sample_lanes: active_lanes,
                ..sample_parallel_params(
                    config,
                    self.triangles_len,
                    self.bvh_nodes_len,
                    self.samples_per_lane,
                    self.sample_lanes,
                    self.tile_min,
                    self.tile_size,
                    self.bounds_min,
                    self.bounds_max,
                    self.ray_epsilon,
                )
            };
            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("xbrdf sample-parallel bake encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("xbrdf sample-parallel bake pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(
                    config.width.div_ceil(WORKGROUP_WIDTH),
                    config.height.div_ceil(WORKGROUP_HEIGHT),
                    active_lanes,
                );
            }
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            dispatch_count += 1;

            let readback_start = Instant::now();
            let partials = read_output_rgba(
                &self.device,
                &self.queue,
                &self.output_buffer,
                active_lanes as u64 * pixel_count,
            )?;
            readback_time += readback_start.elapsed();

            for lane in 0..active_lanes as usize {
                let lane_start = lane * pixel_count as usize;
                for (pixel_index, sum) in sums.iter_mut().enumerate() {
                    let partial = partials[lane_start + pixel_index];
                    sum[0] += partial[0];
                    sum[1] += partial[1];
                    sum[2] += partial[2];
                }
            }

            completed_samples = completed_samples
                .saturating_add(samples_per_wave)
                .min(config.samples);
        }

        let gpu_dispatch_time = dispatch_start.elapsed().saturating_sub(readback_time);
        let inv_samples = 1.0 / config.samples as f32;
        let pixels = sums
            .into_iter()
            .map(|sum| {
                [
                    sum[0] * inv_samples,
                    sum[1] * inv_samples,
                    sum[2] * inv_samples,
                ]
            })
            .collect();

        let max_periodic_copies_per_axis = config.max_repeat_radius as u64 * 2 + 1;
        let max_periodic_copies_per_ray =
            max_periodic_copies_per_axis * max_periodic_copies_per_axis;
        let camera_ray_count = config.width as u64 * config.height as u64 * config.samples as u64;
        let trace_multiplier = if config.enable_shadows { 2 } else { 1 };
        let bvh_build_time = if self.setup_reported {
            Duration::ZERO
        } else {
            self.bvh_build_time
        };
        let gpu_setup_time = if self.setup_reported {
            Duration::ZERO
        } else {
            self.gpu_setup_time
        };
        self.setup_reported = true;

        Ok(GpuBakeResult {
            pixels,
            stats: GpuBakeStats {
                triangle_count: self.triangles_len,
                bvh_node_count: self.bvh_nodes_len,
                width: config.width,
                height: config.height,
                samples: config.samples,
                max_repeat_radius: config.max_repeat_radius,
                rows_per_dispatch: config.height,
                dispatch_count,
                sample_lanes: self.sample_lanes,
                samples_per_lane: self.samples_per_lane,
                camera_ray_count,
                max_periodic_copies_per_ray,
                max_bvh_traces: camera_ray_count * max_periodic_copies_per_ray * trace_multiplier,
                bvh_build_time,
                gpu_setup_time,
                gpu_dispatch_time,
                readback_time,
            },
        })
    }
}

async fn bake_sample_parallel(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
) -> Result<GpuBakeResult, GpuBakeError> {
    let mut context = SampleParallelContext::new(config, mesh).await?;
    context.bake(config)
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
        sample_limit: config.samples,
        sample_lanes: 1,
        tile_min: [mesh.tile_min_x, mesh.tile_min_z],
        tile_size: [mesh.tile_width, mesh.tile_depth],
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
            sampler_code(config.sampler),
            u32::from(config.enable_shadows),
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
    let max_sample_lanes = sample_parallel_lanes_for_pixel_count(pixel_count);
    let output_size =
        pixel_count * max_sample_lanes as u64 * std::mem::size_of::<[f32; 4]>() as u64;
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
        let rows_per_dispatch = rows_per_dispatch_for(
            config.width,
            config.height,
            active_batch,
            config.max_repeat_radius,
        );
        let active_lanes = sample_lanes_for(active_batch).min(max_sample_lanes);
        let samples_per_lane = active_batch.div_ceil(active_lanes).max(1);
        let batch_start = Instant::now();
        let mut y_offset = 0;
        while y_offset < config.height {
            let active_height = rows_per_dispatch.min(config.height - y_offset);
            let params = GpuParams {
                samples: samples_per_lane,
                sample_offset: completed_samples,
                target_samples: config.samples,
                sample_limit: completed_samples + active_batch,
                sample_lanes: active_lanes,
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
                pass.dispatch_workgroups(
                    config.width.div_ceil(WORKGROUP_WIDTH),
                    active_height.div_ceil(WORKGROUP_HEIGHT),
                    active_lanes,
                );
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            y_offset += active_height;
            dispatch_count += 1;
        }

        let readback_start = Instant::now();
        let partials = read_output_rgba(
            &device,
            &queue,
            &output_buffer,
            active_lanes as u64 * pixel_count,
        )?;
        readback_time += readback_start.elapsed();
        completed_samples += active_batch;

        for lane in 0..active_lanes as usize {
            let lane_start = lane * pixel_count as usize;
            for (pixel_index, sum) in accumulated.iter_mut().enumerate() {
                let partial = partials[lane_start + pixel_index];
                sum[0] += partial[0];
                sum[1] += partial[1];
                sum[2] += partial[2];
            }
        }

        for (sum, avg) in accumulated.iter().zip(averaged.iter_mut()) {
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
    let trace_multiplier = if config.enable_shadows { 2 } else { 1 };

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
            sample_lanes: 1,
            samples_per_lane: config.samples,
            camera_ray_count,
            max_periodic_copies_per_ray,
            max_bvh_traces: camera_ray_count * max_periodic_copies_per_ray * trace_multiplier,
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
        sample_limit: config.samples,
        sample_lanes: 1,
        tile_min: [mesh.tile_min_x, mesh.tile_min_z],
        tile_size: [mesh.tile_width, mesh.tile_depth],
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
            sampler_code(config.sampler),
            u32::from(config.enable_shadows),
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
            pass.dispatch_workgroups(
                config.width.div_ceil(WORKGROUP_WIDTH),
                active_height.div_ceil(WORKGROUP_HEIGHT),
                1,
            );
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
    let trace_multiplier = if config.enable_shadows { 2 } else { 1 };

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
            sample_lanes: 1,
            samples_per_lane: config.samples,
            camera_ray_count,
            max_periodic_copies_per_ray,
            max_bvh_traces: camera_ray_count * max_periodic_copies_per_ray * trace_multiplier,
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

fn read_output_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output_buffer: &wgpu::Buffer,
    pixel_count: u64,
) -> Result<Vec<[f32; 4]>, GpuBakeError> {
    let output_size = pixel_count * std::mem::size_of::<[f32; 4]>() as u64;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xbrdf rgba readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xbrdf rgba readback encoder"),
    });
    encoder.copy_buffer_to_buffer(output_buffer, 0, &readback_buffer, 0, output_size);
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
    let result = rgba.to_vec();
    drop(mapped);
    readback_buffer.unmap();

    Ok(result)
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
    let mid = if let Some((axis, mid)) = sah_split(triangles, centroid_min, centroid_max) {
        let mid = mid.min(triangles.len() - 1).max(1);
        partition_by_bin(triangles, centroid_min, centroid_max, axis, mid);
        mid
    } else {
        let axis = longest_axis(centroid_max - centroid_min);
        let mid = triangles.len() / 2;
        triangles.select_nth_unstable_by(mid, |a, b| compare_axis(a.centroid, b.centroid, axis));
        mid
    };
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

fn sah_split(
    triangles: &[BuildTriangle],
    centroid_min: Vec3,
    centroid_max: Vec3,
) -> Option<(usize, usize)> {
    let mut best_axis = 0usize;
    let mut best_left_count = 0usize;
    let mut best_cost = f32::INFINITY;

    for axis in 0..3 {
        let extent = axis_value(centroid_max, axis) - axis_value(centroid_min, axis);
        if extent <= 1.0e-8 {
            continue;
        }

        let mut bins = [SahBin::empty(); SAH_BINS];
        for triangle in triangles {
            let index = centroid_bin(triangle.centroid, centroid_min, centroid_max, axis);
            bins[index].add(triangle.bounds_min, triangle.bounds_max);
        }

        let mut left_counts = [0usize; SAH_BINS - 1];
        let mut right_counts = [0usize; SAH_BINS - 1];
        let mut left_min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut left_max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut left_count = 0usize;
        let mut left_areas = [0.0f32; SAH_BINS - 1];

        for split in 0..(SAH_BINS - 1) {
            if bins[split].count > 0 {
                left_min = left_min.min(bins[split].bounds_min);
                left_max = left_max.max(bins[split].bounds_max);
                left_count += bins[split].count;
            }
            left_counts[split] = left_count;
            left_areas[split] = surface_area(left_min, left_max);
        }

        let mut right_min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut right_max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut right_count = 0usize;
        let mut right_areas = [0.0f32; SAH_BINS - 1];

        for split in (0..(SAH_BINS - 1)).rev() {
            let bin = split + 1;
            if bins[bin].count > 0 {
                right_min = right_min.min(bins[bin].bounds_min);
                right_max = right_max.max(bins[bin].bounds_max);
                right_count += bins[bin].count;
            }
            right_counts[split] = right_count;
            right_areas[split] = surface_area(right_min, right_max);
        }

        for split in 0..(SAH_BINS - 1) {
            let left_count = left_counts[split];
            let right_count = right_counts[split];
            if left_count == 0 || right_count == 0 {
                continue;
            }

            let cost =
                left_areas[split] * left_count as f32 + right_areas[split] * right_count as f32;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_left_count = left_count;
            }
        }
    }

    if best_cost.is_finite() {
        Some((best_axis, best_left_count))
    } else {
        None
    }
}

fn partition_by_bin(
    triangles: &mut [BuildTriangle],
    centroid_min: Vec3,
    centroid_max: Vec3,
    axis: usize,
    mid: usize,
) {
    if mid < triangles.len() {
        triangles.select_nth_unstable_by_key(mid, |triangle| {
            centroid_bin(triangle.centroid, centroid_min, centroid_max, axis)
        });
    }
}

fn centroid_bin(centroid: Vec3, centroid_min: Vec3, centroid_max: Vec3, axis: usize) -> usize {
    let extent = axis_value(centroid_max, axis) - axis_value(centroid_min, axis);
    if extent <= 1.0e-8 {
        return 0;
    }

    let normalized = (axis_value(centroid, axis) - axis_value(centroid_min, axis)) / extent;
    ((normalized * SAH_BINS as f32) as usize).min(SAH_BINS - 1)
}

fn axis_value(value: Vec3, axis: usize) -> f32 {
    match axis {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}

fn surface_area(bounds_min: Vec3, bounds_max: Vec3) -> f32 {
    let extent = bounds_max - bounds_min;
    if extent.x < 0.0 || extent.y < 0.0 || extent.z < 0.0 {
        return 0.0;
    }
    2.0 * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z)
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

impl SahBin {
    const fn empty() -> Self {
        Self {
            count: 0,
            bounds_min: Vec3 {
                x: f32::INFINITY,
                y: f32::INFINITY,
                z: f32::INFINITY,
            },
            bounds_max: Vec3 {
                x: f32::NEG_INFINITY,
                y: f32::NEG_INFINITY,
                z: f32::NEG_INFINITY,
            },
        }
    }

    fn add(&mut self, bounds_min: Vec3, bounds_max: Vec3) {
        self.count += 1;
        self.bounds_min = self.bounds_min.min(bounds_min);
        self.bounds_max = self.bounds_max.max(bounds_max);
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
    rows_per_dispatch_for(
        config.width,
        config.height,
        config.samples,
        config.max_repeat_radius,
    )
}

fn rows_per_dispatch_for(width: u32, height: u32, samples: u32, max_repeat_radius: u32) -> u32 {
    let repeat_diameter = max_repeat_radius as u64 * 2 + 1;
    let traces_per_row = width as u64 * samples as u64 * repeat_diameter * repeat_diameter * 2;
    let rows = (TARGET_RAY_TRACES_PER_DISPATCH / traces_per_row.max(1)).max(1);
    let rows = rows.min(height as u64) as u32;
    let rounded_rows = if height >= WORKGROUP_HEIGHT {
        rows.max(WORKGROUP_HEIGHT)
    } else {
        rows
    };
    let rounded_rows = if rounded_rows >= WORKGROUP_HEIGHT {
        rounded_rows.div_ceil(WORKGROUP_HEIGHT) * WORKGROUP_HEIGHT
    } else {
        rounded_rows
    };
    rounded_rows.max(1).min(height)
}

fn sample_lanes_for(sample_count: u32) -> u32 {
    sample_count
        .div_ceil(SAMPLE_PARALLEL_SAMPLES_PER_LANE)
        .clamp(1, SAMPLE_PARALLEL_LANES)
}

fn sample_parallel_lanes_for_pixel_count(pixel_count: u64) -> u32 {
    let bytes_per_lane = pixel_count.max(1) * std::mem::size_of::<[f32; 4]>() as u64;
    let lanes = (MAX_SAMPLE_PARALLEL_BUFFER_BYTES / bytes_per_lane)
        .max(1)
        .min(SAMPLE_PARALLEL_LANES as u64);
    lanes as u32
}

fn sample_parallel_samples_per_lane(sample_count: u32) -> u32 {
    if sample_count >= HIGH_SAMPLE_THRESHOLD {
        HIGH_SAMPLE_PARALLEL_SAMPLES_PER_LANE
    } else {
        SAMPLE_PARALLEL_SAMPLES_PER_LANE
    }
}

fn sampler_code(sampler: SamplerKind) -> u32 {
    match sampler {
        SamplerKind::Halton => 0,
        SamplerKind::Random => 1,
    }
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
