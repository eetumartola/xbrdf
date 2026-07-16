use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use xbrdf_core::{MaterialKind, Mesh, ResolvedBakeConfig, SamplerKind, Triangle, Vec3};

const SHADER: &str = include_str!("bake.wgsl");
const BVH_BALANCE_DEPTH: u32 = 32;
const BVH_LEAF_SIZE: usize = 4;
const ATLAS_TILE_BATCH_SIZE: usize = 8;
const TARGET_RAY_TRACES_PER_DISPATCH: u64 = 1_024_000_000;
const WORKGROUP_WIDTH: u32 = 8;
const WORKGROUP_HEIGHT: u32 = 8;
const SAMPLE_PARALLEL_THRESHOLD: u32 = 2048;
const SAMPLE_PARALLEL_LANES: u32 = 128;
const SAMPLE_PARALLEL_SAMPLES_PER_LANE: u32 = 64;
const HIGH_SAMPLE_PARALLEL_SAMPLES_PER_LANE: u32 = 512;
const HIGH_SAMPLE_THRESHOLD: u32 = 65_536;
const MAX_SAMPLE_PARALLEL_BUFFER_BYTES: u64 = 512 * 1024 * 1024;
const OUTPUT_PIXEL_BYTES: u64 = std::mem::size_of::<[f32; 4]>() as u64;
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
    #[error("image {width}x{height} exceeds 32-bit shader indexing")]
    ImageTooLarge { width: u32, height: u32 },
    #[error("GPU output requires {required_bytes} bytes but the device storage-buffer limit is {limit_bytes} bytes")]
    OutputBufferTooLarge {
        required_bytes: u64,
        limit_bytes: u64,
    },
    #[error("GPU {resource} buffer requires {required_bytes} bytes but the device storage-buffer limit is {limit_bytes} bytes")]
    SceneBufferTooLarge {
        resource: &'static str,
        required_bytes: u64,
        limit_bytes: u64,
    },
    #[error(
        "scene has {count} {resource}, exceeding the shader limit of {}",
        u32::MAX
    )]
    SceneCountTooLarge {
        resource: &'static str,
        count: usize,
    },
    #[error("bake cancelled")]
    Cancelled,
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
    intersection_params: [f32; 4],
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
    pub gpu_timestamp_time: Option<Duration>,
    pub readback_time: Duration,
}

#[derive(Clone, Default)]
pub struct BakeControl(Arc<AtomicBool>);

impl BakeControl {
    pub fn cancel(&self) {
        self.0.store(true, AtomicOrdering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(AtomicOrdering::Relaxed)
    }

    fn check(&self) -> Result<(), GpuBakeError> {
        if self.is_cancelled() {
            Err(GpuBakeError::Cancelled)
        } else {
            Ok(())
        }
    }
}
struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    pass_count: u32,
    timestamp_period_ns: f32,
}

impl GpuTimer {
    fn optional_feature(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::TIMESTAMP_QUERY
    }

    fn new(device: &wgpu::Device, queue: &wgpu::Queue, pass_count: u32) -> Option<Self> {
        if pass_count == 0 || !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }

        let query_count = pass_count.checked_mul(2)?;
        let buffer_size = query_count as u64 * std::mem::size_of::<u64>() as u64;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("xbrdf compute timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xbrdf timestamp resolve"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xbrdf timestamp readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            pass_count,
            timestamp_period_ns: queue.get_timestamp_period(),
        })
    }

    fn writes(&self, pass_index: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        let first_query = pass_index * 2;
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(first_query),
            end_of_pass_write_index: Some(first_query + 1),
        }
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let query_count = self.pass_count * 2;
        let buffer_size = query_count as u64 * std::mem::size_of::<u64>() as u64;
        encoder.resolve_query_set(&self.query_set, 0..query_count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            buffer_size,
        );
    }

    fn read(&self, device: &wgpu::Device) -> Result<Vec<Duration>, GpuBakeError> {
        let slice = self.readback_buffer.slice(..);
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
        let timestamps: &[u64] = bytemuck::cast_slice(&mapped);
        let durations = timestamps
            .chunks_exact(2)
            .map(|pair| {
                let ticks = pair[1].saturating_sub(pair[0]);
                let nanoseconds = ticks as f64 * self.timestamp_period_ns as f64;
                Duration::from_secs_f64(nanoseconds * 1.0e-9)
            })
            .collect();
        drop(mapped);
        self.readback_buffer.unmap();
        Ok(durations)
    }
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
    checked_pixel_count(config.width, config.height)?;
    if config.samples >= SAMPLE_PARALLEL_THRESHOLD {
        return bake_sample_parallel(config, mesh).await;
    }

    bake_inner(config, mesh, None).await
}

pub async fn bake_atlas(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
) -> Result<GpuBakeResult, GpuBakeError> {
    bake_atlas_inner(config, mesh, None, &BakeControl::default()).await
}

pub async fn bake_atlas_with_progress<F>(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    mut progress: F,
) -> Result<GpuBakeResult, GpuBakeError>
where
    F: FnMut(AtlasProgressFrame),
{
    bake_atlas_inner(config, mesh, Some(&mut progress), &BakeControl::default()).await
}

pub async fn bake_atlas_with_control<F>(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    control: &BakeControl,
    mut progress: F,
) -> Result<GpuBakeResult, GpuBakeError>
where
    F: FnMut(AtlasProgressFrame),
{
    bake_atlas_inner(config, mesh, Some(&mut progress), control).await
}

async fn bake_atlas_inner(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    mut progress: Option<&mut dyn FnMut(AtlasProgressFrame)>,
    control: &BakeControl,
) -> Result<GpuBakeResult, GpuBakeError> {
    control.check()?;
    checked_pixel_count(config.camera_tile_width(), config.camera_tile_height())?;
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
    let atlas_pixel_count = atlas_width as u64 * atlas_height as u64;
    let atlas_capacity =
        usize::try_from(atlas_pixel_count).map_err(|_| GpuBakeError::ImageTooLarge {
            width: atlas_width,
            height: atlas_height,
        })?;
    let mut atlas_pixels = Vec::new();
    atlas_pixels
        .try_reserve_exact(atlas_capacity)
        .map_err(|_| GpuBakeError::ImageTooLarge {
            width: atlas_width,
            height: atlas_height,
        })?;
    atlas_pixels.resize(atlas_capacity, [0.0; 3]);
    let mut combined_stats = None;
    let mut completed_tiles = 0u32;

    let tile_config = config.config_for_tile(0, 0);
    let mut sample_context = SampleParallelContext::new(&tile_config, mesh).await?;

    let batch_capacity = sample_context.atlas_batch_capacity(&tile_config);
    let light_count = config.light_count() as usize;
    for batch_start in (0..light_count).step_by(batch_capacity) {
        control.check()?;
        let batch_end = (batch_start + batch_capacity).min(light_count);
        let batch_configs: Vec<_> = (batch_start..batch_end)
            .map(|tile_index| {
                let light_x = tile_index as u32 % light_width;
                let light_y = tile_index as u32 / light_width;
                config.config_for_tile(light_x, light_y)
            })
            .collect();
        let batch = sample_context.bake_batch(&batch_configs, control)?;

        for (batch_index, tile) in batch.into_iter().enumerate() {
            let tile_index = batch_start + batch_index;
            let light_x = tile_index as u32 % light_width;
            let light_y = tile_index as u32 / light_width;
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
            if let Some(progress) = progress.as_deref_mut() {
                progress(AtlasProgressFrame {
                    completed_tiles,
                    total_tiles: config.light_count(),
                    width: atlas_width,
                    height: atlas_height,
                    pixels: atlas_pixels.clone(),
                });
            }
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
    bake_progressive_inner(
        config,
        mesh,
        options,
        &BakeControl::default(),
        &mut progress,
    )
    .await
}

pub async fn bake_progressive_with_control<F>(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    options: ProgressiveBakeOptions,
    control: &BakeControl,
    mut progress: F,
) -> Result<GpuBakeResult, GpuBakeError>
where
    F: FnMut(ProgressiveFrame),
{
    bake_progressive_inner(config, mesh, options, control, &mut progress).await
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
    total.gpu_timestamp_time = match (total.gpu_timestamp_time, tile.gpu_timestamp_time) {
        (Some(total), Some(tile)) => Some(total + tile),
        _ => None,
    };
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

#[derive(Clone, Copy)]
struct SceneGpuParams {
    triangle_count: usize,
    node_count: usize,
    tile_min: [f32; 2],
    tile_size: [f32; 2],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    ray_epsilon: f32,
    hit_epsilon: f32,
    determinant_epsilon: f32,
}
struct GpuShared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

struct CachedScene {
    key: blake3::Hash,
    params: SceneGpuParams,
    triangle_buffer: wgpu::Buffer,
    bvh_buffer: wgpu::Buffer,
}

#[derive(Default)]
struct GpuResourceCache {
    shared: Option<Arc<GpuShared>>,
    scene: Option<Arc<CachedScene>>,
}

static GPU_RESOURCE_CACHE: LazyLock<Mutex<GpuResourceCache>> =
    LazyLock::new(|| Mutex::new(GpuResourceCache::default()));

async fn cached_gpu_resources(
    mesh: &Mesh,
) -> Result<(Arc<GpuShared>, Arc<CachedScene>, Duration, Duration), GpuBakeError> {
    let (shared, shared_setup_time) = cached_gpu_shared().await?;
    let key = mesh_cache_key(mesh);
    if let Some(scene) = GPU_RESOURCE_CACHE
        .lock()
        .expect("GPU resource cache poisoned")
        .scene
        .as_ref()
        .filter(|scene| scene.key == key)
        .cloned()
    {
        return Ok((shared, scene, Duration::ZERO, shared_setup_time));
    }

    let bvh_start = Instant::now();
    let (triangles, bvh_nodes) = build_bvh(&mesh.triangles);
    let bvh_build_time = bvh_start.elapsed();
    validate_scene_buffers(&shared.device, triangles.len(), bvh_nodes.len())?;
    let upload_start = Instant::now();
    let triangle_buffer = shared
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xbrdf cached triangles"),
            contents: bytemuck::cast_slice(&triangles),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let bvh_buffer = shared
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xbrdf cached BVH"),
            contents: bytemuck::cast_slice(&bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let tolerances = ray_tolerances(mesh);
    let params = SceneGpuParams {
        triangle_count: triangles.len(),
        node_count: bvh_nodes.len(),
        tile_min: [mesh.tile_min_x, mesh.tile_min_z],
        tile_size: [mesh.tile_width, mesh.tile_depth],
        bounds_min: vec4(mesh.bounds.min),
        bounds_max: vec4(mesh.bounds.max),
        ray_epsilon: tolerances.ray_origin,
        hit_epsilon: tolerances.hit,
        determinant_epsilon: tolerances.determinant,
    };
    let scene = Arc::new(CachedScene {
        key,
        params,
        triangle_buffer,
        bvh_buffer,
    });
    let scene_setup_time = upload_start.elapsed();
    GPU_RESOURCE_CACHE
        .lock()
        .expect("GPU resource cache poisoned")
        .scene = Some(scene.clone());
    Ok((
        shared,
        scene,
        bvh_build_time,
        shared_setup_time + scene_setup_time,
    ))
}

async fn cached_gpu_shared() -> Result<(Arc<GpuShared>, Duration), GpuBakeError> {
    if let Some(shared) = GPU_RESOURCE_CACHE
        .lock()
        .expect("GPU resource cache poisoned")
        .shared
        .clone()
    {
        return Ok((shared, Duration::ZERO));
    }

    let setup_start = Instant::now();
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
                label: Some("xbrdf cached device"),
                required_features: GpuTimer::optional_feature(&adapter),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xbrdf cached bake shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("xbrdf cached bind group layout"),
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
        label: Some("xbrdf cached pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("xbrdf cached bake pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let shared = Arc::new(GpuShared {
        device,
        queue,
        bind_group_layout,
        pipeline,
    });
    let setup_time = setup_start.elapsed();
    let mut cache = GPU_RESOURCE_CACHE
        .lock()
        .expect("GPU resource cache poisoned");
    if let Some(cached) = cache.shared.clone() {
        Ok((cached, Duration::ZERO))
    } else {
        cache.shared = Some(shared.clone());
        Ok((shared, setup_time))
    }
}

fn mesh_cache_key(mesh: &Mesh) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    for triangle in &mesh.triangles {
        for value in [
            triangle.v0,
            triangle.v1,
            triangle.v2,
            triangle.normal,
            triangle.color,
        ] {
            hasher.update(&value.x.to_bits().to_le_bytes());
            hasher.update(&value.y.to_bits().to_le_bytes());
            hasher.update(&value.z.to_bits().to_le_bytes());
        }
    }
    for value in [mesh.bounds.min, mesh.bounds.max] {
        hasher.update(&value.x.to_bits().to_le_bytes());
        hasher.update(&value.y.to_bits().to_le_bytes());
        hasher.update(&value.z.to_bits().to_le_bytes());
    }
    for value in [
        mesh.tile_min_x,
        mesh.tile_min_z,
        mesh.tile_width,
        mesh.tile_depth,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.finalize()
}

fn validate_scene_buffers(
    device: &wgpu::Device,
    triangle_count: usize,
    node_count: usize,
) -> Result<(), GpuBakeError> {
    validate_scene_buffer::<GpuTriangle>(device, "triangle", triangle_count)?;
    validate_scene_buffer::<GpuBvhNode>(device, "BVH node", node_count)
}

fn validate_scene_buffer<T>(
    device: &wgpu::Device,
    resource: &'static str,
    count: usize,
) -> Result<(), GpuBakeError> {
    if count > u32::MAX as usize {
        return Err(GpuBakeError::SceneCountTooLarge { resource, count });
    }
    let required_bytes = u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<T>() as u64))
        .ok_or(GpuBakeError::SceneBufferTooLarge {
            resource,
            required_bytes: u64::MAX,
            limit_bytes: device.limits().max_buffer_size,
        })?;
    let limits = device.limits();
    let limit_bytes = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size as u64);
    if required_bytes > limit_bytes {
        return Err(GpuBakeError::SceneBufferTooLarge {
            resource,
            required_bytes,
            limit_bytes,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct RayTolerances {
    ray_origin: f32,
    hit: f32,
    determinant: f32,
}

fn ray_tolerances(mesh: &Mesh) -> RayTolerances {
    let extent = mesh.bounds.max - mesh.bounds.min;
    let geometry_scale = mesh
        .tile_width
        .max(mesh.tile_depth)
        .max(extent.x.abs())
        .max(extent.y.abs())
        .max(extent.z.abs())
        .max(f32::MIN_POSITIVE);
    let coordinate_scale = mesh
        .bounds
        .min
        .x
        .abs()
        .max(mesh.bounds.min.y.abs())
        .max(mesh.bounds.min.z.abs())
        .max(mesh.bounds.max.x.abs())
        .max(mesh.bounds.max.y.abs())
        .max(mesh.bounds.max.z.abs());
    let roundoff_floor = coordinate_scale * f32::EPSILON * 8.0;
    let ray_origin = (geometry_scale * 1.0e-5)
        .max(roundoff_floor)
        .max(f32::MIN_POSITIVE);
    RayTolerances {
        ray_origin,
        hit: ray_origin * 0.1,
        determinant: geometry_scale * geometry_scale * 1.0e-7,
    }
}

fn sample_parallel_params(
    config: &ResolvedBakeConfig,
    scene: SceneGpuParams,
    samples_per_lane: u32,
    sample_lanes: u32,
) -> GpuParams {
    let light = Vec3::from_array(config.light);
    GpuParams {
        width: config.width,
        height: config.height,
        samples: samples_per_lane,
        triangle_count: scene.triangle_count as u32,
        node_count: scene.node_count as u32,
        max_repeat_radius: config.max_repeat_radius,
        y_offset: 0,
        active_height: config.height,
        sample_offset: 0,
        target_samples: config.samples,
        sample_limit: config.samples,
        sample_lanes,
        tile_min: scene.tile_min,
        tile_size: scene.tile_size,
        bounds_min: scene.bounds_min,
        bounds_max: scene.bounds_max,
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
            1,
        ],
        material_params: [
            1.0 / light.y.max(1.0e-4),
            scene.ray_epsilon,
            config.material.roughness.unwrap_or(0.0),
            config.material.phong_exponent().unwrap_or(1.0),
        ],
        intersection_params: [scene.hit_epsilon, scene.determinant_epsilon, 0.0, 0.0],
    }
}

struct SampleParallelContext {
    scene: SceneGpuParams,
    bvh_build_time: Duration,
    gpu_setup_time: Duration,
    shared: Arc<GpuShared>,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    sample_lanes: u32,
    samples_per_lane: u32,
    timer: Option<GpuTimer>,
    setup_reported: bool,
}

impl SampleParallelContext {
    async fn new(config: &ResolvedBakeConfig, mesh: &Mesh) -> Result<Self, GpuBakeError> {
        let (shared, cached_scene, bvh_build_time, cached_setup_time) =
            cached_gpu_resources(mesh).await?;
        let gpu_setup_start = Instant::now();
        let pixel_count = checked_pixel_count(config.width, config.height)?;
        let limits = shared.device.limits();
        let sample_lanes = sample_parallel_lanes_for_pixel_count(pixel_count, &limits)?
            .min(sample_lanes_for(config.samples));
        let samples_per_lane = sample_parallel_samples_per_lane(config.samples);
        let scene = cached_scene.params;
        let base_params = sample_parallel_params(config, scene, samples_per_lane, sample_lanes);
        let params_buffer = shared
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("xbrdf sample-parallel params"),
                contents: bytemuck::bytes_of(&base_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let partial_count = pixel_count * sample_lanes as u64;
        let output_size = partial_count * OUTPUT_PIXEL_BYTES;
        let output_buffer = shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xbrdf sample-parallel partials"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xbrdf sample-parallel bind group"),
            layout: &shared.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cached_scene.triangle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cached_scene.bvh_buffer.as_entire_binding(),
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
        let timer = GpuTimer::new(&shared.device, &shared.queue, 1);
        let gpu_setup_time = cached_setup_time + gpu_setup_start.elapsed();

        Ok(Self {
            scene,
            bvh_build_time,
            gpu_setup_time,
            shared,
            params_buffer,
            output_buffer,
            bind_group,
            sample_lanes,
            samples_per_lane,
            timer,
            setup_reported: false,
        })
    }

    fn bake(
        &mut self,
        config: &ResolvedBakeConfig,
        control: &BakeControl,
    ) -> Result<GpuBakeResult, GpuBakeError> {
        let pixel_count = config.width as u64 * config.height as u64;
        let mut sums = vec![[0.0f32; 3]; pixel_count as usize];
        let mut completed_samples = 0u32;
        let mut dispatch_count = 0u32;
        let mut readback_time = Duration::ZERO;
        let mut gpu_timestamp_time = self.timer.as_ref().map(|_| Duration::ZERO);
        let mut timestamp_readback_time = Duration::ZERO;
        let dispatch_start = Instant::now();
        let samples_per_wave = self.sample_lanes * self.samples_per_lane;

        while completed_samples < config.samples {
            control.check()?;
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
                    self.scene,
                    self.samples_per_lane,
                    self.sample_lanes,
                )
            };
            self.shared
                .queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder =
                self.shared
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("xbrdf sample-parallel bake encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("xbrdf sample-parallel bake pass"),
                    timestamp_writes: self.timer.as_ref().map(|timer| timer.writes(0)),
                });
                pass.set_pipeline(&self.shared.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(
                    config.width.div_ceil(WORKGROUP_WIDTH),
                    config.height.div_ceil(WORKGROUP_HEIGHT),
                    active_lanes,
                );
            }
            if let Some(timer) = &self.timer {
                timer.resolve(&mut encoder);
            }
            self.shared.queue.submit(Some(encoder.finish()));
            self.shared.device.poll(wgpu::Maintain::Wait);
            if let (Some(total), Some(timer)) = (&mut gpu_timestamp_time, &self.timer) {
                let timestamp_read_start = Instant::now();
                *total += timer
                    .read(&self.shared.device)?
                    .into_iter()
                    .sum::<Duration>();
                timestamp_readback_time += timestamp_read_start.elapsed();
            }
            dispatch_count += 1;

            let readback_start = Instant::now();
            let partials = read_output_rgba(
                &self.shared.device,
                &self.shared.queue,
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

        let gpu_dispatch_time = dispatch_start
            .elapsed()
            .saturating_sub(readback_time)
            .saturating_sub(timestamp_readback_time);
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
                triangle_count: self.scene.triangle_count,
                bvh_node_count: self.scene.node_count,
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
                gpu_timestamp_time,
                readback_time,
            },
        })
    }
    fn atlas_batch_capacity(&self, config: &ResolvedBakeConfig) -> usize {
        let pixel_count = config.width as u64 * config.height as u64;
        let bytes_per_tile = pixel_count * self.sample_lanes as u64 * OUTPUT_PIXEL_BYTES;
        let device_capacity = self.shared.device.limits().max_buffer_size / bytes_per_tile.max(1);
        ATLAS_TILE_BATCH_SIZE.min(device_capacity as usize).max(1)
    }

    fn bake_batch(
        &mut self,
        configs: &[ResolvedBakeConfig],
        control: &BakeControl,
    ) -> Result<Vec<GpuBakeResult>, GpuBakeError> {
        if configs.is_empty() {
            return Ok(Vec::new());
        }

        let first = &configs[0];
        debug_assert!(configs.iter().all(|config| {
            config.width == first.width
                && config.height == first.height
                && config.samples == first.samples
        }));
        let pixel_count = first.width as u64 * first.height as u64;
        let pixel_count_usize = pixel_count as usize;
        let mut sums = vec![vec![[0.0f32; 3]; pixel_count_usize]; configs.len()];
        let mut gpu_timestamp_times = vec![
            self.shared
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then_some(Duration::ZERO);
            configs.len()
        ];
        let mut completed_samples = 0u32;
        let mut wave_count = 0u32;
        let mut gpu_dispatch_time = Duration::ZERO;
        let mut readback_time = Duration::ZERO;
        let samples_per_wave = self.sample_lanes * self.samples_per_lane;

        while completed_samples < first.samples {
            control.check()?;
            let remaining = first.samples - completed_samples;
            let active_lanes = remaining
                .div_ceil(self.samples_per_lane)
                .min(self.sample_lanes)
                .max(1);
            let partials_per_tile = pixel_count * active_lanes as u64;
            let bytes_per_tile = partials_per_tile * OUTPUT_PIXEL_BYTES;
            let batch_size = bytes_per_tile * configs.len() as u64;
            let batch_readback = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xbrdf atlas batch readback"),
                size: batch_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let timer = GpuTimer::new(
                &self.shared.device,
                &self.shared.queue,
                configs.len() as u32,
            );
            let dispatch_start = Instant::now();

            for (tile_index, config) in configs.iter().enumerate() {
                control.check()?;
                let params = GpuParams {
                    sample_offset: completed_samples,
                    sample_limit: (completed_samples + samples_per_wave).min(config.samples),
                    sample_lanes: active_lanes,
                    ..sample_parallel_params(
                        config,
                        self.scene,
                        self.samples_per_lane,
                        self.sample_lanes,
                    )
                };
                self.shared
                    .queue
                    .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

                let mut encoder =
                    self.shared
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("xbrdf atlas batch encoder"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("xbrdf atlas batch pass"),
                        timestamp_writes: timer
                            .as_ref()
                            .map(|timer| timer.writes(tile_index as u32)),
                    });
                    pass.set_pipeline(&self.shared.pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(
                        config.width.div_ceil(WORKGROUP_WIDTH),
                        config.height.div_ceil(WORKGROUP_HEIGHT),
                        active_lanes,
                    );
                }
                encoder.copy_buffer_to_buffer(
                    &self.output_buffer,
                    0,
                    &batch_readback,
                    tile_index as u64 * bytes_per_tile,
                    bytes_per_tile,
                );
                self.shared.queue.submit(Some(encoder.finish()));
            }

            if let Some(timer) = &timer {
                let mut encoder =
                    self.shared
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("xbrdf atlas timestamp resolve"),
                        });
                timer.resolve(&mut encoder);
                self.shared.queue.submit(Some(encoder.finish()));
            }
            self.shared.device.poll(wgpu::Maintain::Wait);
            gpu_dispatch_time += dispatch_start.elapsed();

            if let Some(timer) = &timer {
                for (total, elapsed) in gpu_timestamp_times
                    .iter_mut()
                    .zip(timer.read(&self.shared.device)?)
                {
                    if let Some(total) = total {
                        *total += elapsed;
                    }
                }
            }

            let readback_start = Instant::now();
            let partials = read_mapped_rgba(&self.shared.device, &batch_readback)?;
            readback_time += readback_start.elapsed();
            for (tile_index, tile_sums) in sums.iter_mut().enumerate() {
                let tile_start = tile_index * partials_per_tile as usize;
                for lane in 0..active_lanes as usize {
                    let lane_start = tile_start + lane * pixel_count_usize;
                    for (pixel_index, sum) in tile_sums.iter_mut().enumerate() {
                        let partial = partials[lane_start + pixel_index];
                        sum[0] += partial[0];
                        sum[1] += partial[1];
                        sum[2] += partial[2];
                    }
                }
            }

            completed_samples = completed_samples
                .saturating_add(samples_per_wave)
                .min(first.samples);
            wave_count += 1;
        }

        let result_count = configs.len() as u32;
        let dispatch_time_per_tile = gpu_dispatch_time / result_count;
        let readback_time_per_tile = readback_time / result_count;
        let report_setup = !self.setup_reported;
        self.setup_reported = true;
        Ok(configs
            .iter()
            .zip(sums)
            .zip(gpu_timestamp_times)
            .enumerate()
            .map(|(index, ((config, sums), gpu_timestamp_time))| {
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
                let camera_ray_count =
                    config.width as u64 * config.height as u64 * config.samples as u64;
                let trace_multiplier = if config.enable_shadows { 2 } else { 1 };
                GpuBakeResult {
                    pixels,
                    stats: GpuBakeStats {
                        triangle_count: self.scene.triangle_count,
                        bvh_node_count: self.scene.node_count,
                        width: config.width,
                        height: config.height,
                        samples: config.samples,
                        max_repeat_radius: config.max_repeat_radius,
                        rows_per_dispatch: config.height,
                        dispatch_count: wave_count,
                        sample_lanes: self.sample_lanes,
                        samples_per_lane: self.samples_per_lane,
                        camera_ray_count,
                        max_periodic_copies_per_ray,
                        max_bvh_traces: camera_ray_count
                            * max_periodic_copies_per_ray
                            * trace_multiplier,
                        bvh_build_time: if report_setup && index == 0 {
                            self.bvh_build_time
                        } else {
                            Duration::ZERO
                        },
                        gpu_setup_time: if report_setup && index == 0 {
                            self.gpu_setup_time
                        } else {
                            Duration::ZERO
                        },
                        gpu_dispatch_time: dispatch_time_per_tile,
                        gpu_timestamp_time,
                        readback_time: readback_time_per_tile,
                    },
                }
            })
            .collect())
    }
}

async fn bake_sample_parallel(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
) -> Result<GpuBakeResult, GpuBakeError> {
    let mut context = SampleParallelContext::new(config, mesh).await?;
    context.bake(config, &BakeControl::default())
}
async fn bake_progressive_inner(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    options: ProgressiveBakeOptions,
    control: &BakeControl,
    progress: &mut dyn FnMut(ProgressiveFrame),
) -> Result<GpuBakeResult, GpuBakeError> {
    let (shared, cached_scene, bvh_build_time, cached_setup_time) =
        cached_gpu_resources(mesh).await?;
    let gpu_setup_start = Instant::now();
    let device = &shared.device;
    let queue = &shared.queue;
    let pipeline = &shared.pipeline;
    let scene = cached_scene.params;
    let base_params = sample_parallel_params(config, scene, config.samples, 1);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf progressive params"),
        contents: bytemuck::bytes_of(&base_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let pixel_count = checked_pixel_count(config.width, config.height)?;
    let max_sample_lanes = sample_parallel_lanes_for_pixel_count(pixel_count, &device.limits())?;
    let output_size = pixel_count * max_sample_lanes as u64 * OUTPUT_PIXEL_BYTES;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xbrdf progressive output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xbrdf progressive bind group"),
        layout: &shared.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cached_scene.triangle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cached_scene.bvh_buffer.as_entire_binding(),
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
    let timer = GpuTimer::new(device, queue, 1);
    let gpu_setup_time = cached_setup_time + gpu_setup_start.elapsed();

    let rows_per_dispatch = rows_per_dispatch(config);
    let mut completed_samples = 0u32;
    let mut batch_samples = 1u32;
    let mut dispatch_count = 0u32;
    let mut readback_time = Duration::ZERO;
    let mut gpu_timestamp_time = timer.as_ref().map(|_| Duration::ZERO);
    let mut timestamp_readback_time = Duration::ZERO;
    let mut accumulated = vec![[0.0; 3]; pixel_count as usize];
    let mut averaged = vec![[0.0; 3]; pixel_count as usize];
    let dispatch_start = Instant::now();
    let target_interval = options.update_interval.max(Duration::from_millis(1));

    while completed_samples < config.samples {
        control.check()?;
        let remaining = config.samples - completed_samples;
        let active_batch = batch_samples.min(remaining).max(1);
        let rows_per_dispatch = rows_per_dispatch_for(
            config.width,
            config.height,
            active_batch,
            config.max_repeat_radius,
            config.enable_shadows,
        );
        let active_lanes = sample_lanes_for(active_batch).min(max_sample_lanes);
        let samples_per_lane = active_batch.div_ceil(active_lanes).max(1);
        let batch_start = Instant::now();
        let mut y_offset = 0;
        while y_offset < config.height {
            control.check()?;
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
                    timestamp_writes: timer.as_ref().map(|timer| timer.writes(0)),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    config.width.div_ceil(WORKGROUP_WIDTH),
                    active_height.div_ceil(WORKGROUP_HEIGHT),
                    active_lanes,
                );
            }
            if let Some(timer) = &timer {
                timer.resolve(&mut encoder);
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            if let (Some(total), Some(timer)) = (&mut gpu_timestamp_time, &timer) {
                let timestamp_read_start = Instant::now();
                *total += timer.read(device)?.into_iter().sum::<Duration>();
                timestamp_readback_time += timestamp_read_start.elapsed();
            }
            y_offset += active_height;
            dispatch_count += 1;
        }

        let readback_start = Instant::now();
        let partials = read_output_rgba(
            device,
            queue,
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
    let gpu_dispatch_time = dispatch_start
        .elapsed()
        .saturating_sub(readback_time)
        .saturating_sub(timestamp_readback_time);

    let max_periodic_copies_per_axis = config.max_repeat_radius as u64 * 2 + 1;
    let max_periodic_copies_per_ray = max_periodic_copies_per_axis * max_periodic_copies_per_axis;
    let camera_ray_count = config.width as u64 * config.height as u64 * config.samples as u64;
    let trace_multiplier = if config.enable_shadows { 2 } else { 1 };

    Ok(GpuBakeResult {
        pixels: averaged,
        stats: GpuBakeStats {
            triangle_count: scene.triangle_count,
            bvh_node_count: scene.node_count,
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
            gpu_timestamp_time,
            readback_time,
        },
    })
}

async fn bake_inner(
    config: &ResolvedBakeConfig,
    mesh: &Mesh,
    mut progress: Option<&mut dyn FnMut(ProgressChunk)>,
) -> Result<GpuBakeResult, GpuBakeError> {
    let (shared, cached_scene, bvh_build_time, cached_setup_time) =
        cached_gpu_resources(mesh).await?;
    let gpu_setup_start = Instant::now();
    let device = &shared.device;
    let queue = &shared.queue;
    let pipeline = &shared.pipeline;
    let scene = cached_scene.params;
    let mut base_params = sample_parallel_params(config, scene, config.samples, 1);
    base_params.material_kind[3] = 0;
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("xbrdf params"),
        contents: bytemuck::bytes_of(&base_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let pixel_count = checked_pixel_count(config.width, config.height)?;
    let output_size = pixel_count * OUTPUT_PIXEL_BYTES;
    validate_storage_buffer_size(output_size, &device.limits())?;
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
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xbrdf bind group"),
        layout: &shared.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cached_scene.triangle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cached_scene.bvh_buffer.as_entire_binding(),
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
    let timer = GpuTimer::new(device, queue, 1);
    let gpu_setup_time = cached_setup_time + gpu_setup_start.elapsed();

    let rows_per_dispatch = rows_per_dispatch(config);
    let mut y_offset = 0;
    let mut dispatch_count = 0;
    let mut gpu_timestamp_time = timer.as_ref().map(|_| Duration::ZERO);
    let mut timestamp_readback_time = Duration::ZERO;
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
                timestamp_writes: timer.as_ref().map(|timer| timer.writes(0)),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                config.width.div_ceil(WORKGROUP_WIDTH),
                active_height.div_ceil(WORKGROUP_HEIGHT),
                1,
            );
        }
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        if let (Some(total), Some(timer)) = (&mut gpu_timestamp_time, &timer) {
            let timestamp_read_start = Instant::now();
            *total += timer.read(device)?.into_iter().sum::<Duration>();
            timestamp_readback_time += timestamp_read_start.elapsed();
        }

        if let Some(progress) = progress.as_deref_mut() {
            let chunk_pixels = read_output_rows(
                device,
                queue,
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
    let gpu_dispatch_time = dispatch_start
        .elapsed()
        .saturating_sub(timestamp_readback_time);

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
            triangle_count: scene.triangle_count,
            bvh_node_count: scene.node_count,
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
            gpu_timestamp_time,
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

    read_mapped_rgba(device, &readback_buffer)
}

fn read_mapped_rgba(
    device: &wgpu::Device,
    readback_buffer: &wgpu::Buffer,
) -> Result<Vec<[f32; 4]>, GpuBakeError> {
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
    build_bvh_node(&mut build_triangles, &mut ordered_triangles, &mut nodes, 0);
    (ordered_triangles, nodes)
}

fn build_bvh_node(
    triangles: &mut [BuildTriangle],
    ordered_triangles: &mut Vec<GpuTriangle>,
    nodes: &mut Vec<GpuBvhNode>,
    depth: u32,
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
    let mid = if depth >= BVH_BALANCE_DEPTH {
        let axis = longest_axis(centroid_max - centroid_min);
        let mid = triangles.len() / 2;
        triangles.select_nth_unstable_by(mid, |a, b| compare_axis(a.centroid, b.centroid, axis));
        mid
    } else if let Some((axis, mid)) = sah_split(triangles, centroid_min, centroid_max) {
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
    let left = build_bvh_node(left_items, ordered_triangles, nodes, depth + 1);
    let right = build_bvh_node(right_items, ordered_triangles, nodes, depth + 1);

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
        config.enable_shadows,
    )
}

fn rows_per_dispatch_for(
    width: u32,
    height: u32,
    samples: u32,
    max_repeat_radius: u32,
    enable_shadows: bool,
) -> u32 {
    let repeat_diameter = max_repeat_radius as u64 * 2 + 1;
    let traces_per_sample = if enable_shadows { 2 } else { 1 };
    let traces_per_row =
        width as u64 * samples as u64 * repeat_diameter * repeat_diameter * traces_per_sample;
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

fn sample_parallel_lanes_for_pixel_count(
    pixel_count: u64,
    limits: &wgpu::Limits,
) -> Result<u32, GpuBakeError> {
    let bytes_per_lane = pixel_count.max(1) * OUTPUT_PIXEL_BYTES;
    let buffer_limit = MAX_SAMPLE_PARALLEL_BUFFER_BYTES
        .min(limits.max_buffer_size)
        .min(limits.max_storage_buffer_binding_size as u64);
    validate_storage_buffer_size(bytes_per_lane, limits)?;
    Ok((buffer_limit / bytes_per_lane)
        .max(1)
        .min(SAMPLE_PARALLEL_LANES as u64) as u32)
}

fn checked_pixel_count(width: u32, height: u32) -> Result<u64, GpuBakeError> {
    let pixel_count = width as u64 * height as u64;
    if pixel_count > u32::MAX as u64 {
        return Err(GpuBakeError::ImageTooLarge { width, height });
    }
    Ok(pixel_count)
}

fn validate_storage_buffer_size(
    required_bytes: u64,
    limits: &wgpu::Limits,
) -> Result<(), GpuBakeError> {
    let limit_bytes = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size as u64);
    if required_bytes > limit_bytes {
        return Err(GpuBakeError::OutputBufferTooLarge {
            required_bytes,
            limit_bytes,
        });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    fn push_quad(triangles: &mut Vec<Triangle>, vertices: [Vec3; 4], normal: Vec3) {
        let color = Vec3::new(1.0, 1.0, 1.0);
        triangles.push(Triangle {
            v0: vertices[0],
            v1: vertices[1],
            v2: vertices[2],
            normal,
            color,
        });
        triangles.push(Triangle {
            v0: vertices[0],
            v1: vertices[2],
            v2: vertices[3],
            normal,
            color,
        });
    }

    fn periodic_box_mesh(at_seam: bool) -> Mesh {
        let mut triangles = Vec::new();
        push_quad(
            &mut triangles,
            [
                Vec3::new(-0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, 0.5),
                Vec3::new(-0.5, 0.0, 0.5),
            ],
            Vec3::Y,
        );

        let intervals = if at_seam {
            vec![(-0.5, -0.3), (0.3, 0.5)]
        } else {
            vec![(-0.2, 0.2)]
        };
        for &(x0, x1) in &intervals {
            push_quad(
                &mut triangles,
                [
                    Vec3::new(x0, 0.5, -0.2),
                    Vec3::new(x1, 0.5, -0.2),
                    Vec3::new(x1, 0.5, 0.2),
                    Vec3::new(x0, 0.5, 0.2),
                ],
                Vec3::Y,
            );
            push_quad(
                &mut triangles,
                [
                    Vec3::new(x0, 0.0, -0.2),
                    Vec3::new(x1, 0.0, -0.2),
                    Vec3::new(x1, 0.5, -0.2),
                    Vec3::new(x0, 0.5, -0.2),
                ],
                Vec3::new(0.0, 0.0, -1.0),
            );
            push_quad(
                &mut triangles,
                [
                    Vec3::new(x0, 0.0, 0.2),
                    Vec3::new(x0, 0.5, 0.2),
                    Vec3::new(x1, 0.5, 0.2),
                    Vec3::new(x1, 0.0, 0.2),
                ],
                Vec3::new(0.0, 0.0, 1.0),
            );
        }
        let (left, right) = if at_seam { (-0.3, 0.3) } else { (-0.2, 0.2) };
        push_quad(
            &mut triangles,
            [
                Vec3::new(left, 0.0, -0.2),
                Vec3::new(left, 0.5, -0.2),
                Vec3::new(left, 0.5, 0.2),
                Vec3::new(left, 0.0, 0.2),
            ],
            if at_seam {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(-1.0, 0.0, 0.0)
            },
        );
        push_quad(
            &mut triangles,
            [
                Vec3::new(right, 0.0, -0.2),
                Vec3::new(right, 0.0, 0.2),
                Vec3::new(right, 0.5, 0.2),
                Vec3::new(right, 0.5, -0.2),
            ],
            if at_seam {
                Vec3::new(-1.0, 0.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            },
        );

        let bounds = xbrdf_core::Bounds {
            min: Vec3::new(-0.5, 0.0, -0.5),
            max: Vec3::new(0.5, 0.5, 0.5),
        };
        Mesh {
            triangles,
            bounds,
            original_bounds: bounds,
            y_offset_to_zero: 0.0,
            tile_min_x: -0.5,
            tile_min_z: -0.5,
            tile_width: 1.0,
            tile_depth: 1.0,
            color_source: xbrdf_core::ColorSource::None,
        }
    }

    fn renderer_test_config(samples: u32) -> ResolvedBakeConfig {
        ResolvedBakeConfig {
            obj: std::path::PathBuf::new(),
            width: 4,
            height: 2,
            mode: xbrdf_core::BakeMode::Single,
            light_width: 1,
            light_height: 1,
            samples,
            light: [-1.0, 1.0, -1.0],
            max_repeat_radius: 2,
            sampler: SamplerKind::Halton,
            enable_shadows: true,
            material: xbrdf_core::ResolvedMaterial {
                kind: MaterialKind::Lambertian,
                color: [1.0, 1.0, 1.0],
                roughness: None,
            },
        }
    }
    fn scale_mesh(mut mesh: Mesh, scale: f32) -> Mesh {
        for triangle in &mut mesh.triangles {
            triangle.v0 = triangle.v0 * scale;
            triangle.v1 = triangle.v1 * scale;
            triangle.v2 = triangle.v2 * scale;
        }
        mesh.bounds.min = mesh.bounds.min * scale;
        mesh.bounds.max = mesh.bounds.max * scale;
        mesh.original_bounds.min = mesh.original_bounds.min * scale;
        mesh.original_bounds.max = mesh.original_bounds.max * scale;
        mesh.y_offset_to_zero *= scale;
        mesh.tile_min_x *= scale;
        mesh.tile_min_z *= scale;
        mesh.tile_width *= scale;
        mesh.tile_depth *= scale;
        mesh
    }

    fn maximum_pixel_error(left: &[[f32; 3]], right: &[[f32; 3]]) -> f32 {
        left.iter()
            .zip(right)
            .flat_map(|(left, right)| (0..3).map(|channel| (left[channel] - right[channel]).abs()))
            .fold(0.0, f32::max)
    }
    fn mean_pixel_error(left: &[[f32; 3]], right: &[[f32; 3]]) -> f32 {
        let absolute_sum: f32 = left
            .iter()
            .zip(right)
            .flat_map(|(left, right)| (0..3).map(|channel| (left[channel] - right[channel]).abs()))
            .sum();
        absolute_sum / (left.len() * 3) as f32
    }

    #[test]
    fn shader_pixel_indexing_limit_is_enforced() {
        assert_eq!(
            checked_pixel_count(65_535, 65_537).unwrap(),
            u32::MAX as u64
        );
        assert!(matches!(
            checked_pixel_count(65_536, 65_536),
            Err(GpuBakeError::ImageTooLarge { .. })
        ));
    }

    #[test]
    fn sample_lanes_fit_the_storage_binding() {
        let limits = wgpu::Limits::default();
        let pixels = 1024 * 1024;
        let lanes = sample_parallel_lanes_for_pixel_count(pixels, &limits).unwrap();
        let bytes = pixels * lanes as u64 * OUTPUT_PIXEL_BYTES;
        assert!(bytes <= limits.max_storage_buffer_binding_size as u64);
        assert!(bytes <= limits.max_buffer_size);
    }

    #[test]
    fn shadowless_dispatch_budget_does_not_reserve_shadow_traces() {
        let with_shadows = rows_per_dispatch_for(256, 128, 1024, 2, true);
        let without_shadows = rows_per_dispatch_for(256, 128, 1024, 2, false);
        assert_eq!(with_shadows, 80);
        assert_eq!(without_shadows, 128);
    }

    #[test]
    fn skewed_bvh_stays_within_the_shader_stack() {
        let triangles: Vec<_> = (0..100)
            .map(|index| {
                let x = 2.0f32.powi(index);
                Triangle {
                    v0: Vec3::new(x, 0.0, 0.0),
                    v1: Vec3::new(x, 1.0, 0.0),
                    v2: Vec3::new(x, 0.0, 1.0),
                    normal: Vec3::new(1.0, 0.0, 0.0),
                    color: Vec3::new(1.0, 1.0, 1.0),
                }
            })
            .collect();
        let (_, nodes) = build_bvh(&triangles);

        fn depth(nodes: &[GpuBvhNode], index: u32) -> u32 {
            let node = nodes[index as usize];
            if node.triangle_count > 0 {
                1
            } else {
                1 + depth(nodes, node.child_or_first).max(depth(nodes, node.child_b))
            }
        }

        assert!(depth(&nodes, 0) <= 64);
    }

    #[test]
    fn ray_tolerances_follow_uniform_geometry_scale() {
        fn mesh(scale: f32) -> Mesh {
            let bounds = xbrdf_core::Bounds {
                min: Vec3::new(-0.5, -0.25, -0.5) * scale,
                max: Vec3::new(0.5, 0.0, 0.5) * scale,
            };
            Mesh {
                triangles: Vec::new(),
                bounds,
                original_bounds: bounds,
                y_offset_to_zero: 0.0,
                tile_min_x: -0.5 * scale,
                tile_min_z: -0.5 * scale,
                tile_width: scale,
                tile_depth: scale,
                color_source: xbrdf_core::ColorSource::None,
            }
        }

        let small = ray_tolerances(&mesh(1.0e-3));
        let unit = ray_tolerances(&mesh(1.0));
        let large = ray_tolerances(&mesh(1.0e3));
        assert!((small.ray_origin * 1.0e3 - unit.ray_origin).abs() < 1.0e-8);
        assert!((large.ray_origin / 1.0e3 - unit.ray_origin).abs() < 1.0e-8);
        assert!((small.determinant * 1.0e6 - unit.determinant).abs() < 1.0e-8);
        assert!((large.determinant / 1.0e6 - unit.determinant).abs() < 1.0e-8);
    }

    #[test]
    fn one_oversized_lane_is_rejected() {
        let limits = wgpu::Limits::default();
        let pixels = limits.max_storage_buffer_binding_size as u64 / OUTPUT_PIXEL_BYTES + 1;
        assert!(matches!(
            sample_parallel_lanes_for_pixel_count(pixels, &limits),
            Err(GpuBakeError::OutputBufferTooLarge { .. })
        ));
    }
    #[test]
    fn bake_control_is_shared_and_observable() {
        let control = BakeControl::default();
        let worker = control.clone();
        assert!(!worker.is_cancelled());
        control.cancel();
        assert!(worker.is_cancelled());
        assert!(matches!(worker.check(), Err(GpuBakeError::Cancelled)));
    }
    #[test]
    #[ignore = "requires a compatible wgpu adapter"]
    fn periodic_shadowing_is_continuous_across_the_tile_seam() {
        let mut config = renderer_test_config(32_768);
        config.width = 16;
        config.height = 6;
        let centered = pollster::block_on(bake(&config, &periodic_box_mesh(false))).unwrap();
        let seam = pollster::block_on(bake(&config, &periodic_box_mesh(true))).unwrap();

        let seam_error = mean_pixel_error(&centered.pixels, &seam.pixels);
        assert!(
            seam_error < 0.006,
            "translating periodic geometry to the tile seam changed the mean shadowed response: {seam_error}"
        );

        config.enable_shadows = false;
        let unshadowed = pollster::block_on(bake(&config, &periodic_box_mesh(false))).unwrap();
        assert!(
            maximum_pixel_error(&centered.pixels, &unshadowed.pixels) > 0.03,
            "fixture does not exercise periodic shadow occlusion"
        );
    }

    #[test]
    #[ignore = "requires a compatible wgpu adapter"]
    fn uniformly_scaled_geometry_has_the_same_response() {
        let config = renderer_test_config(4_096);
        let unit_mesh = periodic_box_mesh(false);
        let unit = pollster::block_on(bake(&config, &unit_mesh)).unwrap();
        let repeated = pollster::block_on(bake(&config, &unit_mesh)).unwrap();
        assert_eq!(repeated.stats.bvh_build_time, Duration::ZERO);

        for scale in [1.0e-3, 1.0e3] {
            let scaled =
                pollster::block_on(bake(&config, &scale_mesh(unit_mesh.clone(), scale))).unwrap();
            let error = maximum_pixel_error(&unit.pixels, &scaled.pixels);
            assert!(error < 1.0e-5, "scale={scale} maximum error={error}");
        }
    }

    #[test]
    #[ignore = "requires a compatible wgpu adapter"]
    fn progressive_and_dispatch_threshold_paths_match() {
        let mesh = periodic_box_mesh(false);
        for samples in [1, 63, 64, 65, 2_047, 2_048, 2_049, 65_535, 65_536, 65_537] {
            let config = renderer_test_config(samples);
            let complete = pollster::block_on(bake(&config, &mesh)).unwrap();
            let progressive = pollster::block_on(bake_progressive(
                &config,
                &mesh,
                ProgressiveBakeOptions {
                    update_interval: Duration::from_millis(1),
                },
                |_| {},
            ))
            .unwrap();
            let error = maximum_pixel_error(&complete.pixels, &progressive.pixels);
            assert!(error < 1.0e-4, "samples={samples} maximum error={error}");
        }
    }
}
