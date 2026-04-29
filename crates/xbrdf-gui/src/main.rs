use anyhow::{Context, Result};
use exr::prelude::{read_first_rgba_layer_from_file, write_rgb_file};
use fbx::{Node, Property};
use glium::framebuffer::SimpleFrameBuffer;
use glium::index::PrimitiveType;
use glium::texture::DepthTexture2d;
use glium::texture::{ClientFormat, RawImage2d, Texture2d};
use glium::uniform;
use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter, SamplerBehavior};
use glium::Surface;
use glium::{IndexBuffer, Program, VertexBuffer};
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextAttributesBuilder, NotCurrentGlContext},
    display::{GetGlDisplay, GlDisplay},
    surface::{SurfaceAttributesBuilder, WindowSurface},
};
use imgui::{Condition, Image, MouseButton, ProgressBar, TextureId, Ui};
use imgui_glium_renderer::{Renderer, Texture};
use imgui_winit_support::winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use raw_window_handle::HasRawWindowHandle;
use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};
use xbrdf_core::{
    BakeConfigFile, BakeMode, BakeOverrides, Manifest, MaterialKind, Mesh, ResolvedBakeConfig,
    SamplerKind, Vec3,
};
use xbrdf_gpu::{AtlasProgressFrame, GpuBakeStats, ProgressiveBakeOptions, ProgressiveFrame};

const TITLE: &str = "xBRDF Bake";
const MAX_GUI_SAMPLES: i32 = 100_000_000;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
    }
}

fn run() -> Result<()> {
    let (event_loop, window, display) = create_window()?;
    let (mut platform, mut imgui) = imgui_init(&window);
    let mut renderer =
        Renderer::init(&mut imgui, &display).context("failed to initialize ImGui renderer")?;
    let mut app = AppState::new();
    let mut preview = PreviewTexture::default();
    let mut model_preview = ModelPreviewTexture::default();
    let mut last_frame = Instant::now();

    event_loop
        .run(move |event, window_target| match event {
            Event::NewEvents(_) => {
                let now = Instant::now();
                imgui.io_mut().update_delta_time(now - last_frame);
                last_frame = now;
            }
            Event::AboutToWait => {
                if let Err(error) = platform.prepare_frame(imgui.io_mut(), &window) {
                    app.status = format!("Frame preparation failed: {error}");
                }
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                app.receive_bake_events();
                if app.preview_dirty {
                    app.preview_version = app.preview_version.wrapping_add(1);
                    if let Err(error) = preview.update(&display, &mut renderer, &app.preview) {
                        app.status = format!("Preview upload failed: {error:#}");
                    }
                    app.preview_dirty = false;
                }
                if let Err(error) = model_preview.render(
                    &display,
                    &mut renderer,
                    &app.preview,
                    &app.settings,
                    app.selected_preview_model(),
                    app.preview_version,
                ) {
                    app.status = format!("3D preview failed: {error:#}");
                }

                let ui = imgui.frame();
                draw_ui(ui, &mut app, &preview, &model_preview);

                let mut target = display.draw();
                target.clear_color_srgb(0.055, 0.055, 0.06, 1.0);
                platform.prepare_render(ui, &window);
                let draw_data = imgui.render();
                if let Err(error) = renderer.render(&mut target, draw_data) {
                    app.status = format!("Render failed: {error}");
                }
                if let Err(error) = target.finish() {
                    app.status = format!("Swap failed: {error}");
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => window_target.exit(),
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                if new_size.width > 0 && new_size.height > 0 {
                    display.resize((new_size.width, new_size.height));
                }
                platform.handle_event(imgui.io_mut(), &window, &event);
            }
            event => {
                platform.handle_event(imgui.io_mut(), &window, &event);
            }
        })
        .context("event loop failed")
}

#[derive(Clone)]
struct BakeSettings {
    config_path: String,
    obj_path: String,
    out_dir: String,
    width: i32,
    height: i32,
    mode_index: usize,
    light_width: i32,
    light_height: i32,
    samples: i32,
    max_repeat_radius: i32,
    sampler_index: usize,
    enable_shadows: bool,
    light: [f32; 3],
    material_index: usize,
    material_color: [f32; 3],
    roughness: f32,
    update_interval_seconds: f32,
    preview_light: [f32; 3],
    preview_rotation: [f32; 4],
    preview_model_index: usize,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            config_path: "assets/fixtures/specular.toml".to_string(),
            obj_path: String::new(),
            out_dir: "out/gui".to_string(),
            width: 256,
            height: 64,
            mode_index: 0,
            light_width: 8,
            light_height: 4,
            samples: 64,
            max_repeat_radius: 2,
            sampler_index: 0,
            enable_shadows: true,
            light: [0.0, 1.0, -1.0],
            material_index: 0,
            material_color: [1.0, 1.0, 1.0],
            roughness: 0.05,
            update_interval_seconds: 0.5,
            preview_light: [0.0, 1.0, -1.0],
            preview_rotation: quat_from_axis_angle([0.0, 1.0, 0.0], 0.65),
            preview_model_index: 0,
        }
    }
}

impl BakeSettings {
    fn material_kind(&self) -> MaterialKind {
        match self.material_index {
            1 => MaterialKind::SpecularPhong,
            _ => MaterialKind::Lambertian,
        }
    }

    fn bake_mode(&self) -> BakeMode {
        match self.mode_index {
            1 => BakeMode::Full,
            2 => BakeMode::Isotropic,
            _ => BakeMode::Single,
        }
    }

    fn sampler_kind(&self) -> SamplerKind {
        match self.sampler_index {
            1 => SamplerKind::Random,
            _ => SamplerKind::Halton,
        }
    }
}

struct AppState {
    settings: BakeSettings,
    receiver: Option<Receiver<BakeEvent>>,
    baking: bool,
    progress: f32,
    completed_samples: u32,
    total_samples: u32,
    status: String,
    preview: PreviewImage,
    latest_bake_preview: Option<PreviewImage>,
    preview_dirty: bool,
    preview_version: u64,
    last_stats: Option<GpuBakeStats>,
    history: Vec<RenderSnapshot>,
    history_index: i32,
    follow_latest: bool,
    preview_models: Vec<PreviewModelOption>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            settings: BakeSettings::default(),
            receiver: None,
            baking: false,
            progress: 0.0,
            completed_samples: 0,
            total_samples: 0,
            status: "Idle".to_string(),
            preview: PreviewImage::default(),
            latest_bake_preview: None,
            preview_dirty: false,
            preview_version: 0,
            last_stats: None,
            history: Vec::new(),
            history_index: -1,
            follow_latest: true,
            preview_models: discover_preview_models(),
        }
    }
}

impl AppState {
    fn new() -> Self {
        let mut app = Self::default();
        match load_config_into_settings(&mut app.settings) {
            Ok(()) => app.status = "Loaded default config".to_string(),
            Err(error) => app.status = format!("Idle; default config not loaded: {error:#}"),
        }
        app
    }

    fn start_bake(&mut self) {
        if self.baking {
            return;
        }

        let settings = self.settings.clone();
        let (sender, receiver) = mpsc::sync_channel(2);
        self.receiver = Some(receiver);
        self.baking = true;
        self.progress = 0.0;
        self.completed_samples = 0;
        self.total_samples = 0;
        self.last_stats = None;
        self.latest_bake_preview = None;
        self.status = "Starting bake".to_string();
        self.follow_latest = true;

        std::thread::spawn(move || {
            let result = run_bake(settings, |event| {
                let _ = sender.send(event);
            });
            if let Err(error) = result {
                let _ = sender.send(BakeEvent::Error(format!("{error:#}")));
            }
        });
    }

    fn receive_bake_events(&mut self) {
        let Some(receiver) = self.receiver.take() else {
            return;
        };

        let mut keep_receiver = self.baking;
        while let Ok(event) = receiver.try_recv() {
            match event {
                BakeEvent::Status(status) => self.status = status,
                BakeEvent::Started { width, height } => {
                    self.preview = PreviewImage::new(
                        width,
                        height,
                        PreviewMeta::from_settings(&self.settings),
                    );
                    self.latest_bake_preview = Some(self.preview.clone());
                    self.preview_dirty = true;
                    self.total_samples = if self.settings.bake_mode() == BakeMode::Single {
                        self.settings.samples.max(1) as u32
                    } else {
                        let (_, _, _, _, light_count) = preview_extents(&self.settings);
                        light_count
                    };
                    self.completed_samples = 0;
                    self.progress = 0.0;
                    self.status = "Baking".to_string();
                }
                BakeEvent::Frame(frame) => {
                    let image = PreviewImage::from_frame(&frame, self.preview.meta);
                    if self.follow_latest || self.history.is_empty() {
                        self.preview = image.clone();
                        self.preview_dirty = true;
                    }
                    self.latest_bake_preview = Some(image);
                    self.completed_samples = frame.completed_samples;
                    self.total_samples = frame.total_samples;
                    self.progress = if frame.total_samples > 0 {
                        frame.completed_samples as f32 / frame.total_samples as f32
                    } else {
                        0.0
                    };
                    self.status = format!(
                        "Baking samples {} / {}",
                        frame.completed_samples, frame.total_samples
                    );
                }
                BakeEvent::AtlasFrame(frame) => {
                    let image = PreviewImage::from_atlas_frame(&frame, self.preview.meta);
                    if self.follow_latest || self.history.is_empty() {
                        self.preview = image.clone();
                        self.preview_dirty = true;
                    }
                    self.latest_bake_preview = Some(image);
                    self.completed_samples = frame.completed_tiles;
                    self.total_samples = frame.total_tiles;
                    self.progress = if frame.total_tiles > 0 {
                        frame.completed_tiles as f32 / frame.total_tiles as f32
                    } else {
                        0.0
                    };
                    self.status = format!(
                        "Baking atlas tile {} / {}",
                        frame.completed_tiles, frame.total_tiles
                    );
                }
                BakeEvent::Finished {
                    stats,
                    image: image_path,
                    manifest,
                } => {
                    self.baking = false;
                    keep_receiver = false;
                    self.progress = 1.0;
                    self.completed_samples = self.total_samples;
                    self.last_stats = Some(stats);
                    let label = if self.settings.bake_mode() == BakeMode::Single {
                        format!("{} samples", self.total_samples)
                    } else {
                        format!("{} light tiles", self.total_samples)
                    };
                    let history_image = self
                        .latest_bake_preview
                        .take()
                        .unwrap_or_else(|| self.preview.clone());
                    self.push_history(RenderSnapshot {
                        image: history_image,
                        label,
                    });
                    self.status =
                        format!("Wrote {} and {}", image_path.display(), manifest.display());
                }
                BakeEvent::Error(error) => {
                    self.baking = false;
                    keep_receiver = false;
                    self.latest_bake_preview = None;
                    self.status = error;
                }
            }
        }

        if keep_receiver {
            self.receiver = Some(receiver);
        }
    }

    fn push_history(&mut self, snapshot: RenderSnapshot) {
        self.history.push(snapshot);
        if self.follow_latest || self.history_index < 0 {
            self.history_index = self.history.len() as i32 - 1;
            self.preview = self.history[self.history_index as usize].image.clone();
            self.preview_dirty = true;
        }
    }

    fn select_history(&mut self, index: i32) {
        if self.history.is_empty() {
            self.history_index = -1;
            return;
        }

        let index = index.clamp(0, self.history.len() as i32 - 1);
        self.history_index = index;
        self.follow_latest = index == self.history.len() as i32 - 1;
        self.preview = self.history[index as usize].image.clone();
        self.preview_dirty = true;
    }

    fn selected_preview_model(&self) -> &PreviewModelOption {
        let index = self
            .settings
            .preview_model_index
            .min(self.preview_models.len().saturating_sub(1));
        &self.preview_models[index]
    }

    fn save_current_atlas(&mut self) {
        let out_dir = PathBuf::from(self.settings.out_dir.trim());
        if out_dir.as_os_str().is_empty() {
            self.status = "Output directory is empty".to_string();
            return;
        }

        let result = (|| -> Result<PathBuf> {
            std::fs::create_dir_all(&out_dir).with_context(|| {
                format!("failed to create output directory {}", out_dir.display())
            })?;
            let path = out_dir.join("xbrdf_current_atlas.exr");
            self.preview.save_exr(&path)?;
            Ok(path)
        })();

        match result {
            Ok(path) => {
                self.status = format!("Saved current atlas {}", path.display());
            }
            Err(error) => {
                self.status = format!("{error:#}");
            }
        }
    }

    fn load_current_atlas(&mut self) {
        let path = PathBuf::from(self.settings.out_dir.trim()).join("xbrdf_current_atlas.exr");
        let result = PreviewImage::load_exr(&path, PreviewMeta::from_settings(&self.settings));

        match result {
            Ok(image) => {
                self.preview = image.clone();
                self.preview_dirty = true;
                self.push_history(RenderSnapshot {
                    image,
                    label: "loaded atlas".to_string(),
                });
                self.status = format!("Loaded current atlas {}", path.display());
            }
            Err(error) => {
                self.status = format!("{error:#}");
            }
        }
    }
}

#[derive(Clone)]
struct PreviewModelOption {
    label: String,
    source: PreviewModelSource,
}

#[derive(Clone)]
enum PreviewModelSource {
    Torus,
    Fbx(PathBuf),
}

enum BakeEvent {
    Status(String),
    Started {
        width: u32,
        height: u32,
    },
    Frame(ProgressiveFrame),
    AtlasFrame(AtlasProgressFrame),
    Finished {
        stats: GpuBakeStats,
        image: PathBuf,
        manifest: PathBuf,
    },
    Error(String),
}

struct RenderSnapshot {
    image: PreviewImage,
    label: String,
}

#[derive(Clone, Default)]
struct PreviewImage {
    width: u32,
    height: u32,
    rgb: Vec<[f32; 3]>,
    rgba: Vec<u8>,
    exposure: f32,
    meta: PreviewMeta,
}

impl PreviewImage {
    fn new(width: u32, height: u32, meta: PreviewMeta) -> Self {
        Self {
            width,
            height,
            rgb: vec![[0.0; 3]; width as usize * height as usize],
            rgba: vec![0; width as usize * height as usize * 4],
            exposure: 0.0,
            meta,
        }
    }

    fn from_frame(frame: &ProgressiveFrame, meta: PreviewMeta) -> Self {
        let mut image = Self::new(frame.width, frame.height, meta);
        image.write_frame(frame);
        image
    }

    fn from_atlas_frame(frame: &AtlasProgressFrame, meta: PreviewMeta) -> Self {
        let mut image = Self::new(frame.width, frame.height, meta);
        image.write_pixels(&frame.pixels);
        image
    }

    fn write_frame(&mut self, frame: &ProgressiveFrame) {
        if self.width != frame.width || self.height != frame.height || self.height == 0 {
            return;
        }

        self.write_pixels(&frame.pixels);
    }

    fn write_pixels(&mut self, pixels: &[[f32; 3]]) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        let gain = 2.0_f32.powf(self.exposure);
        for (index, src) in pixels.iter().enumerate() {
            if index * 4 + 3 >= self.rgba.len() {
                break;
            }
            if index < self.rgb.len() {
                self.rgb[index] = *src;
            }
            let dst = index * 4;
            self.rgba[dst] = to_preview_byte(src[0] * gain);
            self.rgba[dst + 1] = to_preview_byte(src[1] * gain);
            self.rgba[dst + 2] = to_preview_byte(src[2] * gain);
            self.rgba[dst + 3] = 255;
        }
    }

    fn save_exr(&self, path: &Path) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.rgb.is_empty() {
            anyhow::bail!("no atlas is selected");
        }

        write_rgb_file(path, self.width as usize, self.height as usize, |x, y| {
            let pixel = self.rgb[y * self.width as usize + x];
            (pixel[0], pixel[1], pixel[2])
        })
        .with_context(|| format!("failed to write EXR {}", path.display()))
    }

    fn load_exr(path: &Path, meta: PreviewMeta) -> Result<Self> {
        let image = read_first_rgba_layer_from_file(
            path,
            |resolution, _channels| {
                (
                    resolution.0 as u32,
                    resolution.1 as u32,
                    vec![[0.0_f32; 3]; resolution.0 * resolution.1],
                )
            },
            |pixels, position, (r, g, b, _a): (f32, f32, f32, f32)| {
                let width = pixels.0 as usize;
                pixels.2[position.1 * width + position.0] = [r, g, b];
            },
        )
        .with_context(|| format!("failed to read EXR {}", path.display()))?;

        let (width, height, rgb) = image.layer_data.channel_data.pixels;
        let mut preview = Self::new(width, height, meta);
        preview.write_pixels(&rgb);
        Ok(preview)
    }
}

#[derive(Clone, Copy)]
struct PreviewMeta {
    mode: BakeMode,
    camera_width: u32,
    camera_height: u32,
    light_width: u32,
    light_height: u32,
}

impl Default for PreviewMeta {
    fn default() -> Self {
        Self {
            mode: BakeMode::Single,
            camera_width: 1,
            camera_height: 1,
            light_width: 1,
            light_height: 1,
        }
    }
}

impl PreviewMeta {
    fn from_settings(settings: &BakeSettings) -> Self {
        let (_, _, tile_width, tile_height, _) = preview_extents(settings);
        Self {
            mode: settings.bake_mode(),
            camera_width: tile_width.max(1),
            camera_height: tile_height.max(1),
            light_width: match settings.bake_mode() {
                BakeMode::Single => 1,
                BakeMode::Full | BakeMode::Isotropic => settings.light_width.max(1) as u32,
            },
            light_height: match settings.bake_mode() {
                BakeMode::Single => 1,
                BakeMode::Full | BakeMode::Isotropic => settings.light_height.max(1) as u32,
            },
        }
    }

    fn mode_code(self) -> i32 {
        match self.mode {
            BakeMode::Single => 0,
            BakeMode::Full => 1,
            BakeMode::Isotropic => 2,
        }
    }
}

#[derive(Default)]
struct PreviewTexture {
    id: Option<TextureId>,
    width: u32,
    height: u32,
}

impl PreviewTexture {
    fn update(
        &mut self,
        display: &glium::Display<WindowSurface>,
        renderer: &mut Renderer,
        preview: &PreviewImage,
    ) -> Result<()> {
        if preview.width == 0 || preview.height == 0 || preview.rgba.is_empty() {
            return Ok(());
        }

        let raw = RawImage2d {
            data: Cow::Borrowed(preview.rgba.as_slice()),
            width: preview.width,
            height: preview.height,
            format: ClientFormat::U8U8U8U8,
        };
        let texture = Texture2d::new(display, raw).context("failed to create preview texture")?;
        let texture = Texture {
            texture: Rc::new(texture),
            sampler: SamplerBehavior {
                minify_filter: MinifySamplerFilter::Linear,
                magnify_filter: MagnifySamplerFilter::Linear,
                ..Default::default()
            },
        };

        if let Some(id) = self.id {
            renderer.textures().replace(id, texture);
        } else {
            self.id = Some(renderer.textures().insert(texture));
        }
        self.width = preview.width;
        self.height = preview.height;
        Ok(())
    }
}

#[derive(Copy, Clone)]
struct ModelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

glium::implement_vertex!(ModelVertex, position, normal, tangent, bitangent);

struct ModelPreviewTexture {
    id: Option<TextureId>,
    color: Option<Rc<Texture2d>>,
    depth: Option<DepthTexture2d>,
    pano: Option<Texture2d>,
    source_version: u64,
    program: Option<Program>,
    vertices: Option<VertexBuffer<ModelVertex>>,
    indices: Option<IndexBuffer<u32>>,
    model_key: Option<String>,
    size: u32,
}

impl Default for ModelPreviewTexture {
    fn default() -> Self {
        Self {
            id: None,
            color: None,
            depth: None,
            pano: None,
            source_version: u64::MAX,
            program: None,
            vertices: None,
            indices: None,
            model_key: None,
            size: 512,
        }
    }
}

impl ModelPreviewTexture {
    fn render(
        &mut self,
        display: &glium::Display<WindowSurface>,
        renderer: &mut Renderer,
        preview: &PreviewImage,
        settings: &BakeSettings,
        model: &PreviewModelOption,
        preview_version: u64,
    ) -> Result<()> {
        if preview.width == 0 || preview.height == 0 || preview.rgba.is_empty() {
            return Ok(());
        }

        self.ensure_resources(display, renderer, model)?;
        if self.source_version != preview_version {
            let raw = RawImage2d {
                data: Cow::Borrowed(preview.rgba.as_slice()),
                width: preview.width,
                height: preview.height,
                format: ClientFormat::U8U8U8U8,
            };
            self.pano = Some(
                Texture2d::new(display, raw).context("failed to upload xBRDF preview texture")?,
            );
            self.source_version = preview_version;
        }

        let color = self
            .color
            .as_deref()
            .context("missing 3D preview color texture")?;
        let depth = self
            .depth
            .as_ref()
            .context("missing 3D preview depth texture")?;
        let pano = self
            .pano
            .as_ref()
            .context("missing xBRDF preview texture")?;
        let program = self.program.as_ref().context("missing 3D preview shader")?;
        let vertices = self
            .vertices
            .as_ref()
            .context("missing 3D preview vertices")?;
        let indices = self
            .indices
            .as_ref()
            .context("missing 3D preview indices")?;

        let mut target = SimpleFrameBuffer::with_depth_buffer(display, color, depth)
            .context("failed to create 3D preview framebuffer")?;
        target.clear_color_and_depth((0.06, 0.065, 0.07, 1.0), 1.0);

        let meta = preview.meta;
        let light = normalize3(settings.preview_light);
        let camera_pos = [0.0_f32, 0.6, 3.2];
        let uniforms = uniform! {
            model: quat_to_mat4(settings.preview_rotation),
            view: look_at(camera_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            projection: perspective(35.0_f32.to_radians(), 1.0, 0.05, 20.0),
            camera_pos: camera_pos,
            preview_light: light,
            xbrdf_tex: pano.sampled()
                .magnify_filter(MagnifySamplerFilter::Nearest)
                .minify_filter(MinifySamplerFilter::Nearest),
            mode: meta.mode_code(),
            camera_width: meta.camera_width as i32,
            camera_height: meta.camera_height as i32,
            light_width: meta.light_width as i32,
            light_height: meta.light_height as i32,
        };

        let draw_params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };

        target
            .draw(vertices, indices, program, &uniforms, &draw_params)
            .context("failed to draw 3D preview")?;
        Ok(())
    }

    fn ensure_resources(
        &mut self,
        display: &glium::Display<WindowSurface>,
        renderer: &mut Renderer,
        model: &PreviewModelOption,
    ) -> Result<()> {
        if self.color.is_none() {
            let color = Rc::new(
                Texture2d::empty(display, self.size, self.size)
                    .context("failed to create 3D preview color texture")?,
            );
            let renderer_texture = Texture {
                texture: color.clone(),
                sampler: SamplerBehavior {
                    minify_filter: MinifySamplerFilter::Linear,
                    magnify_filter: MagnifySamplerFilter::Linear,
                    ..Default::default()
                },
            };
            self.id = Some(renderer.textures().insert(renderer_texture));
            self.color = Some(color);
        }
        if self.depth.is_none() {
            self.depth = Some(
                DepthTexture2d::empty(display, self.size, self.size)
                    .context("failed to create 3D preview depth texture")?,
            );
        }
        if self.program.is_none() {
            self.program = Some(
                Program::from_source(display, MODEL_VERTEX_SHADER, MODEL_FRAGMENT_SHADER, None)
                    .context("failed to compile 3D preview shader")?,
            );
        }
        let model_key = model.key();
        if self.model_key.as_deref() != Some(model_key.as_str()) {
            self.vertices = None;
            self.indices = None;
            self.model_key = None;
        }
        if self.vertices.is_none() || self.indices.is_none() {
            let (vertices, indices) = load_preview_model(model)?;
            self.vertices = Some(
                VertexBuffer::new(display, &vertices)
                    .context("failed to create 3D preview vertex buffer")?,
            );
            self.indices = Some(
                IndexBuffer::new(display, PrimitiveType::TrianglesList, &indices)
                    .context("failed to create 3D preview index buffer")?,
            );
            self.model_key = Some(model_key);
        }
        Ok(())
    }
}

impl PreviewModelOption {
    fn key(&self) -> String {
        match &self.source {
            PreviewModelSource::Torus => "synthetic:torus".to_string(),
            PreviewModelSource::Fbx(path) => path.to_string_lossy().to_string(),
        }
    }
}

const MODEL_VERTEX_SHADER: &str = r#"
#version 330 core
in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec3 bitangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;

void main() {
    vec4 world = model * vec4(position, 1.0);
    mat3 basis = mat3(model);
    v_position = world.xyz;
    v_normal = normalize(basis * normal);
    v_tangent = normalize(basis * tangent);
    v_bitangent = normalize(basis * bitangent);
    gl_Position = projection * view * world;
}
"#;

const MODEL_FRAGMENT_SHADER: &str = r#"
#version 330 core
const float PI = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;
const float HALF_PI = 1.57079632679489661923;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;

uniform sampler2D xbrdf_tex;
uniform vec3 camera_pos;
uniform vec3 preview_light;
uniform int mode;
uniform int camera_width;
uniform int camera_height;
uniform int light_width;
uniform int light_height;

out vec4 color;

vec2 dir_to_latlong(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.x, dir.z) / TAU + 0.5;
    float v = 1.0 - asin(clamp(dir.y, 0.0, 1.0)) / HALF_PI;
    return vec2(fract(u), clamp(v, 0.0, 1.0));
}

vec4 fetch_atlas(ivec2 p) {
    ivec2 size = textureSize(xbrdf_tex, 0);
    p.x = ((p.x % size.x) + size.x) % size.x;
    p.y = clamp(p.y, 0, size.y - 1);
    return texelFetch(xbrdf_tex, p, 0);
}

vec4 sample_camera_tile(int light_x, int light_y, vec2 camera_uv) {
    float fx = camera_uv.x * float(camera_width) - 0.5;
    float fy = camera_uv.y * float(camera_height) - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float tx = fract(fx);
    float ty = fract(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int wx0 = ((x0 % camera_width) + camera_width) % camera_width;
    int wx1 = ((x1 % camera_width) + camera_width) % camera_width;
    y0 = clamp(y0, 0, camera_height - 1);
    y1 = clamp(y1, 0, camera_height - 1);

    ivec2 base = ivec2(light_x * camera_width, light_y * camera_height);
    vec4 a = fetch_atlas(base + ivec2(wx0, y0));
    vec4 b = fetch_atlas(base + ivec2(wx1, y0));
    vec4 c = fetch_atlas(base + ivec2(wx0, y1));
    vec4 d = fetch_atlas(base + ivec2(wx1, y1));
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

vec4 sample_iso_tile(int light_x, int light_y, float camera_v) {
    float fy = camera_v * float(camera_height) - 0.5;
    int y0 = clamp(int(floor(fy)), 0, camera_height - 1);
    int y1 = clamp(y0 + 1, 0, camera_height - 1);
    float ty = fract(fy);
    int x = ((light_x % light_width) + light_width) % light_width;
    int base_y = light_y * camera_height;
    return mix(fetch_atlas(ivec2(x, base_y + y0)), fetch_atlas(ivec2(x, base_y + y1)), ty);
}

vec4 sample_light_grid(vec3 light_dir, vec2 camera_uv, bool isotropic) {
    vec2 light_uv = dir_to_latlong(light_dir);
    float gx = light_uv.x * float(light_width) - 0.5;
    float gy = light_uv.y * float(light_height) - 0.5;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    float tx = fract(gx);
    float ty = fract(gy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    x0 = ((x0 % light_width) + light_width) % light_width;
    x1 = ((x1 % light_width) + light_width) % light_width;
    y0 = clamp(y0, 0, light_height - 1);
    y1 = clamp(y1, 0, light_height - 1);

    vec4 a = isotropic ? sample_iso_tile(x0, y0, camera_uv.y) : sample_camera_tile(x0, y0, camera_uv);
    vec4 b = isotropic ? sample_iso_tile(x1, y0, camera_uv.y) : sample_camera_tile(x1, y0, camera_uv);
    vec4 c = isotropic ? sample_iso_tile(x0, y1, camera_uv.y) : sample_camera_tile(x0, y1, camera_uv);
    vec4 d = isotropic ? sample_iso_tile(x1, y1, camera_uv.y) : sample_camera_tile(x1, y1, camera_uv);
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

vec3 stable_perpendicular(vec3 n) {
    vec3 helper = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    return normalize(cross(helper, n));
}

void main() {
    vec3 n = normalize(v_normal);
    vec3 t = normalize(v_tangent);
    vec3 b = normalize(v_bitangent);
    vec3 wo_world = normalize(camera_pos - v_position);
    vec3 wi_world = normalize(preview_light);

    if (mode == 2) {
        float wo_y = dot(wo_world, n);
        float wi_y = dot(wi_world, n);
        if (wo_y <= 0.0 || wi_y <= 0.0) {
            color = vec4(0.015, 0.015, 0.017, 1.0);
            return;
        }

        vec3 view_projected = wo_world - n * wo_y;
        vec3 iso_z = dot(view_projected, view_projected) > 1.0e-8
            ? normalize(view_projected)
            : stable_perpendicular(n);
        vec3 iso_x = normalize(cross(iso_z, n));
        vec3 wi = normalize(vec3(dot(wi_world, iso_x), wi_y, dot(wi_world, iso_z)));

        vec2 camera_uv = vec2(0.5, 1.0 - asin(clamp(wo_y, 0.0, 1.0)) / HALF_PI);
        vec4 response = sample_light_grid(wi, camera_uv, true);
        float rim = 0.08 * pow(1.0 - max(dot(n, wo_world), 0.0), 2.0);
        color = vec4(response.rgb * max(wi_y, 0.0) + vec3(rim), 1.0);
        return;
    }

    vec3 wo = normalize(vec3(dot(wo_world, t), dot(wo_world, n), dot(wo_world, b)));
    vec3 wi = normalize(vec3(dot(wi_world, t), dot(wi_world, n), dot(wi_world, b)));

    if (wo.y <= 0.0 || wi.y <= 0.0) {
        color = vec4(0.015, 0.015, 0.017, 1.0);
        return;
    }

    vec2 camera_uv = dir_to_latlong(wo);
    vec4 response;
    if (mode == 1) {
        response = sample_light_grid(wi, camera_uv, false);
    } else {
        response = sample_camera_tile(0, 0, camera_uv);
    }

    float macro_cosine = max(wi.y, 0.0);
    float rim = 0.08 * pow(1.0 - max(dot(n, wo_world), 0.0), 2.0);
    color = vec4(response.rgb * macro_cosine + vec3(rim), 1.0);
}
"#;

fn draw_ui(
    ui: &Ui,
    app: &mut AppState,
    preview: &PreviewTexture,
    model_preview: &ModelPreviewTexture,
) {
    ui.window("Bake Settings")
        .size([390.0, 620.0], Condition::FirstUseEver)
        .position([12.0, 12.0], Condition::FirstUseEver)
        .build(|| {
            ui.input_text("Config", &mut app.settings.config_path)
                .build();
            if ui.button("Load Config") {
                match load_config_into_settings(&mut app.settings) {
                    Ok(()) => app.status = "Loaded config".to_string(),
                    Err(error) => app.status = format!("{error:#}"),
                }
            }
            ui.input_text("Geometry", &mut app.settings.obj_path)
                .build();
            ui.input_text("Output", &mut app.settings.out_dir).build();
            ui.separator();

            ui.input_int("Width", &mut app.settings.width).build();
            ui.input_int("Height", &mut app.settings.height).build();
            ui.combo_simple_string(
                "Mode",
                &mut app.settings.mode_index,
                &["Single", "Full", "Isotropic"],
            );
            ui.input_int("Light width", &mut app.settings.light_width)
                .build();
            ui.input_int("Light height", &mut app.settings.light_height)
                .build();
            ui.input_int("Samples", &mut app.settings.samples).build();
            ui.input_float("Update sec", &mut app.settings.update_interval_seconds)
                .build();
            ui.input_int("Repeat radius", &mut app.settings.max_repeat_radius)
                .build();
            ui.combo_simple_string(
                "Sampler",
                &mut app.settings.sampler_index,
                &["Halton", "Random"],
            );
            ui.checkbox("Shadows", &mut app.settings.enable_shadows);
            app.settings.width = app.settings.width.clamp(1, 16_384);
            app.settings.height = app.settings.height.clamp(1, 16_384);
            app.settings.light_width = app.settings.light_width.clamp(1, 16_384);
            app.settings.light_height = app.settings.light_height.clamp(1, 16_384);
            app.settings.samples = app.settings.samples.clamp(1, MAX_GUI_SAMPLES);
            app.settings.update_interval_seconds =
                app.settings.update_interval_seconds.clamp(0.05, 60.0);
            app.settings.max_repeat_radius = app.settings.max_repeat_radius.clamp(0, 16);
            let (atlas_width, atlas_height, tile_width, tile_height, light_count) =
                preview_extents(&app.settings);
            ui.text(format!(
                "Output {}x{}; tile {}x{}; {} light directions",
                atlas_width, atlas_height, tile_width, tile_height, light_count
            ));

            ui.input_float3("Light", &mut app.settings.light).build();
            let labels: Vec<_> = app
                .preview_models
                .iter()
                .map(|model| model.label.as_str())
                .collect();
            ui.combo_simple_string(
                "Preview model",
                &mut app.settings.preview_model_index,
                &labels,
            );
            app.settings.preview_model_index = app
                .settings
                .preview_model_index
                .min(app.preview_models.len().saturating_sub(1));
            ui.separator();

            ui.combo_simple_string(
                "Material",
                &mut app.settings.material_index,
                &["Lambertian", "Specular Phong"],
            );
            ui.color_edit3("Color", &mut app.settings.material_color);
            ui.input_float("Roughness", &mut app.settings.roughness)
                .build();
            app.settings.roughness = app.settings.roughness.clamp(0.0, 1.0);

            ui.separator();
            if app.baking {
                ui.text("Bake in progress");
            } else if ui.button("Start Bake") {
                app.start_bake();
            }
            ProgressBar::new(app.progress)
                .size([-1.0, 0.0])
                .overlay_text(progress_label(app))
                .build(ui);
            ui.text_wrapped(&app.status);
            if let Some(stats) = &app.last_stats {
                ui.separator();
                ui.text(format!(
                    "{} triangles, {} BVH nodes",
                    stats.triangle_count, stats.bvh_node_count
                ));
                ui.text(format!(
                    "GPU dispatch {}, throughput {:.0} rays/s",
                    format_duration(stats.gpu_dispatch_time),
                    rays_per_second(stats)
                ));
            }
            if ui.button("Save Current Atlas") {
                app.save_current_atlas();
            }
            ui.same_line();
            if ui.button("Load Current Atlas") {
                app.load_current_atlas();
            }
            if ui.button("Clear History") {
                app.history.clear();
                app.history_index = -1;
                app.follow_latest = true;
                app.preview = PreviewImage::default();
                app.preview_dirty = true;
            }
        });

    ui.window("Viewport")
        .position([414.0, 12.0], Condition::FirstUseEver)
        .size([780.0, 420.0], Condition::FirstUseEver)
        .build(|| {
            let available = ui.content_region_avail();
            if let Some(id) = preview.id {
                let image_aspect = preview.width as f32 / preview.height.max(1) as f32;
                let available_width = available[0].max(32.0);
                let available_height = available[1].max(32.0);
                let mut width = available_width;
                let mut height = width / image_aspect;
                if height > available_height {
                    height = available_height;
                    width = height * image_aspect;
                }
                Image::new(id, [width, height]).build(ui);
                draw_history_controls(ui, app, available_width);
            } else {
                ui.text("No preview yet");
            }
        });

    ui.window("3D Preview")
        .position([414.0, 450.0], Condition::FirstUseEver)
        .size([420.0, 320.0], Condition::FirstUseEver)
        .build(|| {
            ui.set_next_item_width(-1.0);
            ui.slider_config("Light", -1.0, 1.0)
                .display_format("%.3f")
                .build_array(&mut app.settings.preview_light);
            let available = ui.content_region_avail();
            if let Some(id) = model_preview.id {
                let size = available[0].min(available[1]).max(32.0);
                draw_draggable_model_preview(ui, app, id, size);
            } else {
                ui.text("No 3D preview yet");
            }
        });
}

fn draw_draggable_model_preview(ui: &Ui, app: &mut AppState, id: TextureId, size: f32) {
    ui.invisible_button("##model_preview_drag", [size, size]);
    let min = ui.item_rect_min();
    let max = ui.item_rect_max();
    ui.get_window_draw_list()
        .add_image(id, min, max)
        .uv_min([0.0, 1.0])
        .uv_max([1.0, 0.0])
        .build();

    if ui.is_item_hovered() && ui.is_mouse_dragging(MouseButton::Left) {
        let delta = ui.io().mouse_delta;
        let yaw = quat_from_axis_angle([0.0, 1.0, 0.0], delta[0] * 0.01);
        let pitch = quat_from_axis_angle([1.0, 0.0, 0.0], delta[1] * 0.01);
        app.settings.preview_rotation = normalize_quat(quat_mul(
            pitch,
            quat_mul(yaw, app.settings.preview_rotation),
        ));
    }

    if ui.is_item_hovered() && ui.is_mouse_dragging(MouseButton::Right) {
        let delta = ui.io().mouse_delta;
        let roll = quat_from_axis_angle([0.0, 0.0, 1.0], delta[0] * 0.01);
        app.settings.preview_rotation =
            normalize_quat(quat_mul(roll, app.settings.preview_rotation));
    }

    if ui.is_item_hovered() && ui.is_mouse_double_clicked(MouseButton::Left) {
        app.settings.preview_rotation = quat_from_axis_angle([0.0, 1.0, 0.0], 0.65);
    }
}

fn progress_label(app: &AppState) -> String {
    if app.settings.bake_mode() == BakeMode::Single {
        format!("{} / {} samples", app.completed_samples, app.total_samples)
    } else {
        format!("{} / {} tiles", app.completed_samples, app.total_samples)
    }
}

fn draw_history_controls(ui: &Ui, app: &mut AppState, width: f32) {
    if app.history.is_empty() {
        return;
    }

    let max_index = app.history.len() as i32 - 1;
    let mut index = app.history_index.clamp(0, max_index);
    ui.set_next_item_width(width);
    let changed = ui
        .slider_config("##render_history", 0, max_index)
        .display_format("")
        .build(&mut index);
    let slider_min = ui.item_rect_min();
    let slider_max = ui.item_rect_max();
    draw_history_ticks(
        ui,
        slider_min,
        slider_max,
        app.history.len(),
        index as usize,
    );
    ui.dummy([0.0, 9.0]);

    if changed {
        app.select_history(index);
    }

    let label = &app.history[app.history_index.max(0) as usize].label;
    ui.text(format!(
        "Render {} / {} - {}",
        app.history_index + 1,
        app.history.len(),
        label
    ));
}

fn draw_history_ticks(
    ui: &Ui,
    slider_min: [f32; 2],
    slider_max: [f32; 2],
    count: usize,
    selected: usize,
) {
    if count == 0 {
        return;
    }

    let draw_list = ui.get_window_draw_list();
    let usable_width = (slider_max[0] - slider_min[0]).max(1.0);
    let y0 = slider_max[1] + 2.0;
    let max_ticks = 96usize;
    let step = ((count + max_ticks - 1) / max_ticks).max(1);

    for index in (0..count).step_by(step) {
        let t = if count > 1 {
            index as f32 / (count - 1) as f32
        } else {
            0.0
        };
        let x = slider_min[0] + t * usable_width;
        draw_list
            .add_line([x, y0], [x, y0 + 5.0], [0.45, 0.45, 0.48, 1.0])
            .build();
    }

    if count > 1 {
        let t = selected as f32 / (count - 1) as f32;
        let x = slider_min[0] + t * usable_width;
        draw_list
            .add_line([x, y0], [x, y0 + 8.0], [0.95, 0.95, 0.85, 1.0])
            .thickness(2.0)
            .build();
    }
}

fn run_bake<F>(settings: BakeSettings, mut send: F) -> Result<()>
where
    F: FnMut(BakeEvent),
{
    send(BakeEvent::Status("Resolving settings".to_string()));
    let config = resolve_settings(&settings)?;
    let out_dir = PathBuf::from(settings.out_dir.trim());
    if out_dir.as_os_str().is_empty() {
        anyhow::bail!("output directory is empty");
    }

    send(BakeEvent::Status("Loading geometry".to_string()));
    let mesh = Mesh::load(&config.obj)
        .with_context(|| format!("failed to load geometry {}", config.obj.display()))?;

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create output directory {}", out_dir.display()))?;

    send(BakeEvent::Started {
        width: config.atlas_width(),
        height: config.atlas_height(),
    });

    let result = if config.light_count() == 1 && config.mode == BakeMode::Single {
        pollster::block_on(xbrdf_gpu::bake_progressive(
            &config,
            &mesh,
            ProgressiveBakeOptions {
                update_interval: Duration::from_secs_f32(
                    settings.update_interval_seconds.max(0.05),
                ),
            },
            |frame| {
                send(BakeEvent::Frame(frame));
            },
        ))
        .context("GPU bake failed")?
    } else {
        send(BakeEvent::Status("Baking atlas".to_string()));
        let result = pollster::block_on(xbrdf_gpu::bake_atlas_with_progress(
            &config,
            &mesh,
            |frame| {
                send(BakeEvent::AtlasFrame(frame));
            },
        ))
        .context("GPU atlas bake failed")?;
        result
    };

    send(BakeEvent::Status("Writing output".to_string()));
    let image_path = out_dir.join("xbrdf_view.exr");
    write_rgb_file(
        &image_path,
        config.atlas_width() as usize,
        config.atlas_height() as usize,
        |x, y| {
            let pixel = result.pixels[y * config.atlas_width() as usize + x];
            (pixel[0], pixel[1], pixel[2])
        },
    )
    .with_context(|| format!("failed to write EXR {}", image_path.display()))?;

    let manifest = Manifest::new(&config, &mesh);
    let manifest_path = out_dir.join("manifest.toml");
    let manifest_text = toml::to_string_pretty(&manifest).context("failed to encode manifest")?;
    std::fs::write(&manifest_path, manifest_text)
        .with_context(|| format!("failed to write manifest {}", manifest_path.display()))?;

    send(BakeEvent::Finished {
        stats: result.stats,
        image: image_path,
        manifest: manifest_path,
    });
    Ok(())
}

fn resolve_settings(settings: &BakeSettings) -> Result<ResolvedBakeConfig> {
    let config_path = trimmed_path(&settings.config_path);
    let file_config = if let Some(path) = config_path.as_deref() {
        BakeConfigFile::read(path)
            .with_context(|| format!("failed to read config {}", path.display()))?
    } else {
        BakeConfigFile::default()
    };

    file_config
        .resolve(
            config_path.as_deref(),
            BakeOverrides {
                obj: trimmed_path(&settings.obj_path).map(resolve_gui_override_path),
                width: Some(settings.width.max(1) as u32),
                height: Some(settings.height.max(1) as u32),
                mode: Some(settings.bake_mode()),
                light_width: Some(settings.light_width.max(1) as u32),
                light_height: Some(settings.light_height.max(1) as u32),
                samples: Some(settings.samples.max(1) as u32),
                light: Some(settings.light),
                max_repeat_radius: Some(settings.max_repeat_radius.clamp(0, 16) as u32),
                sampler: Some(settings.sampler_kind()),
                enable_shadows: Some(settings.enable_shadows),
                material_kind: Some(settings.material_kind()),
                material_color: Some(settings.material_color),
                material_roughness: Some(settings.roughness),
            },
        )
        .context("failed to resolve bake settings")
}

fn preview_extents(settings: &BakeSettings) -> (u32, u32, u32, u32, u32) {
    let camera_width = settings.width.max(1) as u32;
    let camera_height = settings.height.max(1) as u32;
    let light_width = settings.light_width.max(1) as u32;
    let light_height = settings.light_height.max(1) as u32;

    match settings.bake_mode() {
        BakeMode::Single => (camera_width, camera_height, camera_width, camera_height, 1),
        BakeMode::Full => (
            camera_width * light_width,
            camera_height * light_height,
            camera_width,
            camera_height,
            light_width * light_height,
        ),
        BakeMode::Isotropic => (
            light_width,
            camera_height * light_height,
            1,
            camera_height,
            light_width * light_height,
        ),
    }
}

fn discover_preview_models() -> Vec<PreviewModelOption> {
    let mut models = vec![PreviewModelOption {
        label: "Torus".to_string(),
        source: PreviewModelSource::Torus,
    }];

    let preview_dir = Path::new("assets/preview");
    if let Ok(entries) = std::fs::read_dir(preview_dir) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("fbx"))
            })
            .collect();
        paths.sort();
        for path in paths {
            let label = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("FBX")
                .to_string();
            models.push(PreviewModelOption {
                label,
                source: PreviewModelSource::Fbx(path),
            });
        }
    }

    models
}

fn load_preview_model(model: &PreviewModelOption) -> Result<(Vec<ModelVertex>, Vec<u32>)> {
    match &model.source {
        PreviewModelSource::Torus => Ok(build_torus(64, 24, 0.86, 0.28)),
        PreviewModelSource::Fbx(path) => load_preview_fbx(path)
            .with_context(|| format!("failed to load preview model {}", path.display())),
    }
}

fn build_torus(
    major_segments: u32,
    minor_segments: u32,
    major_radius: f32,
    minor_radius: f32,
) -> (Vec<ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity((major_segments * minor_segments) as usize);
    let mut indices = Vec::with_capacity((major_segments * minor_segments * 6) as usize);

    for i in 0..major_segments {
        let u = i as f32 / major_segments as f32 * std::f32::consts::TAU;
        let (su, cu) = u.sin_cos();
        for j in 0..minor_segments {
            let v = j as f32 / minor_segments as f32 * std::f32::consts::TAU;
            let (sv, cv) = v.sin_cos();
            let ring = major_radius + minor_radius * cv;
            let normal = normalize3([cv * su, sv, cv * cu]);
            let tangent = normalize3([cu, 0.0, -su]);
            let bitangent = cross3(normal, tangent);
            vertices.push(ModelVertex {
                position: [ring * su, minor_radius * sv, ring * cu],
                normal,
                tangent,
                bitangent,
            });
        }
    }

    for i in 0..major_segments {
        let ni = (i + 1) % major_segments;
        for j in 0..minor_segments {
            let nj = (j + 1) % minor_segments;
            let a = i * minor_segments + j;
            let b = ni * minor_segments + j;
            let c = ni * minor_segments + nj;
            let d = i * minor_segments + nj;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    (vertices, indices)
}

fn load_preview_fbx(path: &Path) -> Result<(Vec<ModelVertex>, Vec<u32>)> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let fbx = fbx::File::read_from(reader)?;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for geometry in fbx.children.iter().flat_map(geometry_nodes) {
        parse_preview_fbx_geometry(geometry, &mut vertices, &mut indices);
    }

    if vertices.is_empty() || indices.is_empty() {
        anyhow::bail!("FBX contains no previewable triangle mesh");
    }

    smooth_preview_frames(&mut vertices);
    fit_preview_vertices(&mut vertices);
    Ok((vertices, indices))
}

fn parse_preview_fbx_geometry(
    node: &Node,
    vertices_out: &mut Vec<ModelVertex>,
    indices_out: &mut Vec<u32>,
) {
    let Some(raw_vertices) = child_f64_array(node, "Vertices") else {
        return;
    };
    let Some(polygon_indices) = child_i32_array(node, "PolygonVertexIndex") else {
        return;
    };
    let positions: Vec<_> = raw_vertices
        .chunks_exact(3)
        .map(|chunk| Vec3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32))
        .collect();
    let uv_layer = fbx_uv_layer(node);
    let mut polygon = Vec::new();
    let mut polygon_vertex_start = 0usize;

    for raw_index in polygon_indices {
        let end = raw_index < 0;
        let vertex_index = if end {
            (-raw_index - 1) as usize
        } else {
            raw_index as usize
        };
        polygon.push(vertex_index);

        if end {
            if polygon.len() >= 3 {
                for local in 1..polygon.len() - 1 {
                    let tri_indices = [polygon[0], polygon[local], polygon[local + 1]];
                    if tri_indices.iter().any(|index| *index >= positions.len()) {
                        continue;
                    }
                    let tri_uvs = uv_layer.as_ref().map(|layer| {
                        [
                            layer.uv_at(polygon_vertex_start, &polygon, 0),
                            layer.uv_at(polygon_vertex_start, &polygon, local),
                            layer.uv_at(polygon_vertex_start, &polygon, local + 1),
                        ]
                    });
                    push_preview_triangle(
                        [
                            positions[tri_indices[0]],
                            positions[tri_indices[1]],
                            positions[tri_indices[2]],
                        ],
                        tri_uvs,
                        vertices_out,
                        indices_out,
                    );
                }
            }
            polygon_vertex_start += polygon.len();
            polygon.clear();
        }
    }
}

fn push_preview_triangle(
    positions: [Vec3; 3],
    uvs: Option<[[f32; 2]; 3]>,
    vertices_out: &mut Vec<ModelVertex>,
    indices_out: &mut Vec<u32>,
) {
    let e1 = positions[1] - positions[0];
    let e2 = positions[2] - positions[0];
    let Some(normal) = e1.cross(e2).normalize() else {
        return;
    };

    let (tangent, bitangent) = if let Some(uvs) = uvs {
        tangent_frame_from_uvs(e1, e2, uvs).unwrap_or_else(|| fallback_tangent_frame(normal))
    } else {
        fallback_tangent_frame(normal)
    };

    let base = vertices_out.len() as u32;
    for position in positions {
        vertices_out.push(ModelVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: tangent.to_array(),
            bitangent: bitangent.to_array(),
        });
    }
    indices_out.extend_from_slice(&[base, base + 1, base + 2]);
}

fn tangent_frame_from_uvs(e1: Vec3, e2: Vec3, uvs: [[f32; 2]; 3]) -> Option<(Vec3, Vec3)> {
    let duv1 = [uvs[1][0] - uvs[0][0], uvs[1][1] - uvs[0][1]];
    let duv2 = [uvs[2][0] - uvs[0][0], uvs[2][1] - uvs[0][1]];
    let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
    if det.abs() < 1.0e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tangent = ((e1 * duv2[1] - e2 * duv1[1]) * inv_det).normalize()?;
    let bitangent = ((e2 * duv1[0] - e1 * duv2[0]) * inv_det).normalize()?;
    Some((tangent, bitangent))
}

fn fallback_tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let helper = if normal.y.abs() < 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let tangent = helper
        .cross(normal)
        .normalize()
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let bitangent = normal
        .cross(tangent)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    (tangent, bitangent)
}

fn smooth_preview_frames(vertices: &mut [ModelVertex]) {
    #[derive(Clone, Copy)]
    struct Accum {
        normal: Vec3,
        tangent: Vec3,
        bitangent: Vec3,
    }

    impl Default for Accum {
        fn default() -> Self {
            Self {
                normal: Vec3::ZERO,
                tangent: Vec3::ZERO,
                bitangent: Vec3::ZERO,
            }
        }
    }

    let mut accum = HashMap::<[i32; 3], Accum>::new();
    for vertex in vertices.iter() {
        let key = smooth_position_key(vertex.position);
        let entry = accum.entry(key).or_default();
        entry.normal += Vec3::from_array(vertex.normal);
        entry.tangent += Vec3::from_array(vertex.tangent);
        entry.bitangent += Vec3::from_array(vertex.bitangent);
    }

    for vertex in vertices {
        let Some(entry) = accum.get(&smooth_position_key(vertex.position)).copied() else {
            continue;
        };
        let normal = entry
            .normal
            .normalize()
            .unwrap_or_else(|| Vec3::from_array(vertex.normal));
        let tangent_sum = entry
            .tangent
            .normalize()
            .unwrap_or_else(|| Vec3::from_array(vertex.tangent));
        let mut tangent = tangent_sum - normal * tangent_sum.dot(normal);
        if let Some(projected) = tangent.normalize() {
            tangent = projected;
        } else {
            tangent = fallback_tangent_frame(normal).0;
        }

        let handedness = normal
            .cross(tangent)
            .dot(entry.bitangent)
            .signum()
            .max(-1.0);
        let bitangent = normal.cross(tangent) * if handedness < 0.0 { -1.0 } else { 1.0 };

        vertex.normal = normal.to_array();
        vertex.tangent = tangent.to_array();
        vertex.bitangent = bitangent.to_array();
    }
}

fn smooth_position_key(position: [f32; 3]) -> [i32; 3] {
    [
        (position[0] * 100_000.0).round() as i32,
        (position[1] * 100_000.0).round() as i32,
        (position[2] * 100_000.0).round() as i32,
    ]
}

fn fit_preview_vertices(vertices: &mut [ModelVertex]) {
    if vertices.is_empty() {
        return;
    }
    let mut min = Vec3::from_array(vertices[0].position);
    let mut max = min;
    for vertex in vertices.iter() {
        let position = Vec3::from_array(vertex.position);
        min = min.min(position);
        max = max.max(position);
    }
    let center = (min + max) * 0.5;
    let extent = max - min;
    let max_extent = extent.x.max(extent.y).max(extent.z).max(1.0e-6);
    let scale = 1.8 / max_extent;
    for vertex in vertices {
        let position = (Vec3::from_array(vertex.position) - center) * scale;
        vertex.position = position.to_array();
    }
}

#[derive(Clone)]
struct FbxUvLayer {
    mapping: String,
    reference: String,
    uvs: Vec<[f32; 2]>,
    indices: Vec<i32>,
}

impl FbxUvLayer {
    fn uv_at(&self, polygon_vertex_start: usize, polygon: &[usize], local: usize) -> [f32; 2] {
        let mapped_index = match self.mapping.as_str() {
            "ByPolygonVertex" => polygon_vertex_start + local,
            "ByVertice" | "ByVertex" => polygon[local],
            _ => polygon_vertex_start + local,
        };
        let direct_index = if self.reference == "IndexToDirect" || self.reference == "Index" {
            self.indices
                .get(mapped_index)
                .copied()
                .unwrap_or(mapped_index as i32)
                .max(0) as usize
        } else {
            mapped_index
        };
        self.uvs.get(direct_index).copied().unwrap_or([0.0, 0.0])
    }
}

fn fbx_uv_layer(node: &Node) -> Option<FbxUvLayer> {
    let layer = node
        .children
        .iter()
        .find(|child| child.name == "LayerElementUV")?;
    let mapping = child_string(layer, "MappingInformationType")
        .unwrap_or_else(|| "ByPolygonVertex".to_string());
    let reference =
        child_string(layer, "ReferenceInformationType").unwrap_or_else(|| "Direct".to_string());
    let raw_uvs = child_f64_array(layer, "UV")?;
    let uvs = raw_uvs
        .chunks_exact(2)
        .map(|chunk| [chunk[0] as f32, chunk[1] as f32])
        .collect();
    let indices = child_i32_array(layer, "UVIndex").unwrap_or_default();

    Some(FbxUvLayer {
        mapping,
        reference,
        uvs,
        indices,
    })
}

fn geometry_nodes<'a>(node: &'a Node) -> Vec<&'a Node> {
    let mut nodes = Vec::new();
    collect_geometry_nodes(node, &mut nodes);
    nodes
}

fn collect_geometry_nodes<'a>(node: &'a Node, nodes: &mut Vec<&'a Node>) {
    if node.name == "Geometry" {
        nodes.push(node);
    }
    for child in &node.children {
        collect_geometry_nodes(child, nodes);
    }
}

fn child_f64_array(node: &Node, name: &str) -> Option<Vec<f64>> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::F64Array(values) => Some(values.clone()),
            Property::F32Array(values) => Some(values.iter().map(|value| *value as f64).collect()),
            _ => None,
        })
}

fn child_i32_array(node: &Node, name: &str) -> Option<Vec<i32>> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::I32Array(values) => Some(values.clone()),
            _ => None,
        })
}

fn child_string(node: &Node, name: &str) -> Option<String> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::String(value) => Some(value.clone()),
            _ => None,
        })
}

fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let axis = normalize3(axis);
    let half = angle * 0.5;
    let (s, c) = half.sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1.0e-6 && len.is_finite() {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn quat_to_mat4(q: [f32; 4]) -> [[f32; 4]; 4] {
    let q = normalize_quat(q);
    let x = q[0];
    let y = q[1];
    let z = q[2];
    let w = q[3];
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0],
        [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0],
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (0.5 * fov_y).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
    ]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub3(center, eye));
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let len = dot3(value, value).sqrt();
    if len > 1.0e-6 && len.is_finite() {
        [value[0] / len, value[1] / len, value[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn load_config_into_settings(settings: &mut BakeSettings) -> Result<()> {
    let path = trimmed_path(&settings.config_path).context("config path is empty")?;
    let config = BakeConfigFile::read(&path)
        .with_context(|| format!("failed to read config {}", path.display()))?
        .resolve(Some(&path), BakeOverrides::default())
        .context("failed to resolve config")?;

    settings.obj_path = path_to_gui_string(&config.obj);
    settings.width = config.width as i32;
    settings.height = config.height as i32;
    settings.mode_index = match config.mode {
        BakeMode::Single => 0,
        BakeMode::Full => 1,
        BakeMode::Isotropic => 2,
    };
    settings.light_width = config.light_width as i32;
    settings.light_height = config.light_height as i32;
    settings.samples = config.samples as i32;
    settings.max_repeat_radius = config.max_repeat_radius as i32;
    settings.sampler_index = match config.sampler {
        SamplerKind::Halton => 0,
        SamplerKind::Random => 1,
    };
    settings.enable_shadows = config.enable_shadows;
    settings.light = config.light;
    settings.preview_light = config.light;
    settings.material_index = match config.material.kind {
        MaterialKind::Lambertian => 0,
        MaterialKind::SpecularPhong => 1,
    };
    settings.material_color = config.material.color;
    settings.roughness = config.material.roughness.unwrap_or(0.05);
    Ok(())
}

fn create_window() -> Result<(EventLoop<()>, Window, glium::Display<WindowSurface>)> {
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let window_builder = WindowBuilder::new()
        .with_title(TITLE)
        .with_inner_size(LogicalSize::new(1280.0, 760.0));

    let (window, cfg) = glutin_winit::DisplayBuilder::new()
        .with_window_builder(Some(window_builder))
        .build(&event_loop, ConfigTemplateBuilder::new(), |mut configs| {
            configs.next().expect("no OpenGL configs available")
        })
        .map_err(|error| anyhow::anyhow!("failed to create OpenGL window: {error}"))?;
    let window = window.context("failed to create window")?;

    let context_attribs = ContextAttributesBuilder::new().build(Some(window.raw_window_handle()));
    let context = unsafe {
        cfg.display()
            .create_context(&cfg, &context_attribs)
            .context("failed to create OpenGL context")?
    };

    let size = window.inner_size();
    let surface_attribs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        window.raw_window_handle(),
        NonZeroU32::new(size.width.max(1)).unwrap(),
        NonZeroU32::new(size.height.max(1)).unwrap(),
    );
    let surface = unsafe {
        cfg.display()
            .create_window_surface(&cfg, &surface_attribs)
            .context("failed to create OpenGL surface")?
    };
    let context = context
        .make_current(&surface)
        .context("failed to make OpenGL context current")?;
    let display = glium::Display::from_context_surface(context, surface)
        .context("failed to create display")?;

    Ok((event_loop, window, display))
}

fn imgui_init(window: &Window) -> (imgui_winit_support::WinitPlatform, imgui::Context) {
    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    platform.attach_window(
        imgui.io_mut(),
        window,
        imgui_winit_support::HiDpiMode::Default,
    );
    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);
    (platform, imgui)
}

fn trimmed_path(value: &str) -> Option<PathBuf> {
    let value = value.trim();
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn resolve_gui_override_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }

    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }

    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path,
    }
}

fn path_to_gui_string(path: &PathBuf) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn to_preview_byte(value: f32) -> u8 {
    let value = value.max(0.0).powf(1.0 / 2.2).clamp(0.0, 1.0);
    (value * 255.0 + 0.5) as u8
}

fn rays_per_second(stats: &GpuBakeStats) -> f64 {
    if stats.gpu_dispatch_time.as_secs_f64() > 0.0 {
        stats.camera_ray_count as f64 / stats.gpu_dispatch_time.as_secs_f64()
    } else {
        0.0
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.3}s")
    } else {
        format!("{:.2}ms", seconds * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_pig_fbx_loads_when_present() {
        let path = Path::new("assets/preview/pig.fbx");
        if !path.exists() {
            return;
        }

        let (vertices, indices) = load_preview_fbx(path).unwrap();
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0);
    }
}
