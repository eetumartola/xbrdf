use anyhow::{Context, Result};
use exr::prelude::write_rgb_file;
use glium::texture::{ClientFormat, RawImage2d, Texture2d};
use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter, SamplerBehavior};
use glium::Surface;
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextAttributesBuilder, NotCurrentGlContext},
    display::{GetGlDisplay, GlDisplay},
    surface::{SurfaceAttributesBuilder, WindowSurface},
};
use imgui::{Condition, Image, ProgressBar, TextureId, Ui};
use imgui_glium_renderer::{Renderer, Texture};
use imgui_winit_support::winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use raw_window_handle::HasRawWindowHandle;
use std::borrow::Cow;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};
use xbrdf_core::{BakeConfigFile, BakeOverrides, Manifest, MaterialKind, Mesh, ResolvedBakeConfig};
use xbrdf_gpu::{GpuBakeStats, ProgressiveBakeOptions, ProgressiveFrame};

const TITLE: &str = "xBRDF Bake";

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
                    if let Err(error) = preview.update(&display, &mut renderer, &app.preview) {
                        app.status = format!("Preview upload failed: {error:#}");
                    }
                    app.preview_dirty = false;
                }

                let ui = imgui.frame();
                draw_ui(ui, &mut app, &preview);

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
    samples: i32,
    max_repeat_radius: i32,
    light: [f32; 3],
    tile_width: f32,
    tile_depth: f32,
    material_index: usize,
    material_color: [f32; 3],
    roughness: f32,
    update_interval_seconds: f32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            config_path: "assets/fixtures/specular.toml".to_string(),
            obj_path: String::new(),
            out_dir: "out/gui".to_string(),
            width: 256,
            height: 64,
            samples: 64,
            max_repeat_radius: 2,
            light: [0.0, 1.0, -1.0],
            tile_width: 0.0,
            tile_depth: 0.0,
            material_index: 0,
            material_color: [1.0, 1.0, 1.0],
            roughness: 0.05,
            update_interval_seconds: 0.5,
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
    preview_dirty: bool,
    last_stats: Option<GpuBakeStats>,
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
            preview_dirty: false,
            last_stats: None,
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
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        self.baking = true;
        self.progress = 0.0;
        self.completed_samples = 0;
        self.total_samples = 0;
        self.last_stats = None;
        self.status = "Starting bake".to_string();
        self.preview = PreviewImage::default();
        self.preview_dirty = true;

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
                    self.preview = PreviewImage::new(width, height);
                    self.preview_dirty = true;
                    self.total_samples = self.settings.samples.max(1) as u32;
                    self.completed_samples = 0;
                    self.progress = 0.0;
                    self.status = "Baking".to_string();
                }
                BakeEvent::Frame(frame) => {
                    self.preview.write_frame(&frame);
                    self.preview_dirty = true;
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
                BakeEvent::Finished {
                    stats,
                    image,
                    manifest,
                } => {
                    self.baking = false;
                    keep_receiver = false;
                    self.progress = 1.0;
                    self.completed_samples = self.total_samples;
                    self.last_stats = Some(stats);
                    self.status = format!("Wrote {} and {}", image.display(), manifest.display());
                }
                BakeEvent::Error(error) => {
                    self.baking = false;
                    keep_receiver = false;
                    self.status = error;
                }
            }
        }

        if keep_receiver {
            self.receiver = Some(receiver);
        }
    }
}

enum BakeEvent {
    Status(String),
    Started {
        width: u32,
        height: u32,
    },
    Frame(ProgressiveFrame),
    Finished {
        stats: GpuBakeStats,
        image: PathBuf,
        manifest: PathBuf,
    },
    Error(String),
}

#[derive(Default)]
struct PreviewImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    exposure: f32,
}

impl PreviewImage {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgba: vec![0; width as usize * height as usize * 4],
            exposure: 0.0,
        }
    }

    fn write_frame(&mut self, frame: &ProgressiveFrame) {
        if self.width != frame.width || self.height != frame.height || self.height == 0 {
            return;
        }

        let gain = 2.0_f32.powf(self.exposure);
        for (index, src) in frame.pixels.iter().enumerate() {
            let dst = index * 4;
            self.rgba[dst] = to_preview_byte(src[0] * gain);
            self.rgba[dst + 1] = to_preview_byte(src[1] * gain);
            self.rgba[dst + 2] = to_preview_byte(src[2] * gain);
            self.rgba[dst + 3] = 255;
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

fn draw_ui(ui: &Ui, app: &mut AppState, preview: &PreviewTexture) {
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
            ui.input_int("Samples", &mut app.settings.samples).build();
            ui.input_float("Update sec", &mut app.settings.update_interval_seconds)
                .build();
            ui.input_int("Repeat radius", &mut app.settings.max_repeat_radius)
                .build();
            app.settings.width = app.settings.width.clamp(1, 16_384);
            app.settings.height = app.settings.height.clamp(1, 16_384);
            app.settings.samples = app.settings.samples.clamp(1, 1_000_000);
            app.settings.update_interval_seconds =
                app.settings.update_interval_seconds.clamp(0.05, 60.0);
            app.settings.max_repeat_radius = app.settings.max_repeat_radius.clamp(0, 16);

            ui.input_float3("Light", &mut app.settings.light).build();
            ui.input_float("Tile width", &mut app.settings.tile_width)
                .build();
            ui.input_float("Tile depth", &mut app.settings.tile_depth)
                .build();
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
                .overlay_text(format!(
                    "{} / {} samples",
                    app.completed_samples, app.total_samples
                ))
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
            } else {
                ui.text("No preview yet");
            }
        });
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
    let mesh = Mesh::load(
        &config.obj,
        config.tile_width_override,
        config.tile_depth_override,
    )
    .with_context(|| format!("failed to load geometry {}", config.obj.display()))?;

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create output directory {}", out_dir.display()))?;

    send(BakeEvent::Started {
        width: config.width,
        height: config.height,
    });

    let result = pollster::block_on(xbrdf_gpu::bake_progressive(
        &config,
        &mesh,
        ProgressiveBakeOptions {
            update_interval: Duration::from_secs_f32(settings.update_interval_seconds.max(0.05)),
        },
        |frame| {
            send(BakeEvent::Frame(frame));
        },
    ))
    .context("GPU bake failed")?;

    send(BakeEvent::Status("Writing output".to_string()));
    let image_path = out_dir.join("xbrdf_view.exr");
    write_rgb_file(
        &image_path,
        config.width as usize,
        config.height as usize,
        |x, y| {
            let pixel = result.pixels[y * config.width as usize + x];
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
                samples: Some(settings.samples.max(1) as u32),
                tile_width: positive_override(settings.tile_width),
                tile_depth: positive_override(settings.tile_depth),
                light: Some(settings.light),
                max_repeat_radius: Some(settings.max_repeat_radius.clamp(0, 16) as u32),
                material_kind: Some(settings.material_kind()),
                material_color: Some(settings.material_color),
                material_roughness: Some(settings.roughness),
            },
        )
        .context("failed to resolve bake settings")
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
    settings.samples = config.samples as i32;
    settings.max_repeat_radius = config.max_repeat_radius as i32;
    settings.light = config.light;
    settings.tile_width = config.tile_width_override.unwrap_or(0.0);
    settings.tile_depth = config.tile_depth_override.unwrap_or(0.0);
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

fn positive_override(value: f32) -> Option<f32> {
    (value.is_finite() && value > 0.0).then_some(value)
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
