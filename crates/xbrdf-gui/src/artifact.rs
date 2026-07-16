use anyhow::{Context, Result};
use exr::prelude::{read_first_rgba_layer_from_file, write_rgb_file};
use std::path::Path;
use xbrdf_core::{AtlasMetadata, BakeMode, ResolvedBakeConfig, ARTIFACT_SCHEMA_VERSION};
use xbrdf_gpu::{AtlasProgressFrame, ProgressiveFrame};

#[derive(Clone, Default)]
pub(crate) struct PreviewImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgb: Vec<[f32; 3]>,
    pub(crate) rgba: Vec<u8>,
    pub(crate) exposure: f32,
    pub(crate) meta: PreviewMeta,
}

impl PreviewImage {
    pub(crate) fn new(width: u32, height: u32, meta: PreviewMeta) -> Self {
        Self {
            width,
            height,
            rgb: vec![[0.0; 3]; width as usize * height as usize],
            rgba: vec![0; width as usize * height as usize * 4],
            exposure: 0.0,
            meta,
        }
    }

    pub(crate) fn from_frame(frame: &ProgressiveFrame, meta: PreviewMeta) -> Self {
        let mut image = Self::new(frame.width, frame.height, meta);
        image.write_pixels(&frame.pixels);
        image
    }

    pub(crate) fn from_atlas_frame(frame: &AtlasProgressFrame, meta: PreviewMeta) -> Self {
        let mut image = Self::new(frame.width, frame.height, meta);
        image.write_pixels(&frame.pixels);
        image
    }

    fn write_pixels(&mut self, pixels: &[[f32; 3]]) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let gain = 2.0_f32.powf(self.exposure);
        for (index, src) in pixels.iter().take(self.rgb.len()).enumerate() {
            self.rgb[index] = *src;
            let dst = index * 4;
            self.rgba[dst] = to_preview_byte(src[0] * gain);
            self.rgba[dst + 1] = to_preview_byte(src[1] * gain);
            self.rgba[dst + 2] = to_preview_byte(src[2] * gain);
            self.rgba[dst + 3] = 255;
        }
    }

    pub(crate) fn save_exr(&self, path: &Path) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.rgb.is_empty() {
            anyhow::bail!("no atlas is selected");
        }
        write_rgb_file(path, self.width as usize, self.height as usize, |x, y| {
            let pixel = self.rgb[y * self.width as usize + x];
            (pixel[0], pixel[1], pixel[2])
        })
        .with_context(|| format!("failed to write EXR {}", path.display()))
    }

    pub(crate) fn load_exr(path: &Path, meta: PreviewMeta) -> Result<Self> {
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
pub(crate) struct PreviewMeta {
    pub(crate) mode: BakeMode,
    pub(crate) camera_width: u32,
    pub(crate) camera_height: u32,
    pub(crate) light_width: u32,
    pub(crate) light_height: u32,
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
    pub(crate) fn from_config(config: &ResolvedBakeConfig) -> Self {
        Self {
            mode: config.mode,
            camera_width: config.camera_tile_width(),
            camera_height: config.camera_tile_height(),
            light_width: config.effective_light_width(),
            light_height: config.effective_light_height(),
        }
    }

    pub(crate) fn from_atlas_metadata(metadata: AtlasMetadata) -> Self {
        Self {
            mode: metadata.mode,
            camera_width: metadata.camera_width,
            camera_height: metadata.camera_height,
            light_width: metadata.light_width,
            light_height: metadata.light_height,
        }
    }

    pub(crate) fn to_atlas_metadata(self, atlas_width: u32, atlas_height: u32) -> AtlasMetadata {
        AtlasMetadata {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            mode: self.mode,
            atlas_width,
            atlas_height,
            camera_width: self.camera_width,
            camera_height: self.camera_height,
            light_width: self.light_width,
            light_height: self.light_height,
            camera_azimuth_wraps: self.mode != BakeMode::Isotropic,
            elevation_clamps: true,
            samples_are_texel_centered: true,
            consumer_applies_macro_cosine: true,
        }
    }

    pub(crate) fn mode_code(self) -> i32 {
        match self.mode {
            BakeMode::Single => 0,
            BakeMode::Full => 1,
            BakeMode::Isotropic => 2,
        }
    }
}

fn to_preview_byte(value: f32) -> u8 {
    let value = value.max(0.0).powf(1.0 / 2.2).clamp(0.0, 1.0);
    (value * 255.0 + 0.5) as u8
}
