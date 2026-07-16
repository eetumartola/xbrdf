pub mod atlas;
pub mod config;
pub mod fbx;
pub mod geometry;
pub mod math;
pub mod reference;
pub mod sampling;

pub use atlas::{hemisphere_uv, AtlasMetadata, ARTIFACT_SCHEMA_VERSION};
pub use config::{
    BakeConfigFile, BakeMode, BakeOverrides, Manifest, MaterialConfigFile, MaterialKind,
    ResolvedBakeConfig, ResolvedMaterial, SamplerKind,
};
pub use geometry::{Bounds, ColorSource, Mesh, Triangle};
pub use math::Vec3;
