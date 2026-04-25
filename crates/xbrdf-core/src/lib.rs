pub mod config;
pub mod geometry;
pub mod math;
pub mod reference;
pub mod sampling;

pub use config::{
    BakeConfigFile, BakeOverrides, Manifest, MaterialConfigFile, MaterialKind, ResolvedBakeConfig,
    ResolvedMaterial,
};
pub use geometry::{Bounds, ColorSource, Mesh, Triangle};
pub use math::Vec3;
