use glam::{Mat4, Vec2, Vec4};

pub struct FrameData {
    pub bgcolor: Vec4,
    pub frame_limiter: bool,
    pub frame_limiter_fps: u32,
    pub rotate_model: bool,
    pub cam_matrix: Mat4,
    pub cam_nearfar: Vec2,
    pub cam_fov: f32,
}
