use glam::{Vec2, Vec3, Vec4};

pub struct FrameData {
    pub bgcolor: Vec4,
    pub frame_limiter: bool,
    pub frame_limiter_fps: u32,
    pub rotate_model: bool,
    pub cam_eye: Vec3,
    pub cam_center: Vec3,
    pub cam_up: Vec3,
    pub cam_nearfar: Vec2,
}
