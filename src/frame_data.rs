use glam::Vec4;

pub struct FrameData {
    pub bgcolor: Vec4,
    pub frame_limiter: bool,
    pub frame_limiter_fps: u32,
}
