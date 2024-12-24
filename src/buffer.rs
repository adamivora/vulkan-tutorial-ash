use std::ffi::c_void;

use ash::vk;
use vk_mem::Allocation;

pub struct BoundBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

pub struct BoundBufferMapped {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub ptr: *mut c_void,
}

pub struct BoundImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

pub struct VulkanBuffers {
    pub vertex: BoundBuffer,
    pub index: BoundBuffer,
    pub uniforms: Vec<BoundBufferMapped>,
    pub texture: BoundImage,
    pub color: BoundImage,
    pub depth: BoundImage,
}
