use crate::buffer::{BoundBuffer, BoundBufferMapped, BoundImage, VulkanBuffers};
use crate::frame_data::FrameData;
use crate::obj_loader::ObjLoader;
#[cfg(debug_assertions)]
use ash::ext::debug_utils;
use ash::prelude::VkResult;
use ash::util::Align;
use ash::vk;
use ash::Entry;
use glam::{Mat4, Vec2, Vec3};
use image::{EncodableLayout, ImageReader};
use imgui::DrawData;
use imgui_rs_vulkan_renderer::Renderer as ImguiRenderer;
use serde::Serialize;
use serde_binary::binary_stream::Endian;
use std::borrow::Cow;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::ffi::{c_void, CStr, CString};
use std::fs;
use std::hash::Hash;
use std::iter::zip;
use std::mem::{offset_of, ManuallyDrop};
use std::os::raw::c_char;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::Instant;
use std::u32;
use vk_mem::{Alloc, Allocation, Allocator, AllocatorCreateInfo, MemoryUsage};
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const MODEL_PATH: &str = "models/viking_room.obj";
const TEXTURE_PATH: &str = "textures/viking_room.png";
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

static START_TIME: LazyLock<Instant> = LazyLock::new(|| Instant::now());

#[derive(Copy, Clone, PartialEq, Serialize)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(&serde_binary::to_vec(&self, Endian::default()).unwrap());
    }
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        let binding_description: vk::VertexInputBindingDescription =
            vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX);
        binding_description
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let attribute_descriptions: [vk::VertexInputAttributeDescription; 3] = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, tex_coord) as u32),
        ];
        attribute_descriptions
    }
}

#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

const VALIDATION_LAYERS: [&CStr; 1] =
    [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

const DEVICE_EXTENSIONS: [&CStr; 2] = [vk::KHR_SWAPCHAIN_NAME, vk::KHR_MAINTENANCE1_NAME];

unsafe extern "system" fn messenger_debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message: Cow<'_, str> = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };
    eprintln!("validation layer: {message}\n",);
    vk::FALSE
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    fn query_swap_chain_support(
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> Result<SwapChainSupportDetails, vk::Result> {
        let capabilities =
            unsafe { surface_instance.get_physical_device_surface_capabilities(device, surface)? };
        let formats =
            unsafe { surface_instance.get_physical_device_surface_formats(device, surface)? };
        let present_modes =
            unsafe { surface_instance.get_physical_device_surface_present_modes(device, surface)? };

        Result::Ok(SwapChainSupportDetails {
            capabilities,
            formats,
            present_modes,
        })
    }

    fn choose_swap_surface_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> Result<vk::SurfaceFormatKHR, vk::Result> {
        let format = available_formats
            .iter()
            .find(|&available_format| {
                available_format.format == vk::Format::B8G8R8A8_SRGB
                    && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(available_formats.first().unwrap());
        Result::Ok(*format)
    }

    fn choose_swap_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        let present_mode = available_present_modes
            .iter()
            .find(|&&available_present_mode| available_present_mode == vk::PresentModeKHR::MAILBOX);
        let present_mode = if present_mode.is_none() {
            vk::PresentModeKHR::FIFO
        } else {
            *present_mode.unwrap()
        };
        log::debug!("present mode: {:?}", present_mode);
        present_mode
    }

    fn choose_swap_extent(&self, window: &Window) -> vk::Extent2D {
        match self.capabilities.current_extent.width {
            u32::MAX => {
                let size = window.inner_size();

                let actual_width = size.width.clamp(
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                );
                let actual_height = size.height.clamp(
                    self.capabilities.min_image_extent.height,
                    self.capabilities.max_image_extent.height,
                );
                vk::Extent2D {
                    width: actual_width,
                    height: actual_height,
                }
            }
            _ => self.capabilities.current_extent,
        }
    }
}

pub struct Vulkan {
    allocator: Arc<Mutex<vk_mem::Allocator>>,
    instance: ash::Instance,
    #[cfg(debug_assertions)]
    debug_utils_loader: ash::ext::debug_utils::Instance,
    surface_instance: ash::khr::surface::Instance,
    swapchain_device: ash::khr::swapchain::Device,
    device: ash::Device,
    // handles from now on
    #[cfg(debug_assertions)]
    debug_callback: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    present_queue: vk::Queue,
    graphics_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    texture_sampler: vk::Sampler,

    vertices: Vec<Vertex>,
    indices: Vec<u32>,

    mip_levels: u32,
    msaa_samples: vk::SampleCountFlags,
    current_frame: usize,
    framebuffer_resized: bool,
    is_rendering: bool,

    #[allow(dead_code)]
    entry: Entry,
}

struct VulkanInit;
impl VulkanInit {
    fn init_window(event_loop: &ActiveEventLoop) -> Window {
        let window_attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan")
            .with_resizable(true);
        event_loop.create_window(window_attributes).unwrap()
    }

    fn create_instance(
        entry: &Entry,
        window: &Window,
    ) -> Result<ash::Instance, Box<dyn std::error::Error>> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support(entry)? {
            return Err("validation layers requested, but not available!".into());
        }

        let application_name = CString::new("Hello Triangle").unwrap();
        let engine_name = CString::new("No Engine").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);

        #[allow(unused_mut)]
        let mut extensions =
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }
        #[cfg(debug_assertions)]
        {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let layer_names = if ENABLE_VALIDATION_LAYERS {
            Vec::from(VALIDATION_LAYERS)
        } else {
            Vec::new()
        };
        let layer_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names_raw)
            .flags(create_flags);

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
        if ENABLE_VALIDATION_LAYERS {
            debug_create_info = Self::populate_debug_messenger_create_info(debug_create_info);
            create_info = create_info.push_next(&mut debug_create_info);
        }

        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let supported_extensions: Vec<vk::ExtensionProperties> =
            unsafe { entry.enumerate_instance_extension_properties(None)? };

        println!(
            "Supported extensions:\n{}",
            supported_extensions
                .iter()
                .fold(String::new(), |acc, &num| acc
                    + "\t"
                    + num.extension_name_as_c_str().unwrap().to_str().unwrap()
                    + "\n")
        );

        Result::Ok(instance)
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: ash::vk::PhysicalDevice,
    ) -> Result<bool, vk::Result> {
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(device)? };
        let required_extensions = DEVICE_EXTENSIONS;

        let mut required_extensions = HashSet::from(required_extensions);
        available_extensions.iter().for_each(|extension| {
            required_extensions.remove(extension.extension_name_as_c_str().unwrap());
        });

        Result::Ok(required_extensions.is_empty())
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: ash::vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<bool, vk::Result> {
        let indices = Self::find_queue_families(instance, device, surface_instance, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, device)?;
        let mut swapchain_adequate: bool = false;
        if extensions_supported {
            let swapchain_support = SwapChainSupportDetails::query_swap_chain_support(
                surface_instance,
                surface,
                device,
            )?;
            swapchain_adequate = !swapchain_support.formats.is_empty()
                && !swapchain_support.present_modes.is_empty();
        }

        let supported_features = unsafe { instance.get_physical_device_features(device) };
        Result::Ok(
            indices.is_complete()
                && extensions_supported
                && swapchain_adequate
                && supported_features.sampler_anisotropy != 0,
        )
    }

    fn get_max_usable_sample_count(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::SampleCountFlags {
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let counts = physical_device_properties
            .limits
            .framebuffer_color_sample_counts
            & physical_device_properties
                .limits
                .framebuffer_depth_sample_counts;

        let count = match counts {
            flag if flag.contains(vk::SampleCountFlags::TYPE_64) => vk::SampleCountFlags::TYPE_64,
            flag if flag.contains(vk::SampleCountFlags::TYPE_32) => vk::SampleCountFlags::TYPE_32,
            flag if flag.contains(vk::SampleCountFlags::TYPE_16) => vk::SampleCountFlags::TYPE_16,
            flag if flag.contains(vk::SampleCountFlags::TYPE_8) => vk::SampleCountFlags::TYPE_8,
            flag if flag.contains(vk::SampleCountFlags::TYPE_4) => vk::SampleCountFlags::TYPE_4,
            flag if flag.contains(vk::SampleCountFlags::TYPE_2) => vk::SampleCountFlags::TYPE_2,
            _ => vk::SampleCountFlags::TYPE_1,
        };
        count
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<ash::vk::PhysicalDevice, Box<dyn std::error::Error>> {
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical_device = devices
            .iter()
            .find(|&&device| {
                Self::is_device_suitable(instance, device, surface_instance, surface)
                    .expect("failed to find a suitable GPU!")
            })
            .ok_or("failed to find a suitable GPU!")?;
        Result::Ok(physical_device.clone())
    }

    fn check_validation_layer_support(entry: &Entry) -> Result<bool, vk::Result> {
        let available_layers: Vec<vk::LayerProperties> =
            unsafe { entry.enumerate_instance_layer_properties()? };

        let layers_found = VALIDATION_LAYERS.iter().all(|&layer| {
            available_layers
                .iter()
                .any(|&av_layer| av_layer.layer_name_as_c_str().unwrap() == layer)
        });

        Result::Ok(layers_found)
    }

    fn populate_debug_messenger_create_info<'a>(
        create_info: vk::DebugUtilsMessengerCreateInfoEXT<'a>,
    ) -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
        create_info
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(messenger_debug_callback))
    }

    #[cfg(debug_assertions)]
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &ash::Instance,
    ) -> Result<
        (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT),
        Box<dyn std::error::Error>,
    > {
        if !ENABLE_VALIDATION_LAYERS {
            return Result::Err("validation layers not enabled".into());
        }

        let create_info = Self::populate_debug_messenger_create_info(
            vk::DebugUtilsMessengerCreateInfoEXT::default(),
        );
        let debug_utils_loader = debug_utils::Instance::new(&entry, instance);
        let debug_callback =
            unsafe { debug_utils_loader.create_debug_utils_messenger(&create_info, None)? };

        Result::Ok((debug_utils_loader, debug_callback))
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(ash::Device, vk::Queue, vk::Queue), vk::Result> {
        let indices =
            Self::find_queue_families(instance, physical_device, surface_instance, surface)?;

        let unique_queue_families = HashSet::from([
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ]);

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_queue_families
            .iter()
            .map(|&queue_family| {
                vk::DeviceQueueCreateInfo {
                    queue_family_index: queue_family,
                    queue_count: 1,
                    ..vk::DeviceQueueCreateInfo::default()
                }
                .queue_priorities(&[1.0])
            })
            .collect();

        let device_features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .sample_rate_shading(true);

        #[allow(unused_mut)]
        let mut extensions: Vec<*const c_char> = DEVICE_EXTENSIONS
            .iter()
            .map(|&ext| ext.as_ptr() as *const c_char)
            .collect();
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&extensions);

        let device = unsafe { instance.create_device(physical_device, &create_info, None) }?;
        let graphics_queue =
            unsafe { device.get_device_queue(indices.graphics_family.unwrap(), 0) };
        let present_queue = unsafe { device.get_device_queue(indices.present_family.unwrap(), 0) };
        Result::Ok((device, graphics_queue, present_queue))
    }

    fn create_surface(
        entry: &Entry,
        window: &Window,
        instance: &ash::Instance,
    ) -> Result<(ash::khr::surface::Instance, vk::SurfaceKHR), Box<dyn std::error::Error>> {
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }?;
        let surface_instance = ash::khr::surface::Instance::new(&entry, instance);
        Result::Ok((surface_instance, surface))
    }

    fn create_swapchain(
        window: &Window,
        instance: &ash::Instance,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
        logical_device: &ash::Device,
    ) -> Result<
        (
            ash::khr::swapchain::Device,
            vk::SwapchainKHR,
            Vec<vk::Image>,
            vk::Format,
            vk::Extent2D,
        ),
        vk::Result,
    > {
        let swapchain_support =
            SwapChainSupportDetails::query_swap_chain_support(surface_instance, surface, device)?;
        let surface_format =
            SwapChainSupportDetails::choose_swap_surface_format(&swapchain_support.formats)?;
        let present_mode =
            SwapChainSupportDetails::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = swapchain_support.choose_swap_extent(window);
        log::debug!("New swapchain extent: {} {}", extent.width, extent.height);

        let image_count: u32 = if swapchain_support.capabilities.max_image_count > 0
            && swapchain_support.capabilities.min_image_count
                >= swapchain_support.capabilities.max_image_count
        {
            swapchain_support.capabilities.max_image_count
        } else {
            swapchain_support.capabilities.min_image_count + 1
        };

        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let indices = Self::find_queue_families(instance, device, surface_instance, surface)?;
        let queue_family_indices = [
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];
        if indices.graphics_family != indices.present_family {
            create_info = create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        } else {
            create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        }

        let swapchain_device = ash::khr::swapchain::Device::new(&instance, &logical_device);
        let swapchain = unsafe { swapchain_device.create_swapchain(&create_info, None)? };

        let swapchain_images = unsafe { swapchain_device.get_swapchain_images(swapchain)? };
        Result::Ok((
            swapchain_device,
            swapchain,
            swapchain_images,
            surface_format.format,
            extent,
        ))
    }

    fn create_allocator(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Arc<Mutex<Allocator>>, vk::Result> {
        let allocator_create_info = AllocatorCreateInfo::new(&instance, &device, physical_device);
        let allocator = { unsafe { Allocator::new(allocator_create_info)? } };
        Result::Ok(Arc::new(Mutex::new(allocator)))
    }

    fn find_queue_families(
        instance: &ash::Instance,
        device: ash::vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<QueueFamilyIndices, vk::Result> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device) };

        let graphics_family = queue_families.iter().position(|queue_family| {
            queue_family.queue_flags & ash::vk::QueueFlags::GRAPHICS
                == ash::vk::QueueFlags::GRAPHICS
        });
        let graphics_family = if graphics_family.is_some() {
            Some(u32::try_from(graphics_family.unwrap()).unwrap())
        } else {
            None
        };

        let present_family = queue_families.iter().enumerate().find_map(
            |(queue_family_index, _queue_family)| unsafe {
                let idx = u32::try_from(queue_family_index).unwrap();
                let supports_present = surface_instance
                    .get_physical_device_surface_support(device, idx, surface)
                    .unwrap();
                if supports_present {
                    Some(idx)
                } else {
                    None
                }
            },
        );

        // Logic to find graphics queue family
        Result::Ok(QueueFamilyIndices {
            graphics_family,
            present_family,
        })
    }

    fn init_vulkan(window: &Window) -> Result<(Vulkan, VulkanBuffers), Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry, window)?;
        #[cfg(debug_assertions)]
        let (debug_utils_loader, debug_callback) = Self::setup_debug_messenger(&entry, &instance)?;
        let (surface_instance, surface) = Self::create_surface(&entry, window, &instance)?;
        let physical_device = Self::pick_physical_device(&instance, &surface_instance, surface)?;
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device(&instance, physical_device, &surface_instance, surface)?;
        let (
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = Self::create_swapchain(
            window,
            &instance,
            &surface_instance,
            surface,
            physical_device,
            &device,
        )?;
        let allocator = VulkanInit::create_allocator(&instance, &device, physical_device)?;
        let msaa_samples = VulkanInit::get_max_usable_sample_count(&instance, physical_device);

        let mut vulkan = Vulkan {
            allocator,
            entry,
            instance,
            device,
            physical_device,
            #[cfg(debug_assertions)]
            debug_utils_loader,
            #[cfg(debug_assertions)]
            debug_callback,
            surface_instance,
            surface,
            graphics_queue,
            present_queue,
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            msaa_samples,
            swapchain_image_views: Vec::new(),
            swapchain_framebuffers: Vec::new(),
            render_pass: vk::RenderPass::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            graphics_pipeline: vk::Pipeline::null(),
            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),
            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            texture_sampler: vk::Sampler::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            mip_levels: 0,
            current_frame: 0,
            framebuffer_resized: false,
            is_rendering: true,
        };

        vulkan.create_image_views()?;
        vulkan.create_render_pass()?;
        vulkan.create_descriptor_set_layout()?;
        vulkan.create_graphics_pipeline()?;
        vulkan.create_command_pool()?;
        let allocator_mutex = vulkan.allocator.clone();
        let mut allocator = allocator_mutex.lock().expect("cannot lock allocator");
        let color_image = vulkan.create_color_resources(&mut allocator)?;
        let depth_image = vulkan.create_depth_resources(&mut allocator)?;
        vulkan.create_framebuffers(color_image.image_view, depth_image.image_view)?;
        let texture_image = vulkan.create_texture_image(&mut allocator)?;
        vulkan.create_texture_sampler()?;
        vulkan.load_model()?;
        let vertex_buffer = vulkan.create_vertex_buffer(&mut allocator, &vulkan.vertices)?;
        let index_buffer = vulkan.create_index_buffer(&mut allocator, &vulkan.indices)?;
        let uniform_buffers = vulkan.create_uniform_buffers(&mut allocator)?;
        vulkan.create_descriptor_pool()?;
        vulkan.create_descriptor_sets(texture_image.image_view, &uniform_buffers)?;
        vulkan.create_command_buffers()?;
        vulkan.create_sync_objects()?;

        let buffers = VulkanBuffers {
            vertex: vertex_buffer,
            index: index_buffer,
            uniforms: uniform_buffers,
            texture: texture_image,
            color: color_image,
            depth: depth_image,
        };

        Result::Ok((vulkan, buffers))
    }
}

impl Vulkan {
    pub fn new(
        event_loop: &ActiveEventLoop,
    ) -> Result<(Window, Vulkan, VulkanBuffers), Box<dyn std::error::Error>> {
        let window = VulkanInit::init_window(event_loop);
        let (vulkan, buffers) = VulkanInit::init_vulkan(&window)?;
        Result::Ok((window, vulkan, buffers))
    }

    fn read_file(filename: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
        let data = fs::File::open(filename)?;
        Result::Ok(data)
    }

    fn create_shader_module<R: std::io::Read + std::io::Seek>(
        mut file: &mut R,
        device: &ash::Device,
    ) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        let code = ash::util::read_spv(&mut file)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let shader_module = unsafe { device.create_shader_module(&create_info, None)? };
        Result::Ok(shader_module)
    }

    fn record_command_buffer(
        &self,
        image_index: u32,
        imgui_renderer: &mut ImguiRenderer,
        imgui_draw_data: &DrawData,
        frame_data: &FrameData,
        buffers: &VulkanBuffers,
    ) -> VkResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: frame_data.bgcolor.into(),
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(
                *self
                    .swapchain_framebuffers
                    .get(image_index as usize)
                    .unwrap(),
            )
            .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
        }

        let vertex_buffers = [buffers.vertex.buffer];
        let offsets = [0];
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            self.device.cmd_bind_index_buffer(
                command_buffer,
                buffers.index.buffer,
                0,
                vk::IndexType::UINT32,
            );
        }

        let viewport = vk::Viewport::default()
            .y(self.swapchain_extent.height as f32)
            .width(self.swapchain_extent.width as f32)
            .height(-1.0 * self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        unsafe {
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        }

        let scissor = vk::Rect2D::default().extent(self.swapchain_extent);
        unsafe {
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.current_frame]],
                &[],
            );
            self.device
                .cmd_draw_indexed(command_buffer, self.indices.len() as u32, 1, 0, 0, 0);
        }

        imgui_renderer
            .cmd_draw(command_buffer, imgui_draw_data)
            .unwrap();

        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }

        Result::Ok(())
    }

    pub fn init_imgui_renderer(
        &self,
        imgui: &mut imgui::Context,
    ) -> Result<ImguiRenderer, Box<dyn std::error::Error>> {
        let imgui_renderer = {
            ImguiRenderer::with_vk_mem_allocator(
                self.allocator.clone(),
                self.device.clone(),
                self.graphics_queue,
                self.command_pool,
                self.render_pass,
                imgui,
                Some(imgui_rs_vulkan_renderer::Options {
                    in_flight_frames: MAX_FRAMES_IN_FLIGHT as usize,
                    sample_count: self.msaa_samples,
                    ..Default::default()
                }),
            )?
        };
        Result::Ok(imgui_renderer)
    }

    fn create_buffer(
        allocator: &mut MutexGuard<Allocator>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<BoundBuffer, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocate_info = vk_mem::AllocationCreateInfo {
            required_flags: properties,
            ..Default::default()
        };

        let (buffer, buffer_allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &allocate_info)? };

        Result::Ok(BoundBuffer {
            buffer,
            allocation: buffer_allocation,
        })
    }

    fn begin_single_time_commands(&self) -> VkResult<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        Result::Ok(command_buffer)
    }

    fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) -> VkResult<()> {
        let command_buffers = [command_buffer];
        let submit_infos = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        unsafe {
            self.device.end_command_buffer(command_buffer)?;
            self.device
                .queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.graphics_queue)?;
            self.device
                .free_command_buffers(self.command_pool, &command_buffers);
        }
        Result::Ok(())
    }

    fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) -> VkResult<()> {
        let command_buffer = self.begin_single_time_commands()?;

        let copy_regions = [vk::BufferCopy::default().size(size)];
        unsafe {
            self.device
                .cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);
        }

        self.end_single_time_commands(command_buffer)?;
        Result::Ok(())
    }

    fn copy_cpu_to_gpu<T: Copy>(
        &self,
        allocator: &mut MutexGuard<Allocator>,
        allocation: &mut Allocation,
        from: &[T],
        size: vk::DeviceSize,
    ) -> VkResult<()> {
        unsafe {
            let data = allocator.map_memory(allocation)? as *mut c_void;
            Self::copy_to_ptr(data, from, size);
            allocator.unmap_memory(allocation);
        }

        Result::Ok(())
    }

    unsafe fn copy_to_ptr<T: Copy>(to: *mut c_void, from: &[T], size: vk::DeviceSize) -> () {
        let mut index_slice = Align::new(to, align_of::<T>() as u64, size);
        index_slice.copy_from_slice(from);
    }

    fn create_image_views(&mut self) -> VkResult<()> {
        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .map(|&image| {
                self.create_image_view(
                    image,
                    self.swapchain_image_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<VkResult<Vec<vk::ImageView>>>()?;
        Result::Ok(())
    }

    fn create_descriptor_set_layout(&mut self) -> VkResult<()> {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [ubo_layout_binding, sampler_layout_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        self.descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        Result::Ok(())
    }

    fn create_graphics_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut vert_shader_code = Self::read_file("./shaders/vert.spv")?;
        let mut frag_shader_code = Self::read_file("./shaders/frag.spv")?;

        let vert_shader_module = Self::create_shader_module(&mut vert_shader_code, &self.device)?;
        let frag_shader_module = Self::create_shader_module(&mut frag_shader_code, &self.device)?;

        let p_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(p_name);
        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(p_name);
        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default().extent(self.swapchain_extent)];

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(self.msaa_samples)
            .sample_shading_enable(true)
            .min_sample_shading(0.2);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let descriptor_set_layouts = [self.descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0)
            .depth_stencil_state(&depth_stencil);

        let graphics_pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("failed to create graphics pipeline!")
        };
        let graphics_pipeline = graphics_pipelines[0];

        unsafe {
            self.device.destroy_shader_module(vert_shader_module, None);
            self.device.destroy_shader_module(frag_shader_module, None);
        }

        self.pipeline_layout = pipeline_layout;
        self.graphics_pipeline = graphics_pipeline;
        Result::Ok(())
    }

    fn create_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(self.swapchain_image_format)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachment_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let depth_attachment = vk::AttachmentDescription::default()
            .format(self.find_depth_format()?)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescription::default()
            .format(self.swapchain_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_resolve_ref = [vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_ref)
            .depth_stencil_attachment(&depth_attachment_ref)
            .resolve_attachments(&color_attachment_resolve_ref)];

        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )];

        let attachments = [color_attachment, depth_attachment, color_attachment_resolve];
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None)? };

        self.render_pass = render_pass;
        Result::Ok(())
    }

    fn create_framebuffers(
        &mut self,
        color_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
    ) -> VkResult<()> {
        self.swapchain_framebuffers = self
            .swapchain_image_views
            .iter()
            .map(|&view| {
                let attachments = [color_image_view, depth_image_view, view];

                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&attachments)
                    .width(self.swapchain_extent.width)
                    .height(self.swapchain_extent.height)
                    .layers(1);

                unsafe { self.device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect::<VkResult<Vec<vk::Framebuffer>>>()?;
        Result::Ok(())
    }

    fn create_command_pool(&mut self) -> VkResult<()> {
        let queue_family_indices = VulkanInit::find_queue_families(
            &self.instance,
            self.physical_device,
            &self.surface_instance,
            self.surface,
        )?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_family.unwrap());

        self.command_pool = unsafe { self.device.create_command_pool(&pool_info, None)? };
        Result::Ok(())
    }

    fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format, Box<dyn std::error::Error>> {
        let result = candidates
            .iter()
            .find(|&&format| {
                let props = unsafe {
                    self.instance
                        .get_physical_device_format_properties(self.physical_device, format)
                };
                match tiling {
                    vk::ImageTiling::LINEAR => {
                        (props.linear_tiling_features & features) == features
                    }
                    vk::ImageTiling::OPTIMAL => {
                        (props.optimal_tiling_features & features) == features
                    }
                    _ => false,
                }
            })
            .ok_or("failed to find supported format!")?;
        return Result::Ok(*result);
    }

    fn find_depth_format(&self) -> Result<vk::Format, Box<dyn std::error::Error>> {
        self.find_supported_format(
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    fn has_stencil_component(format: vk::Format) -> bool {
        match format {
            vk::Format::D32_SFLOAT_S8_UINT => true,
            vk::Format::D24_UNORM_S8_UINT => true,
            _ => false,
        }
    }

    fn create_color_resources(
        &mut self,
        allocator: &mut MutexGuard<Allocator>,
    ) -> Result<BoundImage, Box<dyn std::error::Error>> {
        let color_format = self.swapchain_image_format;

        let (color_image, color_allocation) = Self::create_image(
            allocator,
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            1,
            self.msaa_samples,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let color_image_view =
            self.create_image_view(color_image, color_format, vk::ImageAspectFlags::COLOR, 1)?;

        Result::Ok(BoundImage {
            image: color_image,
            image_view: color_image_view,
            allocation: color_allocation,
        })
    }

    fn create_depth_resources(
        &mut self,
        allocator: &mut MutexGuard<Allocator>,
    ) -> Result<BoundImage, Box<dyn std::error::Error>> {
        let depth_format = self.find_depth_format()?;
        let (depth_image, depth_image_allocation) = Self::create_image(
            allocator,
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            1,
            self.msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let depth_image_view =
            self.create_image_view(depth_image, depth_format, vk::ImageAspectFlags::DEPTH, 1)?;
        self.transition_image_layout(
            depth_image,
            depth_format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1,
        )?;
        Result::Ok(BoundImage {
            image: depth_image,
            image_view: depth_image_view,
            allocation: depth_image_allocation,
        })
    }

    fn create_image(
        allocator: &mut MutexGuard<Allocator>,
        width: u32,
        height: u32,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Image, vk_mem::Allocation), Box<dyn std::error::Error>> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: width,
                height: height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(num_samples);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            required_flags: properties,
            ..Default::default()
        };

        let (image, image_allocation) =
            unsafe { allocator.create_image(&image_info, &alloc_info)? };

        Result::Ok((image, image_allocation))
    }

    fn transition_image_layout(
        &self,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) -> VkResult<()> {
        let command_buffer = self.begin_single_time_commands()?;

        let aspect_mask = match new_layout {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                if Self::has_stencil_component(format) {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                } else {
                    vk::ImageAspectFlags::DEPTH
                }
            }
            _ => vk::ImageAspectFlags::COLOR,
        };
        let (src_access_mask, dst_access_mask, source_stage, destination_stage) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => {
                    (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    )
                }
                (old_layout, new_layout) => {
                    panic!(
                        "unsupported layout transition {:?} -> {:?}!",
                        old_layout, new_layout
                    );
                }
            };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_commands(command_buffer)?;
        Result::Ok(())
    }

    fn copy_buffer_to_image(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> VkResult<()> {
        let command_buffer = self.begin_single_time_commands()?;

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        self.end_single_time_commands(command_buffer)?;
        Result::Ok(())
    }

    fn create_texture_image(
        &mut self,
        allocator: &mut MutexGuard<Allocator>,
    ) -> Result<BoundImage, Box<dyn std::error::Error>> {
        let img = ImageReader::open(TEXTURE_PATH)?.decode()?.into_rgba8();
        let (tex_width, tex_height) = (img.width(), img.height());
        let pixels = img.as_bytes();
        let image_size = (tex_width as vk::DeviceSize) * (tex_height as vk::DeviceSize) * 4;
        self.mip_levels = cmp::max(tex_width, tex_height).ilog2() + 1;

        let mut staging_buffer = Self::create_buffer(
            allocator,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.copy_cpu_to_gpu(
            allocator,
            &mut staging_buffer.allocation,
            pixels,
            image_size,
        )?;

        let (texture_image, texture_image_allocation) = Self::create_image(
            allocator,
            tex_width,
            tex_height,
            self.mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        self.transition_image_layout(
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            self.mip_levels,
        )?;
        self.copy_buffer_to_image(staging_buffer.buffer, texture_image, tex_width, tex_height)?;
        // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

        self.generate_bitmaps(
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            tex_width as i32,
            tex_height as i32,
            self.mip_levels,
        )?;

        unsafe {
            allocator.destroy_buffer(staging_buffer.buffer, &mut staging_buffer.allocation);
        }

        let texture_image_view = self.create_image_view(
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            self.mip_levels,
        )?;

        Result::Ok(BoundImage {
            image: texture_image,
            image_view: texture_image_view,
            allocation: texture_image_allocation,
        })
    }

    fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> VkResult<vk::ImageView> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_flags)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe { self.device.create_image_view(&view_info, None)? };
        Result::Ok(image_view)
    }

    fn create_texture_sampler(&mut self) -> VkResult<()> {
        let properties = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        log::debug!(
            "max anisotropy: {}",
            properties.limits.max_sampler_anisotropy
        );

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(self.mip_levels as f32);

        self.texture_sampler = unsafe { self.device.create_sampler(&sampler_info, None)? };
        Result::Ok(())
    }

    fn create_vertex_buffer(
        &self,
        allocator: &mut MutexGuard<Allocator>,
        vertices: &Vec<Vertex>,
    ) -> Result<BoundBuffer, Box<dyn std::error::Error>> {
        let buffer_size: u64 = size_of::<Vertex>() as u64 * vertices.len() as u64;
        let mut staging_buffer = Self::create_buffer(
            allocator,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        self.copy_cpu_to_gpu(
            allocator,
            &mut staging_buffer.allocation,
            vertices,
            buffer_size,
        )?;

        let vertex_buffer = Self::create_buffer(
            allocator,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.copy_buffer(staging_buffer.buffer, vertex_buffer.buffer, buffer_size)?;

        unsafe {
            allocator.destroy_buffer(staging_buffer.buffer, &mut staging_buffer.allocation);
        }

        Result::Ok(vertex_buffer)
    }

    fn create_index_buffer(
        &self,
        allocator: &mut MutexGuard<Allocator>,
        indices: &Vec<u32>,
    ) -> Result<BoundBuffer, Box<dyn std::error::Error>> {
        let buffer_size: u64 = size_of::<u32>() as u64 * indices.len() as u64;
        let mut staging_buffer = Self::create_buffer(
            allocator,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        self.copy_cpu_to_gpu(
            allocator,
            &mut staging_buffer.allocation,
            indices,
            buffer_size,
        )?;

        let index_buffer = Self::create_buffer(
            allocator,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.copy_buffer(staging_buffer.buffer, index_buffer.buffer, buffer_size)?;

        unsafe {
            allocator.destroy_buffer(staging_buffer.buffer, &mut staging_buffer.allocation);
        }
        Result::Ok(index_buffer)
    }

    fn create_uniform_buffers(
        &self,
        allocator: &mut MutexGuard<Allocator>,
    ) -> Result<Vec<BoundBufferMapped>, Box<dyn std::error::Error>> {
        let buffer_size = size_of::<UniformBufferObject>() as vk::DeviceSize;

        let mut uniform_buffers = Vec::new();
        for _i in 0..MAX_FRAMES_IN_FLIGHT {
            let mut uniform_buffer = Self::create_buffer(
                allocator,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let uniform_buffer = BoundBufferMapped {
                ptr: unsafe {
                    allocator.map_memory(&mut uniform_buffer.allocation)? as *mut c_void
                },
                buffer: uniform_buffer.buffer,
                allocation: uniform_buffer.allocation,
            };

            uniform_buffers.push(uniform_buffer);
        }
        Result::Ok(uniform_buffers)
    }

    fn create_descriptor_pool(&mut self) -> VkResult<()> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT);

        self.descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };
        Result::Ok(())
    }

    fn create_descriptor_sets(
        &mut self,
        texture_image_view: vk::ImageView,
        uniform_buffers: &Vec<BoundBufferMapped>,
    ) -> VkResult<()> {
        let layouts = [self.descriptor_set_layout; MAX_FRAMES_IN_FLIGHT as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        self.descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };
        for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
            let buffer_infos = [vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i].buffer)
                .offset(0)
                .range(size_of::<UniformBufferObject>() as vk::DeviceSize)];
            let image_infos = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture_image_view)
                .sampler(self.texture_sampler)];
            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&buffer_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&image_infos),
            ];
            unsafe {
                self.device
                    .update_descriptor_sets(&descriptor_writes, &Vec::new());
            }
        }
        Result::Ok(())
    }

    fn create_command_buffers(&mut self) -> VkResult<()> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT);
        self.command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Result::Ok(())
    }

    fn create_sync_objects(&mut self) -> VkResult<()> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores
                .push(unsafe { self.device.create_semaphore(&semaphore_info, None)? });
            render_finished_semaphores
                .push(unsafe { self.device.create_semaphore(&semaphore_info, None)? });
            in_flight_fences.push(unsafe { self.device.create_fence(&fence_info, None)? });
        }

        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;
        self.in_flight_fences = in_flight_fences;
        Result::Ok(())
    }

    fn load_model(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mesh = ObjLoader::load(MODEL_PATH)?;
        let vertices = &mesh.vertices;
        let texcoords = &mesh.tex_coords;
        let mut unique_vertices = HashMap::new();
        mesh.indices.iter().for_each(|index| {
            let vertex = Vertex {
                pos: Vec3::from_slice(
                    &vertices[3 * index.vertex_index..3 * (index.vertex_index + 1)],
                ),
                tex_coord: if index.texcoord_index == usize::MAX {
                    Vec2::ZERO
                } else {
                    Vec2 {
                        x: texcoords[2 * index.texcoord_index + 0],
                        y: 1.0 - texcoords[2 * index.texcoord_index + 1],
                    }
                },
                color: Vec3::ONE,
            };
            if !unique_vertices.contains_key(&vertex) {
                unique_vertices.insert(vertex, self.vertices.len());
                self.vertices.push(vertex);
            }

            self.indices.push(unique_vertices[&vertex] as u32);
        });
        log::info!(
            "Loaded {} vertices, {} indices.",
            self.vertices.len(),
            self.indices.len()
        );
        Result::Ok(())
    }

    fn generate_bitmaps(
        &self,
        image: vk::Image,
        format: vk::Format,
        tex_width: i32,
        tex_height: i32,
        mip_levels: u32,
    ) -> VkResult<()> {
        let format_properties = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device, format)
        };
        if format_properties.optimal_tiling_features
            & vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            == vk::FormatFeatureFlags::empty()
        {
            panic!("texture image does not support linear blitting!");
        }

        let command_buffer = self.begin_single_time_commands()?;

        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut mip_width = tex_width;
        let mut mip_height = tex_height;
        for i in 1..mip_levels {
            let barrier = vk::ImageMemoryBarrier {
                subresource_range: vk::ImageSubresourceRange {
                    base_mip_level: i - 1,
                    ..barrier.subresource_range
                },
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                ..barrier
            };

            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            unsafe {
                self.device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_READ,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..barrier
            };

            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        let barrier = vk::ImageMemoryBarrier {
            subresource_range: vk::ImageSubresourceRange {
                base_mip_level: mip_levels - 1,
                ..barrier.subresource_range
            },
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..barrier
        };

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_commands(command_buffer)?;
        Result::Ok(())
    }

    pub fn cleanup(self, buffers: &mut VulkanBuffers, renderer: &mut ManuallyDrop<ImguiRenderer>) {
        log::debug!("Destroying Vulkan...");
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        unsafe {
            ManuallyDrop::drop(renderer);
            {
                let mut allocator = self.allocator.lock().expect("cannot lock allocator");
                self.cleanup_swapchain(&mut allocator, buffers);

                self.device.destroy_sampler(self.texture_sampler, None);
                self.device
                    .destroy_image_view(buffers.texture.image_view, None);

                allocator.destroy_image(buffers.texture.image, &mut buffers.texture.allocation);

                buffers.uniforms.iter_mut().for_each(|uniform_buffer| {
                    allocator.unmap_memory(&mut uniform_buffer.allocation);
                    allocator.destroy_buffer(uniform_buffer.buffer, &mut uniform_buffer.allocation);
                });

                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

                allocator.destroy_buffer(buffers.index.buffer, &mut buffers.index.allocation);
                allocator.destroy_buffer(buffers.vertex.buffer, &mut buffers.vertex.allocation);
            }
            drop(self.allocator);

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);

            zip(
                zip(
                    &self.image_available_semaphores,
                    &self.render_finished_semaphores,
                ),
                &self.in_flight_fences,
            )
            .for_each(
                |((&image_available_semaphore, &render_finished_semaphore), &in_flight_fence)| {
                    self.device
                        .destroy_semaphore(image_available_semaphore, None);
                    self.device
                        .destroy_semaphore(render_finished_semaphore, None);
                    self.device.destroy_fence(in_flight_fence, None);
                },
            );

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);

            #[cfg(debug_assertions)]
            {
                if ENABLE_VALIDATION_LAYERS {
                    self.debug_utils_loader
                        .destroy_debug_utils_messenger(self.debug_callback, None);
                }
            }

            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }

    fn update_uniform_buffer(
        &self,
        current_image: u32,
        frame_data: &FrameData,
        buffers: &VulkanBuffers,
    ) {
        let start_time = *START_TIME;
        let current_time = Instant::now();
        let time = current_time - start_time;

        let model = if frame_data.rotate_model {
            Mat4::from_axis_angle(Vec3::Z, time.as_secs_f32() * (90.0_f32.to_radians()))
        } else {
            Mat4::from_axis_angle(Vec3::Z, 0.0)
        };
        let ubo = UniformBufferObject {
            model,
            view: frame_data.cam_matrix,
            proj: Mat4::perspective_rh(
                frame_data.cam_fov.to_radians(),
                self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
                frame_data.cam_nearfar.x,
                frame_data.cam_nearfar.y,
            ),
        };

        unsafe {
            Self::copy_to_ptr(
                buffers.uniforms[current_image as usize].ptr,
                &[ubo],
                size_of::<UniformBufferObject>() as vk::DeviceSize,
            );
        }
    }
    pub fn draw_frame(
        &mut self,
        window: &Window,
        renderer: &mut ImguiRenderer,
        draw_data: &DrawData,
        frame_data: &FrameData,
        buffers: &mut VulkanBuffers,
    ) -> Result<(), vk::Result> {
        if !self.is_rendering {
            return Result::Ok(());
        }
        unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            )?;
        }

        let image_index = unsafe {
            let image_index_result = self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            let image_index = match image_index_result {
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window, buffers)?;
                    None
                }
                Err(_) => {
                    image_index_result?;
                    None
                }
                _ => Some(image_index_result.unwrap().0),
            };
            if image_index.is_none() {
                return Result::Ok(());
            }

            self.update_uniform_buffer(self.current_frame as u32, frame_data, buffers);

            // Only reset the fence if we are submitting work
            self.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            self.device.reset_command_buffer(
                self.command_buffers[self.current_frame],
                vk::CommandBufferResetFlags::empty(),
            )?;
            image_index.unwrap()
        };

        self.record_command_buffer(image_index, renderer, draw_data, frame_data, buffers)?;

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let submit_info: vk::SubmitInfo = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame],
            )?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.swapchain_device
                .queue_present(self.present_queue, &present_info)
        };
        match present_result {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.framebuffer_resized = false;
                self.recreate_swapchain(window, buffers)?;
            }
            Err(_) => {
                present_result?;
            }
            Ok(_) => {}
        }
        if self.framebuffer_resized {
            self.recreate_swapchain(window, buffers)?;
        }

        self.current_frame = (self.current_frame + 1) % (MAX_FRAMES_IN_FLIGHT as usize);
        Result::Ok(())
    }

    fn cleanup_swapchain(
        &self,
        allocator: &mut MutexGuard<Allocator>,
        buffers: &mut VulkanBuffers,
    ) {
        unsafe {
            self.device
                .destroy_image_view(buffers.color.image_view, None);
            allocator.destroy_image(buffers.color.image, &mut buffers.color.allocation);
            self.device
                .destroy_image_view(buffers.depth.image_view, None);
            allocator.destroy_image(buffers.depth.image, &mut buffers.depth.allocation);
        }

        self.swapchain_framebuffers.iter().for_each(|&framebuffer| {
            unsafe { self.device.destroy_framebuffer(framebuffer, None) };
        });

        self.swapchain_image_views.iter().for_each(|&image_view| {
            unsafe { self.device.destroy_image_view(image_view, None) };
        });

        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        window: &Window,
        buffers: &mut VulkanBuffers,
    ) -> Result<(), vk::Result> {
        unsafe {
            self.device.device_wait_idle()?;
        }

        let allocator_mutex = self.allocator.clone();
        let mut allocator = allocator_mutex.lock().expect("cannot lock allocator");
        self.cleanup_swapchain(&mut allocator, buffers);

        let (
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = VulkanInit::create_swapchain(
            window,
            &self.instance,
            &self.surface_instance,
            self.surface,
            self.physical_device,
            &self.device,
        )?;
        self.swapchain_device = swapchain_device;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_format = swapchain_image_format;
        self.swapchain_extent = swapchain_extent;

        self.create_image_views()?;
        buffers.color = self.create_color_resources(&mut allocator).unwrap();
        buffers.depth = self.create_depth_resources(&mut allocator).unwrap();
        self.create_framebuffers(buffers.color.image_view, buffers.depth.image_view)?;

        Result::Ok(())
    }

    pub fn pause_rendering(&mut self) {
        log::info!("Stopped rendering.");
        self.is_rendering = false;
    }

    pub fn resume_rendering(&mut self) {
        log::info!("Resumed rendering.");
        self.is_rendering = true;
    }

    pub fn wait(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}
