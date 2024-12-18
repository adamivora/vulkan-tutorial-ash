use crate::frame_data::FrameData;
#[cfg(debug_assertions)]
use ash::ext::debug_utils;
use ash::vk;
use ash::Entry;
use glam::{Vec2, Vec3};
use imgui::DrawData;
use imgui_rs_vulkan_renderer::Renderer;
use std::borrow::Cow;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::ffi::{CStr, CString};
use std::fs;
use std::mem::offset_of;
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};
use std::u32;
use vk_mem::{Allocator, AllocatorCreateInfo};
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    const fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    fn get_binding_description() -> vk::VertexInputBindingDescription {
        let binding_description: vk::VertexInputBindingDescription =
            vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX);
        binding_description
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let attribute_descriptions: [vk::VertexInputAttributeDescription; 2] = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
        ];
        attribute_descriptions
    }
}

const VERTICES: [Vertex; 3] = [
    Vertex::new(Vec2::new(0.5, -0.5), Vec3::new(1.0, 0.0, 0.0)),
    Vertex::new(Vec2::new(0.5, 0.5), Vec3::new(0.0, 1.0, 0.0)),
    Vertex::new(Vec2::new(-0.5, 0.5), Vec3::new(0.0, 0.0, 1.0)),
];

const VALIDATION_LAYERS: [&CStr; 1] =
    [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

const DEVICE_EXTENSIONS: [&CStr; 1] = [vk::KHR_SWAPCHAIN_NAME];

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
    instance: ash::Instance,
    #[cfg(debug_assertions)]
    debug_utils_loader: ash::ext::debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_callback: vk::DebugUtilsMessengerEXT,
    surface_instance: ash::khr::surface::Instance,
    swapchain_device: ash::khr::swapchain::Device,

    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    present_queue: vk::Queue,
    graphics_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    framebuffer_resized: bool,
    is_rendering: bool,
}

impl Vulkan {
    pub fn new(
        event_loop: &ActiveEventLoop,
    ) -> Result<(Window, Vulkan), Box<dyn std::error::Error>> {
        let window = Vulkan::init_window(event_loop);
        let vulkan = Vulkan::init_vulkan(&window)?;
        Result::Ok((window, vulkan))
    }

    fn init_window(event_loop: &ActiveEventLoop) -> Window {
        let window_attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan")
            .with_resizable(true);
        event_loop.create_window(window_attributes).unwrap()
    }

    fn create_instance(window: &Window) -> Result<ash::Instance, Box<dyn std::error::Error>> {
        if ENABLE_VALIDATION_LAYERS && !Vulkan::check_validation_layer_support()? {
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
            debug_create_info = Vulkan::populate_debug_messenger_create_info(debug_create_info);
            create_info = create_info.push_next(&mut debug_create_info);
        }

        let entry = Entry::linked();

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
        let indices = Vulkan::find_queue_families(instance, device, surface_instance, surface)?;
        let extensions_supported = Vulkan::check_device_extension_support(instance, device)?;
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
        Result::Ok(indices.is_complete() && extensions_supported && swapchain_adequate)
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

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<ash::vk::PhysicalDevice, Box<dyn std::error::Error>> {
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical_device = devices
            .iter()
            .find(|&&device| {
                Vulkan::is_device_suitable(instance, device, surface_instance, surface)
                    .expect("failed to find a suitable GPU!")
            })
            .ok_or("failed to find a suitable GPU!")?;
        Result::Ok(physical_device.clone())
    }

    fn check_validation_layer_support() -> Result<bool, vk::Result> {
        let entry = Entry::linked();
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
        instance: &ash::Instance,
    ) -> Result<
        (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT),
        Box<dyn std::error::Error>,
    > {
        if !ENABLE_VALIDATION_LAYERS {
            return Result::Err("validation layers not enabled".into());
        }

        let create_info = Vulkan::populate_debug_messenger_create_info(
            vk::DebugUtilsMessengerCreateInfoEXT::default(),
        );
        let entry = Entry::linked();
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
            Vulkan::find_queue_families(instance, physical_device, surface_instance, surface)?;

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

        let device_features = vk::PhysicalDeviceFeatures::default();

        let mut extensions: Vec<*const i8> =
            DEVICE_EXTENSIONS.iter().map(|&ext| ext.as_ptr()).collect();
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
        window: &Window,
        instance: &ash::Instance,
    ) -> Result<(ash::khr::surface::Instance, vk::SurfaceKHR), Box<dyn std::error::Error>> {
        let entry = Entry::linked();
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

        let indices = Vulkan::find_queue_families(instance, device, surface_instance, surface)?;
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

    fn create_image_views(
        device: &ash::Device,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        swapchain_images
            .iter()
            .map(|&image| unsafe {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_image_format)
                    .components(
                        vk::ComponentMapping::default()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                device.create_image_view(&create_info, None)
            })
            .collect()
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

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn std::error::Error>> {
        let mut vert_shader_code = Vulkan::read_file("./shaders/vert.spv")?;
        let mut frag_shader_code = Vulkan::read_file("./shaders/frag.spv")?;

        let vert_shader_module = Vulkan::create_shader_module(&mut vert_shader_code, device)?;
        let frag_shader_module = Vulkan::create_shader_module(&mut frag_shader_code, device)?;

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
            .width(swapchain_extent.width as f32)
            .height(swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default().extent(swapchain_extent)];

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
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

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

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

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
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("failed to create graphics pipeline!")
        };
        let graphics_pipeline = graphics_pipelines[0];

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        Result::Ok((pipeline_layout, graphics_pipeline))
    }

    fn create_render_pass(
        device: &ash::Device,
        swapchain_image_format: vk::Format,
    ) -> Result<vk::RenderPass, vk::Result> {
        let color_attachments = [vk::AttachmentDescription::default()
            .format(swapchain_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

        let color_attachment_refs = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)];

        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&color_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };

        Result::Ok(render_pass)
    }

    fn create_framebuffers(
        device: &ash::Device,
        swapchain_image_views: &Vec<vk::ImageView>,
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> Result<Vec<vk::Framebuffer>, vk::Result> {
        swapchain_image_views
            .iter()
            .map(|&view| {
                let attachments = [view];

                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect()
    }

    fn create_command_pool(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::CommandPool, vk::Result> {
        let queue_family_indices =
            Vulkan::find_queue_families(instance, physical_device, surface_instance, surface)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_family.unwrap());

        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        Result::Ok(command_pool)
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        Result::Ok(command_buffers)
    }

    fn create_sync_objects(
        device: &ash::Device,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>), vk::Result> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_info, None)? });
            render_finished_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_info, None)? });
            in_flight_fences.push(unsafe { device.create_fence(&fence_info, None)? });
        }

        Result::Ok((
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        ))
    }

    fn record_command_buffer(
        &self,
        image_index: u32,
        imgui_renderer: &mut Renderer,
        imgui_draw_data: &DrawData,
        frame_data: &FrameData,
    ) -> Result<(), vk::Result> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        let clear_color_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: frame_data.bgcolor.into(),
            },
        }];
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(
                *self
                    .swapchain_framebuffers
                    .get(image_index as usize)
                    .unwrap(),
            )
            .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
            .clear_values(&clear_color_values);

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

        let viewport = vk::Viewport::default()
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
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
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
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
    ) -> Result<Renderer, Box<dyn std::error::Error>> {
        let imgui_renderer = {
            let allocator = {
                let allocator_create_info =
                    AllocatorCreateInfo::new(&self.instance, &self.device, self.physical_device);

                unsafe { Allocator::new(allocator_create_info)? }
            };

            Renderer::with_vk_mem_allocator(
                Arc::new(Mutex::new(allocator)),
                self.device.clone(),
                self.graphics_queue,
                self.command_pool,
                self.render_pass,
                imgui,
                Some(imgui_rs_vulkan_renderer::Options {
                    in_flight_frames: MAX_FRAMES_IN_FLIGHT as usize,
                    ..Default::default()
                }),
            )?
        };
        Result::Ok(imgui_renderer)
    }

    fn init_vulkan(window: &Window) -> Result<Vulkan, Box<dyn std::error::Error>> {
        let instance = Vulkan::create_instance(window)?;
        #[cfg(debug_assertions)]
        let (debug_utils_loader, debug_callback) = Vulkan::setup_debug_messenger(&instance)?;
        let (surface_instance, surface) = Vulkan::create_surface(window, &instance)?;
        let physical_device = Vulkan::pick_physical_device(&instance, &surface_instance, surface)?;
        let (device, graphics_queue, present_queue) =
            Vulkan::create_logical_device(&instance, physical_device, &surface_instance, surface)?;
        let (
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = Vulkan::create_swapchain(
            window,
            &instance,
            &surface_instance,
            surface,
            physical_device,
            &device,
        )?;
        let swapchain_image_views =
            Vulkan::create_image_views(&device, &swapchain_images, swapchain_image_format)?;
        let render_pass = Vulkan::create_render_pass(&device, swapchain_image_format)?;
        let (pipeline_layout, graphics_pipeline) =
            Vulkan::create_graphics_pipeline(&device, swapchain_extent, render_pass)?;
        let swapchain_framebuffers = Vulkan::create_framebuffers(
            &device,
            &swapchain_image_views,
            swapchain_extent,
            render_pass,
        )?;
        let command_pool = Vulkan::create_command_pool(
            &instance,
            &device,
            physical_device,
            &surface_instance,
            surface,
        )?;
        let command_buffers = Vulkan::create_command_buffers(&device, command_pool)?;
        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Vulkan::create_sync_objects(&device)?;

        Result::Ok(Self {
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
            swapchain_image_views,
            swapchain_framebuffers,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            framebuffer_resized: false,
            is_rendering: true,
        })
    }

    fn cleanup(&mut self) {
        unsafe {
            self.cleanup_swapchain();

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i as usize], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i as usize], None);
                self.device
                    .destroy_fence(self.in_flight_fences[i as usize], None);
            }

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

    pub fn draw_frame(
        &mut self,
        window: &Window,
        renderer: &mut Renderer,
        draw_data: &DrawData,
        frame_data: &FrameData,
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
                    self.recreate_swapchain(window)?;
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

            // Only reset the fence if we are submitting work
            self.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            self.device.reset_command_buffer(
                self.command_buffers[self.current_frame],
                vk::CommandBufferResetFlags::empty(),
            )?;
            image_index.unwrap()
        };

        self.record_command_buffer(image_index, renderer, draw_data, frame_data)?;

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
                self.recreate_swapchain(window)?;
            }
            Err(_) => {
                present_result?;
            }
            Ok(_) => {}
        }
        if self.framebuffer_resized {
            self.recreate_swapchain(window)?;
        }

        self.current_frame = (self.current_frame + 1) % (MAX_FRAMES_IN_FLIGHT as usize);
        Result::Ok(())
    }

    fn cleanup_swapchain(&self) {
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

    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), vk::Result> {
        unsafe {
            self.device.device_wait_idle()?;
        }

        self.cleanup_swapchain();

        let (
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = Vulkan::create_swapchain(
            window,
            &self.instance,
            &self.surface_instance,
            self.surface,
            self.physical_device,
            &self.device,
        )?;
        let swapchain_image_views =
            Vulkan::create_image_views(&self.device, &swapchain_images, swapchain_image_format)?;
        let swapchain_framebuffers = Vulkan::create_framebuffers(
            &self.device,
            &swapchain_image_views,
            swapchain_extent,
            self.render_pass,
        )?;

        self.swapchain_device = swapchain_device;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_format = swapchain_image_format;
        self.swapchain_extent = swapchain_extent;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain_framebuffers = swapchain_framebuffers;

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

impl Drop for Vulkan {
    fn drop(&mut self) {
        log::debug!("Destroying Vulkan...");
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.cleanup();
    }
}
