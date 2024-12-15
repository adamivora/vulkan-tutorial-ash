use ash::vk::{self, SurfaceFormatKHR, KHR_SWAPCHAIN_NAME};

use ash::ext::debug_utils;
use ash::Entry;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

use std::borrow::Cow;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::{process, u32};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: [&CStr; 1] =
    [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

const DEVICE_EXTENSIONS: [&CStr; 1] = [KHR_SWAPCHAIN_NAME];

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
        available_formats: &Vec<SurfaceFormatKHR>,
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
        if present_mode.is_none() {
            vk::PresentModeKHR::FIFO
        } else {
            *present_mode.unwrap()
        }
    }

    fn choose_swap_extent(&self) -> vk::Extent2D {
        match self.capabilities.current_extent.width {
            u32::MAX => {
                let actual_width = WIDTH.clamp(
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                );
                let actual_height = HEIGHT.clamp(
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

struct Vulkan {
    instance: ash::Instance,
    debug_utils_loader: ash::ext::debug_utils::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
    surface_instance: ash::khr::surface::Instance,
    swapchain_device: ash::khr::swapchain::Device,

    device: ash::Device,
    surface: vk::SurfaceKHR,
    _present_queue: vk::Queue,
    _graphics_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    _swapchain_image_format: vk::Format,
    _swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan: Option<Vulkan>,
}

impl Vulkan {
    fn new(event_loop: &ActiveEventLoop) -> Result<(Window, Vulkan), Box<dyn std::error::Error>> {
        let window = Vulkan::init_window(event_loop);
        let vulkan = Vulkan::init_vulkan(&window)?;
        Result::Ok((window, vulkan))
    }

    fn init_window(event_loop: &ActiveEventLoop) -> Window {
        let window_attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan")
            .with_resizable(false);
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
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());

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
        let extent = swapchain_support.choose_swap_extent();

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

    fn init_vulkan(window: &Window) -> Result<Vulkan, Box<dyn std::error::Error>> {
        let instance = Vulkan::create_instance(window)?;
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
            &instance,
            &surface_instance,
            surface,
            physical_device,
            &device,
        )?;
        let swapchain_image_views =
            Vulkan::create_image_views(&device, &swapchain_images, swapchain_image_format)?;

        Result::Ok(Self {
            instance,
            device,
            debug_utils_loader,
            debug_callback,
            surface_instance,
            surface,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            swapchain_device,
            swapchain,
            _swapchain_images: swapchain_images,
            _swapchain_image_format: swapchain_image_format,
            _swapchain_extent: swapchain_extent,
            swapchain_image_views,
        })
    }

    fn cleanup(&mut self) {
        unsafe {
            self.swapchain_image_views.iter().for_each(|&image_view| {
                self.device.destroy_image_view(image_view, None);
            });
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            if ENABLE_VALIDATION_LAYERS {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_callback, None);
            }
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl App {}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (window, vulkan) = Vulkan::new(event_loop).unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
        self.window = Some(window);
        self.vulkan = Some(vulkan);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                self.vulkan = None;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {}
            _ => (),
        }
    }
}

fn main() -> std::result::Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}
