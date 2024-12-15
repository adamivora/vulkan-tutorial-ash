use ash::vk;

use ash::ext::debug_utils;
use ash::Entry;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};

use std::borrow::Cow;
use std::convert::TryFrom;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::process;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: [&CStr; 1] =
    [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
const ENABLE_VALIDATION_LAYERS: bool = cfg!(debug_assertions);

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
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

struct Vulkan {
    instance: ash::Instance,
    device: ash::Device,
    debug_utils_loader: ash::ext::debug_utils::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
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

    fn is_device_suitable(
        instance: &ash::Instance,
        device: &ash::vk::PhysicalDevice,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let indices = Vulkan::find_queue_families(instance, device)?;
        Result::Ok(indices.is_complete())
    }

    fn find_queue_families(
        instance: &ash::Instance,
        device: &ash::vk::PhysicalDevice,
    ) -> Result<QueueFamilyIndices, Box<dyn std::error::Error>> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };
        let graphics_family = queue_families.iter().position(|queue_family| {
            queue_family.queue_flags & ash::vk::QueueFlags::GRAPHICS
                == ash::vk::QueueFlags::GRAPHICS
        });

        let graphics_family = if graphics_family.is_some() {
            Some(u32::try_from(graphics_family.unwrap())?)
        } else {
            None
        };

        // Logic to find graphics queue family
        Result::Ok(QueueFamilyIndices { graphics_family })
    }

    fn pick_physical_device(
        instance: &ash::Instance,
    ) -> Result<ash::vk::PhysicalDevice, Box<dyn std::error::Error>> {
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical_device = devices
            .iter()
            .find(|&device| {
                Vulkan::is_device_suitable(instance, device)
                    .expect("failed to find a suitable GPU!")
            })
            .ok_or("failed to find a suitable GPU!")?;
        Result::Ok(physical_device.clone())
    }

    fn check_validation_layer_support() -> Result<bool, Box<dyn std::error::Error>> {
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
    ) -> Result<ash::Device, Box<dyn std::error::Error>> {
        let indices = Vulkan::find_queue_families(instance, &physical_device)?;
        let queue_create_infos = [vk::DeviceQueueCreateInfo {
            queue_family_index: indices.graphics_family.unwrap(),
            queue_count: 1,
            ..vk::DeviceQueueCreateInfo::default()
        }
        .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::default();

        let mut extensions = Vec::new();
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&extensions);

        let device = unsafe { instance.create_device(physical_device, &create_info, None) }?;
        Result::Ok(device)
    }

    fn init_vulkan(window: &Window) -> Result<Vulkan, Box<dyn std::error::Error>> {
        let instance = Vulkan::create_instance(window)?;
        let physical_device = Vulkan::pick_physical_device(&instance)?;
        let device = Vulkan::create_logical_device(&instance, physical_device)?;
        let (loader, callback) = Vulkan::setup_debug_messenger(&instance)?;

        Result::Ok(Self {
            instance,
            device,
            debug_utils_loader: loader,
            debug_callback: callback,
        })
    }

    fn cleanup(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            if ENABLE_VALIDATION_LAYERS {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_callback, None);
            }
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
