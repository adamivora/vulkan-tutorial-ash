use ash::vk;

use ash::Entry;
use ash::Instance;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};

use std::ffi::CString;
use std::process;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[derive(Default)]
struct App {
    window: Option<Window>,
    instance: Option<Instance>,
}

impl App {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan")
            .with_resizable(false);
        self.window = Some(event_loop.create_window(window_attributes).unwrap());
    }

    fn create_instance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let application_name = CString::new("Hello Triangle").unwrap();
        let engine_name = CString::new("No Engine").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_0);

        let mut extensions = ash_window::enumerate_required_extensions(
            self.window.as_ref().unwrap().display_handle()?.as_raw(),
        )
        .unwrap()
        .to_vec();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let layer_names = Vec::new();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names)
            .flags(create_flags);

        let entry = Entry::linked();

        self.instance = Some(unsafe { entry.create_instance(&create_info, None)? });
        let supported_extensions = unsafe { entry.enumerate_instance_extension_properties(None)? };

        println!(
            "Supported extensions:\n{}",
            supported_extensions
                .iter()
                .fold(String::new(), |acc, &num| acc
                    + "\t"
                    + num.extension_name_as_c_str().unwrap().to_str().unwrap()
                    + "\n")
        );

        Result::Ok(())
    }

    fn init_vulkan(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.create_instance()
    }

    fn cleanup(&mut self) {
        unsafe {
            if self.instance.is_some() {
                self.instance.as_mut().unwrap().destroy_instance(None);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        App::init_window(self, event_loop);
        App::init_vulkan(self).unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                App::cleanup(self);
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
