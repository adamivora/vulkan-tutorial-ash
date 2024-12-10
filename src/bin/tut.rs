use ash::util::*;
use ash::vk;

use winit::application::ApplicationHandler;
use winit::dpi::Size;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl App {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Vulkan")
            .with_resizable(false);
        self.window = Some(event_loop.create_window(window_attributes).unwrap());
    }

    fn init_vulkan(&mut self) {}

    fn cleanup(&self) {}
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        App::init_window(self, event_loop);
        App::init_vulkan(self);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
                App::cleanup(self);
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
