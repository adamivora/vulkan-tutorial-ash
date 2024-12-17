use crate::vulkan::Vulkan;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use std::process;

#[derive(Default)]
pub struct App {
    window: Option<Window>,
    vulkan: Option<Vulkan>,
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
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                self.vulkan = None;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(vulkan) = &mut self.vulkan {
                    vulkan
                        .draw_frame(self.window.as_ref().unwrap())
                        .unwrap_or_else(|err| {
                            eprintln!("{err}");
                            process::exit(1);
                        });
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(_) => {
                if let Some(vulkan) = &mut self.vulkan {
                    vulkan
                        .recreate_swapchain(self.window.as_ref().unwrap())
                        .unwrap_or_else(|err| {
                            eprintln!("{err}");
                            process::exit(1);
                        });
                }
            }
            WindowEvent::Occluded(true) => {
                if let Some(vulkan) = &mut self.vulkan {
                    vulkan.pause_rendering();
                }
            }
            WindowEvent::Occluded(false) => {
                if let Some(vulkan) = &mut self.vulkan {
                    vulkan.resume_rendering();
                }
            }
            _ => (),
        }
    }
}
