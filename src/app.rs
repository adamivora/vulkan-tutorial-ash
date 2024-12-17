use crate::ui::UiBuilder;
use crate::vulkan::Vulkan;
use imgui::{FontConfig, FontSource};
use imgui_rs_vulkan_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use std::process;
use std::time::Instant;

struct AppContext {
    imgui: imgui::Context,
    imgui_renderer: Renderer,
    platform: WinitPlatform,
    vulkan: Vulkan,
    window: Window,
}

impl AppContext {
    fn init_imgui(
        window: &Window,
    ) -> Result<(imgui::Context, WinitPlatform), Box<dyn std::error::Error>> {
        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);

        let mut platform = WinitPlatform::new(&mut imgui);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        }]);
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);

        Result::Ok((imgui, platform))
    }

    fn init(event_loop: &ActiveEventLoop) -> Result<AppContext, Box<dyn std::error::Error>> {
        let (window, vulkan) = Vulkan::new(event_loop)?;
        let (mut imgui, platform) = Self::init_imgui(&window)?;
        let imgui_renderer = vulkan.init_imgui_renderer(&mut imgui)?;

        Result::Ok(AppContext {
            imgui_renderer,
            vulkan,
            window,
            imgui,
            platform,
        })
    }
}

pub struct App {
    context: Option<AppContext>,
    last_frame: std::time::Instant,
    ui_builder: Box<dyn UiBuilder>,
}

impl App {
    pub fn new(ui_builder: impl UiBuilder + 'static) -> Self {
        Self {
            context: None,
            last_frame: Instant::now(),
            ui_builder: Box::new(ui_builder),
        }
    }

    fn pass_event_to_platform(&mut self, event: WindowEvent) {
        if let Some(context) = &mut self.context {
            let event: winit::event::Event<App> = winit::event::Event::WindowEvent {
                window_id: context.window.id(),
                event: event,
            };
            context
                .platform
                .handle_event(context.imgui.io_mut(), &context.window, &event);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let context = AppContext::init(event_loop).unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
        self.context = Some(context);
        if let Some(context) = &mut self.context {
            context.window.request_redraw();
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        let now = Instant::now();
        if let Some(context) = &mut self.context {
            context
                .imgui
                .io_mut()
                .update_delta_time(now - self.last_frame);
            self.last_frame = now;
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(context) = &mut self.context {
            context
                .platform
                .prepare_frame(context.imgui.io_mut(), &context.window)
                .expect("Failed to prepare frame");
            context.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("The close button was pressed; stopping");
                if let Some(context) = &mut self.context {
                    context.vulkan.wait();
                    self.context = None;
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(context) = &mut self.context {
                    let ui = context.imgui.frame();
                    self.ui_builder.build(ui);
                    context.platform.prepare_render(ui, &context.window);
                    let imgui_draw_data = context.imgui.render();
                    context
                        .vulkan
                        .draw_frame(
                            &context.window,
                            &mut context.imgui_renderer,
                            imgui_draw_data,
                        )
                        .unwrap_or_else(|err| {
                            eprintln!("{err}");
                            process::exit(1);
                        });

                    context.window.request_redraw();
                }
            }
            WindowEvent::Resized(_) => {
                if let Some(context) = &mut self.context {
                    context
                        .vulkan
                        .recreate_swapchain(&context.window)
                        .unwrap_or_else(|err| {
                            eprintln!("{err}");
                            process::exit(1);
                        });
                }
                self.pass_event_to_platform(event);
            }
            WindowEvent::Occluded(true) => {
                if let Some(context) = &mut self.context {
                    context.vulkan.pause_rendering();
                }
            }
            WindowEvent::Occluded(false) => {
                if let Some(context) = &mut self.context {
                    context.vulkan.resume_rendering();
                }
            }
            event => {
                self.pass_event_to_platform(event);
            }
        }
    }
}
