use crate::buffer::VulkanBuffers;
use crate::imgui::camera_manipulator::{Inputs, CAMERA_MANIPULATOR_INSTANCE};
use crate::ui::UiBuilder;
use crate::vulkan::Vulkan;
use imgui::{FontConfig, FontSource};
use imgui_rs_vulkan_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::NamedKey;
use winit::window::{Window, WindowId};

use std::mem::ManuallyDrop;
use std::process;
use std::time::{Duration, Instant};

struct AppContext {
    imgui: imgui::Context,
    imgui_renderer: ManuallyDrop<Renderer>,
    platform: WinitPlatform,
    vulkan: Vulkan,
    buffers: VulkanBuffers,
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
        let (window, vulkan, buffers) = Vulkan::new(event_loop)?;
        let (mut imgui, platform) = Self::init_imgui(&window)?;
        let imgui_renderer = vulkan.init_imgui_renderer(&mut imgui)?;

        Result::Ok(AppContext {
            imgui_renderer: ManuallyDrop::new(imgui_renderer),
            vulkan,
            buffers,
            window,
            imgui,
            platform,
        })
    }
}

pub struct App {
    context: Option<AppContext>,
    mouse_position: PhysicalPosition<f64>,
    frame_start: std::time::Instant,
    ui_builder: Box<dyn UiBuilder>,
    key_state: Inputs,
}

impl App {
    pub fn new(ui_builder: impl UiBuilder + 'static) -> Self {
        Self {
            context: None,
            frame_start: Instant::now(),
            ui_builder: Box::new(ui_builder),
            mouse_position: PhysicalPosition::default(),
            key_state: Inputs::default(),
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

    fn render(&mut self) {
        if let Some(context) = &mut self.context {
            let ui = context.imgui.new_frame();
            self.ui_builder.build(ui);
            context.platform.prepare_render(ui, &context.window);
            CAMERA_MANIPULATOR_INSTANCE
                .lock()
                .expect("cannot lock mutex")
                .update_anim();
            let imgui_draw_data = context.imgui.render();
            context
                .vulkan
                .draw_frame(
                    &context.window,
                    &mut context.imgui_renderer,
                    imgui_draw_data,
                    &self.ui_builder.frame_data(),
                    &mut context.buffers,
                )
                .unwrap_or_else(|err| {
                    eprintln!("{err}");
                    process::exit(1);
                });

            let frame_data = self.ui_builder.frame_data();
            if frame_data.frame_limiter {
                let frame_time = Duration::from_secs_f32(1.0 / frame_data.frame_limiter_fps as f32);
                let remaining_time = Instant::now() - self.frame_start + Duration::from_millis(1);
                if frame_time > remaining_time {
                    std::thread::sleep(frame_time - remaining_time);
                }
            }
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
            let mut camera_m = CAMERA_MANIPULATOR_INSTANCE
                .lock()
                .expect("cannot lock mutex");
            camera_m.set_window_size(
                context.vulkan.swapchain_extent.width,
                context.vulkan.swapchain_extent.height,
            );
            context.window.request_redraw();
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        let now = Instant::now();
        if let Some(context) = &mut self.context {
            context
                .imgui
                .io_mut()
                .update_delta_time(now - self.frame_start);
            self.frame_start = now;
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        window_event: WindowEvent,
    ) {
        match window_event {
            WindowEvent::CloseRequested => {
                log::info!("The close button was pressed; stopping");
                let context: Option<AppContext> = std::mem::replace(&mut self.context, None);
                if let Some(mut context) = context {
                    let vulkan: Vulkan = context.vulkan;
                    vulkan.cleanup(&mut context.buffers, &mut context.imgui_renderer);
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(context) = &mut self.context {
                    context.window.request_redraw();
                }
                self.pass_event_to_platform(window_event);
            }
            WindowEvent::Resized(_) => {
                if let Some(context) = &mut self.context {
                    context
                        .vulkan
                        .recreate_swapchain(&context.window, &mut context.buffers)
                        .unwrap_or_else(|err| {
                            eprintln!("{err}");
                            process::exit(1);
                        });
                    CAMERA_MANIPULATOR_INSTANCE
                        .lock()
                        .expect("cannot lock mutex")
                        .set_window_size(
                            context.vulkan.swapchain_extent.width,
                            context.vulkan.swapchain_extent.height,
                        );
                }
                self.pass_event_to_platform(window_event);
            }
            WindowEvent::Occluded(true) => {
                if let Some(context) = &mut self.context {
                    context.vulkan.pause_rendering();
                }
                self.pass_event_to_platform(window_event);
            }
            WindowEvent::Occluded(false) => {
                if let Some(context) = &mut self.context {
                    context.vulkan.resume_rendering();
                }
                self.pass_event_to_platform(window_event);
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                ref event,
                is_synthetic: _,
            } => {
                if let Some(context) = &self.context {
                    if context.imgui.io().want_capture_keyboard {
                        self.pass_event_to_platform(window_event);
                        return;
                    }
                }

                let key_state = &mut self.key_state;
                let pressed = event.state == ElementState::Pressed;
                match event.logical_key {
                    winit::keyboard::Key::Named(NamedKey::Shift) => key_state.shift = pressed,
                    winit::keyboard::Key::Named(NamedKey::Alt) => key_state.alt = pressed,
                    winit::keyboard::Key::Named(NamedKey::Control) => key_state.ctrl = pressed,
                    _ => {}
                }
                self.pass_event_to_platform(window_event);
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.mouse_position = position;
                CAMERA_MANIPULATOR_INSTANCE
                    .lock()
                    .expect("cannot unlock mutex")
                    .mouse_move(
                        self.mouse_position.x as f32,
                        self.mouse_position.y as f32,
                        &self.key_state,
                    );

                self.pass_event_to_platform(window_event);
            }
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                if let Some(context) = &self.context {
                    if context.imgui.io().want_capture_mouse {
                        self.pass_event_to_platform(window_event);
                        return;
                    }
                }

                let key_state = &mut self.key_state;
                let pressed = state == ElementState::Pressed;
                let mut mouse_down = true;
                match button {
                    MouseButton::Left => key_state.lmb = pressed,
                    MouseButton::Right => key_state.rmb = pressed,
                    MouseButton::Middle => key_state.mmb = pressed,
                    _ => mouse_down = false,
                }
                if pressed && mouse_down {
                    CAMERA_MANIPULATOR_INSTANCE
                        .lock()
                        .expect("cannot unlock mutex")
                        .set_mouse_position(
                            self.mouse_position.x as f32,
                            self.mouse_position.y as f32,
                        );
                }
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
            } => {
                if let Some(context) = &self.context {
                    if context.imgui.io().want_capture_mouse {
                        self.pass_event_to_platform(window_event);
                        return;
                    }
                }

                let mut camera_m = CAMERA_MANIPULATOR_INSTANCE
                    .lock()
                    .expect("cannot unlock mutex");
                let delta_positive = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => y > 0.0,
                    MouseScrollDelta::PixelDelta(PhysicalPosition { x: _, y }) => y > 0.0,
                };
                camera_m.wheel(if delta_positive { 1 } else { -1 }, self.key_state);
            }
            event => {
                self.pass_event_to_platform(event);
            }
        }
    }
}
