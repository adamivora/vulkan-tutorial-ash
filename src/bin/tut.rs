use imgui::Condition;
use rust_vulkan::ui::UiBuilder;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

use glam::Vec4;
use rust_vulkan::app::App;
use rust_vulkan::frame_data::FrameData;

struct UiTut {
    color: Vec4,
    frame_limiter: bool,
    frame_limiter_limit: i32,
}

impl Default for UiTut {
    fn default() -> Self {
        Self {
            color: Vec4::new(0.0, 0.0, 0.0, 0.0),
            frame_limiter: true,
            frame_limiter_limit: 60 as i32,
        }
    }
}

impl UiBuilder for UiTut {
    fn build(&mut self, ui: &mut imgui::Ui) {
        ui.window("Hello world")
            .always_auto_resize(true)
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(|| {
                ui.text_wrapped("Hello world!");
                ui.separator();
                ui.checkbox("Frame Limiter", &mut self.frame_limiter);
                if self.frame_limiter {
                    ui.same_line();
                    ui.set_next_item_width(75.0);
                    ui.slider("FPS Limit", 10, 120, &mut self.frame_limiter_limit);
                }
                ui.text_wrapped(format!("FPS: {:.0}", ui.io().framerate));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                if mouse_pos[0] != f32::MIN {
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                }

                ui.separator();
                ui.color_picker4("##picker", &mut self.color);
            });
    }

    fn frame_data(&self) -> FrameData {
        return FrameData {
            bgcolor: self.color,
            frame_limiter: self.frame_limiter,
            frame_limiter_fps: self.frame_limiter_limit as u32,
        };
    }
}

fn main() -> std::result::Result<(), EventLoopError> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(UiTut::default());

    event_loop.run_app(&mut app)
}
