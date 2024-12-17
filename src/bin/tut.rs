use imgui::Condition;
use rust_vulkan::ui::UiBuilder;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

use glam::Vec4;
use rust_vulkan::app::App;
use rust_vulkan::frame_data::FrameData;

struct UiTut {
    value: usize,
    color: Vec4,
}

impl Default for UiTut {
    fn default() -> Self {
        Self {
            value: 0,
            color: Vec4::new(0.0, 0.0, 0.0, 0.0),
        }
    }
}

impl UiBuilder for UiTut {
    fn build(&mut self, ui: &mut imgui::Ui) {
        let choices = ["test test this is 1", "test test this is 2"];
        ui.window("Hello world")
            .always_auto_resize(true)
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(|| {
                ui.text_wrapped("Hello world!");
                if ui.button(choices[self.value]) {
                    self.value = (self.value + 1) % 2;
                }

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
