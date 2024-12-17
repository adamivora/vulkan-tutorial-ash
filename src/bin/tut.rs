use imgui::Condition;
use rust_vulkan::ui::UiBuilder;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

use rust_vulkan::app::App;

struct UiTut {
    value: usize,
}

impl Default for UiTut {
    fn default() -> Self {
        Self { value: 0 }
    }
}

impl UiBuilder for UiTut {
    fn build(&mut self, ui: &mut imgui::Ui) {
        let choices = ["test test this is 1", "test test this is 2"];
        ui.window("Hello world")
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(|| {
                ui.text_wrapped("Hello world!");
                if ui.button(choices[self.value]) {
                    self.value = (self.value + 1) % 2;
                }

                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
    }
}

fn main() -> std::result::Result<(), EventLoopError> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(UiTut::default());

    event_loop.run_app(&mut app)
}
