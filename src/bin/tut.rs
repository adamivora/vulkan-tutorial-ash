use imgui::Condition;
use rust_vulkan::ui::UiBuilder;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

use glam::{Vec2, Vec3, Vec4};
use rust_vulkan::app::App;
use rust_vulkan::frame_data::FrameData;

struct UiTut {
    color: Vec4,
    frame_limiter: bool,
    frame_limiter_limit: i32,
    rotate_model: bool,
    cam_eye: Vec3,
    cam_center: Vec3,
    cam_up: Vec3,
    cam_nearfar: Vec2,
}

impl Default for UiTut {
    fn default() -> Self {
        Self {
            color: Vec4::new(0.0, 0.0, 0.0, 0.0),
            frame_limiter: true,
            frame_limiter_limit: 60 as i32,
            rotate_model: true,
            cam_eye: Vec3::splat(2.0),
            cam_center: Vec3::new(0.5, 0.0, 0.5),
            cam_up: Vec3::Z,
            cam_nearfar: Vec2::new(0.1, 10.0),
        }
    }
}

impl UiBuilder for UiTut {
    fn build(&mut self, ui: &mut imgui::Ui) {
        ui.window("Debug")
            .position([0.0, 0.0], Condition::FirstUseEver)
            .always_auto_resize(true)
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(|| {
                ui.checkbox("Frame Limiter", &mut self.frame_limiter);
                if self.frame_limiter {
                    ui.same_line();
                    ui.set_next_item_width(75.0);
                    ui.slider("FPS Limit", 10, 120, &mut self.frame_limiter_limit);
                }
                ui.text_wrapped(format!("FPS: {:.0}", ui.io().framerate));
                ui.separator();
                ui.checkbox("Rotate Model", &mut self.rotate_model);

                ui.separator();
                ui.text_wrapped("Background color");
                ui.color_picker4("##picker", &mut self.color);

                ui.separator();
                ui.text_wrapped("Camera Settings");
                ui.input_float3("Eye", &mut self.cam_eye).build();
                ui.input_float3("Center", &mut self.cam_center).build();
                ui.input_float3("Up", &mut self.cam_up).build();
                ui.input_float2("Near/Far", &mut self.cam_nearfar).build();

                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                if mouse_pos[0] != f32::MIN {
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                }
            });
    }

    fn frame_data(&self) -> FrameData {
        FrameData {
            bgcolor: self.color,
            frame_limiter: self.frame_limiter,
            frame_limiter_fps: self.frame_limiter_limit as u32,
            rotate_model: self.rotate_model,
            cam_eye: self.cam_eye,
            cam_center: self.cam_center,
            cam_up: self.cam_up,
            cam_nearfar: self.cam_nearfar,
        }
    }
}

fn main() -> std::result::Result<(), EventLoopError> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(UiTut::default());

    event_loop.run_app(&mut app)
}
