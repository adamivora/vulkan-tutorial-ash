use imgui::Condition;
use rust_vulkan::imgui::camera_manipulator::CAMERA_MANIPULATOR_INSTANCE;
use rust_vulkan::imgui::camera_widget::CameraWidgetUi;
use rust_vulkan::ui::UiBuilder;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

use glam::{Vec3, Vec4};
use rust_vulkan::app::App;
use rust_vulkan::frame_data::FrameData;

struct UiTut {
    color: Vec3,
    frame_limiter: bool,
    frame_limiter_limit: i32,
    rotate_model: bool,
}

impl Default for UiTut {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.0, 0.0, 0.0),
            frame_limiter: true,
            frame_limiter_limit: 60 as i32,
            rotate_model: true,
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
                ui.color_edit3("Clear color", &mut self.color);

                ui.separator();
                ui.text(format!(
                    "Application average {:.3} ms/frame ({:.1} FPS)",
                    1000.0 / ui.io().framerate,
                    ui.io().framerate
                ));
                ui.checkbox("Frame Limiter", &mut self.frame_limiter);
                if self.frame_limiter {
                    ui.same_line();
                    ui.set_next_item_width(75.0);
                    ui.slider("FPS Limit", 10, 120, &mut self.frame_limiter_limit);
                }
                ui.checkbox("Rotate Model", &mut self.rotate_model);

                ui.separator();
                ui.text_wrapped("Camera Settings");
                ui.camera_widget();

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
        let camera_m = CAMERA_MANIPULATOR_INSTANCE
            .lock()
            .expect("cannot lock mutex");
        let cam_matrix = camera_m.matrix;
        let cam_nearfar = camera_m.clip_planes;
        let cam_fov = camera_m.current.fov;

        FrameData {
            bgcolor: Vec4::new(self.color.x, self.color.y, self.color.z, 0.0),
            frame_limiter: self.frame_limiter,
            frame_limiter_fps: self.frame_limiter_limit as u32,
            rotate_model: self.rotate_model,
            cam_matrix,
            cam_nearfar,
            cam_fov,
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
