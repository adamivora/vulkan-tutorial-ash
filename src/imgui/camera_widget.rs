// Rewritten from https://github.com/nvpro-samples/nvpro_core

use std::sync::{LazyLock, Mutex};

use glam::Vec3;
use imgui::SliderFlags;

use super::camera_manipulator::{Camera, CameraManipulator};
use super::camera_manipulator::{Modes, CAMERA_MANIPULATOR_INSTANCE as CAMERA_M};

#[derive(Default)]
struct CameraManager {
    cameras: Vec<Camera>,
}

impl CameraManager {
    fn update(&mut self, camera_m: &CameraManipulator) {
        if self.cameras.is_empty() {
            self.cameras.push(camera_m.current);
        }
    }

    #[allow(unused)]
    fn remove_saved_cameras(&mut self) {
        self.cameras.drain(1..);
    }

    #[allow(unused)]
    fn set_home_camera(&mut self, camera: &Camera) {
        if self.cameras.is_empty() {
            self.cameras.push(*camera);
        } else {
            self.cameras[0] = *camera;
        }
    }

    #[allow(unused)]
    fn add_camera(&mut self, camera: &Camera) {
        let unique = self.cameras.iter().all(|c| c != camera);
        if unique {
            self.cameras.push(*camera);
        }
    }

    #[allow(unused)]
    fn remove_camera(&mut self, delete_item: usize) {
        self.cameras.remove(delete_item);
    }
}

static CAM_MGR: LazyLock<Mutex<CameraManager>> = LazyLock::new(Default::default);

fn current_camera_tab(
    ui: &imgui::Ui,
    camera_m: &mut CameraManipulator,
    camera: &mut Camera,
) -> (bool, bool) {
    let mut y_is_up = camera.up.y == 1.0;
    let mut instant_set = false;

    let mut changed = false;
    ui.input_float3("Eye", &mut camera.eye)
        .display_format("%.5f")
        .build();
    changed |= ui.is_item_deactivated_after_edit();
    ui.input_float3("Center", &mut camera.center)
        .display_format("%.5f")
        .build();
    changed |= ui.is_item_deactivated_after_edit();
    ui.input_float3("Up", &mut camera.up)
        .display_format("%.5f")
        .build();
    changed |= ui.is_item_deactivated_after_edit();

    if ui.checkbox("Y is UP", &mut y_is_up) {
        camera.up = if y_is_up { Vec3::Y } else { Vec3::Z };
        changed = true;
        instant_set = true;
    }
    if camera.up.length() < 0.0001 {
        camera.up = if y_is_up { Vec3::Y } else { Vec3::Z };
        changed = true;
    }
    if ui
        .slider_config("FOV", 1.0, 179.0)
        .display_format("%.1f deg")
        .flags(SliderFlags::LOGARITHMIC)
        .build(&mut camera.fov)
    {
        instant_set = true;
        changed = true;
    }

    if let Some(node) = ui.tree_node("Clip planes") {
        let mut clip = camera_m.clip_planes;
        ui.input_float("Near", &mut clip.x).build();
        changed |= ui.is_item_deactivated_after_edit();
        ui.input_float("Far", &mut clip.y).build();
        changed |= ui.is_item_deactivated_after_edit();
        node.pop();
        camera_m.clip_planes = clip;
    }

    if camera_m.is_animated() {
        changed = false;
    }

    ui.text_disabled("(?)");
    if ui.is_item_hovered() {
        ui.tooltip(|| {
            ui.text(CameraManipulator::get_help());
        });
    }

    (changed, instant_set)
}

#[allow(unused)]
fn saved_camera_tab(camera_m: &CameraManipulator, camera: &mut Camera) -> bool {
    false
}

fn extra_camera_tab(ui: &imgui::Ui, camera_m: &mut CameraManipulator) -> bool {
    let mut changed = false;
    changed |= ui.radio_button("Examine", &mut camera_m.mode, Modes::Examine);
    if ui.is_item_hovered() {
        ui.tooltip(|| ui.text("The camera orbits around a point of interest."));
    }
    changed |= ui.radio_button("Fly", &mut camera_m.mode, Modes::Fly);
    if ui.is_item_hovered() {
        ui.tooltip(|| ui.text("The camera is free and move toward the looking direction."));
    }
    changed |= ui.radio_button("Walk", &mut camera_m.mode, Modes::Walk);
    if ui.is_item_hovered() {
        ui.tooltip(|| ui.text("The camera is free but stays on a plane."));
    }

    changed |= ui
        .slider_config("Speed", 0.1, 10.0)
        .display_format("%.3f")
        .build(&mut camera_m.speed);
    changed |= ui
        .slider_config("Transition", 0.0, 20.0)
        .display_format("%.3f")
        .build(&mut camera_m.duration);
    changed
}

pub trait CameraWidgetUi {
    fn camera_widget(&self) -> bool;
}

impl CameraWidgetUi for imgui::Ui {
    fn camera_widget(&self) -> bool {
        let ui = self;
        let mut changed = false;
        let mut instant_set = false;
        let mut camera_m = CAMERA_M.lock().expect("cannot lock mutex");
        let mut camera = camera_m.get_camera();

        let mut cam_mgr = CAM_MGR.lock().expect("cannot lock mutex");
        cam_mgr.update(&camera_m);

        if let Some(tab_bar) = ui.tab_bar("Hello") {
            if let Some(item) = ui.tab_item("Current") {
                (changed, instant_set) = current_camera_tab(ui, &mut camera_m, &mut camera);
                item.end();
            }

            /*if let Some(item) = ui.tab_item("Cameras") {
                changed = saved_camera_tab(&camera_m, &mut camera);
                item.end();
            }*/

            if let Some(item) = ui.tab_item("Extra") {
                changed = extra_camera_tab(ui, &mut camera_m);
                item.end();
            }
            tab_bar.end();
        }

        if changed {
            if instant_set {
                camera_m.set_camera(camera);
            } else {
                camera_m.set_camera_interp(camera);
            }
        }
        ui.separator();

        return changed;
    }
}
