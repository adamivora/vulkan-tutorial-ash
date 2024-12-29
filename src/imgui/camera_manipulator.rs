// Rewritten from https://github.com/nvpro-samples/nvpro_core

use core::f32;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use std::{
    ops::{Add, Mul},
    sync::{LazyLock, Mutex},
    time::Instant,
};

#[derive(PartialEq, Clone, Copy)]
pub enum Modes {
    Examine = 0,
    Fly = 1,
    Walk = 2,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Actions {
    NoAction,
    Orbit,
    Dolly,
    Pan,
    LookAround,
}

#[derive(Default, Copy, Clone)]
pub struct Inputs {
    pub lmb: bool,
    pub mmb: bool,
    pub rmb: bool,
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

#[derive(PartialEq, Copy, Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub center: Vec3,
    pub up: Vec3,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::splat(2.0),
            center: Vec3::ZERO,
            up: Vec3::Z,
            fov: 60.0,
        }
    }
}

impl Eq for Camera {}

pub struct CameraManipulator {
    pub matrix: Mat4,

    pub current: Camera,
    goal: Camera,
    snapshot: Camera,

    // Animation
    bezier: [Vec3; 3],
    start_time: Instant,
    pub duration: f64,
    anim_done: bool,
    key_vec: Vec3,

    // Screen
    pub width: u32,
    pub height: u32,

    // Other
    pub speed: f32,
    pub mouse: Vec2,
    pub clip_planes: Vec2,

    pub mode: Modes,
}

impl Default for CameraManipulator {
    fn default() -> Self {
        let mut manip = Self {
            matrix: Mat4::IDENTITY,
            current: Camera::default(),
            goal: Camera::default(),
            snapshot: Camera::default(),
            bezier: [Vec3::ZERO; 3],
            start_time: Instant::now(),
            duration: 0.5,
            anim_done: true,
            key_vec: Vec3::ZERO,
            width: 1,
            height: 1,
            speed: 0.3,
            mouse: Vec2::ZERO,
            clip_planes: Vec2::new(0.001, 100000000.0),
            mode: Modes::Examine,
        };
        manip.update();
        manip
    }
}

impl CameraManipulator {
    fn update(&mut self) {
        let eye = self.current.eye;
        let dir = self.current.center - eye;
        let up = self.current.up;

        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let s = if s.is_nan() {
            let up = up + 0.5 * Vec3::NEG_X;
            f.cross(up).normalize()
        } else {
            s
        };
        let u = s.cross(f);

        self.matrix = Mat4::from_cols(
            Vec4::new(s.x, u.x, -f.x, 0.0),
            Vec4::new(s.y, u.y, -f.y, 0.0),
            Vec4::new(s.z, u.z, -f.z, 0.0),
            Vec4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0),
        );
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let (dx, dy) = if self.mode == Modes::Fly {
            (dx * -1.0, dy * -1.0)
        } else {
            (dx, dy)
        };

        let z = self.current.eye - self.current.center;
        let length = z.length() / 0.785; // 45 degrees
        let z = z.normalize();
        let x = Vec3::cross(self.current.up, z);
        let y = Vec3::cross(z, x);
        let x = x.normalize();
        let y = y.normalize();

        let pan_vector = (-dx * x + dy * y) * length;
        self.current.eye += pan_vector;
        self.current.center += pan_vector;
    }

    fn orbit(&mut self, dx: f32, dy: f32, invert: bool) {
        if dx == 0.0 && dy == 0.0 {
            return;
        }

        let dx = dx * f32::consts::TAU;
        let dy = dx * f32::consts::TAU;

        let origin = if invert {
            self.current.eye
        } else {
            self.current.center
        };
        let position = if invert {
            self.current.center
        } else {
            self.current.eye
        };

        let center_to_eye = position - origin;
        let radius = center_to_eye.length();
        let center_to_eye = center_to_eye.normalize();
        let axe_z = center_to_eye;

        let rot_y = Mat4::from_axis_angle(self.current.up, -dx);
        let center_to_eye = (rot_y * Vec4::from((center_to_eye, 0.0))).xyz();

        let axe_x = Vec3::cross(self.current.up, axe_z).normalize();
        let rot_x = Mat4::from_axis_angle(axe_x, -dy);

        let vect_rot = (rot_x * Vec4::from((center_to_eye, 0.0))).xyz();
        let center_to_eye = if vect_rot.x.signum() == center_to_eye.x.signum() {
            vect_rot
        } else {
            center_to_eye
        };

        let center_to_eye = center_to_eye * radius;
        let new_position = center_to_eye + origin;

        if !invert {
            self.current.eye = new_position;
        } else {
            self.current.center = new_position;
        }
    }

    fn dolly(&mut self, dx: f32, dy: f32) {
        let mut z = self.current.center - self.current.eye;
        let length = z.length();

        if length < 0.000001 {
            return;
        }

        let dd = if self.mode != Modes::Examine {
            -dy
        } else if dx.abs() > dy.abs() {
            dx
        } else {
            -dy
        };
        let factor = self.speed * dd;

        if self.mode == Modes::Examine {
            if factor >= 1.0 {
                return;
            }

            z *= factor;
        } else {
            z *= factor / length * 10.0;
        }

        if self.mode == Modes::Walk {
            if self.current.up.y > self.current.up.z {
                z.y = 0.0;
            } else {
                z.z = 0.0;
            }
        }

        self.current.eye += z;

        if self.mode != Modes::Examine {
            self.current.center += z;
        }
    }

    fn compute_bezier(t: f32, p0: &Vec3, p1: &Vec3, p2: &Vec3) -> Vec3 {
        let u = 1.0 - t;
        let tt = t * t;
        let uu = u * u;

        let p = uu * p0 + 2.0 * u * t * p1 + tt * p2;
        p
    }

    fn find_bezier_points(&mut self) {
        let p0 = self.current.eye;
        let p2 = self.goal.eye;

        let pi = (self.goal.center + self.current.center) * 0.5;
        let p02 = (p0 + p2) * 0.5;
        let radius = ((p0 - pi).length() + (p2 - pi).length()) * 0.5;
        let p02pi = (p02 - pi).normalize() * radius;
        let pc = pi + p02pi;
        let p1 = 2.0 * pc - p0 * 0.5 - p2 * 0.5;
        let p1 = Vec3 { y: p02.y, ..p1 };

        self.bezier = [p0, p1, p2];
    }

    fn set_look_at_impl(&mut self, eye: Vec3, center: Vec3, up: Vec3, instant_set: bool) {
        let camera = Camera {
            eye,
            center,
            up,
            fov: self.current.fov,
        };
        self.set_camera_impl(camera, instant_set);
    }

    pub fn get_camera(&self) -> Camera {
        self.current
    }

    pub fn set_look_at_interp(&mut self, eye: Vec3, center: Vec3, up: Vec3) {
        self.set_look_at_impl(eye, center, up, false);
    }

    pub fn set_look_at(&mut self, eye: Vec3, center: Vec3, up: Vec3) {
        self.set_look_at_impl(eye, center, up, true);
    }

    fn mix<T>(x: T, y: T, a: f32) -> T
    where
        T: Mul<f32, Output = T>,
        T: Add<Output = T>,
    {
        x * (1.0 - a) + y * a
    }

    pub fn update_anim(&mut self) {
        let elapse = (Instant::now() - self.start_time).as_secs_f32();

        // Key animation
        if self.key_vec != Vec3::ZERO {
            self.current.eye += self.key_vec * elapse;
            self.current.center += self.key_vec * elapse;
            self.update();
            self.start_time = Instant::now();
            return;
        }

        if self.anim_done {
            return;
        }

        let t = f32::min(elapse / self.duration as f32, 1.0);
        // Evaluate polynomial (smoother step from Perlin)
        let t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        if t >= 1.0 {
            self.current = self.goal;
            self.anim_done = true;
            self.update();
            return;
        }

        // Interpolate camera position and interest
        // The distance of the camera between the interest is preserved to
        // create a nicer interpolation
        self.current.center = Self::mix(self.snapshot.center, self.goal.center, t);
        self.current.up = Self::mix(self.snapshot.up, self.goal.up, t);
        self.current.eye =
            Self::compute_bezier(t, &self.bezier[0], &self.bezier[1], &self.bezier[2]);
        self.current.fov = Self::mix(self.snapshot.fov, self.goal.fov, t);
        self.update();
    }

    pub fn set_window_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse = Vec2::new(x, y);
    }

    fn set_camera_impl(&mut self, camera: Camera, instant_set: bool) {
        self.anim_done = true;

        if instant_set || self.duration == 0.0 {
            self.current = camera;
            self.update();
        } else if camera != self.current {
            self.goal = camera;
            self.snapshot = self.current;
            self.anim_done = false;
            self.start_time = Instant::now();
            self.find_bezier_points();
        }
    }

    pub fn set_camera_interp(&mut self, cam: Camera) {
        self.set_camera_impl(cam, false);
    }

    pub fn set_camera(&mut self, cam: Camera) {
        self.set_camera_impl(cam, true);
    }

    fn set_matrix_impl(&mut self, matrix: &Mat4, center_distance: f32, instant_set: bool) {
        let eye = matrix.col(3).xyz();

        let rot_mat = Mat3::from_mat4(*matrix);
        let ctr = Vec3::new(0.0, 0.0, -center_distance);
        let ctr = eye + (rot_mat * ctr);
        let camera = Camera {
            eye,
            center: ctr,
            up: Vec3::Y,
            fov: self.current.fov,
        };

        self.anim_done = instant_set;
        if instant_set {
            self.current = camera;
        } else {
            self.goal = camera;
            self.snapshot = self.current;
            self.start_time = Instant::now();
            self.find_bezier_points();
        }
        self.update();
    }

    pub fn set_matrix_interp(&mut self, matrix: &Mat4, center_distance: f32) {
        self.set_matrix_impl(matrix, center_distance, false);
    }

    pub fn set_matrix(&mut self, matrix: &Mat4, center_distance: f32) {
        self.set_matrix_impl(matrix, center_distance, true);
    }

    pub fn motion(&mut self, x: f32, y: f32, action: Actions) {
        let dx = (x - self.mouse[0]) / self.width as f32;
        let dy = (y - self.mouse[1]) / self.height as f32;

        match action {
            Actions::Orbit => self.orbit(dx, dy, false),
            Actions::Dolly => self.dolly(dx, dy),
            Actions::Pan => self.pan(dx, dy),
            Actions::LookAround => self.orbit(dx, -dy, true),
            _ => {}
        }

        self.anim_done = true;
        self.update();

        self.mouse = Vec2 {
            x: x as f32,
            y: y as f32,
        };
    }

    pub fn key_motion(&mut self, dx: f32, dy: f32, action: Actions) {
        if action == Actions::NoAction {
            self.key_vec = Vec3::ZERO;
        }

        let d = (self.current.center - self.current.eye).normalize();
        let dx = dx * self.speed * 2.0;
        let dy = dy * self.speed * 2.0;

        let key_vec = match action {
            Actions::Dolly => {
                let mut key_vec = d * dx;
                if self.mode == Modes::Walk {
                    if self.current.up.y > self.current.up.z {
                        key_vec.y = 0.0;
                    } else {
                        key_vec.z = 0.0;
                    }
                }
                key_vec
            }
            Actions::Pan => {
                let r = Vec3::cross(d, self.current.up);
                r * dx + self.current.up * dy
            }
            _ => Vec3::ZERO,
        };

        self.key_vec += key_vec;
        self.start_time = Instant::now();
    }

    pub fn mouse_move(&mut self, x: f32, y: f32, inputs: &Inputs) -> Actions {
        if !inputs.lmb && !inputs.rmb && !inputs.mmb {
            self.set_mouse_position(x, y);
            return Actions::NoAction;
        }

        let cur_action = if inputs.lmb {
            if (inputs.ctrl && inputs.shift) || inputs.alt {
                if self.mode == Modes::Examine {
                    Actions::LookAround
                } else {
                    Actions::Orbit
                }
            } else if inputs.shift {
                Actions::Dolly
            } else if inputs.ctrl {
                Actions::Pan
            } else {
                if self.mode == Modes::Examine {
                    Actions::Orbit
                } else {
                    Actions::LookAround
                }
            }
        } else if inputs.mmb {
            Actions::Pan
        } else if inputs.rmb {
            Actions::Dolly
        } else {
            Actions::NoAction
        };

        if cur_action != Actions::NoAction {
            self.motion(x, y, cur_action);
        }
        cur_action
    }

    pub fn wheel(&mut self, value: i32, inputs: Inputs) {
        let fval = value as f32;
        let dx = (fval * fval.abs()) / self.width as f32;

        if inputs.shift {
            self.set_fov(self.current.fov + fval);
        } else {
            self.dolly(dx * self.speed, dx * self.speed);
            self.update();
        }
    }

    pub fn get_aspect_ratio(&self) -> f32 {
        return self.width as f32 / self.height as f32;
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.current.fov = f32::clamp(fov, 0.01, 179.0);
    }

    pub fn is_animated(&self) -> bool {
        !self.anim_done
    }

    pub fn fit(
        &mut self,
        box_min: &Vec3,
        box_max: &Vec3,
        instant_fit: bool,
        tight_fit: bool,
        aspect: f32,
    ) {
        let box_half_size = 0.5 * (box_max - box_min);
        let box_center = 0.5 * (box_min + box_max);

        let yfov = f32::tan(self.current.fov * 0.5).to_radians();
        let xfov = yfov * aspect;

        let mut ideal_distance = 0.0;
        if tight_fit {
            let view = Mat3::from_mat4(Mat4::look_at_rh(
                self.current.eye,
                box_center,
                self.current.up,
            ));
            for i in 0..8 {
                let vct = Vec3 {
                    x: if i & 1 != 0 {
                        box_half_size.x
                    } else {
                        -box_half_size.x
                    },
                    y: if i & 2 != 0 {
                        box_half_size.y
                    } else {
                        -box_half_size.y
                    },
                    z: if i & 4 != 0 {
                        box_half_size.z
                    } else {
                        -box_half_size.z
                    },
                };
                let vct = view * vct;

                if vct.z < 0.0 {
                    ideal_distance = f32::max(vct.y.abs() / yfov + vct.z.abs(), ideal_distance);
                    ideal_distance = f32::max(vct.x.abs() / xfov + vct.z.abs(), ideal_distance);
                }
            }
        } else {
            let radius = box_half_size.length();
            ideal_distance = f32::max(radius / xfov, radius / yfov);
        }

        let new_eye = box_center - ideal_distance * (box_center - self.current.eye).normalize();
        self.set_look_at_impl(new_eye, box_center, self.current.up, instant_fit);
    }

    pub fn get_help() -> String {
        String::new()
            + "LMB: Rotate around the target\n"
            + "RMB: Dolly in/out\n"
            + "MMB: Pan along view plane\n"
            + "LMB + Shift: Dolly in/out\n"
            + "LMB + Ctrl: Pan\n"
            + "LMB + Alt: Look around/Pan\n"
            + "Mouse wheel: Dolly in/out\n"
            + "Mouse wheel + Shift: Zoom in/out\n"
    }
}

pub static CAMERA_MANIPULATOR_INSTANCE: LazyLock<Mutex<CameraManipulator>> =
    LazyLock::new(Default::default);
