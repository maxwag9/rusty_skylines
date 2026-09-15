use crate::data::Settings;
use crate::helpers::hsv::lerp;
use crate::helpers::positions::*;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::{Mat4, Vec3};
use std::time::Instant;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    /// `target` is the pivot point; the eye orbits around it at `orbit_radius`.
    Orbit,
    /// `target` is the eye itself (the "head"); yaw/pitch just steer the look direction.
    FirstPerson,
}
pub struct Camera {
    pub mode: CameraMode,
    pub target: WorldPos,
    pub orbit_radius: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub near: f32,
    pub far: f32,
    pub fov: f32,
    pub prev_view_proj: Mat4,
    view: Mat4,
    proj: Mat4,
    view_proj: Mat4,
    prev_eye_world: WorldPos,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Orbit,
            target: WorldPos::zero(),
            orbit_radius: 50.0,
            yaw: 1f32.to_radians(),
            pitch: 80f32.to_radians(),
            near: 2.5,
            far: 10_000.0,
            fov: 55.0,
            prev_view_proj: Default::default(),
            view: Default::default(),
            proj: Default::default(),
            view_proj: Default::default(),
            prev_eye_world: Default::default(),
        }
    }

    /// Switches camera mode, adjusting `target` so the eye position doesn't
    /// jump on the frame of the switch.
    pub fn set_mode(&mut self, mode: CameraMode) {
        if self.mode == mode {
            return;
        }
        match mode {
            CameraMode::FirstPerson => {
                // The head takes over the eye's current world position.
                self.target = self.eye_world();
            }
            CameraMode::Orbit => {
                // Push the pivot out in front of the head by orbit_radius,
                // so the orbiting eye lands back on the head's position.
                self.target = self
                    .target
                    .add_render_offset(self.forward() * self.orbit_radius);
            }
        }
        self.mode = mode;
    }
    // pub fn toggle_mode(&mut self) {
    //     if self.mode == CameraMode::FirstPerson {
    //         self.set_mode(CameraMode::Orbit);
    //     } else { self.set_mode(CameraMode::FirstPerson); }
    // }
    /// Point gizmo/debug code should treat as "the ground point the camera
    /// is centered on." In Orbit mode that's the pivot (`target`). In
    /// FirstPerson mode `target` *is* the eye, so anchoring debug shapes
    /// there draws them inside the camera — project a point out along the
    /// look direction instead.
    pub fn debug_anchor(&self, distance: f32) -> WorldPos {
        match self.mode {
            CameraMode::Orbit => self.target,
            CameraMode::FirstPerson => self.target.add_render_offset(self.forward() * distance),
        }
    }
    #[inline]
    pub fn is_first_person(&self) -> bool {
        matches!(self.mode, CameraMode::FirstPerson)
    }
    /// Unit look direction from yaw/pitch, used in FirstPerson mode
    /// (and to keep set_mode's target math consistent with orbit_offset).
    #[inline]
    pub fn forward(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();

        -Vec3::new(cp * cy, sp, cp * sy)
    }
    #[inline]
    pub fn end_frame(&mut self) {
        self.prev_view_proj = self.view_proj;
        self.prev_eye_world = self.eye_world();
    }

    /// Vector from `target` to the eye, used in Orbit mode.
    #[inline]
    pub fn orbit_offset(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();

        Vec3::new(
            self.orbit_radius * cp * cy,
            self.orbit_radius * sp,
            self.orbit_radius * cp * sy,
        )
    }

    pub fn compute_matrices(&mut self, aspect: f32, settings: &Settings) {
        let eye = Vec3::ZERO;
        let look_target = match self.mode {
            CameraMode::Orbit => -self.orbit_offset(),
            CameraMode::FirstPerson => self.forward(),
        };

        let view = glam::camera::rh::view::look_at_mat4(eye, look_target, Vec3::Y);
        let proj = if settings.reversed_depth_z {
            glam::camera::rh::proj::directx::perspective_infinite_reverse(
                self.fov.to_radians(),
                aspect,
                self.near,
            )
        } else {
            glam::camera::rh::proj::directx::perspective(
                self.fov.to_radians(),
                aspect,
                self.near,
                self.far,
            )
        };
        self.view = view;
        self.proj = proj;
        self.view_proj = proj * view;
    }

    #[inline]
    pub fn view(&self) -> Mat4 {
        self.view
    }
    #[inline]
    pub fn proj(&self) -> Mat4 {
        self.proj
    }
    #[inline]
    pub fn view_proj(&self) -> Mat4 {
        self.view_proj
    }
    #[inline]
    pub fn matrices(&self) -> (Mat4, Mat4, Mat4) {
        (self.view, self.proj, self.view_proj)
    }

    #[inline]
    pub fn eye_world(&self) -> WorldPos {
        match self.mode {
            CameraMode::Orbit => self.target.add_render_offset(self.orbit_offset()),
            CameraMode::FirstPerson => self.target,
        }
    }

    #[inline]
    pub fn prev_eye_world(&self) -> WorldPos {
        self.prev_eye_world
    }
    #[inline]
    pub fn _world_to_render(&self, pos: WorldPos) -> Vec3 {
        let eye = self.eye_world();
        pos.to_relative_pos(eye) // subtract eye, not target
    }

    #[inline]
    pub fn _render_to_world(&self, render_pos: Vec3) -> WorldPos {
        self.eye_world().add_render_offset(render_pos)
    }
}
/// Eases FOV/near-plane across a mode switch. Position and look direction
/// need no blending (see `Camera::set_mode`), only these "feel" parameters.
#[derive(Debug, Clone)]
struct ModeTransition {
    start: Instant,
    duration: f32,
    from_fov: f32,
    to_fov: f32,
    from_near: f32,
    to_near: f32,
}

#[derive(Debug, Clone)]
pub struct CameraController {
    pub velocity: Vec3,
    pub zoom_velocity: f32,
    pub target_yaw: f32,
    pub target_pitch: f32,
    pub orbit_smoothness: f32,
    pub yaw_velocity: f32,
    pub pitch_velocity: f32,
    pub orbit_damping_release: f32,
    pub zoom_damping: f32,
    prev_fov: f32,
    zoom_from_fov: f32,
    pub(crate) zoom_time_start: Option<Instant>,
    zoom_time_end: Option<Instant>,
    target_zoom_fov: f32,

    /// Resting FOV per mode; `switch_mode` blends `camera.fov` to the new
    /// mode's value and updates `prev_fov` so zoom/unzoom return to it.
    pub orbit_fov: f32,
    pub first_person_fov: f32,
    mode_transition: Option<ModeTransition>,
}

impl CameraController {
    pub fn new(camera: &Camera) -> Self {
        Self {
            velocity: Vec3::ZERO,
            zoom_velocity: 0.0,
            target_yaw: camera.yaw,
            target_pitch: camera.pitch,
            orbit_smoothness: 0.35,
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
            orbit_damping_release: 3.0,
            zoom_damping: 13.0,
            prev_fov: camera.fov,
            target_zoom_fov: 15.0,
            zoom_time_start: None,
            zoom_time_end: None,
            zoom_from_fov: 14.0,
            orbit_fov: 55.0,
            first_person_fov: 90.0,
            mode_transition: None,
        }
    }

    /// Entry point for input code: switches `camera.mode` and starts a short
    /// eased blend of FOV/near so the switch reads as a transition, not a cut.
    pub fn switch_mode(&mut self, camera: &mut Camera, mode: CameraMode, duration: f32) {
        if camera.mode == mode {
            return;
        }

        // Any in-flight ADS zoom shouldn't fight the mode blend.
        self.zoom_time_start = None;
        self.zoom_time_end = None;

        let from_fov = camera.fov;
        let from_near = camera.near;

        camera.set_mode(mode);

        let to_fov = match mode {
            CameraMode::Orbit => self.orbit_fov,
            CameraMode::FirstPerson => self.first_person_fov,
        };
        let to_near = match mode {
            CameraMode::Orbit => from_near.max(2.5),
            CameraMode::FirstPerson => 0.1,
        };

        self.prev_fov = to_fov;
        self.mode_transition = Some(ModeTransition {
            start: Instant::now(),
            duration: duration.max(0.0001),
            from_fov,
            to_fov,
            from_near,
            to_near,
        });
    }

    /// Advances the mode-switch blend. Call once per frame before zoom/unzoom
    /// so an active "Zoom" input still takes priority, same as it already
    /// overrides `prev_fov`.
    pub fn update_mode_transition(&mut self, camera: &mut Camera) {
        let Some(tr) = &self.mode_transition else {
            return;
        };
        let elapsed = (Instant::now() - tr.start).as_secs_f32();
        let t = (elapsed / tr.duration).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);

        camera.fov = lerp(tr.from_fov, tr.to_fov, t);
        camera.near = lerp(tr.from_near, tr.to_near, t);

        if t >= 1.0 {
            self.mode_transition = None;
        }
    }

    #[inline]
    pub fn is_transitioning(&self) -> bool {
        self.mode_transition.is_some()
    }

    pub fn zoom(&mut self, camera: &mut Camera, zoom_speed: f32) {
        if self.zoom_time_start.is_none() {
            self.zoom_time_end = None;
            self.zoom_time_start = Some(Instant::now());
            self.zoom_from_fov = camera.fov;
        }
        let Some(start) = self.zoom_time_start else {
            return;
        };
        let zoom_time = (0.5 * zoom_speed).max(0.0001);
        let elapsed = (Instant::now() - start).as_secs_f32();
        let t = (elapsed / zoom_time).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        camera.fov = lerp(self.zoom_from_fov, self.target_zoom_fov, t);
    }

    pub fn zoom_deactivate(&mut self, camera: &Camera) {
        self.zoom_from_fov = camera.fov;
        self.zoom_time_start = None;
        self.zoom_time_end = Some(Instant::now());
    }

    pub fn zoom_end(&mut self, camera: &mut Camera, unzoom_speed: f32) {
        if self.zoom_time_start.is_some() {
            return;
        }
        let Some(end_time) = self.zoom_time_end else {
            return;
        };
        let unzoom_time = (0.5 * unzoom_speed).max(0.0001);
        let elapsed = (Instant::now() - end_time).as_secs_f32();
        let t = (elapsed / unzoom_time).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        camera.fov = lerp(self.zoom_from_fov, self.prev_fov, t);
        if t >= 1.0 {
            self.zoom_time_end = None;
        }
    }
}

pub fn ground_camera_target(
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    terrain: &Terrain,
    min_clearance: f32,
) {
    let ground_y = terrain.get_height_at(camera.target, true);
    let penetration = (ground_y + min_clearance) - camera.target.local.y;

    if penetration > 0.0 {
        camera.target.local.y += penetration;
        camera_controller.velocity.y = camera_controller.velocity.y.max(0.0);
    }
}
pub fn resolve_pitch_by_search(
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    world_renderer: &Terrain,
) {
    let target = camera.target;
    let orbit_radius = camera.orbit_radius;

    let samples = 8;
    let mut max_terrain_y = f32::MIN;

    let offset = camera.orbit_offset(); // Vec3 in meters

    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let sample_pos: WorldPos = target.add_vec3(offset * t); // uses WorldPos + Vec3

        let terrain_y = world_renderer.get_height_at(sample_pos, true);
        max_terrain_y = max_terrain_y.max(terrain_y);
    }

    let min_clearance = 1.0;
    let desired_y = max_terrain_y + min_clearance;

    let dy = desired_y - target.local.y;
    let horizontal_dist = orbit_radius;

    let new_pitch = (dy / horizontal_dist)
        .asin()
        .clamp(-85.0_f32.to_radians(), 85.0_f32.to_radians());

    if new_pitch > camera.pitch || new_pitch > camera_controller.target_pitch {
        camera_controller.target_pitch = new_pitch;
        camera.pitch = new_pitch;
        camera_controller.pitch_velocity = 0.0;
    }
}
