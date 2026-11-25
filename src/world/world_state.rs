use crate::data::{CameraTrackingTarget, Settings};
use crate::resources::Time;
use crate::ui::ui_editor::Ui;
use crate::ui::variables::update_ui_variables;
use crate::world::astronomy::{
    Astronomy, ObserverParams, TimeScales, compute_astronomy,
};
use crate::world::camera::camera::{Camera, CameraController};
use crate::world::satellites::{
    BODY_MARS,
    BODY_PLUTO,
    BODY_NEPTUNE,
    BODY_URANUS,
    BODY_JUPITER,
    BODY_VENUS,
    BODY_MERCURY,
    BODY_SATURN,
    BODY_IO,
    BODY_EUROPA,
    BODY_GANYMEDE,
    BODY_CALLISTO,
    BODY_TITAN,
    BODY_RHEA,
    BODY_TETHYS,
    BODY_DIONE,
    BODY_IAPETUS,
    BODY_ENCELADUS,
    BODY_TRITON,
    BODY_CHARON,
};
use glam::{Mat4, Vec3};

pub struct WorldState {
    pub camera: Camera,
    pub cam_controller: CameraController,
}

impl WorldState {
    pub fn new() -> Self {
        let camera = Camera::new();
        let cam_controller = CameraController::new(&camera);

        Self {
            camera,
            cam_controller,
        }
    }

    pub fn update(
        &mut self,
        ui: Option<&mut Ui>,
        time: &mut Time,
        settings: &Settings,
        proj: Mat4,
    ) {
        let time_scales = TimeScales::from_game_time(
            time.total_game_time,
            time.day_length,
            settings.always_day,
        );

        let observer = ObserverParams::from_jd(time_scales.jd);

        time.astronomy = compute_astronomy(&time_scales, proj);

        let tracking_direction = camera_tracking_direction(
            settings.camera_tracking_target,
            &time.astronomy,
        );

        self.cam_controller.update_celestial_tracking(
            &mut self.camera,
            tracking_direction,
        );
        if let Some(ui) = ui {
            update_ui_variables(
                ui,
                &time_scales,
                &time.astronomy,
                observer.obliquity,
                settings,
            );
        }
    }
}

fn camera_tracking_direction(
    target: CameraTrackingTarget,
    astronomy: &Astronomy,
) -> Option<Vec3> {
    match target {
        CameraTrackingTarget::None => None,

        CameraTrackingTarget::Sol => Some(astronomy.sun_dir),
        CameraTrackingTarget::Luna => Some(astronomy.moon_dir),

        CameraTrackingTarget::Earth => Some(Vec3::NEG_Y),

        CameraTrackingTarget::Mars =>
            Some(astronomy.celestial_bodies[BODY_MARS].direction),

        CameraTrackingTarget::Pluto =>
            Some(astronomy.celestial_bodies[BODY_PLUTO].direction),
        CameraTrackingTarget::PlutoCharonBarycenter => {
            let pluto = astronomy.celestial_bodies[BODY_PLUTO];
            let charon = astronomy.celestial_bodies[BODY_CHARON];

            const PLUTO_MASS_KG: f64 = 1.303e22;
            const CHARON_MASS_KG: f64 = 1.586e21;

            let charon_mass_fraction =
                (CHARON_MASS_KG / (PLUTO_MASS_KG + CHARON_MASS_KG)) as f32;

            let pluto_position =
                pluto.direction * pluto.distance_km as f32;

            let charon_position =
                charon.direction * charon.distance_km as f32;

            Some(
                pluto_position
                    .lerp(charon_position, charon_mass_fraction)
                    .normalize_or_zero()
            )
        },
        CameraTrackingTarget::Neptune =>
            Some(astronomy.celestial_bodies[BODY_NEPTUNE].direction),

        CameraTrackingTarget::Uranus =>
            Some(astronomy.celestial_bodies[BODY_URANUS].direction),

        CameraTrackingTarget::Jupiter =>
            Some(astronomy.celestial_bodies[BODY_JUPITER].direction),

        CameraTrackingTarget::Venus =>
            Some(astronomy.celestial_bodies[BODY_VENUS].direction),

        CameraTrackingTarget::Mercury =>
            Some(astronomy.celestial_bodies[BODY_MERCURY].direction),

        CameraTrackingTarget::Saturn =>
            Some(astronomy.celestial_bodies[BODY_SATURN].direction),

        CameraTrackingTarget::Io =>
            Some(astronomy.celestial_bodies[BODY_IO].direction),

        CameraTrackingTarget::Europa =>
            Some(astronomy.celestial_bodies[BODY_EUROPA].direction),

        CameraTrackingTarget::Ganymede =>
            Some(astronomy.celestial_bodies[BODY_GANYMEDE].direction),

        CameraTrackingTarget::Callisto =>
            Some(astronomy.celestial_bodies[BODY_CALLISTO].direction),

        CameraTrackingTarget::Titan =>
            Some(astronomy.celestial_bodies[BODY_TITAN].direction),

        CameraTrackingTarget::Rhea =>
            Some(astronomy.celestial_bodies[BODY_RHEA].direction),

        CameraTrackingTarget::Tethys =>
            Some(astronomy.celestial_bodies[BODY_TETHYS].direction),

        CameraTrackingTarget::Dione =>
            Some(astronomy.celestial_bodies[BODY_DIONE].direction),

        CameraTrackingTarget::Iapetus =>
            Some(astronomy.celestial_bodies[BODY_IAPETUS].direction),

        CameraTrackingTarget::Enceladus =>
            Some(astronomy.celestial_bodies[BODY_ENCELADUS].direction),

        CameraTrackingTarget::Triton =>
            Some(astronomy.celestial_bodies[BODY_TRITON].direction),

        CameraTrackingTarget::Charon =>
            Some(astronomy.celestial_bodies[BODY_CHARON].direction),
    }
}
