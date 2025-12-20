use crate::resources::Resources;
use crate::world::World;

pub fn render_system(world: &mut World, resources: &mut Resources) {
    let camera_entity = world.main_camera();
    if let Some(camera_bundle) = world.camera_and_controller_mut(camera_entity) {
        resources.renderer.render(
            camera_bundle,
            &mut resources.ui_loader,
            &resources.time,
            &resources.input.mouse,
            &resources.settings,
        );
    }
}
