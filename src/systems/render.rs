use crate::resources::Resources;
use crate::world::World;

pub fn render_system(world: &mut World, resources: &mut Resources) {
    let camera_entity = world.main_camera();
    if let Some(camera) = world.camera(camera_entity) {
        resources.renderer.render(
            camera,
            &mut resources.ui_loader,
            &resources.time,
            &resources.mouse,
        );
    }
}
