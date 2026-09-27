use crate::helpers::positions::WorldPos;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::Props;
use crate::resources::Resources;
use crate::ui::input::Input;
use crate::ui::variables::Variables;
use crate::world::buildings::buildings::{BuildingStorage, Buildings};
use crate::world::buildings::lot_fitting::collect_road_points;
use crate::world::buildings::zoning::{LotId, Zoning, draw_area, point_in_polygon_xz};
use crate::world::cars::car_render::interpolate_cars;
use crate::world::roads::road_mesh_manager::RoadMeshManager;
use crate::world::roads::road_structs::{NodeId, RoadEditorCommand, SegmentId};
use crate::world::roads::road_subsystem::Roads;
use crate::world::statisticals::CityState;
use crate::world::statisticals::money::calculate_road_cost;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use glam::Vec3;
use winit::event_loop::ActiveEventLoop;

pub fn run_ticked(resources: &mut Resources) {
    let world = &mut resources.world;
    let renderer = &mut resources.render_core;
    let camera = &world.world_state.camera;
    let variables = &mut resources.ui.variables;
    let (time, terrain, roads, zoning, buildings, cars, input, city_state, sounds) = (
        &mut world.time,
        &mut world.terrain,
        &mut world.roads,
        &mut world.zoning,
        &mut world.buildings,
        &mut world.cars,
        &mut world.input,
        &mut world.city_state,
        &mut world.sounds,
    );
    let (settings, gizmo, road_mesh_manager, props) = (
        &mut resources.settings,
        &mut renderer.gizmo,
        &renderer.road_renderer.mesh_manager,
        &mut renderer.props,
    );
    let aspect = renderer.config.width as f32 / renderer.config.height as f32;

    // for pending in terrain.terrain_jobs.workers.pending_chunks().iter() {
    //     let half_chunk_size = chunk_size() as f32 * 0.5;
    //     let center = WorldPos::new(*pending, LocalPos::new(half_chunk_size, camera.target.local.y, half_chunk_size));
    //     gizmo.square(center, half_chunk_size, [1.0, 0.0, 0.0, 1.0], 0.0, 0.0);
    // }
    if resources.simulation.running() {
        time.timer.checkpoint("Terrain", false);
        terrain.update(
            gizmo,
            camera,
            aspect,
            settings,
            input,
            time,
            roads,
            props,
            variables,
            &resources.game_state.mod_manager,
        );
        time.timer.checkpoint("Terrain", true);
    }
    handle_destruction(
        gizmo,
        variables,
        input,
        terrain,
        zoning,
        buildings,
        roads,
        road_mesh_manager,
        city_state,
        props,
    );
    roads.update(
        terrain, cars, city_state, input, time, settings, gizmo, sounds, buildings,
    );
}

fn handle_destruction(
    gizmo: &mut Gizmo,
    variables: &Variables,
    input: &mut Input,
    terrain: &mut Terrain,
    zoning: &mut Zoning,
    buildings: &mut Buildings,
    roads: &mut Roads,
    road_mesh_manager: &RoadMeshManager,
    city_state: &mut CityState,
    props: &mut Props,
) {
    if !matches!(terrain.cursor.mode, CursorMode::Destruction) {
        return;
    }

    let Some(picked) = terrain.last_picked.as_ref() else {
        return;
    };

    let mut inside_lot_id: Option<LotId> = None;
    for lot in zoning
        .zoning_storage
        .lots_in_chunk_plus(picked.chunk.chunk_coord)
        .iter()
        .flat_map(|lot_id| zoning.zoning_storage.get_lot(*lot_id))
    {
        if point_in_polygon_xz(picked.pos, lot.bounds.as_slice()) {
            inside_lot_id = Some(lot.id);
            break;
        }
    }
    if let Some(lot_id) = inside_lot_id {
        let removing_building = input.action_down("Destroy");
        //let finished_removing_lot = input.action_released("Destroy");
        if let Some(lot) = zoning.zoning_storage.get_lot(lot_id) {
            gizmo.polyline(
                lot.bounds.as_slice(),
                [0.8, 0.3, 0.7, 0.8],
                10.0,
                false,
                0.15,
                0.0,
            );
            if removing_building {
                draw_area(
                    lot.bounds.as_slice(),
                    lot.zoning_type,
                    variables,
                    gizmo,
                    Some([3.0, 0.2, 0.2, 1.0]),
                    None,
                );
                BuildingStorage::despawn(
                    buildings,
                    zoning,
                    &mut terrain.terrain_editor,
                    props,
                    lot.building_id,
                );
            }
        }
    } else {
        let mut closest_road: Option<RoadDestroyType> = None;
        let mut closest_distance = 5.0;

        let segment_ids = roads
            .road_manager
            .roads
            .segment_ids_touching_chunk(picked.chunk.chunk_coord);

        for segment_id in segment_ids {
            let segment = roads.road_manager.roads.segment(segment_id);

            // Check nodes first
            for node_id in [segment.start, segment.end] {
                let node = roads.road_manager.roads.node(node_id);
                let distance = node.pos().distance_to(picked.pos);

                if distance < closest_distance {
                    closest_distance = distance;
                    closest_road = Some(RoadDestroyType::Node(node_id));
                }
            }

            // Only allow segment picking if I haven't already found a close node
            if matches!(closest_road, Some(RoadDestroyType::Node(_))) {
                continue;
            }

            let Some(road_edges) = road_mesh_manager.road_edge_storage.get(&segment_id) else {
                continue;
            };

            let points: Vec<WorldPos> = collect_road_points(road_edges)
                .into_iter()
                .copied()
                .collect();

            if point_in_polygon_xz(picked.pos, points.as_slice()) {
                closest_road = Some(RoadDestroyType::Segment(segment_id));
            }
        }
        let sign_pos = picked.pos.add_vec3(Vec3::new(0.0, 10.0, 0.0));
        let scale = 2.0;
        let destroy = input.action_pressed_once("Destroy");
        if let Some(destroy_type) = closest_road {
            match destroy_type {
                RoadDestroyType::Segment(segment_id) => {
                    if let Some(edges) = road_mesh_manager.road_edge_storage.get(&segment_id) {
                        let points: Vec<WorldPos> =
                            collect_road_points(edges).into_iter().copied().collect();
                        //gizmo.area(points.as_slice(), [0.5, 0.2, 0.0, 0.3], 0.0);
                        let segment = roads.road_manager.roads.segment(segment_id);
                        if let Some(lane) = segment
                            .lanes()
                            .first()
                            .map(|&lane_id| roads.road_manager.roads.lane(lane_id))
                        {
                            if let Some(road_type) = roads
                                .road_manager
                                .road_types
                                .get_road_type(segment.road_type_id)
                            {
                                let cost =
                                    (calculate_road_cost(road_type, &[], lane.geometry().total_len)
                                        as f64
                                        * 0.2) as i64;
                                gizmo.text(
                                    format!("Refund: {}€", cost),
                                    sign_pos,
                                    scale,
                                    [0.03, 0.97, 0.03, 0.85],
                                    None,
                                    true,
                                    0.0,
                                    if destroy { 2.0 } else { 0.0 },
                                );
                                if destroy {
                                    city_state.economy.add_money(cost);
                                };
                            };
                        };
                        roads
                            .road_editor
                            .pending_outside_commands
                            .push(RoadEditorCommand::PreviewDestruction(destroy_type));
                    }
                    if destroy {
                        roads.road_manager.roads.delete_segment(
                            segment_id,
                            &roads.road_manager.road_types,
                            gizmo,
                            true,
                        )
                    }
                }
                RoadDestroyType::Node(node_id) => {
                    let node = roads.road_manager.roads.node(node_id);
                    //let points: Vec<WorldPos> = collect_road_points(edges).into_iter().copied().collect();
                    gizmo.circle(node.pos(), 4.0, [0.5, 0.2, 0.0, 0.3], 0.6, 0.0);
                    let mut total_cost = 0;
                    for segment in node
                        .arms()
                        .iter()
                        .map(|arm| arm.segment())
                        .map(|segment_id| roads.road_manager.roads.segment(segment_id))
                    {
                        let Some(lane) = segment
                            .lanes()
                            .first()
                            .map(|&lane_id| roads.road_manager.roads.lane(lane_id))
                        else {
                            continue;
                        };
                        if let Some(road_type) = roads
                            .road_manager
                            .road_types
                            .get_road_type(segment.road_type_id)
                        {
                            let cost =
                                (calculate_road_cost(road_type, &[], lane.geometry().total_len)
                                    as f64
                                    * 0.2) as i64;
                            total_cost += cost;
                        };
                    }
                    gizmo.text(
                        format!("Refund: {}€", total_cost),
                        sign_pos,
                        scale,
                        [0.03, 0.97, 0.03, 0.85],
                        None,
                        true,
                        0.0,
                        if destroy { 2.0 } else { 0.0 },
                    );
                    if destroy {
                        city_state.economy.add_money(total_cost);
                    };
                    roads
                        .road_editor
                        .pending_outside_commands
                        .push(RoadEditorCommand::PreviewDestruction(destroy_type));

                    if input.action_pressed_once("Destroy") {
                        roads.road_manager.roads.delete_node(
                            node_id,
                            &roads.road_manager.road_types,
                            gizmo,
                        )
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub enum RoadDestroyType {
    Segment(SegmentId),
    Node(NodeId),
}
pub fn run_sim(resources: &mut Resources) {
    let world = &mut resources.world;
    resources.simulation.update(
        world,
        &mut resources.render_core,
        &mut resources.ui,
        &resources.settings,
    );
}

pub fn run_ui(resources: &mut Resources, event_loop: &dyn ActiveEventLoop) {
    let input = &mut resources.world.input;
    let time = &resources.world.time;
    let dt = time.target_frametime;
    input.now = time.total_time;
    resources.ui.handle_touches(
        dt,
        &mut resources.world,
        &mut resources.render_core,
        resources.window.surface_size().cast::<f32>(),
        &mut resources.command_queues,
        &mut resources.settings,
        event_loop,
        &mut resources.game_state,
        &mut resources.simulation,
    );
}

pub fn run_interpolation(resources: &mut Resources) {
    interpolate_cars(
        &mut resources.world,
        &resources.ui.variables,
        &resources.settings,
    );
    if resources.settings.render_signfinding_gizmo {
        resources
            .render_core
            .gizmo
            .visualize_signfinding_trajectories(
                resources.world.cars.car_storage(),
                &resources.world.terrain,
                &resources.world.roads.road_manager.roads,
                &resources.world.buildings,
                &resources.world.zoning,
            )
    }
}

pub fn run_render(resources: &mut Resources) {
    let aspect =
        resources.render_core.config.width as f32 / resources.render_core.config.height as f32;
    resources
        .world
        .world_state
        .camera
        .compute_matrices(aspect, &resources.settings);

    resources.render_core.render(
        &resources.surface,
        &mut resources.world,
        &mut resources.ui,
        &resources.settings,
        &resources.game_state.mod_manager,
    );

    resources.world.world_state.camera.end_frame();
}
