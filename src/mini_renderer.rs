use std::collections::{HashMap, VecDeque};
use std::mem::take;
use crate::data::Settings;
use glam::Vec3;
use rusty_skylines_utils::positions::WorldPos;
use wgpu::{Device, Queue, SurfaceConfiguration, Texture, TextureView};
use winit::dpi::PhysicalSize;
use crate::helpers::modpack::ModManager;
use crate::renderer::renderer::Renderer;
use crate::resources::Timer;
use crate::ui::action_parser::{find_matching_close_paren, make_ui_command, parse_arguments};
use crate::ui::actions::load_save;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::game_state::GameState;
use crate::world::roads::road_subsystem::Roads;
use crate::world::world::World;

pub struct MiniRender {
    pub id: u64,
    pub pos: WorldPos,
    pub rotation: Vec3,
    pub size: PhysicalSize<u32>,
    pub fov: f32,
    pub mini_world_name: String,
    pub commands: Vec<String>,
}
impl MiniRender {
    pub fn new(pos: WorldPos, rotation: Vec3, size: PhysicalSize<u32>, fov: f32, mini_world_name: String, commands: Vec<String>) -> Self {
        Self {
            id: 0,
            pos,
            rotation,
            size,
            fov,
            mini_world_name,
            commands,
        }
    }
}
impl Default for MiniRender {
    fn default() -> Self {
        Self {
            id: 0,
            pos: Default::default(),
            rotation: Default::default(),
            size: PhysicalSize::new(256, 256),
            fov: 80.0,
            mini_world_name: "".to_string(),
            commands: vec![],
        }
    }
}
pub struct MiniRenderOutput {
    pub view: TextureView,
}

pub struct MiniRenderer {
    pub world: World,
    pub renderer: Renderer,
    pub game_state: GameState,
    pub queued_renders: Vec<MiniRender>,
    pub outputs: HashMap<u64, MiniRenderOutput>,
    pub next_id: u64,
    pub timer: Timer,
}

impl MiniRenderer {
    pub fn new(
        device: &Device,
        queue: &Queue,
        settings: &Settings,
        config: &SurfaceConfiguration,
        mod_manager: &ModManager,
        real_world: &World,
    ) -> Self {
        let mut world = World::new(device, queue, settings);
        world.reload(settings, mod_manager);

        let mut mini_config = config.clone();
        mini_config.width = 256;
        mini_config.height = 256;
        let game_state = GameState::new_off();
        let renderer = Renderer::new(
            device,
            queue,
            settings,
            &mini_config,
            PhysicalSize::new(256.0, 256.0),
            &world.world_state.camera,
            mod_manager,
            false,
        );

        let mut mini_renderer = Self {
            world,
            renderer,
            game_state,
            queued_renders: Vec::new(),
            outputs: HashMap::new(),
            next_id: 0,
            timer: Timer::default(),
        };
        mini_renderer.render_single(MiniRender::default(), settings, mod_manager, real_world);
        mini_renderer
    }

    fn recreate_world(&mut self, settings: &Settings, mod_manager: &ModManager) {
        self.world.recreate_light();
        self.renderer.reset_scene_caches();
    }

    pub fn queue_render(&mut self, mut mini_render: MiniRender) -> u64 {
        let id = self.next_id;
        mini_render.id = id;
        self.queued_renders.push(mini_render);
        self.next_id += 1;
        id
    }
    pub fn get_view(&self, id: u64) -> Option<&TextureView> {
        self.outputs.get(&id).map(|o| &o.view)
    }
    pub fn render(&mut self, settings: &Settings, mod_manager: &ModManager, real_world: &World) {
        let queued_renders = take(&mut self.queued_renders);

        for mini_render in queued_renders {
            self.render_single(mini_render, settings, mod_manager, real_world);
        }
    }
    pub fn render_single_id(&mut self, id: u64, settings: &Settings, mod_manager: &ModManager, real_world: &World) {
        if let Some(mini_render_idx) = self.queued_renders.iter().rposition(|r| r.id == id) {
            let mini_render = self.queued_renders.remove(mini_render_idx);
            self.render_single(mini_render, settings, mod_manager, real_world);
        }
    }
    fn execute_commands(&mut self, commands: &[String], real_world: &World) {
        for s in commands {
            let (func_name, args) = if let Some(open_paren) = s.find('(') {
                let func_name = s[..open_paren].trim();

                if let Some(close_paren) = find_matching_close_paren(s, open_paren) {
                    let args_str = s[open_paren + 1..close_paren].trim();
                    let args = parse_arguments(args_str);
                    (func_name, args)
                } else {
                    (func_name, vec![])
                }
            } else { continue };
            match func_name {
                "replace_all_road_types" => {
                    let Some(road_type_arg) = args.first() else {
                        continue;
                    };

                    let Some(road_type) = Value::from_str_pure(road_type_arg).as_array() else {
                        continue;
                    };

                    let [branch, category, item] = road_type.as_slice() else {
                        continue;
                    };

                    let (Some(branch), Some(category), Some(item)) = (
                        branch.as_str(),
                        category.as_str(),
                        item.as_str(),
                    ) else {
                        continue;
                    };
                    if branch != "Roads" { continue; };

                    let namespace = format!("{}:{}:{}", branch, category, item);
                    if let Some(r_id) = real_world.roads.road_manager.road_types.data.categories.get(category).and_then(|c| c.iter().find(|r| r.id == item)).map(|r| r.stable_id) {
                        println!("[Mini Renderer] Replacing all road types with: '{}' ({})", namespace, r_id);
                        self.world.roads.road_manager.roads.replace_all_road_types(r_id);
                    }
                }
                _ => {
                    println!("[Mini Renderer] Unknown command: '{}'", func_name);
                }
            }
        }
    }
    fn render_single(
        &mut self,
        mini_render: MiniRender,
        settings: &Settings,
        mod_manager: &ModManager,
        real_world: &World,
    ) {
        self.timer.clear();
        self.timer.checkpoint(["Creation", "Recreate"], false);
        self.recreate_world(settings, mod_manager);
        self.timer.checkpoint(["Creation", "Recreate"], true);
        self.timer.checkpoint(["Creation", "Load World"], false);
        load_save(&mut self.game_state, &mut self.world, settings, mini_render.mini_world_name.as_str());
        self.timer.checkpoint(["Creation", "Load World"], true);
        self.execute_commands(mini_render.commands.as_slice(), real_world);
        let aspect = mini_render.size.width as f32 / mini_render.size.height as f32;

        let camera = &mut self.world.world_state.camera;
        camera.set_mode(crate::world::camera::camera::CameraMode::FirstPerson);
        camera.target = mini_render.pos;
        camera.fov = mini_render.fov;
        // camera.pitch = mini_render.dir.y;
        // camera.yaw = mini_render.dir.x;

        camera.yaw = mini_render.rotation.x.to_radians();
        camera.pitch = mini_render.rotation.y.clamp(-89.99, 89.99).to_radians();
        //camera.roll = mini_render.rotation.z;
        //println!("FOV: {}", camera.fov);
        camera.compute_matrices(aspect, settings);

        let actual_forward = camera.forward();

        // println!(
        //     "Requested: {:?} | Actual: {:?} | Dot: {} | Yaw: {} | Pitch: {}",
        //     dir,
        //     actual_forward,
        //     dir.dot(actual_forward),
        //     camera.yaw,
        //     camera.pitch,
        // );
        self.timer.checkpoint(["Update", "World State"], false);
        self.world.world_state.update(
            None,
            &mut self.world.time, // Updates astronomy and UI Variables
            &settings,
            self.world.world_state.camera.proj(),
        );
        self.timer.checkpoint(["Update", "World State"], true);
        let camera = &mut self.world.world_state.camera;
        let mut variables = Variables::new();
        for _ in 0..2 {
            self.timer.checkpoint(["Update", "Terrain"], false);
            self.world.terrain.update(&mut self.renderer.gizmo, camera, aspect, settings, &mut self.world.input, &mut self.world.time, &mut self.world.roads, &mut self.world.props, &variables, &self.game_state.mod_manager);
            self.timer.checkpoint(["Update", "Terrain"], true);
            self.timer.checkpoint(["Update", "Roads"], false);
            self.world.roads.update(&mut self.world.terrain, &mut self.world.cars, &mut self.world.city_state, &mut self.world.input, settings, &mut self.renderer.gizmo, &mut self.world.sounds, &mut self.world.buildings, &mut variables, &self.game_state.mod_manager);
            self.timer.checkpoint(["Update", "Roads"], true);
        }
        //println!("SUN DIR: {:?}", self.world.time.astronomy.sun_dir);
        self.timer.checkpoint("Create Output", false);
        let output = self.create_output(mini_render.size);
        self.timer.checkpoint("Create Output", true);
        self.timer.checkpoint("Render", false);
        self.renderer.render_preview(
            &mut self.world,
            &output.view,
            settings,
            mod_manager,
            mini_render.size,
            &mut self.timer,
        );
        self.timer.checkpoint("Render", true);
        self.outputs.insert(mini_render.id, output);

        let all_totals = self.timer.totals();

        let mut totals: Vec<_> = all_totals
            .iter()
            .filter(|(name, _)| {
                !all_totals.iter().any(|(other, _)| {
                    other.len() > name.len() && other.starts_with(name)
                })
            })
            .cloned()
            .collect();

        totals.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        totals.truncate(5);

        println!(
            "Mini Render: {:.3}ms total; Top 5: {:?}",
            self.timer.total_elapsed().as_secs_f64() * 1000.0,
            totals.iter().map(|(name, duration)| (
                name,
                format!("{:.2}ms", duration.as_secs_f64() * 1000.0)
            )).collect::<Vec<_>>()
        );
    }

    fn create_output(&self, size: PhysicalSize<u32>) -> MiniRenderOutput {
        let texture = self.renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mini Render Output"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.renderer.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        MiniRenderOutput { view }
    }
}