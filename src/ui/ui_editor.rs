use crate::paths::project_path;
use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::resources::{InputState, TimeSystem};
use crate::ui::helper::{calc_move_speed, triangulate_polygon};
use crate::ui::input::MouseState;
use crate::ui::parser::resolve_template;
use crate::ui::touches::{
    EditorInteractionResult, HitResult, MouseSnapshot, apply_pending_circle_updates, find_top_hit,
    handle_editor_mode_interactions, handle_scroll_resize, handle_text_editing, near_handle,
    press_began_on_ui,
};
use crate::ui::vertex::UiElement::*;
pub(crate) use crate::ui::vertex::*;
use glam::Vec2;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, PI};
use std::fs;
use std::path::PathBuf;

pub struct UiVariableRegistry {
    pub(crate) vars: HashMap<String, String>,
}

impl UiVariableRegistry {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        self.vars.insert(name.to_string(), value.into());
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(|s| s.as_str())
    }
}

pub fn style_to_u32(style: &str) -> u32 {
    match style {
        "Hue Circle" => 1,
        "None" => 0,
        &_ => 0,
    }
}

pub fn execute_action(
    loader: &mut UiButtonLoader,
    top_hit: &Option<HitResult>,
    mouse_state: &MouseState,
) {
    let actions: Vec<String> = loader.ui_runtime.action_states.keys().cloned().collect();

    for action in actions {
        match action.as_str() {
            "Drag Hue Point" => {
                drag_hue_point(loader, mouse_state, top_hit);
            }
            "None" => {}
            _ => {}
        }
    }
}

fn activate_action(loader: &mut UiButtonLoader, top_hit: &Option<HitResult>) {
    if let Some(hit) = top_hit {
        let action = hit.action.clone().unwrap_or("None".to_string());
        match action.as_str() {
            "Drag Hue Point" => {
                if let Some(action_state) =
                    loader.ui_runtime.action_states.get_mut("Drag Hue Point")
                {
                    action_state.active = true;
                } else {
                    let action_state = ActionState {
                        action_name: "Drag Hue Point".to_string(),
                        position: Default::default(),
                        last_pos: Default::default(),
                        radius: 5.0,
                        original_radius: 5.0,
                        active: true,
                    };
                    loader
                        .ui_runtime
                        .action_states
                        .insert("Drag Hue Point".to_string(), action_state);
                }
            }
            "None" => {}
            &_ => {}
        }
    }
}

fn deactivate_action(loader: &mut UiButtonLoader, action_name: &str) {
    if let Some(action_state) = loader.ui_runtime.action_states.get_mut(action_name) {
        action_state.active = false;
    }
}

fn drag_hue_point(
    loader: &mut UiButtonLoader,
    mouse_state: &MouseState,
    top_hit: &Option<HitResult>,
) {
    // if loader.ui_runtime.selected_ui_element.element_type != ElementKind::Circle {
    //     return;
    // }

    let mut new_x = None;
    let mut new_y = None;
    let mut fill_color = None;
    let border_color = None;
    let mut radius = None;

    if loader.ui_runtime.selected_ui_element.dragging {
        if let Some(action_state) = loader.ui_runtime.action_states.get_mut("Drag Hue Point") {
            if action_state.last_pos.x == 0.0 {
                new_x = Some(lerp(mouse_state.last_pos.x, mouse_state.pos.x, 0.5));
                new_y = Some(lerp(mouse_state.last_pos.y, mouse_state.pos.y, 0.4));
            } else {
                new_x = Some(lerp(action_state.last_pos.x, mouse_state.pos.x, 0.5));
                new_y = Some(lerp(action_state.last_pos.y, mouse_state.pos.y, 0.4));
            }

            action_state.last_pos = Vec2::from((
                new_x.unwrap_or(mouse_state.last_pos.x),
                new_y.unwrap_or(mouse_state.last_pos.y),
            ));
        }
    }
    if let Some(circle_layer) = loader
        .menus
        .get("Editor_Menu")
        .and_then(|m| m.layers.iter().find(|l| l.name == "Color Picker"))
    {
        let hue_circle = circle_layer
            .circles
            .iter()
            .find(|c| c.style == "Hue Circle");

        let handle_circle = circle_layer
            .circles
            .iter()
            .find(|c| c.id.as_deref() == Some("color picker handle circle"));

        if let (Some(hue), Some(handle)) = (hue_circle, handle_circle) {
            // Use smoothed position if available, otherwise raw mouse
            let pointer_x = new_x.unwrap_or(mouse_state.pos.x);
            let pointer_y = new_y.unwrap_or(mouse_state.pos.y);

            // Hover radius stuff still uses pointer, not raw mouse
            let dmx = handle.x - pointer_x;
            let dmy = handle.y - pointer_y;
            let dist_to_mouse = (dmx * dmx + dmy * dmy).sqrt();
            if let Some(action_state) = loader.ui_runtime.action_states.get_mut("Drag Hue Point") {
                if dist_to_mouse < action_state.original_radius + 4.0 {
                    radius = Some(action_state.original_radius + 4.0);
                } else {
                    radius = Some(action_state.original_radius);
                }
            }
            if loader.ui_runtime.selected_ui_element.dragging {
                radius = Some(loader.ui_runtime.original_radius + 5.0);

                // 1. Vector from center to (smoothed) pointer
                let mx = pointer_x - hue.x;
                let my = pointer_y - hue.y;

                let angle = my.atan2(mx);
                let dist_mouse = (mx * mx + my * my).sqrt();

                // Clamp to hue radius
                let clamped_r = dist_mouse.min(hue.radius);

                // New, clamped handle position
                let hx = hue.x + angle.cos() * clamped_r;
                let hy = hue.y + angle.sin() * clamped_r;

                new_x = Some(hx);
                new_y = Some(hy);

                // 2. Now compute HSV from the *new* handle position
                let dx = hx - hue.x;
                let dy = hy - hue.y;

                let angle = dy.atan2(dx);
                let angle_shifted = angle + FRAC_PI_2;
                let angle_wrapped = angle_shifted.sin().atan2(angle_shifted.cos());

                let h = angle_wrapped / (PI * 2.0) + 0.5;

                let dist = (dx * dx + dy * dy).sqrt();
                let s_linear = (dist / hue.radius).clamp(0.0, 1.0);
                let s = s_linear.powf(0.47);

                let v = 1.0;
                let rgb = hsv_to_rgb(h, s, v);

                fill_color = Some([rgb[0], rgb[1], rgb[2], 1.0]);

                loader.variables.set("color_picker.r", rgb[0].to_string());
                loader.variables.set("color_picker.g", rgb[1].to_string());
                loader.variables.set("color_picker.b", rgb[2].to_string());

                loader.variables.set("color_picker.h", h.to_string());
                loader.variables.set("color_picker.s", s.to_string());
                loader.variables.set("color_picker.v", v.to_string());
            }
        }

        loader.ui_runtime.action_states.get("Drag Hue Point");
    }

    let result = loader.edit_circle(
        "Editor_Menu",
        "Color Picker",
        "color picker handle circle",
        new_x,
        new_y,
        radius,
        fill_color,
        border_color,
    );

    if let Some(hit) = top_hit {
        if hit.action != Some("Drag Hue Point".to_string()) {
            if loader.ui_runtime.selected_ui_element.just_selected {
                deactivate_action(loader, "Drag Hue Point");
                let _ = loader.delete_element(
                    "Editor_Menu",
                    "Color Picker",
                    "color picker handle circle",
                );
            }
        }
    }

    // Handle delete on deselection
    if loader.ui_runtime.selected_ui_element.just_deselected {
        deactivate_action(loader, "Drag Hue Point");
        let _ = loader.delete_element("Editor_Menu", "Color Picker", "color picker handle circle");
    }

    if let Some(action) = loader.ui_runtime.action_states.get("Drag Hue Point") {
        let active = action.active.clone();
        if !result {
            if loader.ui_runtime.selected_ui_element.just_selected && active {
                let handle_circle = UiButtonCircle {
                    id: Some("color picker handle circle".to_string()),
                    action: "None".to_string(),
                    style: "None".to_string(),
                    z_index: 990,
                    x: mouse_state.pos.x,
                    y: mouse_state.pos.y,
                    radius: 6.0,
                    inside_border_thickness: 0.002,
                    border_thickness: 1.0,
                    fade: 0.0,
                    fill_color: [0.2, 0.2, 0.2, 0.0],
                    inside_border_color: [0.4; 4],
                    border_color: [0.1, 0.1, 0.1, 0.8],
                    glow_color: [0.0; 4],
                    glow_misc: Default::default(),
                    misc: MiscButtonSettings {
                        active: true,
                        touched_time: 0.0,
                        is_touched: false,
                        pressable: false,
                        editable: false,
                    },
                };
                loader.ui_runtime.last_pos = (mouse_state.pos.x, mouse_state.pos.y);
                loader.ui_runtime.original_radius = handle_circle.radius;
                let _ = loader.add_element(
                    "Editor_Menu",
                    "Color Picker",
                    Circle(handle_circle),
                    mouse_state,
                    false,
                );
            }
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    if i == 0.0 {
        [v, t, p]
    } else if i == 1.0 {
        [q, v, p]
    } else if i == 2.0 {
        [p, v, t]
    } else if i == 3.0 {
        [p, q, v]
    } else if i == 4.0 {
        [t, p, v]
    } else {
        [v, p, q]
    }
}

#[derive(Debug)]
pub struct Menu {
    pub layers: Vec<RuntimeLayer>,
    pub active: bool,
}

impl Menu {
    pub fn rebuild_layer_cache_index(&mut self, layer_index: usize, runtime: &UiRuntime) {
        // Split so we can have &mut to this layer and & to all others.
        let (before, rest) = self.layers.split_at_mut(layer_index);
        let (layer, after) = rest.split_first_mut().unwrap();
        let l = layer;

        let dirty = l.dirty;
        if !dirty.any() {
            return;
        }

        let outlines_dirty = dirty.outlines || dirty.polygons;
        let mut rebuilt = LayerDirty::none();

        let runtime_info = |id: &Option<String>| {
            let id_str = id.as_deref().unwrap_or("");
            let runtime = runtime.get(id_str);
            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            (runtime, hash)
        };

        // ------- TEXTS -------
        if dirty.texts {
            l.cache.texts.clear();

            for t in &l.texts {
                let (rt, hash) = runtime_info(&t.id);

                l.cache.texts.push(TextParams {
                    pos: [t.x, t.y],
                    px: t.px,
                    color: t.color,
                    id_hash: hash,
                    misc: [
                        f32::from(t.misc.active),
                        rt.touched_time,
                        f32::from(rt.is_down),
                        hash,
                    ],
                    text: t.text.clone(),

                    natural_width: t.natural_width,
                    natural_height: t.natural_height,
                    id: t.id.clone(),
                    caret: t.text.len(),
                    glyph_bounds: vec![],
                });
            }

            rebuilt.mark_texts();
        }

        // ------- CIRCLES -------
        if dirty.circles {
            l.cache.circle_params.clear();

            for c in &l.circles {
                let (rt, hash) = runtime_info(&c.id);

                l.cache.circle_params.push(CircleParams {
                    center_radius_border: [c.x, c.y, c.radius, c.border_thickness],
                    fill_color: c.fill_color,
                    inside_border_color: c.inside_border_color,
                    border_color: c.border_color,
                    glow_color: c.glow_color,
                    glow_misc: [
                        c.glow_misc.glow_size,
                        c.glow_misc.glow_speed,
                        c.glow_misc.glow_intensity,
                        1.0,
                    ],
                    misc: [
                        f32::from(c.misc.active),
                        rt.touched_time,
                        f32::from(rt.is_down),
                        hash,
                    ],
                    fade: c.fade,
                    style: style_to_u32(&c.style),
                    inside_border_thickness: c.inside_border_thickness,
                    _pad: 1,
                });
            }

            rebuilt.mark_circles();
        }

        // Local helper that searches ALL OTHER layers (before + after) for a polygon id
        fn find_polygon_by_id<'a>(
            id: &Option<String>,
            before: &'a [RuntimeLayer],
            after: &'a [RuntimeLayer],
        ) -> Option<&'a UiButtonPolygon> {
            let target = match id {
                Some(s) => s,
                None => return None,
            };

            for layer in before.iter().chain(after.iter()) {
                for p in &layer.polygons {
                    if let Some(pid) = &p.id {
                        if pid == target {
                            return Some(p);
                        }
                    }
                }
            }
            None
        }

        // ------- OUTLINES -------
        if outlines_dirty {
            l.cache.outline_params.clear();
            l.cache.outline_poly_vertices.clear();

            for o in &mut l.outlines {
                if o.mode == 1.0 {
                    if let Some(poly) = find_polygon_by_id(&o.parent_id, before, after) {
                        o.vertex_offset = l.cache.outline_poly_vertices.len() as u32;

                        for v in &poly.vertices {
                            l.cache.outline_poly_vertices.push([v.pos[0], v.pos[1]]);
                        }

                        o.vertex_count = poly.vertices.len() as u32;
                    }
                }

                let (rt, hash) = runtime_info(&o.id);

                l.cache.outline_params.push(OutlineParams {
                    mode: o.mode,
                    vertex_offset: o.vertex_offset,
                    vertex_count: o.vertex_count,
                    _pad0: 0, // u32 padding, must be integer
                    shape_data: [
                        o.shape_data.x,
                        o.shape_data.y,
                        o.shape_data.radius,
                        o.shape_data.border_thickness,
                    ],
                    dash_color: o.dash_color,
                    dash_misc: [
                        o.dash_misc.dash_len,
                        o.dash_misc.dash_spacing,
                        o.dash_misc.dash_roundness,
                        o.dash_misc.dash_speed,
                    ],
                    sub_dash_color: o.sub_dash_color,
                    sub_dash_misc: [
                        o.sub_dash_misc.dash_len,
                        o.sub_dash_misc.dash_spacing,
                        o.sub_dash_misc.dash_roundness,
                        o.sub_dash_misc.dash_speed,
                    ],
                    misc: [
                        f32::from(o.misc.active),
                        rt.touched_time,
                        f32::from(rt.is_down),
                        hash,
                    ],
                });
            }

            rebuilt.mark_outlines();
        }

        // ------- HANDLES -------
        if dirty.handles {
            l.cache.handle_params.clear();

            for h in &l.handles {
                let (rt, hash) = runtime_info(&h.id);

                l.cache.handle_params.push(HandleParams {
                    center_radius_mode: [h.x, h.y, h.radius, 1.0],
                    handle_color: h.handle_color,
                    handle_misc: [
                        h.handle_misc.handle_len,
                        h.handle_misc.handle_width,
                        h.handle_misc.handle_roundness,
                        h.handle_misc.handle_speed,
                    ],
                    sub_handle_color: h.sub_handle_color,
                    sub_handle_misc: [
                        h.sub_handle_misc.handle_len,
                        h.sub_handle_misc.handle_width,
                        h.sub_handle_misc.handle_roundness,
                        h.sub_handle_misc.handle_speed,
                    ],
                    misc: [
                        f32::from(h.misc.active),
                        rt.touched_time,
                        f32::from(rt.is_down),
                        hash,
                    ],
                });
            }

            rebuilt.mark_handles();
        }

        // Common builder for vertex-emitting shapes
        let push_with_misc = |v: &UiVertex,
                              roundness_px: f32,
                              poly_index_f: f32,
                              misc: [f32; 4],
                              out: &mut Vec<UiVertexPoly>| {
            out.push(UiVertexPoly {
                pos: v.pos,
                data: [roundness_px, poly_index_f],
                color: v.color,
                misc,
            });
        };

        // ------- POLYGONS (N verts) -------
        if dirty.polygons {
            l.cache.polygon_vertices.clear();

            for (poly_index, poly) in l.polygons.iter_mut().enumerate() {
                let (rt, hash) = runtime_info(&poly.id);

                let misc = [
                    f32::from(poly.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ];

                let poly_index_f = poly_index as f32;

                let tris = triangulate_polygon(&mut poly.vertices);
                poly.tri_count = tris.len() as u32 / 3;

                for vertex in &tris {
                    let roundness_px = vertex.roundness;
                    push_with_misc(
                        vertex,
                        roundness_px,
                        poly_index_f,
                        misc,
                        &mut l.cache.polygon_vertices,
                    );
                }
            }

            rebuilt.mark_polygons();
        }

        l.dirty.clear(rebuilt);
    }

    pub fn sort_layers(&mut self) {
        self.layers.sort_by_key(|l| l.order);
    }

    pub fn bump_layer_order(
        &mut self,
        layer_name: &str,
        delta: i32,
        variables: &mut UiVariableRegistry,
    ) {
        for layer in &mut self.layers {
            if layer.name == layer_name {
                let new = layer.order as i32 + delta;
                layer.order = new.max(0) as u32;
                variables.set("selected_layer.order", layer.order.to_string());
                return;
            }
        }
    }
}

#[derive(Debug)]
pub struct ActionState {
    action_name: String,
    position: Vec2,
    last_pos: Vec2,
    radius: f32,
    original_radius: f32,

    pub active: bool,
}

#[derive(Debug)]
pub struct UiRuntime {
    pub elements: HashMap<String, ButtonRuntime>,
    pub selected_ui_element: SelectedUiElement,
    pub active_vertex: Option<usize>,
    pub drag_offset: Option<(f32, f32)>,
    pub editor_mode: bool,
    pub editing_text: bool,
    pub clipboard: String,
    pub dragging_text: bool,
    pub last_pos: (f32, f32),
    pub original_radius: f32,
    pub action_states: HashMap<String, ActionState>,
}

impl UiRuntime {
    pub fn new(editor_mode: bool) -> Self {
        Self {
            elements: HashMap::new(),
            selected_ui_element: SelectedUiElement::default(),
            active_vertex: None,
            drag_offset: None,
            editor_mode,
            editing_text: false,
            clipboard: "".to_string(),
            dragging_text: false,
            last_pos: (0.0, 0.0),
            original_radius: 0.0,
            action_states: HashMap::new(),
        }
    }

    pub fn update_touch(
        &mut self,
        id: &str,
        touched_now: bool,
        dt: f32,
        _layer_name: &String,
    ) -> TouchState {
        let entry = self
            .elements
            .entry(id.to_string())
            .or_insert_with(ButtonRuntime::default);

        entry.just_pressed = false;
        entry.just_released = false;

        match (entry.is_down, touched_now) {
            (false, true) => {
                entry.is_down = true;
                entry.just_pressed = true;
                entry.touched_time = 0.0;
                TouchState::Pressed
            }
            (true, true) => {
                entry.touched_time += dt;
                TouchState::Held
            }
            (true, false) => {
                entry.is_down = false;
                entry.just_released = true;
                TouchState::Released
            }
            (false, false) => TouchState::Idle,
        }
    }

    pub fn update_editor_mode(&mut self, editor_mode: bool) {
        self.editor_mode = editor_mode;
    }

    pub fn get(&self, id: &str) -> ButtonRuntime {
        *self.elements.get(id).unwrap_or(&ButtonRuntime::default())
    }
}

pub struct UiButtonLoader {
    pub menus: HashMap<String, Menu>,

    pub console_lines: VecDeque<String>,
    pub ui_runtime: UiRuntime,
    pub variables: UiVariableRegistry,
}

impl UiButtonLoader {
    pub fn new(editor_mode: bool) -> Self {
        let layout_path = project_path("data/ui_data/gui_layout.json");
        let layout = Self::load_gui_from_file(layout_path).unwrap_or_else(|e| {
            eprintln!("❌ Failed to load GUI layout: {e}");
            GuiLayout { menus: vec![] }
        });

        let mut loader = Self {
            menus: Default::default(),
            ui_runtime: UiRuntime::new(editor_mode),
            console_lines: VecDeque::new(),
            variables: UiVariableRegistry::new(),
        };

        // JSON layers to runtime layers...…
        let mut id_gen: usize = 1;
        for menu_json in layout.menus {
            let mut layers = Vec::new();

            for l in menu_json.layers {
                let texts = l
                    .texts
                    .unwrap_or_default()
                    .into_iter()
                    .map(UiButtonText::from_json)
                    .collect();

                let circles = l
                    .circles
                    .unwrap_or_default()
                    .into_iter()
                    .map(UiButtonCircle::from_json)
                    .collect();

                let outlines = l
                    .outlines
                    .unwrap_or_default()
                    .into_iter()
                    .map(UiButtonOutline::from_json)
                    .collect();

                let handles = l
                    .handles
                    .unwrap_or_default()
                    .into_iter()
                    .map(UiButtonHandle::from_json)
                    .collect();

                let polygons = l
                    .polygons
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| UiButtonPolygon::from_json(p, &mut id_gen))
                    .collect();

                layers.push(RuntimeLayer {
                    name: l.name,
                    order: l.order,
                    active: l.active.unwrap_or(true),
                    opaque: l.opaque.unwrap_or(false),
                    texts,
                    circles,
                    outlines,
                    handles,
                    polygons,
                    cache: LayerCache::default(),
                    gpu: LayerGpu::default(),
                    dirty: LayerDirty::all(),
                    saveable: true,
                });
            }

            layers.sort_by_key(|l| l.order); // SORT!
            // Insert this menu into the loader
            loader.menus.insert(
                menu_json.name.clone(),
                Menu {
                    layers,
                    active: true,
                },
            );
        }

        loader.add_editor_layers();
        loader.ensure_console_layer();

        loader
    }

    pub fn load_gui_from_file(path: PathBuf) -> Result<GuiLayout, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(&path)?;
        let parsed: GuiLayout = serde_json::from_str(&data)?;
        Ok(parsed)
    }

    pub fn save_gui_to_file(&self, path: PathBuf) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Convert runtime → serializable JSON
        let layout = self.to_json_gui_layout();

        let json = serde_json::to_string_pretty(&layout)?;
        fs::write(&path, json)?;

        println!("GUI saved to {}", path.display());
        Ok(())
    }

    fn to_json_gui_layout(&self) -> GuiLayout {
        let mut menus = Vec::new();
        for (menu_name, menu) in &self.menus {
            let mut layers = Vec::new();

            for l in &menu.layers {
                // Skip editor-only layers to avoid saving internal crap
                if !l.saveable {
                    continue;
                }

                layers.push(UiLayerJson {
                    name: l.name.clone(),
                    order: l.order,
                    active: Some(l.active),
                    opaque: Some(l.opaque),

                    texts: Some(l.texts.iter().map(|t| t.to_json()).collect()),
                    circles: Some(l.circles.iter().map(|c| c.to_json()).collect()),
                    outlines: Some(l.outlines.iter().map(|o| o.to_json()).collect()),
                    handles: Some(l.handles.iter().map(|h| h.to_json()).collect()),
                    polygons: Some(l.polygons.iter().map(|p| p.to_json()).collect()),
                });
            }

            menus.push(MenuJson {
                name: menu_name.to_string(),
                layers,
            })
        }

        GuiLayout { menus }
    }

    fn add_editor_layers(&mut self) {
        let menu = self.menus.entry("Editor_Menu".into()).or_insert(Menu {
            layers: Vec::new(),
            active: true,
        });

        // Insert layers
        menu.layers.push(RuntimeLayer {
            name: "editor_selection".into(),
            order: 900,
            active: true,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            outlines: vec![],
            handles: vec![],
            polygons: vec![],
            dirty: LayerDirty::all(),
            gpu: LayerGpu::default(),
            opaque: true,
            saveable: false,
        });

        menu.layers.push(RuntimeLayer {
            name: "editor_handles".into(),
            order: 950,
            active: true,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            outlines: vec![],
            handles: vec![],
            polygons: vec![],
            dirty: LayerDirty::all(),
            gpu: LayerGpu::default(),
            opaque: true,
            saveable: false,
        });

        // ensure correct draw order
        menu.layers.sort_by_key(|l| l.order);
    }

    pub fn ensure_console_layer(&mut self) -> &mut RuntimeLayer {
        // 1. Ensure Debug_Menu exists
        let menu = self.menus.entry("Debug_Menu".into()).or_insert(Menu {
            layers: Vec::new(),
            active: true,
        });

        // 2. If shader_console layer exists, return it
        if let Some(idx) = menu.layers.iter().position(|l| l.name == "shader_console") {
            return &mut menu.layers[idx];
        }

        // 3. Create shader_console layer
        menu.layers.push(RuntimeLayer {
            name: "shader_console".into(),
            order: 980,
            active: true,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            outlines: vec![],
            handles: vec![],
            polygons: vec![],
            dirty: LayerDirty::all(),
            gpu: LayerGpu::default(),
            opaque: false,
            saveable: false,
        });

        // 4. Sort by order
        menu.layers.sort_by_key(|l| l.order);

        // 5. Return newly created layer
        let idx = menu
            .layers
            .iter()
            .position(|l| l.name == "shader_console")
            .expect("shader_console layer missing");

        &mut menu.layers[idx]
    }

    pub fn sync_console_ui(&mut self) {
        let lines: Vec<String> = self.console_lines.iter().cloned().collect();

        let layer = self.ensure_console_layer();

        layer.texts.clear();
        for (i, line) in lines.into_iter().enumerate() {
            layer.texts.push(UiButtonText {
                id: Some(format!("console_line_{}", i)),
                action: "None".to_string(),
                style: "None".to_string(),
                z_index: 980 + i as i32,
                x: 20.0,
                y: 20.0 + i as f32 * 22.0,
                top_left_offset: [0.0, 0.0],
                bottom_left_offset: [0.0, 0.0],
                top_right_offset: [0.0, 0.0],
                bottom_right_offset: [0.0, 0.0],
                px: 18,
                color: [0.95, 0.9, 0.8, 0.95],
                text: line.to_string(),
                template: line.to_string(),
                misc: MiscButtonSettings {
                    active: true,
                    touched_time: 0.0,
                    is_touched: false,
                    pressable: false,
                    editable: false,
                },
                natural_width: 50.0,
                natural_height: 20.0,
                being_edited: false,
                caret: line.len(),
                being_hovered: false,
                just_unhovered: false,
                sel_start: 0,
                sel_end: 0,
                has_selection: false,
                glyph_bounds: vec![],
            });
        }

        layer.dirty.mark_texts();
    }

    pub(crate) fn mark_editor_layers_dirty(&mut self) {
        const TARGETS: [&str; 3] = ["editor_selection", "editor_handles", "shader_console"];

        let menu = self.menus.get_mut("Debug_Menu");
        if let Some(menu) = menu {
            for layer in &mut menu.layers {
                if TARGETS.contains(&layer.name.as_str()) {
                    layer.dirty.mark_all();
                }
            }
        }
    }
    pub fn log_console(&mut self, message: impl Into<String>) {
        self.console_lines.push_back(message.into());

        while self.console_lines.len() > 6 {
            self.console_lines.pop_front();
        }
    }

    pub fn update_dynamic_texts(&mut self) {
        let mut being_hovered = false;
        let mut selected_being_hovered = false;
        for (_, menu) in &mut self.menus {
            for layer in &mut menu.layers {
                let mut any_changed = false;

                for t in &mut layer.texts {
                    if self.ui_runtime.selected_ui_element.just_deselected {
                        t.clear_selection();
                        layer.dirty.mark_texts();
                    }
                    if t.being_edited || t.being_hovered || t.just_unhovered {
                        any_changed = true;
                    }

                    // Skip if no template braces exist!
                    if !t.template.contains('{') || !t.template.contains('}') || t.being_edited {
                        continue;
                    }

                    // Resolve template
                    let new_text = resolve_template(&t.template, &self.variables);
                    if new_text != t.text {
                        t.text = new_text;
                        any_changed = true;
                    }
                    if !being_hovered && t.being_hovered {
                        being_hovered = true;
                        if t.id
                            == Option::from(self.ui_runtime.selected_ui_element.element_id.clone())
                        {
                            selected_being_hovered = true;
                        }
                    }
                }

                if any_changed {
                    layer.dirty.mark_texts();
                }
            }
        }
        self.variables
            .set("any_text.being_hovered", being_hovered.to_string());
        self.variables.set(
            "selected_text.being_hovered",
            selected_being_hovered.to_string(),
        );
    }

    pub fn handle_touches(
        &mut self,
        dt: f32,
        input_state: &mut InputState,
        time_system: &TimeSystem,
    ) {
        self.ui_runtime.selected_ui_element.just_deselected = false;
        self.ui_runtime.selected_ui_element.just_selected = false;
        let mouse_snapshot = MouseSnapshot::from_mouse(&input_state.mouse);
        let editor_mode = self.ui_runtime.editor_mode;

        let press_started_on_ui = if mouse_snapshot.just_pressed {
            press_began_on_ui(&self.menus, &mouse_snapshot, editor_mode)
        } else {
            (false, "None".to_string())
        };

        if mouse_snapshot.just_pressed && !press_started_on_ui.0 {
            if !near_handle(&self.menus, &mouse_snapshot) {
                self.ui_runtime.selected_ui_element.active = false;
                self.ui_runtime.editing_text = false;
                println!("deselection");
                self.ui_runtime.selected_ui_element.dragging = false;
                self.ui_runtime.selected_ui_element.just_deselected = true;
                self.ui_runtime.selected_ui_element.just_selected = false;
                self.update_selection();
            }
        }

        let top_hit = find_top_hit(&mut self.menus, &mouse_snapshot, editor_mode);

        let EditorInteractionResult {
            trigger_selection: mut selection,
            pending_circle_updates,
            moved_any_selected_object,
        } = handle_editor_mode_interactions(
            self,
            time_system,
            &mouse_snapshot,
            top_hit.clone(),
            input_state,
        );
        self.variables
            .set("editing_text", format!("{}", self.ui_runtime.editing_text));
        if editor_mode {
            handle_text_editing(
                &mut self.ui_runtime,
                &mut self.menus,
                input_state,
                mouse_snapshot,
            );

            apply_pending_circle_updates(self, dt, pending_circle_updates);
        }

        if moved_any_selected_object {
            self.mark_editor_layers_dirty();
            selection = true;
        }

        if editor_mode {
            if handle_scroll_resize(self, mouse_snapshot.scroll) {
                selection = true;
            }
        }

        let trigger_selection = selection;

        // if self.ui_runtime.selected_ui_element.active {
        //     self.ui_runtime.selected_ui_element.active = false;
        //     self.update_selection();
        //
        // }

        if trigger_selection {
            self.update_selection();
        }

        activate_action(self, &top_hit);

        execute_action(self, &top_hit, &input_state.mouse);

        if editor_mode {
            self.apply_ui_edit_movement(input_state);

            if input_state.action_pressed_once("Delete selected GUI Element")
                && self.ui_runtime.selected_ui_element.active
                && !self.ui_runtime.editing_text
            {
                println!("deleting");
                let element_id = self.ui_runtime.selected_ui_element.element_id.clone();
                let layer_name = self.ui_runtime.selected_ui_element.layer_name.clone();
                let menu_name = self.ui_runtime.selected_ui_element.menu_name.clone();
                self.ui_runtime.selected_ui_element.active = false;

                let _ = self.delete_element(&menu_name, &layer_name, &element_id);
            }
        }
    }

    pub fn add_element(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        mut element: UiElement,
        mouse: &MouseState,
        on_mouse_pos: bool,
    ) -> Result<(), String> {
        // 1. Get selected menu
        let menu = self
            .menus
            .get_mut(menu_name)
            .ok_or_else(|| format!("Menu *{}* doesn't exist", menu_name))?;

        // 2. Get selected layer
        let layer = menu
            .layers
            .iter_mut()
            .find(|l| l.name == layer_name)
            .ok_or_else(|| format!("Layer *{}* not found in *{}* menu", layer_name, menu_name))?;
        let id = mouse.pos.x as u32 - mouse.pos.y as u32;

        // 3. Apply mouse positioning (editor placement)
        if on_mouse_pos {
            match &mut element {
                UiElement::Text(t) => {
                    t.x = mouse.pos.x;
                    t.y = mouse.pos.y;
                    t.id = Option::from(id.to_string());
                }
                UiElement::Circle(c) => {
                    c.x = mouse.pos.x;
                    c.y = mouse.pos.y;
                    c.id = Option::from(id.to_string());
                }
                UiElement::Outline(o) => {
                    o.id = Option::from(id.to_string());
                }
                UiElement::Handle(h) => {
                    h.x = mouse.pos.x;
                    h.y = mouse.pos.y;
                    h.id = Option::from(id.to_string());
                }
                UiElement::Polygon(p) => {
                    p.id = Option::from(id.to_string());
                }
            }
        }
        // 4. Insert element (NO ids needed)
        match element {
            UiElement::Text(t) => {
                layer.texts.push(t);
                layer.dirty.mark_texts();
            }
            UiElement::Circle(c) => {
                layer.circles.push(c);
                layer.dirty.mark_circles();
            }
            UiElement::Outline(o) => {
                layer.outlines.push(o);
                layer.dirty.mark_outlines();
            }
            UiElement::Handle(h) => {
                layer.handles.push(h);
                layer.dirty.mark_handles();
            }
            UiElement::Polygon(p) => {
                layer.polygons.push(p);
                layer.dirty.mark_polygons();
            }
        }

        Ok(())
    }

    pub(crate) fn hash_id(id: &str) -> f32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        let hash_u64 = hasher.finish(); // 64-bit
        // map to [0, 1]
        (hash_u64 as f64 / u64::MAX as f64) as f32
    }

    pub fn update_selection(&mut self) {
        self.ui_runtime.editing_text = false;
        if !self.ui_runtime.selected_ui_element.active {
            if let Some(editor_menu) = self.menus.get_mut("Editor_Menu") {
                if let Some(editor_layer) = editor_menu
                    .layers
                    .iter_mut()
                    .find(|l| l.name == "editor_selection")
                {
                    editor_layer.active = false;
                    editor_layer.circles.clear();
                    editor_layer.outlines.clear();
                    editor_layer.handles.clear();
                    editor_layer.polygons.clear();
                    editor_layer.dirty.mark_all()
                }
            }
            return;
        }
        let sel = self.ui_runtime.selected_ui_element.clone();

        if let Some(menu) = self.menus.get_mut(&sel.menu_name) {
            if let Some(layer) = menu
                .layers
                .iter()
                .find(|l| l.name == sel.layer_name.to_string())
            {
                self.variables
                    .set("selected_layer.order", layer.order.to_string());
            }
        }
        if let Some(element) = self.find_element(&sel.menu_name, &sel.layer_name, &sel.element_id) {
            if let Some(editor_menu) = self.menus.get_mut("Editor_Menu") {
                if let Some(editor_layer) = editor_menu
                    .layers
                    .iter_mut()
                    .find(|l| l.name == "editor_selection")
                {
                    editor_layer.active = true;
                    editor_layer.dirty.mark_all();
                    editor_layer.circles.clear();
                    editor_layer.outlines.clear();
                    editor_layer.handles.clear();
                    editor_layer.polygons.clear();
                    match element {
                        Circle(c) => {
                            if self.ui_runtime.editor_mode && c.misc.editable {
                                let circle_outline = UiButtonOutline {
                                    id: Some("Circle Outline".to_string()),
                                    z_index: c.z_index,
                                    parent_id: c.id.clone(),
                                    mode: 0.0,
                                    vertex_offset: 0,
                                    vertex_count: 0,
                                    dash_color: [0.2, 0.0, 0.4, 0.8],
                                    dash_misc: DashMisc {
                                        dash_len: 4.0,
                                        dash_spacing: 1.5,
                                        dash_roundness: 1.0,
                                        dash_speed: 4.0,
                                    },
                                    sub_dash_color: [0.8, 1.0, 1.0, 0.5],
                                    sub_dash_misc: DashMisc {
                                        dash_len: 1.0,
                                        dash_spacing: 1.0,
                                        dash_roundness: 0.0,
                                        dash_speed: -2.0,
                                    },
                                    misc: c.misc,
                                    shape_data: ShapeData {
                                        x: c.x,
                                        y: c.y,
                                        radius: c.radius,
                                        border_thickness: c.border_thickness,
                                    },
                                };
                                editor_layer.outlines.push(circle_outline);

                                let handle = UiButtonHandle {
                                    id: Some("Circle Handle".to_string()),
                                    parent_id: c.id,
                                    x: c.x,
                                    y: c.y,
                                    radius: c.radius,
                                    handle_thickness: 10.0,
                                    handle_color: [0.65, 0.22, 0.05, 1.0],
                                    handle_misc: HandleMisc {
                                        handle_len: 0.09,
                                        handle_width: 0.2,
                                        handle_roundness: 0.3,
                                        handle_speed: 0.0,
                                    },
                                    sub_handle_color: [0.0, 0.0, 0.0, 0.7],
                                    sub_handle_misc: HandleMisc {
                                        handle_len: 0.08,
                                        handle_width: 0.05,
                                        handle_roundness: 0.5,
                                        handle_speed: 0.0,
                                    },
                                    misc: MiscButtonSettings {
                                        active: true,
                                        touched_time: 0.0,
                                        is_touched: false,
                                        pressable: true,
                                        editable: false,
                                    },
                                    z_index: c.z_index,
                                };
                                editor_layer.handles.push(handle);
                            }
                            self.variables
                                .set("selected_element.z_index", c.z_index.to_string());
                        }
                        Handle(_h) => {}
                        Polygon(p) => {
                            if self.ui_runtime.editor_mode && p.misc.editable {
                                let mut cx = 0.0;
                                let mut cy = 0.0;
                                for v in &p.vertices {
                                    cx += v.pos[0];
                                    cy += v.pos[1];
                                }
                                cx /= p.vertices.len() as f32;
                                cy /= p.vertices.len() as f32;

                                let mut radius: f32 = 0.0;
                                for (i, v) in p.vertices.iter().enumerate() {
                                    let dx = v.pos[0] - cx;
                                    let dy = v.pos[1] - cy;
                                    let distance = (dx * dx + dy * dy).sqrt();

                                    radius = radius.max(distance);
                                    let vertex_outline = UiButtonCircle {
                                        id: Some(format!("vertex_outline_{}", i)),
                                        action: "None".to_string(),
                                        style: "None".to_string(),
                                        z_index: i as i32,
                                        x: v.pos[0],
                                        y: v.pos[1],
                                        radius: 10.0,
                                        inside_border_thickness: 2.0,
                                        border_thickness: 0.0,
                                        fade: 1.0,
                                        fill_color: [0.0, 0.8, 0.0, 0.6],
                                        inside_border_color: [0.0; 4],
                                        border_color: [0.0, 0.0, 0.0, 0.0],
                                        glow_color: [0.0, 0.0, 0.5, 0.0],
                                        glow_misc: GlowMisc {
                                            glow_size: 0.0,
                                            glow_speed: 0.0,
                                            glow_intensity: 0.0,
                                        },
                                        misc: MiscButtonSettings {
                                            active: false,
                                            touched_time: 0.0,
                                            is_touched: false,
                                            pressable: false,
                                            editable: false,
                                        },
                                    };
                                    editor_layer.circles.push(vertex_outline);
                                }
                                let polygon_outline = UiButtonOutline {
                                    id: Some("Polygon Outline".to_string()),
                                    z_index: p.z_index,
                                    parent_id: p.id.clone(),
                                    mode: 1.0,
                                    vertex_offset: 0,
                                    vertex_count: p.vertices.len() as u32,
                                    dash_color: [0.2, 0.0, 0.4, 0.8],
                                    dash_misc: DashMisc {
                                        dash_len: 4.0,
                                        dash_spacing: 1.5,
                                        dash_roundness: 1.0,
                                        dash_speed: 4.0,
                                    },
                                    sub_dash_color: [0.8, 1.0, 1.0, 0.5],
                                    sub_dash_misc: DashMisc {
                                        dash_len: 1.0,
                                        dash_spacing: 1.0,
                                        dash_roundness: 0.0,
                                        dash_speed: -2.0,
                                    },
                                    misc: p.misc,
                                    shape_data: ShapeData {
                                        x: cx,
                                        y: cy,
                                        radius,
                                        border_thickness: 2.0,
                                    },
                                };
                                editor_layer.outlines.push(polygon_outline);
                            }
                            self.variables
                                .set("selected_element.z_index", p.z_index.to_string());
                        }
                        Text(tx) => {
                            // if self.ui_runtime.editor_mode {
                            //
                            // }

                            self.variables
                                .set("selected_element.z_index", tx.z_index.to_string());
                        }
                        Outline(_o) => {}
                    }
                }
            }
        }

        self.ui_runtime.selected_ui_element.active = true;
        if !self.ui_runtime.selected_ui_element.just_deselected {
            self.ui_runtime.selected_ui_element.just_selected = true;
        }
    }

    pub fn find_element(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        element_id: &str,
    ) -> Option<UiElement> {
        if let Some(menu) = self.menus.get_mut(menu_name) {
            let layer = menu
                .layers
                .iter()
                .find(|l| l.name == layer_name.to_string())?;

            // Circles
            for c in &layer.circles {
                if let Some(id) = &c.id {
                    if id == element_id {
                        return Some(Circle(c.clone()));
                    }
                }
            }

            // Polygons
            for p in &layer.polygons {
                if let Some(id) = &p.id {
                    if id == element_id {
                        return Some(UiElement::Polygon(p.clone()));
                    }
                }
            }

            // Texts
            for tx in &layer.texts {
                if let Some(id) = &tx.id {
                    if id == element_id {
                        return Some(UiElement::Text(tx.clone()));
                    }
                }
            }
        }
        None
    }

    pub fn edit_circle(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        element_id: &str,
        x: Option<f32>,
        y: Option<f32>,
        radius: Option<f32>,
        fill_color: Option<[f32; 4]>,
        border_color: Option<[f32; 4]>,
    ) -> bool {
        if let Some(menu) = self.menus.get_mut(menu_name) {
            let layer = menu.layers.iter_mut().find(|l| {
                //println!("COMPARING [{}] with [{}]", l.name, layer_name);
                l.name == layer_name
            });

            // Circles
            if let Some(layer) = layer {
                for c in &mut layer.circles {
                    if let Some(id) = &c.id {
                        if id.as_str() == element_id {
                            c.x = x.unwrap_or(c.x);
                            c.y = y.unwrap_or(c.y);
                            c.radius = radius.unwrap_or(c.radius);
                            c.fill_color = fill_color.unwrap_or(c.fill_color);
                            c.border_color = border_color.unwrap_or(c.border_color);
                            layer.dirty.mark_circles();
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    pub fn delete_element(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        element_id: &str,
    ) -> Option<UiElement> {
        // get mutable layer
        if let Some(menu) = self.menus.get_mut(menu_name) {
            let layer = menu.layers.iter_mut().find(|l| l.name == layer_name)?;

            // Circles
            if let Some(pos) = layer
                .circles
                .iter()
                .position(|c| c.id.as_deref() == Some(element_id))
            {
                let removed = layer.circles.remove(pos);
                return Some(UiElement::Circle(removed));
            }

            // Polygons
            if let Some(pos) = layer
                .polygons
                .iter()
                .position(|p| p.id.as_deref() == Some(element_id))
            {
                let removed = layer.polygons.remove(pos);
                return Some(UiElement::Polygon(removed));
            }

            // Texts
            if let Some(pos) = layer
                .texts
                .iter()
                .position(|t| t.id.as_deref() == Some(element_id))
            {
                let removed = layer.texts.remove(pos);
                return Some(UiElement::Text(removed));
            }
        }
        None
    }
    pub fn apply_ui_edit_movement(&mut self, input_state: &mut InputState) {
        let sel = &self.ui_runtime.selected_ui_element;
        if !sel.active {
            return;
        }

        let mut changed = false;

        // ============================================================
        // BLOCK 1: Element XY movement, resizing, z-index movement
        // ============================================================
        {
            let menu = match self.menus.get_mut(&sel.menu_name) {
                Some(m) => m,
                None => return,
            };

            let layer = match menu.layers.iter_mut().find(|l| l.name == sel.layer_name) {
                Some(l) => l,
                None => return,
            };

            // Movement (WASD)
            let speed = calc_move_speed(input_state);
            let mut dx = 0.0;
            let mut dy = 0.0;

            if input_state.action_repeat("Move Element Left") {
                dx -= speed;
            }
            if input_state.action_repeat("Move Element Right") {
                dx += speed;
            }
            if input_state.action_repeat("Move Element Up") {
                dy -= speed;
            }
            if input_state.action_repeat("Move Element Down") {
                dy += speed;
            }

            if dx != 0.0 || dy != 0.0 {
                layer.bump_element_xy(&sel.element_id, dx, dy);
                layer.dirty.mark_all();
                changed = true;
            }

            // -----------------------------
            // Resizing (+, -, scroll)
            // -----------------------------
            let mut scale = 1.0;

            if input_state.action_repeat("Resize Element Bigger") {
                scale = 1.05;
            }
            if input_state.action_repeat("Resize Element Smaller") {
                scale = 0.95;
            }
            if input_state.action_repeat("Resize Element Bigger Scroll") {
                scale = 1.05;
            }
            if input_state.action_repeat("Resize Element Smaller Scroll") {
                print!("Scroleld down");
                scale = 0.95;
            }

            if scale != 1.0 {
                layer.resize_element(&sel.element_id, scale);
                layer.dirty.mark_all();
                changed = true;
            }

            // -----------------------------
            // Element Z movement
            // -----------------------------
            if !input_state.shift {
                if input_state.action_repeat("Move Element Z Up") {
                    layer.bump_element_z(&sel.element_id, 1, &mut self.variables);
                    layer.sort_by_z();
                    layer.dirty.mark_all();
                    changed = true;
                }

                if input_state.action_repeat("Move Element Z Down") {
                    layer.bump_element_z(&sel.element_id, -1, &mut self.variables);
                    layer.sort_by_z();
                    layer.dirty.mark_all();
                    changed = true;
                }
            }
        }
        // ====== end block 1 (layer + menu borrow released) ======

        // ============================================================
        // BLOCK 2: LAYER ORDERING
        // ============================================================
        {
            let menu = match self.menus.get_mut(&sel.menu_name) {
                Some(m) => m,
                None => return,
            };

            if !input_state.shift {
                if input_state.action_repeat("Move Layer Up") {
                    menu.bump_layer_order(&sel.layer_name, 1, &mut self.variables);
                    menu.sort_layers();
                    changed = true;
                }

                if input_state.action_repeat("Move Layer Down") {
                    menu.bump_layer_order(&sel.layer_name, -1, &mut self.variables);
                    menu.sort_layers();
                    changed = true;
                }
            }
        }
        // ====== end block 2 ======

        // ============================================================
        // FINAL: only dirty if something changed
        // ============================================================
        if changed {
            self.update_selection();
            self.mark_editor_layers_dirty();
        }
    }
}
