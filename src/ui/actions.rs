#![allow(dead_code, unused_variables)]
pub mod drag_hue_point;

use crate::data::{SettingKey, SettingOp, Settings};
use crate::helpers::get_splash_message;
use crate::helpers::paths::rusty_skylines_dir;
use crate::renderer::renderer::Renderer;
use crate::renderer::ui::ui_text_rendering::Anchor;
use crate::simulation::Simulation;
use crate::ui::action_parser::{compile_actions, run_action};
use crate::ui::parser::Value;
use crate::ui::ui_edit_manager::{delete_aps_in_layer, ChangeColorCommand, ColorComponent, CreateAPCommand, CreateElementCommand, DeleteAPCommand, DeleteElementCommand, DuplicateElementCommand, MoveElementCommand, ResizeElementCommand};
use crate::ui::ui_editor::{
    Ui, get_element, get_element_main_size, get_element_mut, get_element_mut_dirty_layer,
    get_element_position,
};
use crate::ui::ui_edits::{SizeProperty, create_element, delete_element, get_element_color};
use crate::ui::ui_touch_manager::{ElementEvent, ElementRef, GlobalEvent, Touchable};
use crate::ui::variables::{Variables, initialize_value, save_colors};
use crate::ui::vertex::{get_rect_texture, AdvancedPrimitive, ElementKind, UiButtonCircle, UiButtonHandle, UiButtonOutline, UiButtonPolygon, UiButtonRect, UiButtonText, UiElement};
use crate::world::buildings::zoning::ZoningType;
use crate::world::game_state::{
    GameState, LoadResult, NewSavePackage, SaveInfo, SaveResult, get_available_saves,
    make_safe_save_name,
};
use crate::world::props::props::Props;
use crate::world::roads::road_structs::{BuildMode, LeftLaneCount, RightLaneCount};
use crate::world::terrain::terrain_subsystem::CursorMode;
use crate::world::world::World;
use glam::{DVec3, Vec2, Vec3};
use std::cmp::{Ordering, PartialEq, max, min};
use std::collections::{HashMap, VecDeque};
use std::env::{args, set_var};
use std::fmt::Display;
use std::str::FromStr;
use rusty_skylines_utils::positions::WorldPos;
use tracing::error;
use tracing_subscriber::fmt::layer;
use usvg::filter::CompositeOperator::In;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use crate::mini_renderer::{MiniRender, MiniRenderer};

#[derive(Debug, Clone, PartialEq)]
pub enum UiCommand {
    TextUtils {
        menu: String,
        layer: String,
        id: String,
        action: String,
    },
    DeselectAll,
    OpenMenu {
        menu_name: String,
    },
    CloseMenu {
        menu_name: String,
    },
    CloseAllMenus,
    CloseAllLayers {
        menu_name: String,
    },
    ToggleMenu {
        menu_name: String,
    },
    MenuActive {
        menu_name: String,
    },
    LayerActive {
        menu_name: String,
        layer_name: String,
    },
    UnDirty { menu_name: String, layer_name: String },
    // ===== LAYER COMMANDS =====
    OpenLayer {
        menu_name: String,
        layer_name: String,
    },
    CloseLayer {
        menu_name: String,
        layer_name: String,
    },
    ToggleLayer {
        menu_name: String,
        layer_name: String,
    },

    // ===== VARIABLE COMMANDS =====
    SetVar {
        name: String,
        value: String,
    },
    IncVar {
        name: String,
        amount: String,
    },
    DecVar {
        name: String,
        amount: String,
    },
    MulVar {
        name: String,
        factor: String,
    },
    ToggleVar {
        name: String,
    },
    Clamp {
        name: String,
        min: String,
        max: String,
    },

    // ===== FLOW CONTROL =====
    Delay {
        seconds: String,
    },
    Halt,
    Skip {
        count: usize,
    },
    If {
        condition: String,
        then: Vec<UiCommand>,
        else_branch: Vec<UiCommand>,
    },
    IfVarEq {
        var_name: String,
        value: String,
        then: Vec<UiCommand>,
        else_branch: Vec<UiCommand>,
    },
    For {
        value: String,
        commands: Vec<UiCommand>,
    },
    // Element Commands
    AddElement {
        menu: String,
        layer: String,
        id: String,
        kind: String,
        center: String,
        actions: String,
        undoable: bool,
    },
    AddAP {
        menu: String,
        name: String,
        ap_name: String,
        ap_var: String,
        center: String,
        scale: String,
        is_temporary: bool,
    },
    DeleteAP {
        menu: String,
        layer: String,
        reference_id: String,
    },
    DeleteAPsInLayer {
        menu: String,
        layer: String,
    },
    CloneElement {
        from_menu: String,
        from_layer: String,
        from_id: String,
        to_menu: String,
        to_layer: String,
        to_id: String,
        center: String,
        actions: String,
        undoable: bool,
    },
    CloneLayer {
        from_menu: String,
        from_layer: String,
        to_menu: String,
        to_layer: String,
        undoable: bool,
    },
    DeleteLayer {
        menu: String,
        layer: String,
        undoable: bool,
    },
    DeleteElement {
        menu: String,
        layer: String,
        id: String,
        undoable: bool,
    },
    SaveGame {
        and_exit: String,
    },
    LoadSave {
        save_name: String,
        without_saving: bool,
    },
    ExitGame,
    Print {
        statement: String,
    },
    PrintNoChat {
        statement: String,
    },
    DebugVars,
    DebugMenus,

    Call {
        function_name: String,
        args: Option<String>,
    },
    Noop,

}

// ==================== ACTION STATE ====================

#[derive(Debug, Clone)]
pub struct ActionState {
    pub action_name: String,
    pub active: bool,
    pub started_at: f64,
    pub position: Option<Vec2>,
    pub last_pos: Option<Vec2>,
    pub custom_data: HashMap<String, Value>,
}

impl ActionState {
    pub fn new(name: &str) -> Self {
        Self {
            action_name: name.to_string(),
            active: true,
            started_at: 0.0,
            position: None,
            last_pos: None,
            custom_data: HashMap::new(),
        }
    }

    pub fn with_time(name: &str, time: f64) -> Self {
        Self {
            action_name: name.to_string(),
            active: true,
            started_at: time,
            position: None,
            last_pos: None,
            custom_data: HashMap::new(),
        }
    }

    pub fn set_data(&mut self, key: &str, value: Value) {
        self.custom_data.insert(key.to_string(), value);
    }

    pub fn get_data(&self, key: &str) -> Option<&Value> {
        self.custom_data.get(key)
    }
}

// ==================== DELAYED COMMAND ====================

#[derive(Debug, Clone)]
struct DelayedCommands {
    commands: Vec<UiCommand>,
    execute_at: f64,
}

// ==================== COMMAND RESULT ====================

#[derive(Debug, Clone, PartialEq)]
pub enum CommandResult {
    Ok,
    Stop,
    Delay {
        seconds: f64,
        remaining: Vec<UiCommand>,
    },
    Skip(usize),
    Error(String),
    AnnoyingError(String),
}

// ==================== COMMAND CONTEXT ====================

/// Context provided only during command execution (drain phase).
pub struct CommandContext<'a> {
    pub world: &'a mut World,
    pub renderer: &'a mut Renderer,
    pub ui: &'a mut Ui,
    pub window_size: PhysicalSize<f32>,
    pub settings: &'a mut Settings,
    pub event_loop: &'a dyn ActiveEventLoop,
    pub game_state: &'a mut GameState,
    pub simulation: &'a mut Simulation,
    pub element_ctx: ElementContext,
    pub mini_renderer: &'a mut MiniRenderer,
}

// ==================== COMMAND QUEUE ====================

/// The central command queue that processes commands.
/// Commands are queued without context, then drained with context each frame.
pub struct CommandQueue {
    queue: VecDeque<UiCommand>,
    delayed: Vec<DelayedCommands>,
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            delayed: Vec::new(),
        }
    }

    // ==================== QUEUEING (no context needed) ====================

    /// Queue a single command.
    pub fn push(&mut self, cmd: UiCommand) {
        self.queue.push_back(cmd);
    }

    /// Queue a single optional command.
    pub fn push_optional(&mut self, cmd: Option<UiCommand>) {
        if let Some(cmd) = cmd {
            self.push(cmd);
        }
    }

    /// Queue multiple commands.
    pub fn push_many(&mut self, cmds: impl IntoIterator<Item = UiCommand>) {
        for cmd in cmds {
            self.push(cmd);
        }
    }

    /// Check if queue is empty (including delayed).
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty() && self.delayed.is_empty()
    }

    /// Get pending command count.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    // ==================== EXECUTION (context required) ====================

    /// Drain and execute all pending commands.
    /// Call this once per frame.
    pub fn drain(&mut self, ctx: &mut CommandContext) {
        // Process delayed commands that are ready
        self.process_delayed(ctx);

        // Drain the main queue
        while let Some(cmd) = self.queue.pop_front() {
            match self.execute_one(cmd, ctx) {
                CommandResult::Ok => continue,
                CommandResult::Stop => {
                    self.queue.clear();
                    break;
                }
                CommandResult::Skip(n) => {
                    for _ in 0..n {
                        self.queue.pop_front();
                    }
                }
                CommandResult::Delay { seconds, remaining } => {
                    if !remaining.is_empty() {
                        self.delayed.push(DelayedCommands {
                            commands: remaining,
                            execute_at: ctx.world.time.total_time + seconds,
                        });
                    }
                    break;
                }
                CommandResult::Error(msg) => {
                    eprintln!("[CommandQueue] Error: {}", msg);
                }
                CommandResult::AnnoyingError(msg) => {
                    //eprintln!("[CommandQueue] Annoying Error: {}", msg);
                }
            }
        }
    }

    fn process_delayed(&mut self, ctx: &mut CommandContext) {
        let current_time = ctx.world.time.total_time;

        let ready: Vec<DelayedCommands> = self
            .delayed
            .iter()
            .filter(|d| d.execute_at <= current_time)
            .cloned()
            .collect();

        self.delayed.retain(|d| d.execute_at > current_time);

        for delayed in ready {
            for cmd in delayed.commands {
                self.queue.push_back(cmd);
            }
        }
    }

    fn execute_multiple_for(
        &mut self,
        val: Vec<Value>,
        commands: Vec<UiCommand>,
        ctx: &mut CommandContext,
    ) {
        for (idx, value) in val.into_iter().enumerate() {
            ctx.ui.variables.set_var("idx", idx);
            ctx.ui.variables.set_var("val", value);
            for cmd in commands.iter() {
                match self.execute_one(cmd.clone(), ctx) {
                    CommandResult::Ok => {}
                    CommandResult::Stop => {}
                    CommandResult::Skip(n) => {}
                    CommandResult::Delay { seconds, remaining } => {}
                    CommandResult::Error(msg) => {
                        eprintln!("[CommandQueue] Error inside for loop: {}", msg);
                    }
                    CommandResult::AnnoyingError(msg) => {
                        //eprintln!("[CommandQueue] Annoying Error inside for loop: {}", msg);
                    }
                }
            }
        }
    }
    pub fn execute_multiple(&mut self, commands: Vec<UiCommand>, ctx: &mut CommandContext) {
        for cmd in commands {
            match self.execute_one(cmd, ctx) {
                CommandResult::Ok => {}
                CommandResult::Stop => {}
                CommandResult::Skip(n) => {}
                CommandResult::Delay { seconds, remaining } => {}
                CommandResult::Error(msg) => {
                    eprintln!("[CommandQueue] Error inside command: {}", msg);
                }
                CommandResult::AnnoyingError(msg) => {
                    //eprintln!("[CommandQueue] Annoying Error inside command: {}", msg);
                }
            }
        }
    }
    pub fn execute_one(&mut self, cmd: UiCommand, ctx: &mut CommandContext) -> CommandResult {
        //println!("{:?}", cmd);
        match cmd {
            UiCommand::TextUtils {
                menu,
                layer,
                id,
                action,
            } => {
                let menu = string_to_value(ctx, menu).into_string();
                let layer = string_to_value(ctx, layer).into_string();
                let id = string_to_value(ctx, id).into_string();
                let element_ref = ElementRef {
                    menu,
                    layer,
                    id,
                    kind: ElementKind::Text,
                };
                let action = string_to_value(ctx, action).into_string();
                let Some(el) = get_element_mut(&mut ctx.ui.menus, &element_ref) else {
                    return CommandResult::Error(format!(
                        "In TextUtils UI Action: Element not found: [{}, {}, {}]",
                        element_ref.menu, element_ref.layer, element_ref.id
                    ));
                };
                let Some(text) = el.as_text_mut() else {
                    return CommandResult::Error(format!(
                        "In TextUtils UI Action: Element found, but is not a text: [{}, {}, {}, {}]",
                        element_ref.menu,
                        element_ref.layer,
                        element_ref.id,
                        el.kind()
                    ));
                };

                match action.to_lowercase().as_str() {
                    "texteditstart" => ctx
                        .ui
                        .touch_manager
                        .push_event(element_ref, ElementEvent::TextEditStart),
                    "blocklinebreak" => text.block_line_break = true,
                    "allowlinebreak" => text.block_line_break = false,
                    _ => {
                        return CommandResult::Error(format!(
                            "In TextUtils UI Action: Text action is not a valid action: {}",
                            action
                        ));
                    }
                }

                CommandResult::Ok
            }
            UiCommand::DeselectAll => {
                ctx.ui
                    .touch_manager
                    .next_frame_global_events
                    .push(GlobalEvent::DeselectAllRequested);
                CommandResult::Ok
            }

            UiCommand::OpenMenu { menu_name } => {
                let menu_name = string_to_value(ctx, menu_name);

                let menu_names: Vec<String> = if let Some(name) = menu_name.as_str() {
                    vec![name.to_string()]
                } else if let Some(names) = menu_name.as_array() {
                    let mut result = Vec::with_capacity(names.len());

                    for value in names {
                        let Some(name) = value.as_str() else {
                            return CommandResult::Error(
                                "Menu name in open_menu() array wasn't a string".to_string(),
                            );
                        };

                        result.push(name.to_string());
                    }

                    result
                } else {
                    return CommandResult::Error(
                        "Menu name in open_menu() wasn't resolved to string or array of strings"
                            .to_string(),
                    );
                };

                let mut errors = Vec::new();
                //println!("Opening: {:?}", menu_names);
                for menu_name in &menu_names {
                    if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                        menu.active = true;
                        // for layer in menu.layers.iter_mut() {
                        //     layer.active = true;
                        //     layer.activate_all_elements();
                        //     layer.dirty.mark_all();
                        // }
                    } else {
                        errors.push(format!("Menu '{}' not found", menu_name));
                    }
                }

                if errors.is_empty() {
                    CommandResult::Ok
                } else {
                    CommandResult::Error(errors.join("\n"))
                }
            }

            UiCommand::CloseMenu { menu_name } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu_name = string_to_value(ctx, menu_name);

                let menu_names: Vec<String> = if let Some(name) = menu_name.as_str() {
                    vec![name.to_string()]
                } else if let Some(names) = menu_name.as_array() {
                    let mut result = Vec::with_capacity(names.len());

                    for value in names {
                        let Some(name) = value.as_str() else {
                            return CommandResult::Error(
                                "Menu name in close_menu() array wasn't a string".to_string(),
                            );
                        };

                        result.push(name.to_string());
                    }

                    result
                } else {
                    return CommandResult::Error(
                        "Menu name in close_menu() wasn't resolved to string or array of strings"
                            .to_string(),
                    );
                };

                let mut errors = Vec::new();

                for menu_name in &menu_names {
                    if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                        menu.active = false;
                    } else {
                        errors.push(format!("Menu '{}' not found", menu_name));
                    }
                }

                if errors.is_empty() {
                    CommandResult::Ok
                } else {
                    CommandResult::Error(errors.join("\n"))
                }
            }
            UiCommand::CloseAllMenus => {
                //println!("Closing all menus");
                for (_, menu) in ctx.ui.menus.iter_mut() {
                    menu.active = false;
                }
                CommandResult::Ok
            }
            UiCommand::CloseAllLayers { menu_name } => {
                //println!("Hi, te menu name is: {}", menu_name);
                let element_ctx = &ctx.element_ctx.clone();
                let menu_name = string_to_value(ctx, menu_name);
                let Some(menu_name) = menu_name.as_str() else {
                    return CommandResult::Error(
                        "Menu name in close_all_layers() wasn't resolved to string".to_string(),
                    );
                };
                if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                    for layer in menu.layers.iter_mut() {
                        layer.active = false;
                    }
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!(
                        "Menu '{}' not found for close_all_layers()",
                        menu_name
                    ))
                }
            }

            UiCommand::ToggleMenu { menu_name } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu_name = string_to_value(ctx, menu_name);
                let Some(menu_name) = menu_name.as_str() else {
                    return CommandResult::Error(
                        "Menu name in toggle_menu() wasn't resolved to string".to_string(),
                    );
                };
                if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                    menu.active = !menu.active;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::MenuActive { menu_name } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu_names = string_to_value(ctx, menu_name);
                if let Some(menu_name) = menu_names.as_str() {
                    let is_active = ctx
                        .ui
                        .menus
                        .get(menu_name)
                        .map(|m| m.active)
                        .unwrap_or(false);
                    ctx.ui.variables.set_bool("_result", is_active);
                    CommandResult::Ok
                } else if let Some(menu_names) = menu_names.as_array() {
                    let mut is_active = false;
                    for menu_name in menu_names {
                        let Some(menu_name) = menu_name.as_str() else {
                            return CommandResult::Error(
                                "Menu name in menu_active() in array wasn't resolved to string"
                                    .to_string(),
                            );
                        };
                        is_active |= ctx
                            .ui
                            .menus
                            .get(menu_name)
                            .map(|m| m.active)
                            .unwrap_or(false);
                    }
                    ctx.ui.variables.set_bool("_result", is_active);
                    CommandResult::Ok
                } else {
                    CommandResult::Error(
                        "Menu name in menu_active() wasn't resolved to string".to_string(),
                    )
                }
            }
            UiCommand::LayerActive { menu_name, layer_name } => {
                let menu_names = string_to_value(ctx, menu_name);
                let layer_names = string_to_value(ctx, layer_name);

                let menu_names: Vec<String> = if let Some(name) = menu_names.as_str() {
                    vec![name.to_string()]
                } else if let Some(names) = menu_names.as_array() {
                    let mut result = Vec::with_capacity(names.len());
                    for name in names {
                        let Some(name) = name.as_str() else {
                            return CommandResult::Error(
                                "Menu name in layer_active() array wasn't resolved to string".to_string(),
                            );
                        };
                        result.push(name.to_string());
                    }
                    result
                } else {
                    return CommandResult::Error(
                        "Menu name in layer_active() wasn't resolved to string".to_string(),
                    );
                };

                let layer_names: Vec<String> = if let Some(name) = layer_names.as_str() {
                    vec![name.to_string()]
                } else if let Some(names) = layer_names.as_array() {
                    let mut result = Vec::with_capacity(names.len());
                    for name in names {
                        let Some(name) = name.as_str() else {
                            return CommandResult::Error(
                                "Layer name in layer_active() array wasn't resolved to string".to_string(),
                            );
                        };
                        result.push(name.to_string());
                    }
                    result
                } else {
                    return CommandResult::Error(
                        "Layer name in layer_active() wasn't resolved to string".to_string(),
                    );
                };

                let is_active = menu_names.iter().any(|menu_name| {
                    ctx.ui.menus.get(menu_name).is_some_and(|menu| {
                        layer_names.iter().any(|layer_name| {
                            menu.layers
                                .iter()
                                .any(|layer| layer.name == *layer_name && layer.active)
                        })
                    })
                });

                ctx.ui.variables.set_bool("_result", is_active);
                CommandResult::Ok
            }
            UiCommand::UnDirty { menu_name, layer_name } => {
                let menu_names = string_to_value(ctx, menu_name);
                let layer_names = string_to_value(ctx, layer_name);

                let menu_names = if let Some(name) = menu_names.as_str() {
                    name.to_string()
                } else {
                    return CommandResult::Error(
                        "Menu name in undirty() wasn't resolved to string".to_string(),
                    );
                };

                let layer_names: Vec<String> = if let Some(name) = layer_names.as_str() {
                    vec![name.to_string()]
                } else if let Some(names) = layer_names.as_array() {
                    let mut result = Vec::with_capacity(names.len());

                    for name in names {
                        let Some(name) = name.as_str() else {
                            return CommandResult::Error(
                                "Layer name in undirty() array wasn't resolved to string".to_string(),
                            );
                        };

                        result.push(name.to_string());
                    }

                    result
                } else {
                    return CommandResult::Error(
                        "Layer name in undirty() wasn't resolved to string".to_string(),
                    );
                };

                match ctx.renderer.ui_renderer.rebuild_layers_now(
                    ctx.ui,
                    &menu_names,
                    &layer_names,
                    &ctx.world.time,
                    &ctx.renderer.device,
                    &ctx.renderer.queue,
                    ctx.window_size,
                    ctx.settings,
                    &ctx.game_state.mod_manager,
                ) {
                    Ok(()) => CommandResult::Ok,
                    Err(error) => CommandResult::Error(error),
                }
            }
            // ===== LAYER COMMANDS =====
            UiCommand::OpenLayer {
                menu_name,
                layer_name,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu_name = string_to_value(ctx, menu_name);
                let Some(menu_name) = menu_name.as_str() else {
                    return CommandResult::Error(
                        "Menu name in open_layer() wasn't resolved to string".to_string(),
                    );
                };
                let layer_name = string_to_value(ctx, layer_name);
                let Some(layer_name) = layer_name.as_str() else {
                    return CommandResult::Error(
                        "Layer name in open_layer() wasn't resolved to string".to_string(),
                    );
                };
                if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                    let mut aps_to_activate = vec![];
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        menu.active = true;
                        layer.active = true;
                        aps_to_activate = layer.iter_aps().map(|ap| ap.id.clone()).collect();
                    } else {
                        return CommandResult::Error(format!(
                            "Layer '{}' not found in '{}'",
                            layer_name, menu_name
                        ));
                    }
                    for id in aps_to_activate {
                        menu.layers
                            .iter_mut()
                            .filter(|l| l.name == id)
                            .for_each(|l| l.active = true);
                    }
                    return CommandResult::Ok;
                }
                CommandResult::Error(format!("Menu '{}' not found", menu_name))
            }

            UiCommand::CloseLayer {
                menu_name,
                layer_name,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu_name = string_to_value(ctx, menu_name);
                let Some(menu_name) = menu_name.as_str() else {
                    return CommandResult::Error(
                        "Menu name in close_layer() wasn't resolved to string".to_string(),
                    );
                };
                let layer_name = string_to_value(ctx, layer_name);
                let Some(layer_name) = layer_name.as_str() else {
                    return CommandResult::Error(
                        "Layer name in close_layer() wasn't resolved to string".to_string(),
                    );
                };
                if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = false;
                        return CommandResult::Ok;
                    }
                    return CommandResult::Error(format!("Layer '{}' not found", layer_name));
                }
                CommandResult::Error(format!("Menu '{}' not found", menu_name))
            }

            UiCommand::ToggleLayer {
                menu_name,
                layer_name,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu_name = string_to_value(ctx, menu_name);
                let Some(menu_name) = menu_name.as_str() else {
                    return CommandResult::Error(
                        "Menu name in toggle_layer() wasn't resolved to string".to_string(),
                    );
                };
                let layer_name = string_to_value(ctx, layer_name);
                let Some(layer_name) = layer_name.as_str() else {
                    return CommandResult::Error(
                        "Layer name in toggle_layer() wasn't resolved to string".to_string(),
                    );
                };
                if let Some(menu) = ctx.ui.menus.get_mut(menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = !layer.active;
                        return CommandResult::Ok;
                    }
                    return CommandResult::Error(format!("Layer '{}' not found", layer_name));
                }
                CommandResult::Error(format!("Menu '{}' not found", menu_name))
            }

            UiCommand::SetVar { name, value } => {
                let element_ctx = &ctx.element_ctx.clone();
                let initial_name = name.clone();
                let name = string_to_value(ctx, name);
                //println!("After string to value: {}", name);
                let Some(name) = name.as_str() else {
                    return CommandResult::Error(format!(
                        "'{initial_name}' in set_var() wasn't resolved to string"
                    ));
                };
                //println!("Pre to-value: {}", value);
                let value = string_to_value(ctx, value);
                //println!("Post to-value: {}", value);

                let (field_type, name) = match name.split_once(':') {
                    Some((field_type, base)) => (field_type, base),
                    None => ("None", name),
                };
                let value = initialize_value(field_type, Some(value));
                //println!("Post init-value: {}", value);
                if let Some(key) = get_setting_key(&name) {
                    //println!("In setvar setting key: {:?}", key);
                    if let Some(setting_value) = key.from_value(&value) {
                        //println!("Value: {:?}", setting_value);
                        ctx.settings
                            .apply_setting(key, SettingOp::Set(setting_value));

                        return CommandResult::Ok;
                    }

                    return CommandResult::Error(format!(
                        "Cannot convert {:?} for setting '{}'",
                        value, name
                    ));
                }

                set_variable_or_property(ctx, &name, value)
            }

            UiCommand::IncVar { name, amount } => {
                let element_ctx = &ctx.element_ctx.clone();
                let name = string_to_value(ctx, name);
                let Some(name) = name.as_str() else {
                    return CommandResult::Error("Name in inc() wasn't resolved to string, use 'str' or 'strexpr:' or do something else".to_string());
                };
                let val = string_to_value(ctx, amount);
                if let Some(amount) = val.as_f64() {
                    if let Some(key) = get_setting_key(&name) {
                        let current = ctx.settings.read_setting(key);

                        if let Some(new_value) = current.add(amount) {
                            ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                        } else {
                            let steps = amount as isize;

                            match steps.cmp(&0) {
                                Ordering::Greater => {
                                    for _ in 0..steps {
                                        ctx.settings.apply_setting(key, SettingOp::CycleNext);
                                    }
                                }
                                Ordering::Less => {
                                    for _ in 0..steps.unsigned_abs() {
                                        ctx.settings.apply_setting(key, SettingOp::CyclePrev);
                                    }
                                }
                                Ordering::Equal => {}
                            }
                        }

                        return CommandResult::Ok;
                    }
                }

                let var_val =
                    Value::load_variable(&ctx.ui.variables, &ctx.ui.menus, element_ctx, name);
                //println!("{:?} {:?}", var_val, name);
                let new_val = match var_val {
                    Some(Value::Array(mut vec)) => {
                        vec.push(val);
                        Value::Array(vec)
                    }
                    _ => {
                        if var_val.is_none() {
                            Value::Array(vec![val])
                        } else {
                            let Some(amount) = val.as_f64() else {
                                return CommandResult::Error(
                                    "Value in inc() wasn't resolved to f64".to_string(),
                                );
                            };
                            match var_val {
                                Some(Value::F64(f)) => Value::F64(f + amount),
                                Some(Value::I64(i)) => Value::F64(i as f64 + amount),
                                _ => Value::F64(amount),
                            }
                        }
                    }
                };

                set_variable_or_property(ctx, &name, new_val)
            }

            UiCommand::DecVar { name, amount } => {
                let element_ctx = &ctx.element_ctx.clone();
                let name = string_to_value(ctx, name);
                let Some(name) = name.as_str() else {
                    return CommandResult::Error(
                        "Name in dec_var() wasn't resolved to string".to_string(),
                    );
                };
                let Some(amount) = string_to_value(ctx, amount).as_f64() else {
                    return CommandResult::Error(
                        "Value in dec_var() wasn't resolved to f64".to_string(),
                    );
                };
                if let Some(key) = get_setting_key(&name) {
                    let current = ctx.settings.read_setting(key);

                    if let Some(new_value) = current.subtract(amount) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    } else {
                        let steps = amount as isize;

                        match steps.cmp(&0) {
                            Ordering::Greater => {
                                for _ in 0..steps {
                                    ctx.settings.apply_setting(key, SettingOp::CycleNext);
                                }
                            }
                            Ordering::Less => {
                                for _ in 0..steps.unsigned_abs() {
                                    ctx.settings.apply_setting(key, SettingOp::CyclePrev);
                                }
                            }
                            Ordering::Equal => {}
                        }
                    }

                    return CommandResult::Ok;
                }

                let new_val =
                    match Value::load_variable(&ctx.ui.variables, &ctx.ui.menus, element_ctx, name)
                    {
                        Some(Value::F64(f)) => Value::F64(f - amount),
                        Some(Value::I64(i)) => Value::F64(i as f64 - amount),
                        _ => Value::F64(-amount),
                    };

                set_variable_or_property(ctx, &name, new_val)
            }

            UiCommand::MulVar { name, factor } => {
                let element_ctx = &ctx.element_ctx.clone();
                let name = string_to_value(ctx, name);
                let Some(name) = name.as_str() else {
                    return CommandResult::Error(
                        "Name in mul_var() wasn't resolved to string".to_string(),
                    );
                };
                let Some(factor) = string_to_value(ctx, factor).as_f64() else {
                    return CommandResult::Error(
                        "Factor in mul_var() wasn't resolved to f64".to_string(),
                    );
                };
                if let Some(key) = get_setting_key(&name) {
                    let current = ctx.settings.read_setting(key);

                    if let Some(new_value) = current.multiply(factor) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    }

                    return CommandResult::Ok;
                }

                let new_val =
                    match Value::load_variable(&ctx.ui.variables, &ctx.ui.menus, element_ctx, name)
                    {
                        Some(Value::F64(f)) => Value::F64(f * factor),
                        Some(Value::I64(i)) => Value::F64(i as f64 * factor),
                        _ => Value::F64(factor),
                    };

                set_variable_or_property(ctx, &name, new_val)
            }

            UiCommand::ToggleVar { name } => {
                let element_ctx = &ctx.element_ctx.clone();
                let name = string_to_value(ctx, name);
                let Some(name) = name.as_str() else {
                    return CommandResult::Error(
                        "Name in toggle_var() wasn't resolved to string".to_string(),
                    );
                };

                if let Some(key) = get_setting_key(&name) {
                    //println!("{} {:?}", name, key);
                    ctx.settings.apply_setting(key, SettingOp::Toggle);
                    //println!("{:?}", ctx.settings.read_setting(key));
                    return CommandResult::Ok;
                }
                //println!("{} {:?}", name, ctx.ui.variables.get(&name));
                let new_val =
                    match Value::load_variable(&ctx.ui.variables, &ctx.ui.menus, element_ctx, name)
                    {
                        Some(Value::Bool(b)) => Value::Bool(!b),
                        Some(Value::None) => Value::Bool(true),
                        Some(Value::I64(i)) => {
                            if i == 0 {
                                Value::I64(1)
                            } else if i == 1 {
                                Value::I64(0)
                            } else {
                                return CommandResult::Ok;
                            }
                        }
                        Some(Value::String(str)) => {
                            if str == "false" {
                                Value::Bool(true)
                            } else if str == "true" {
                                Value::Bool(false)
                            } else {
                                return CommandResult::Ok;
                            }
                        }
                        None => Value::Bool(true),
                        _ => return CommandResult::Ok,
                    };

                set_variable_or_property(ctx, &name, new_val)
            }

            UiCommand::Clamp { name, min, max } => {
                let element_ctx = &ctx.element_ctx.clone();
                let name = string_to_value(ctx, name);
                let Some(name) = name.as_str() else {
                    return CommandResult::Error(
                        "Name in clamp() wasn't resolved to string".to_string(),
                    );
                };
                let Some(min) = string_to_value(ctx, min).as_f64() else {
                    return CommandResult::Error(
                        "Min in clamp() wasn't resolved to f64".to_string(),
                    );
                };
                let Some(max) = string_to_value(ctx, max).as_f64() else {
                    return CommandResult::Error(
                        "Max in clamp() wasn't resolved to f64".to_string(),
                    );
                };
                if let Some(key) = get_setting_key(&name) {
                    let current = ctx.settings.read_setting(key);

                    if let Some(new_value) = current.clamp_range(min, max) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    }

                    return CommandResult::Ok;
                }

                let new_val =
                    match Value::load_variable(&ctx.ui.variables, &ctx.ui.menus, element_ctx, name)
                    {
                        Some(Value::F64(f)) => Value::F64(f.clamp(min, max)),
                        Some(Value::I64(i)) => Value::F64((i as f64).clamp(min, max)),
                        _ => Value::F64(min),
                    };

                set_variable_or_property(ctx, &name, new_val)
            }

            UiCommand::Delay { seconds } => {
                let element_ctx = &ctx.element_ctx.clone();
                let Some(seconds) = string_to_value(ctx, seconds).as_f64() else {
                    return CommandResult::Error(
                        "Seconds in delay() wasn't resolved to f64".to_string(),
                    );
                };
                let remaining: Vec<UiCommand> = self.queue.drain(..).collect();
                CommandResult::Delay { seconds, remaining }
            }

            UiCommand::Halt => CommandResult::Stop,

            UiCommand::Skip { count } => CommandResult::Skip(count),

            UiCommand::If {
                condition,
                then,
                else_branch,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                //println!("{} {:?} {:?}", condition, then, else_branch);
                //println!("Before: {}", condition);
                let condition = string_to_value(ctx, condition);
                //println!("After: {}", condition);
                if condition.is_truthy() {
                    self.execute_multiple(then, ctx);
                } else {
                    self.execute_multiple(else_branch, ctx);
                }
                CommandResult::Ok
            }

            UiCommand::IfVarEq {
                var_name,
                value,
                then,
                else_branch,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                //println!("{} {}", var_name, value);
                //println!("{:?} {:?}", then, else_branch);
                let var_value = string_to_value(ctx, var_name);
                //let Some(var_value) = var_name.as_string() else { return CommandResult::Error(format!("Var Name in ifvareq() wasn't resolved to string, instead to: {}", var_name)) };
                //println!("{} {}", var_name, value);
                // let var_value = if let Some(key) = get_setting_key(&var_name) {
                //     Some(ctx.settings.read_setting(key).to_value())
                // } else {
                //     ctx.ui.variables.get(&var_name).map(|v| v.into_owned())
                // };

                let compare_value = string_to_value(ctx, value);
                //println!("{:?} {:?}", var_value, compare_value);
                if var_value == compare_value {
                    //println!("{:#?}", then);
                    self.execute_multiple(then, ctx);
                } else {
                    self.execute_multiple(else_branch, ctx);
                }
                CommandResult::Ok
            }

            UiCommand::For { value, commands } => {
                let element_ctx = &ctx.element_ctx.clone();
                let val = string_to_value(ctx, value);
                match val {
                    Value::None => {}
                    Value::F64(n) => {
                        let vec = (0..n as i64).map(|i| Value::F64(i as f64)).collect();
                        self.execute_multiple_for(vec, commands, ctx);
                    }
                    Value::I64(n) => {
                        let vec = (0..n).map(|i| Value::I64(i)).collect();
                        self.execute_multiple_for(vec, commands, ctx);
                    }
                    Value::Bool(_) => {}
                    Value::String(s) => {
                        self.execute_multiple_for(
                            s.chars()
                                .into_iter()
                                .map(|char| Value::String(char.to_string()))
                                .collect::<Vec<Value>>(),
                            commands,
                            ctx,
                        );
                    }
                    Value::Array(arr) => {
                        self.execute_multiple_for(arr, commands, ctx);
                    }
                }

                CommandResult::Ok
            }
            UiCommand::AddElement {
                menu,
                layer,
                id,
                kind,
                center,
                actions,
                undoable,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let kind = string_to_value(ctx, kind).into_string();
                let kind = ElementKind::from_string(kind.to_string().as_str());
                if kind == ElementKind::None {
                    return CommandResult::Error(
                        "Element Kind is None in AddElement kind argument".to_string(),
                    );
                }
                let center = string_to_value(ctx, center);
                let Some(center) = center.as_pos() else {
                    return CommandResult::Error(
                        "Couldn't unpack center pos from AddElement center argument".to_string(),
                    );
                };
                let menu = string_to_value(ctx, menu).into_string();
                let layer = string_to_value(ctx, layer).into_string();
                let id = string_to_value(ctx, id).into_string();
                let Some(element) = make_element(id.to_string(), kind, center) else {
                    return CommandResult::Error("Couldn't make element in AddElement".to_string());
                };
                ctx.ui.ui_edit_manager.execute_command(
                    CreateElementCommand {
                        affected_element: ElementRef::new(
                            menu.as_str(),
                            layer.as_str(),
                            id.as_str(),
                            kind,
                        ),
                        element,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                CommandResult::Ok
            }
            UiCommand::AddAP {
                menu,
                name,
                ap_name,
                ap_var,
                center,
                scale,
                is_temporary,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let center = string_to_value(ctx, center);
                let Some(center) = center.as_pos() else {
                    return CommandResult::Error(format!(
                        "Couldn't unpack center pos from AddAP center argument, parsed to: {}",
                        center
                    ));
                };
                let scale = string_to_value(ctx, scale);
                let Some(scale) = scale.as_f64() else {
                    return CommandResult::Error(
                        "Couldn't unpack scale from AddAP scale argument".to_string(),
                    );
                };
                let menu = string_to_value(ctx, menu).into_string();
                let name = string_to_value(ctx, name).into_string();
                let ap_name = string_to_value(ctx, ap_name).into_string();
                //println!("[AddAP before string_to_value()] << {} >>", ap_var);
                let ap_vars = string_to_value(ctx, ap_var);

                let Some(ap_vars) = ap_vars
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.to_literal()).collect())
                    .or(ap_vars.as_str().map(|str| vec![str.to_string()]))
                else {
                    return CommandResult::Error(
                        "AP Vars in AddAP was not an array or single argument".to_string(),
                    );
                };
                //println!("[AddAP after string_to_value()] << {:?} >>", ap_vars);
                //println!("Adding AP: {} {} {} {:?} {}", name, ap_name, ap_var, center, scale);
                let ap = {
                    let mut ap = AdvancedPrimitive::default();
                    ap.id = name.clone();
                    ap.set_pos(center);
                    ap.ap_name = ap_name;
                    ap.ap_vars = ap_vars;
                    ap.scale = scale as f32;
                    ap.discard_reference = is_temporary;
                    ap.scale_my_coords = false;
                    ap
                };
                let Some(layer) = element_ctx
                    .self_element
                    .as_ref()
                    .map(|e| e.layer.clone())
                    .or_else(|| element_ctx.as_element.as_ref().map(|e| e.layer.clone()))
                else {
                    return CommandResult::Ok;
                };
                ctx.ui.ui_edit_manager.execute_command(
                    CreateAPCommand {
                        menu,
                        layer,
                        element: UiElement::Advanced(ap),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                CommandResult::Ok
            }
            UiCommand::DeleteAP {
                menu,
                layer,
                reference_id,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu = string_to_value(ctx, menu).into_string();
                let layer = string_to_value(ctx, layer).into_string();
                let reference_id = string_to_value(ctx, reference_id).into_string();
                ctx.ui.ui_edit_manager.execute_command(
                    DeleteAPCommand {
                        menu,
                        layer,
                        reference_id,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                CommandResult::Ok
            }
            UiCommand::DeleteAPsInLayer {
                menu,
                layer,
            } => {
                let menu = string_to_value(ctx, menu).into_string();
                let layer = string_to_value(ctx, layer).into_string();
                if let Some(menu) = ctx.ui.menus.get_mut(&menu) {
                    delete_aps_in_layer(menu, &layer);
                } else {
                    return CommandResult::Error(format!("Menu '{}' not found in DeleteAPsInLayer!", menu));
                }
                CommandResult::Ok
            }
            UiCommand::CloneElement {
                from_menu,
                from_layer,
                from_id,
                to_menu,
                to_layer,
                to_id,
                center,
                actions,
                undoable,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let from_menu = string_to_value(ctx, from_menu).into_string();
                let from_layer = string_to_value(ctx, from_layer).into_string();
                let from_id = string_to_value(ctx, from_id).into_string();
                let from_element = ElementRef::new(
                    from_menu.as_str(),
                    from_layer.as_str(),
                    from_id.as_str(),
                    ElementKind::None,
                );
                let to_menu = string_to_value(ctx, to_menu).into_string();
                let to_layer = string_to_value(ctx, to_layer).into_string();
                let to_id = string_to_value(ctx, to_id).into_string();
                if undoable {
                    let to_element = ElementRef::new(
                        to_menu.as_str(),
                        to_layer.as_str(),
                        to_id.as_str(),
                        ElementKind::None,
                    );
                    let center = string_to_value(ctx, center);
                    let actions = string_to_value(ctx, actions);
                    ctx.ui.ui_edit_manager.execute_command(
                        DuplicateElementCommand {
                            from_element,
                            to_element,
                            cached_element: None,
                            optional_center: center.as_pos(),
                            optional_actions: actions.as_array().map(|a| {
                                a.iter()
                                    .flat_map(|v| v.as_str())
                                    .map(|str| str.to_string())
                                    .collect::<Vec<String>>()
                            }),
                        },
                        &mut ctx.ui.touch_manager,
                        &mut ctx.ui.menus,
                        &mut ctx.ui.variables,
                        &ctx.world.input.mouse,
                    );
                    CommandResult::Ok
                } else {
                    let Some(mut element) = get_element(&ctx.ui.menus, &from_element) else {
                        return CommandResult::Error(
                            "Couldn't unpack element in CloneElement".to_string(),
                        );
                    };
                    let to_element = ElementRef::new(
                        to_menu.as_str(),
                        to_layer.as_str(),
                        to_id.as_str(),
                        element.kind(),
                    );

                    element.set_id(&to_id.to_string());
                    let center = string_to_value(ctx, center);
                    let actions = string_to_value(ctx, actions);
                    if let Some(center) = center.as_pos() {
                        element.set_pos(center[0], center[1])
                    };
                    if let Some(actions) = actions.as_array().map(|a| {
                        a.iter()
                            .flat_map(|v| v.as_str())
                            .map(|str| str.to_string())
                            .collect::<Vec<String>>()
                    }) {
                        element.set_actions(actions);
                    }
                    let result = create_element(
                        &mut ctx.ui.menus,
                        &to_element.menu,
                        &to_element.layer,
                        element,
                        &ctx.world.input.mouse,
                    );
                    match result {
                        Ok(ok) => CommandResult::Ok,
                        Err(err) => CommandResult::Error(err.to_string()),
                    }
                }
            }

            UiCommand::CloneLayer {
                from_menu,
                from_layer,
                to_menu,
                to_layer,
                undoable,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let from_menu = string_to_value(ctx, from_menu).into_string();
                let from_layer = string_to_value(ctx, from_layer).into_string();
                let to_menu = string_to_value(ctx, to_menu).into_string();
                let to_layer = string_to_value(ctx, to_layer).into_string();
                let Some(mut layer) = (if let Some(menu) = ctx.ui.menus.get(&from_menu) {
                    if let Some(layer) = menu.layers.iter().find(|l| l.name == from_layer) {
                        Some(layer.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }) else {
                    return CommandResult::Error(
                        "Couldn't find from_layer or from_menu".to_string(),
                    );
                };

                layer.name = to_layer;

                if let Some(menu) = ctx.ui.menus.get_mut(&to_menu) {
                    menu.layers.push(layer);
                    menu.sort_layers();
                    CommandResult::Ok
                } else {
                    CommandResult::Error("Couldn't target Menu".to_string())
                }
            }
            UiCommand::DeleteLayer {
                menu,
                layer,
                undoable,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu = string_to_value(ctx, menu).into_string();
                let layer = string_to_value(ctx, layer).into_string();
                if let Some(menu) = ctx.ui.menus.get_mut(&menu) {
                    if let Some(idx) = menu.layers.iter().position(|l| l.name == layer) {
                        menu.layers.remove(idx);
                        menu.sort_layers();
                    } else {
                        return CommandResult::AnnoyingError(
                            "Couldn't find layer in menu to delete...".to_string(),
                        );
                    }
                    CommandResult::Ok
                } else {
                    CommandResult::AnnoyingError("Couldn't find Menu to delete...".to_string())
                }
            }
            UiCommand::DeleteElement {
                menu,
                layer,
                id,
                undoable,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let menu = string_to_value(ctx, menu).into_string();
                let layer = string_to_value(ctx, layer).into_string();
                let id = string_to_value(ctx, id).into_string();
                let element = ElementRef::new(
                    menu.as_str(),
                    layer.as_str(),
                    id.as_str(),
                    ElementKind::None,
                );
                //println!("Deleting element: {:?}", element);
                if undoable {
                    ctx.ui.ui_edit_manager.execute_command(
                        DeleteElementCommand {
                            affected_element: element,
                            cached_element: None,
                        },
                        &mut ctx.ui.touch_manager,
                        &mut ctx.ui.menus,
                        &mut ctx.ui.variables,
                        &ctx.world.input.mouse,
                    );
                    CommandResult::Ok
                } else {
                    let result = delete_element(&mut ctx.ui.menus, &element);
                    CommandResult::Ok
                    // match result {
                    //     Ok(ok) => CommandResult::Ok,
                    //     Err(err) => CommandResult::Error(err.to_string()),
                    // }
                }
            }

            UiCommand::SaveGame { and_exit } => {
                let and_exit = string_to_value(ctx, and_exit).is_truthy();
                save_game(
                    ctx.game_state,
                    ctx.world,
                    ctx.settings,
                    and_exit,
                );
                CommandResult::Ok
            }
            UiCommand::LoadSave {
                save_name,
                without_saving,
            } => {
                let element_ctx = &ctx.element_ctx.clone();
                let save_name = string_to_value(ctx, save_name);
                let Some(save_name) = save_name.as_str() else {
                    if save_name.is_none() {
                        return CommandResult::Ok;
                    }
                    return CommandResult::Error(format!(
                        "Save name '{}' in load_save() wasn't resolved to string",
                        save_name
                    ));
                };
                if !without_saving {
                    save_game(
                        ctx.game_state,
                        ctx.world,
                        ctx.settings,
                        false,
                    );
                }
                load_save(
                    ctx.game_state,
                    ctx.world,
                    ctx.settings,
                    save_name,
                );
                CommandResult::Ok
            }
            UiCommand::ExitGame => {
                exit_game(
                    ctx.game_state,
                    ctx.settings,
                    &mut ctx.ui.variables,
                    ctx.world,
                    ctx.event_loop,
                );
                CommandResult::Ok
            }
            UiCommand::Print { statement } => {
                let element_ctx = &ctx.element_ctx.clone();
                //println!("printing: {}", statement);
                let msg: String = string_to_value(ctx, statement).to_string();
                ctx.ui
                    .variables
                    .push_array("CHAT", Value::Array(vec![Value::String(msg.clone()), Value::F64(ctx.world.time.total_time)]));
                println!("[UI Chat] {}", msg);
                CommandResult::Ok
            }
            UiCommand::PrintNoChat { statement } => {
                let element_ctx = &ctx.element_ctx.clone();
                //println!("printing: {}", statement);
                let msg: String = string_to_value(ctx, statement).to_string();
                println!("[UI PrintNC] {}", msg);
                CommandResult::Ok
            }

            UiCommand::DebugVars => {
                println!("[Debug] Variables: {:#?}", ctx.ui.variables.dump());
                CommandResult::Ok
            }

            UiCommand::DebugMenus => {
                for (name, menu) in &ctx.ui.menus {
                    println!("[Debug] Menu '{}': active={}", name, menu.active);
                    for layer in &menu.layers {
                        println!("  Layer '{}': active={}", layer.name, layer.active);
                        for elem in &layer.elements {
                            println!("    Elem '{}': active={}", elem.id(), elem.is_active());
                        }
                    }
                }
                CommandResult::Ok
            }

            UiCommand::Call {
                function_name,
                args,
            } => {
                fn parse_args_call(ctx: &mut CommandContext, args: Option<String>) -> Vec<Value> {
                    if let Some(args) = args {
                        let args = string_to_value(ctx, args);
                        args.as_array().unwrap_or_else(|| vec![args])
                    } else {
                        vec![]
                    }
                }
                fn parse_action_call(
                    cq: &mut CommandQueue,
                    ctx: &mut CommandContext,
                    function_name: String,
                    args: Option<String>,
                ) -> CommandResult {
                    let element_ctx = &ctx.element_ctx.clone();
                    //println!("[Action Call before] Name: {}, Args: {:?}", function_name, args);
                    let actions = string_to_value(ctx, function_name);
                    //println!("[Action Call after] {:?}", actions);
                    let actions: Vec<String> = if let Some(action) = actions.as_str() {
                        vec![action.to_string()]
                    } else if let Some(actions) = actions.as_array() {
                        let mut result = Vec::with_capacity(actions.len());

                        for a in actions {
                            let Some(action) = a.as_str() else {
                                return CommandResult::Error(
                                    "In call(), action in array wasn't a string".to_string(),
                                );
                            };

                            result.push(action.to_string());
                        }

                        result
                    } else {
                        return CommandResult::Error(
                            "actions in call() weren't resolved to string or array of strings"
                                .to_string(),
                        );
                    };
                    if actions.is_empty() {
                        return CommandResult::Ok;
                    }

                    let args = parse_args_call(ctx, args);

                    ctx.ui.variables.set_array("args", args); // So I can use args.4 for example.
                    let compiled_actions = compile_actions(
                        &ctx.ui.menus,
                        ctx.element_ctx.self_element.clone(),
                        actions,
                    );
                    //println!("{:?} {:?}", compiled_actions, ctx.ui.action_events);
                    for compiled_action in compiled_actions.into_iter() {
                        run_action(cq, compiled_action, ctx);
                    }
                    CommandResult::Ok
                }

                fn parse_rust_call(
                    ctx: &mut CommandContext,
                    function_name: String,
                    args: Option<String>,
                ) -> CommandResult {
                    let function_name = string_to_value(ctx, function_name);
                    let Some(function_name) = function_name.as_str() else {
                        return CommandResult::Error(
                            "Rust function in call() wasn't resolved to string, use 'str:' as in 'call(rust:str:function_name, [args])'".to_string(),
                        );
                    };

                    let args = parse_args_call(ctx, args);
                    //println!("{:?} {:?}", function_name, args);
                    //ctx.ui.variables.set_array("args", args); // So I can use args.4 for example. not needed in rust... most likely...

                    call_rust(ctx, function_name, args);

                    CommandResult::Ok
                }

                if let Some((left, right)) = function_name.split_once(':') {
                    match left {
                        "rust" | "RUST" => {
                            //println!("{}", right);
                            parse_rust_call(ctx, right.to_string(), args)
                        }
                        _ => parse_action_call(self, ctx, function_name, args),
                    }
                } else {
                    parse_action_call(self, ctx, function_name, args)
                }
            }
            UiCommand::Noop => CommandResult::Ok,
        }
    }
}

fn call_rust(ctx: &mut CommandContext, function_name: &str, args: Vec<Value>) {
    match function_name {
        "queue_mini_render" => {
            let Some(pos_arg) = args.get(0) else {
                println!("[MiniRender] queue_mini_render(): Missing argument 0 'pos'. Expected [f64, f64, f64].");
                return;
            };

            let Some(pos_array) = pos_arg.as_array() else {
                println!("[MiniRender] queue_mini_render(): Argument 0 'pos' must be an Array [f64, f64, f64], got: {:?}.", pos_arg);
                return;
            };

            if pos_array.len() != 3 {
                println!("[MiniRender] queue_mini_render(): Argument 0 'pos' must contain exactly 3 values, got {}.", pos_array.len());
                return;
            }

            let Some(pos_x) = pos_array[0].as_f64() else {
                println!("[MiniRender] queue_mini_render(): Argument 0 'pos'[0] (X) must be an f64, got: {:?}.", pos_array[0]);
                return;
            };

            let Some(pos_y) = pos_array[1].as_f64() else {
                println!("[MiniRender] queue_mini_render(): Argument 0 'pos'[1] (Y height) must be an f64, got: {:?}.", pos_array[1]);
                return;
            };

            let Some(pos_z) = pos_array[2].as_f64() else {
                println!("[MiniRender] queue_mini_render(): Argument 0 'pos'[2] (Z) must be an f64, got: {:?}.", pos_array[2]);
                return;
            };
            let Some(dir_arg) = args.get(1) else {
                println!("[MiniRender] queue_mini_render(): Missing argument 1 'dir'. Expected [f64, f64, f64].");
                return;
            };

            let Some(dir_array) = dir_arg.as_array() else {
                println!("[MiniRender] queue_mini_render(): Argument 1 'dir' must be an Array [f64, f64, f64], got: {:?}.", dir_arg);
                return;
            };

            if dir_array.len() != 3 {
                println!("[MiniRender] queue_mini_render(): Argument 1 'dir' must contain exactly 3 values, got {}.", dir_array.len());
                return;
            }

            let mut rotation_values = [0.0; 3];

            for (i, value) in dir_array.iter().enumerate() {
                let Some(value) = value.as_f64() else {
                    println!("[MiniRender] queue_mini_render(): Argument 1 'rotation'[{}] must be an f64, got: {:?}.", i, value);
                    return;
                };

                rotation_values[i] = value as f32;
            }

            let rotation = Vec3::from_array(rotation_values);

            let Some(size_arg) = args.get(2) else {
                println!("[MiniRender] queue_mini_render(): Missing argument 2 'size'. Expected [f64, f64].");
                return;
            };

            let Some(size_array) = size_arg.as_array() else {
                println!("[MiniRender] queue_mini_render(): Argument 2 'size' must be an Array [f64, f64], got: {:?}.", size_arg);
                return;
            };

            if size_array.len() != 2 {
                println!("[MiniRender] queue_mini_render(): Argument 2 'size' must contain exactly 2 values, got {}.", size_array.len());
                return;
            }

            let Some(size_x) = size_array[0].as_f64() else {
                println!("[MiniRender] queue_mini_render(): Argument 2 'size'[0] must be an f64 or i64, got: {:?}.", size_array[0]);
                return;
            };

            let Some(size_y) = size_array[1].as_f64() else {
                println!("[MiniRender] queue_mini_render(): Argument 2 'size'[1] must be an f64 or i64, got: {:?}.", size_array[1]);
                return;
            };
            let Some(fov_arg) = args.get(3) else {
                println!("[MiniRender] queue_mini_render(): Missing argument 3 'fov'. Expected f64.");
                return;
            };
            let Some(fov) = fov_arg.as_f64() else {
                println!("[MiniRender] queue_mini_render(): Argument 3 'fov' must be an f64 or i64, got: {:?}.", fov_arg);
                return;
            };
            let Some(mw_arg) = args.get(4) else {
                println!("[MiniRender] queue_mini_render(): Missing argument 4 'mini_world_name'. Expected String.");
                return;
            };
            let Some(mwn) = mw_arg.as_string() else {
                println!("[MiniRender] queue_mini_render(): Argument 4 'mini_world_name' must be a String, got: {:?}.", mw_arg);
                return;
            };
            let Some(commands_arg) = args.get(5) else {
                println!("[MiniRender] queue_mini_render(): Missing argument 5 'commands'. Expected String.");
                return;
            };
            let commands = commands_arg.as_array().map(|a| a.into_iter().map(|s| s.to_string()).collect::<Vec<String>>()).unwrap_or_else(|| vec![commands_arg.to_string()]); // else {
            //     println!("[MiniRender] queue_mini_render(): Argument 5 'commands' must be a String or Array, got: {:?}.", commands_arg);
            //     return;
            // };
            let pos = WorldPos::from_dvec3(DVec3::new(pos_x, pos_y, pos_z));
            let size = PhysicalSize::new(size_x as u32, size_y as u32);
            println!("Queueing Mini Render with: Pos: '{}', Rotation: '{}', Size: '{:?}', FOV: '{}', Mini World Name: '{}, {} commands.'", pos, rotation, size, fov, mwn, commands.len());
            let mini_render = MiniRender::new(pos, rotation, size, fov as f32, mwn.clone(), commands);
            let id = ctx.mini_renderer.queue_render(mini_render);
            ctx.ui.variables.set_i64("_mini_render_id", id as i64);
        }
        "render_mini_render" => {
            let Some(id_arg) = args.get(0) else {
                println!("[MiniRender] render_mini_render(): Missing argument 0 'id'. Expected f64 or i64.");
                return;
            };
            let Some(id) = id_arg.as_i64() else {
                println!("[MiniRender] render_mini_render(): Argument 0 'id' must be an f64 or i64, got: {:?}.", id_arg);
                return;
            };

            ctx.mini_renderer.render_single_id(id as u64, &ctx.settings, &ctx.game_state.mod_manager, &ctx.world);
        }
        "get_saves" => {
            let saves: Vec<Value> = get_available_saves()
                .into_iter()
                .map(|save| Value::Array(save.to_values()))
                .collect();
            ctx.ui.variables.set_array("saves", saves);
        }
        "get_road_types" => {
            let rts: Value = ctx.world.roads.road_manager.road_types.to_value();
            ctx.ui.variables.set_var("road_types", rts);
        }
        "create_save" => {
            let save_name = args
                .get(0)
                .and_then(|s| s.as_str())
                .unwrap_or_else(|| {
                    println!("Argument 0 of create_save() must be a String, it's the save name.");
                    "No save name"
                })
                .to_string();
            let new_save_package = NewSavePackage {
                name: save_name.clone(),
                difficulty: args
                    .get(1)
                    .and_then(|s| s.as_str())
                    .unwrap_or_else(|| {
                        println!("Argument 1 of create_save() must be a String, it's the Difficulty.");
                        "Easy"
                    }).to_string(),
                preset_name: args
                    .get(2)
                    .and_then(|s| s.as_str())
                    .unwrap_or_else(|| {
                        println!("Argument 2 of create_save() must be a String, it's the Terrain Preset name.");
                        "Default"
                    }).to_string(),
            };
            save_game(
                ctx.game_state,
                ctx.world,
                ctx.settings,
                false,
            );
            let create_result = ctx.game_state.create_save(
                ctx.world,
                ctx.settings,
                new_save_package,
            );
            println!("Created Save '{}': {:?}", save_name, create_result);
        }
        "play_sound" => {
            let sound_name = args.get(0).and_then(|s| s.as_str());
            let Some(sound_name) = sound_name else {
                println!(
                    "[Sounds] Argument 0 of {function_name} must be a sound_name: String, but it was: '{:?}'.",
                    args.get(0)
                );
                return;
            };

            let succeeded = ctx.world.sounds.queue_sfx(sound_name.to_string());
            if !succeeded {
                println!(
                    "[Sounds] In the RUST:play_sound function in UI actions: Sound name '{sound_name}' couldn't be found in the sound folder. In element: {:?}",
                    ctx.element_ctx.self_element
                );
            }
        }
        "stop_sounds" => {
            ctx.world.sounds.clear_sfx_queue();
        }
        "get_new_splash" => {
            let splash_message = get_splash_message();
            ctx.ui
                .variables
                .set_string("SPLASH_MESSAGE", splash_message);
        }
        "start_panorama" => {
            ctx.world.start_panorama(&ctx.game_state);
        }
        "end_panorama" => {
            ctx.world.end_panorama();
        }
        // "convert_song" => {
        //     let path = args.get(0).and_then(|s| s.as_string());
        //     let Some(path) = path else {
        //         println!(
        //             "[Sounds] Argument 0 of {function_name} must be a path: String, but it was: '{:?}'.",
        //             args.get(0)
        //         );
        //         return;
        //     };
        //     let path = PathBuf::from(path);
        //     let yaml = song_to_sfx_yaml(
        //         path.as_path(),
        //         None,
        //         8
        //     );
        //     let (yaml, sfx) = match yaml {
        //         Ok(good) => {good}
        //         Err(err) => {
        //             println!(
        //                 "[Sounds] {function_name}: Couldn't convert song to yaml'{:?}': {:?}",
        //                 path, err
        //             ); return;
        //         }
        //     };
        //     let path = data_dir(format!("Rusty Skylines Mod/sounds/{}.yaml", sanitize(sfx.name)));
        //     if let Err(err) = std::fs::OpenOptions::new().write(true).create_new(true).open(path.as_path()).and_then(|mut file| file.write_all(yaml.as_bytes())) {
        //         println!(
        //             "[Sounds] {function_name}: Couldn't write to file '{:?}': '{:?}'",
        //             path, err
        //         );
        //     }
        // }
        // "place_building" => {
        //     let building_name = args.get(0).and_then(|s| s.as_string());
        //     let Some(building_name) = building_name else {
        //         println!(
        //             "[Buildings] Argument 0 of {function_name} must be a building_name: String, but it was: '{:?}'.",
        //             args.get(0)
        //         );
        //         return;
        //     };
        //     let Some(last_picked) = ctx.world.terrain.last_picked.as_ref() else {
        //         return;
        //     };
        //     // let center = last_picked.pos;
        //     // let half_size = 30.0;
        //     // let corners = [
        //     //     center.add_vec3(Vec3::new(-half_size, 0.0, -half_size)),
        //     //     center.add_vec3(Vec3::new(half_size, 0.0, -half_size)),
        //     //     center.add_vec3(Vec3::new(half_size, 0.0, half_size)),
        //     //     center.add_vec3(Vec3::new(-half_size, 0.0, half_size)),
        //     // ];
        //     // let entrance = LotEntrance {
        //     //     pos: center.add_vec3(Vec3::new(0.0, 0.0, -half_size)),
        //     //     dir: SerializableVec3::from_vec3(Vec3::new(1.0, 0.0, 0.0)),
        //     // };
        //     // let lot = Lot {
        //     //     id: 696,
        //     //     bounds: corners.to_vec(),
        //     //     center,
        //     //     entrance,
        //     //     layout: None,
        //     //     zoning_type: Some(ZoningType::Commercial),
        //     //     segment_id: Default::default(),
        //     //     district_id: 0,
        //     //     building_id: None,
        //     //     land_value: 0.0,
        //     // };
        //     // let lot_id = ctx.world.zoning.zoning_storage.spawn_lot(lot);
        //     // let building = Building {
        //     //     id: 0,
        //     //     pos: center,
        //     //     segment_id: Default::default(),
        //     //     lot_id,
        //     //     level: Default::default(),
        //     //     design_source: BuildingDesignSource::Design(building_name.to_string()),
        //     //     edit_id: None,
        //     //     prop_instance_ids: vec![],
        //     //     occupancy: Default::default(),
        //     // };
        //     // //println!("{:?}", building);
        //     // let building_id =
        //     //     BuildingStorage::spawn(&mut ctx.world.buildings, &mut ctx.world.zoning, building);
        //     // let lot = ctx.world.zoning.zoning_storage.get_mut_lot(lot_id).unwrap();
        //     // lot.building_id = Some(building_id);
        // }
        _ => {}
    }
}

pub fn string_to_value(ctx: &mut CommandContext, s: String) -> Value {
    let val = Value::from_str(
        ctx.settings,
        &ctx.ui.variables,
        &ctx.ui.menus,
        &ctx.element_ctx,
        s.as_str(),
        false,
    );
    //println!("String to value; Input: '{}' Output: '{}'", s, val);
    val
}
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ElementContext {
    pub self_element: Option<ElementRef>,
    pub as_element: Option<ElementRef>,
}
impl Display for ElementContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Self: {}, As: {}", if let Some(s) = self.self_element.as_ref() { format!("Some({s})") } else { "None".to_string() }, if let Some(a) = self.as_element.as_ref() { format!("Some({a})") } else { "None".to_string() })
    }
}
impl ElementContext {
    pub fn from_self(self_element: ElementRef) -> ElementContext {
        ElementContext {
            self_element: Some(self_element),
            as_element: None,
        }
    }
}
/// Canonicalize action names for legacy string conversion.
fn canonicalize_action_name(name: &str) -> String {
    let mut s = name.trim().replace(['-', ' '], "_");

    if !s.contains('_') && s.chars().any(|c| c.is_ascii_uppercase()) {
        let mut out = String::with_capacity(s.len() + 8);
        for (i, ch) in s.chars().enumerate() {
            if ch.is_ascii_uppercase() {
                if i != 0 {
                    out.push('_');
                }
                out.push(ch.to_ascii_lowercase());
            } else {
                out.push(ch.to_ascii_lowercase());
            }
        }
        s = out;
    } else {
        s = s.to_lowercase();
    }

    while s.contains("__") {
        s = s.replace("__", "_");
    }

    s
}

pub fn style_to_u32(style: &str) -> u32 {
    match style {
        "Hue Circle" | "1" => 1,
        _ => 0,
    }
}

/// Process commands and continuous actions. Call once per frame.
pub fn process_commands(
    command_queue: &mut CommandQueue,
    ui: &mut Ui,
    world: &mut World,
    renderer: &mut Renderer,
    window_size: PhysicalSize<f32>,
    settings: &mut Settings,
    event_loop: &dyn ActiveEventLoop,
    game_state: &mut GameState,
    simulation: &mut Simulation,
    mini_renderer: &mut MiniRenderer,
) {
    let mut ctx = CommandContext {
        world,
        renderer,
        ui,
        window_size,
        settings,
        event_loop,
        game_state,
        simulation,
        element_ctx: Default::default(),
        mini_renderer
    };

    command_queue.drain(&mut ctx);
}

pub fn exit_game(
    game_state: &mut GameState,
    settings: &mut Settings,
    variables: &mut Variables,
    world: &mut World,
    event_loop: &dyn ActiveEventLoop,
) {
    match settings.save(rusty_skylines_dir("settings.toml")) {
        Ok(_) => println!("Settings saved"),
        Err(e) => eprintln!("Failed to save Settings: {e}"),
    }
    save_colors(rusty_skylines_dir("colors.toml"), variables);
    save_game(game_state, world, settings, false);

    event_loop.exit();

    //std::process::exit(69); // Die.
}
pub fn save_game(
    game_state: &mut GameState,
    world: &mut World,
    settings: &Settings,
    and_exit: bool,
) {
    let name = game_state.current_save_info.as_ref()
        .map(|s| s.name.clone())
        .unwrap_or("No save!!! Report to maxwag9!!".to_string());
    match game_state.save(world, settings, and_exit) {
        SaveResult::Success => println!("World '{}' saved", name),
        SaveResult::TriedToSaveEmptySave => {}
        e => eprintln!("Failed to save World '{}': {:?}", name, e),
    }
}
pub fn load_save(
    game_state: &mut GameState,
    world: &mut World,
    settings: &Settings,
    save_name: &str,
) {
    let save_name = make_safe_save_name(save_name);
    match game_state.load(save_name.as_str(), world, settings) {
        LoadResult::Success(version) => println!(
            "World '{}', Version {} loaded, {} Terrain Edited Chunks, {} Road Nodes, Terrain preset '{}'",
            game_state.current_save_info.as_ref().unwrap().name,
            version,
            world.terrain.terrain_editor.edits.len(),
            world.roads.road_manager.roads.nodes.len(), game_state.current_save_info.as_ref().unwrap().terrain_preset_name
        ),
        LoadResult::FileNonExistent(e) => {
            eprintln!("Failed to load World '{}': {:?}", save_name, e);
            let mut save_info = SaveInfo::default();
            save_info.name = save_name.clone();
            game_state.current_save_info = Some(save_info);
            game_state.save(world, settings, false);
            load_save(
                game_state,
                world,
                settings,
                save_name.as_str(),
            );
        }

        e => eprintln!("Failed to load World '{}': {:?}", save_name, e),
    }
}
/// ONLY USE EXECUTE_COMMAND SO IT EXECUTES IMMEDIATELY!!
pub fn set_element_property(
    ctx: &mut CommandContext,
    name: &str,
    new_val: &Value,
) -> CommandResult {
    //return CommandResult::Ok;
    match name {
        "new_zone_type" => {
            let zoning_type = ZoningType::from_value(new_val);
            ctx.world.terrain.cursor.zoning_type = zoning_type;
            ctx.ui.variables.set_var(
                name,
                zoning_type
                    .map(|z| z.to_string())
                    .unwrap_or("None".to_string()),
            );
        }
        "target_pos.x" => {
            if let Some(x) = new_val.as_f64() {
                ctx.world.world_state.camera.target.set_x(x);
                ctx.ui.variables.set_f64(name, x);
            }
        }
        "target_pos.y" => {
            if let Some(y) = new_val.as_f64() {
                ctx.world.world_state.camera.target.set_y(y);
                ctx.ui.variables.set_f64(name, y);
            }
        }
        "target_pos.z" => {
            if let Some(z) = new_val.as_f64() {
                ctx.world.world_state.camera.target.set_z(z);
                println!("Set to: {}", z);
                ctx.ui.variables.set_f64(name, z);
            }
        }
        "cursor_mode" => {
            let mode = new_val.to_string();
            ctx.world.terrain.cursor.mode = CursorMode::from(mode.clone());
            ctx.ui.variables.set_string(name, mode);
        }
        "new_building" => {
            let building_name = new_val.to_string();
            ctx.ui.variables.set_string(name, building_name.clone());
            let pos = ctx
                .world
                .terrain
                .last_picked
                .as_ref()
                .map(|p| p.pos)
                .unwrap_or(ctx.world.world_state.camera.target);
            ctx.world.buildings.cancel_preview(
                &mut ctx.world.terrain.cursor,
                &mut ctx.world.zoning,
                &mut ctx.world.terrain.terrain_editor,
                &mut ctx.world.props,
                &mut ctx.ui.variables,
                &mut ctx.world.roads.road_manager.roads,
            );
            ctx.world.buildings.start_preview(
                &mut ctx.world.terrain.cursor,
                building_name,
                pos,
                &mut ctx.world.zoning,
                &mut ctx.ui.variables,
                &mut ctx.world.roads,
                &ctx.renderer.road_renderer.mesh_manager,
            );
            //println!("{:?}", ctx.world.terrain.cursor.preview_building);
        }
        "road_mode" => {
            let road_mode = new_val.to_string();
            //println!("Road Mode: {}", road_mode);
            if let Ok(road_mode) = BuildMode::from_str(road_mode.as_str()) {
                ctx.world.roads.road_editor.style.set_mode(road_mode);
            }
            ctx.ui.variables.set_string(name, road_mode);
        }
        // "cloud_debug_mode" => {
        //     let road_mode = new_val.to_string();
        //     //println!("Road Mode: {}", road_mode);
        //     if let Ok(road_mode) = BuildMode::from_str(road_mode.as_str()) {
        //         ctx.world.roads.road_editor.style.set_mode(road_mode);
        //     }
        //     ctx.ui.variables.set_string(name, road_mode);
        // }
        // "road_type" => {
        //     let road_type = new_val.to_string();
        //     ctx.world.terrain.cursor.road_type = road_type;
        //     ctx.ui.variables.set_string(name, road_type);
        // }
        // "lanes" => {
        //     if let Some(lanes) = new_val.as_i64() {
        //         let road_type = &mut ctx.world.terrain.cursor.road_type;
        //         road_type.lanes_each_direction = (lanes as LeftLaneCount, lanes as RightLaneCount);
        //
        //         ctx.ui.variables.set_i64(name, lanes);
        //     }
        // }
        // "left_lanes" => {
        //     if let Some(lanes) = new_val.as_i64() {
        //         let road_type = &mut ctx.world.terrain.cursor.road_type;
        //         road_type.lanes_each_direction.0 = lanes as LeftLaneCount;
        //
        //         ctx.ui.variables.set_i64(name, lanes);
        //     }
        // }
        // "right_lanes" => {
        //     if let Some(lanes) = new_val.as_i64() {
        //         let road_type = &mut ctx.world.terrain.cursor.road_type;
        //         road_type.lanes_each_direction.1 = lanes as RightLaneCount;
        //
        //         ctx.ui.variables.set_i64(name, lanes);
        //     }
        // }
        "sim_running" => {
            let sim_running = new_val.is_truthy();

            ctx.simulation.set_running(sim_running);
        }
        "sim_speed" => {
            if let Some(sim_speed) = new_val.as_f64() {
                ctx.simulation.set_speed_permanent(sim_speed as f32);
                ctx.ui.variables.set_f64(name, sim_speed);
            }
        }
        _ => {}
    }
    // ONLY USE EXECUTE_COMMAND SO IT EXECUTES IMMEDIATELY!!
    let Some((base, suffix)) = name.split_once('.') else {
        return CommandResult::Ok;
        // return CommandResult::AnnoyingError(format!(
        //     "set_element_property: invalid property name '{}', expected '.' separator",
        //     name
        // ));
    };
    let (property, component) = match suffix.split_once('.') {
        Some((property, component)) => (property, component),
        None => (suffix, ""),
    };

    let selections: Vec<ElementRef>;
    let element_ctx = &ctx.element_ctx;
    selections = match base {
        "self" => {
            if let Some(self_element_ref) = element_ctx.self_element.clone() {
                vec![self_element_ref]
            } else {
                vec![]
            }
        }
        "as" => {
            if let Some(as_element_ref) = element_ctx.as_element.clone() {
                vec![as_element_ref]
            } else {
                vec![]
            }
        }
        "editing" => {
            ctx.ui.touch_manager.selection.selected.clone()
        }
        _ => {
            return CommandResult::AnnoyingError(format!(
                "set_element_property: unknown base '{}', expected 'self', 'as' or 'editing'",
                base
            ));
        }
    };

    for element_ref in selections {
        let after = match property {
            "center" | "topleft" | "bottomleft" | "topright" | "bottomright" => {
                let Some(element) = get_element(&ctx.ui.menus, &element_ref) else {
                    return CommandResult::Error(format!(
                        "get_element() failed in set_element_property for element {:?}",
                        element_ref
                    ));
                };

                let (corner_x, corner_y) = match property {
                    "topleft" => (0.0, 0.0),
                    "topright" => (1.0, 0.0),
                    "bottomleft" => (0.0, 1.0),
                    "bottomright" => (1.0, 1.0),
                    _ => (0.5, 0.5),
                };

                let (before, text_anchor, element_size) =
                    if let Some(text) = element.as_text() {
                        let width = text.width;
                        let height = text.height;
                        let anchor = text.anchor.clone();

                        let top_left = anchor.anchor_to(
                            text.center(),
                            width,
                            height,
                        );

                        (
                            [
                                top_left[0] + width * corner_x,
                                top_left[1] + height * corner_y,
                            ],
                            Some((anchor, width, height)),
                            None,
                        )
                    } else {
                        let center = element.center();

                        if property == "center" {
                            (center, None, None)
                        } else {
                            let Some(size) = element.size2() else {
                                return CommandResult::Error(format!(
                                    "set_element_property: cannot determine bounding size of element {:?}",
                                    element_ref
                                ));
                            };

                            (
                                [
                                    center[0] + size[0] * (corner_x - 0.5),
                                    center[1] + size[1] * (corner_y - 0.5),
                                ],
                                None,
                                Some(size),
                            )
                        }
                    };

                let after = match Variables::component_index(component) {
                    None => {
                        let Some(arr) = new_val.as_array() else {
                            return CommandResult::Error(format!(
                                "set_element_property: '{}' requires an array value, got: {}",
                                property,
                                new_val
                            ));
                        };

                        if arr.len() != 2 {
                            return CommandResult::Error(format!(
                                "set_element_property: '{}' array must have exactly 2 elements, got {}",
                                property,
                                arr.len()
                            ));
                        }

                        let Some(x) = arr[0].as_f64() else {
                            return CommandResult::Error(format!(
                                "set_element_property: '{}' array[0] must be a number, got: {}",
                                property,
                                arr[0]
                            ));
                        };

                        let Some(y) = arr[1].as_f64() else {
                            return CommandResult::Error(format!(
                                "set_element_property: '{}' array[1] must be a number, got: {}",
                                property,
                                arr[1]
                            ));
                        };

                        [x as f32, y as f32]
                    }
                    Some(idx) => {
                        let Some(val) = new_val.as_f64() else {
                            return CommandResult::Error(format!(
                                "set_element_property: {} component must be a number, got: {}",
                                property,
                                new_val
                            ));
                        };

                        let mut after = before;

                        match idx {
                            0 => after[0] = val as f32,
                            1 => after[1] = val as f32,
                            _ => {
                                return CommandResult::Error(format!(
                                    "set_element_property: invalid {} component index {}",
                                    property,
                                    idx
                                ));
                            }
                        }

                        after
                    }
                };

                let command_after = if let Some((anchor, width, height)) = text_anchor {
                    let center = [
                        after[0] + width * (0.5 - corner_x),
                        after[1] + height * (0.5 - corner_y),
                    ];

                    anchor.anchor_position_from_center(center, width, height)
                } else if let Some(size) = element_size {
                    [
                        after[0] + size[0] * (0.5 - corner_x),
                        after[1] + size[1] * (0.5 - corner_y),
                    ]
                } else {
                    after
                };

                ctx.ui.ui_edit_manager.execute_command(
                    MoveElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after: command_after,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );

                Value::from_vec(after)
            }

            "size" | "rect" => {
                let Some(before) = get_element_main_size(&ctx.ui.menus, &element_ref) else {
                    return CommandResult::Error(
                        format!(
                            "get_element_size() failed in set_element_property for element {:?}",
                            element_ref
                        )
                        .to_string(),
                    );
                };
                let Some(before) = before.size2() else {
                    return CommandResult::Error(
                        format!("Size is not square in set_element_property for element {:?}, the element doesn't have a [f32; 2] size.", element_ref).to_string()
                    );
                };
                let after = match Variables::component_index(component) {
                    // Setting full vector
                    None => {
                        let Some(arr) = new_val.as_array() else {
                            return CommandResult::Error(format!(
                                "set_element_property: size requires an array value when no component specified, but got: {}",
                                new_val.to_string()
                            ));
                        };
                        if arr.len() != 2 {
                            return CommandResult::Error(format!(
                                "set_element_property: size array must have exactly 2 elements, got {}",
                                arr.len()
                            ));
                        }

                        let Some(x) = arr[0].as_f64() else {
                            return CommandResult::Error(format!(
                                "set_element_property: size array[0] must be a number, got: {}",
                                arr[0].to_string()
                            ));
                        };

                        let Some(y) = arr[1].as_f64() else {
                            return CommandResult::Error(format!(
                                "set_element_property: size array[1] must be a number, got: {}",
                                arr[1].to_string()
                            ));
                        };

                        [x as f32, y as f32]
                    }

                    // Setting single component
                    Some(idx) => {
                        //println!("Val: {}, idx: {}", new_val, idx);
                        let Some(val) = new_val.as_f64() else {
                            return CommandResult::Error(format!(
                                "set_element_property: size component value must be a number, got: {}",
                                new_val.to_string()
                            ));
                        };

                        let mut after = before;
                        match idx {
                            0 => after[0] = val as f32,
                            1 => after[1] = val as f32,
                            _ => {
                                return CommandResult::Error(format!(
                                    "set_element_property: invalid size component index {}, expected 0 or 1",
                                    idx
                                ));
                            }
                        }
                        after
                    }
                };
                //println!("Setting size to: {:?}, with {:?} and {:?}", after, name, new_val);
                ctx.ui.ui_edit_manager.execute_command(
                    ResizeElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after: SizeProperty::Rect(after),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                Value::from_vec(after)
            }

            "radius" => {
                let Some(radius) = new_val.as_f64() else {
                    return CommandResult::Error(format!(
                        "set_element_property: radius value must be a number, got: {}",
                        new_val.to_string()
                    ));
                };

                ctx.ui.ui_edit_manager.execute_command(
                    ResizeElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after: SizeProperty::Radius(radius as f32),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                Value::F64(radius)
            }

            "border" => {
                let Some(size) = new_val.as_f64() else {
                    return CommandResult::Error(format!(
                        "set_element_property: border value must be a number, got: {}",
                        new_val.to_string()
                    ));
                };

                ctx.ui.ui_edit_manager.execute_command(
                    ResizeElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after: SizeProperty::Border(size as f32),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                Value::F64(size)
            }
            "inside_border" => {
                let Some(size) = new_val.as_f64() else {
                    return CommandResult::Error(format!(
                        "set_element_property: inside_border value must be a number, got: {}",
                        new_val.to_string()
                    ));
                };

                ctx.ui.ui_edit_manager.execute_command(
                    ResizeElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after: SizeProperty::InsideBorder(size as f32),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                Value::F64(size)
            }
            "pt" => {
                let Some(size) = new_val.as_f64() else {
                    return CommandResult::Error(format!(
                        "set_element_property: pt value must be a number, got: {}",
                        new_val.to_string()
                    ));
                };

                ctx.ui.ui_edit_manager.execute_command(
                    ResizeElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after: SizeProperty::Pt(size as f32),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );
                Value::F64(size)
            }
            "color" => {
                let (color_name, channel_index) = match component.rsplit_once('.') {
                    Some((color_name, channel_name))

                    if !(
                        (component.starts_with("vertex_index.") && color_name == "vertex_index")
                            || (component.starts_with("deco.") && color_name == "deco")
                    ) =>
                        {
                            //println!("Balls: {} {}", color_name, channel_name); // It just works all of a sudden, are you kidding me? 😩
                            match Variables::component_index(channel_name) {
                                Some(index) if index < 4 => (color_name, Some(index)),
                                Some(index) => {
                                    return CommandResult::Error(format!(
                                        "set_element_property: invalid color channel index {}, expected 0..3",
                                        index
                                    ));
                                }
                                None => (component, None),
                            }
                        }
                    _ => (component, None),
                };

                let color_property = ColorComponent::from_str(color_name);

                let new_color = if let Some(channel_index) = channel_index {
                    let Some(mut color) =
                        get_element_color(&ctx.ui.menus, &element_ref, &color_property)
                    else {
                        return CommandResult::Error(format!(
                            "set_element_property: couldn't retrieve existing '{}' color",
                            color_property
                        ));
                    };

                    let Some(channel_value) = new_val.as_f64() else {
                        return CommandResult::Error(format!(
                            "set_element_property: color channel requires a numeric value, got: {}",
                            new_val
                        ));
                    };

                    color[channel_index] = channel_value as f32;
                    color
                } else {
                    let Some(color) = new_val.as_color4() else {
                        return CommandResult::Error(format!(
                            "set_element_property: expected color4 value, but got: {}",
                            new_val
                        ));
                    };

                    color
                };

                ctx.ui.ui_edit_manager.execute_command(
                    ChangeColorCommand {
                        affected_element: element_ref,
                        property: color_property,
                        before: None,
                        after: new_color,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.world.input.mouse,
                );

                Value::from_vec(new_color)
            }

            "offset_order" => {
                if let Some(menu) = ctx.ui.menus.get_mut(element_ref.menu.as_str()) {
                    if let Some(layer) = menu.get_layer_mut(element_ref.layer.as_str()) {
                        if let Some(idx) =
                            layer.elements.iter().position(|e| e.id() == element_ref.id)
                        {
                            let offset = new_val.as_i64().unwrap_or(0);
                            let len = layer.elements.len() as i64;

                            let target_idx =
                                (idx as i64 + offset).clamp(0, len.saturating_sub(1)) as usize;

                            //println!("Move {} -> {}", idx, target_idx);

                            if idx != target_idx {
                                let element = layer.elements.remove(idx);
                                layer.elements.insert(target_idx, element); // TO/DO: Sus?

                                layer.dirty.mark_all();
                            }
                        }
                    }
                }
                Value::I64(0)
            }

            "actions" => {
                if let Some(new_actions) = new_val.as_array() {
                    if let Some(element) = get_element_mut(&mut ctx.ui.menus, &element_ref) {
                        let actions = new_actions
                            .iter()
                            .flat_map(|a| a.as_str())
                            .map(|str| str.to_string())
                            .collect::<Vec<_>>();
                        element.set_actions(actions);
                    } else {
                        return CommandResult::Error(format!(
                            "set_element_property: element with ID: '{}' doesn't exist",
                            element_ref.id
                        ));
                    }
                } else {
                    return CommandResult::Error(format!(
                        "set_element_property: expected array value, but got: {}",
                        new_val.to_string()
                    ));
                }
                new_val.clone()
            }
            "text" => {
                if let Some(new_text) = new_val.as_str() {
                    if let Some(element) =
                        get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref)
                    {
                        element.set_text(new_text.to_string());
                    } else {
                        return CommandResult::Error(format!(
                            "set_element_property: element with ID: '{}' doesn't exist",
                            element_ref.id
                        ));
                    }
                } else {
                    return CommandResult::Error(format!(
                        "set_element_property: {}.{} expected string value, but got: {}",
                        base,
                        property,
                        new_val.to_string()
                    ));
                }
                new_val.clone()
            }
            "tt" => {
                if let Some(new_text) = new_val.as_str() {
                    if let Some(element) =
                        get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref)
                    {
                        element.set_text(new_text.to_string());
                        element.set_template(new_text.to_string());
                    } else {
                        return CommandResult::Error(format!(
                            "set_element_property: element with ID: '{}' doesn't exist",
                            element_ref.id
                        ));
                    }
                } else {
                    return CommandResult::Error(format!(
                        "set_element_property: {}.{} expected string value, but got: {}",
                        base,
                        property,
                        new_val.to_string()
                    ));
                }
                new_val.clone()
            }
            "template" => {
                //println!("Trying to set template of element '{element_ref:?}' with '{new_val:?}'");
                if let Some(new_template) = new_val.as_str() {
                    if let Some(element) =
                        get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref)
                    {
                        element.set_template(new_template.to_string());
                    } else {
                        return CommandResult::Error(format!(
                            "set_element_property: element with ID: '{}' doesn't exist",
                            element_ref.id
                        ));
                    }
                } else {
                    return CommandResult::Error(format!(
                        "set_element_property: {}.{} expected string value, but got: {}",
                        base,
                        property,
                        new_val.to_string()
                    ));
                }
                new_val.clone()
            }

            "anchor" => {
                ////println!("Trying to set template of element '{element_ref:?}' with '{new_val:?}'");
                if let Some(new_anchor) = new_val.as_str() {
                    let anchor = Anchor::from_str(new_anchor);
                    let anchor = match anchor {
                        Ok(anchor) => anchor,
                        Err(err) => {
                            return CommandResult::Error(format!(
                                "set_element_property: Text Anchor '{}' doesn't exist. Error: '{}'",
                                new_anchor, err
                            ));
                        }
                    };
                    if let Some(element) =
                        get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref)
                    {
                        if let Some(text) = element.as_text_mut() {
                            text.anchor = anchor;
                        }
                    } else {
                        return CommandResult::Error(format!(
                            "set_element_property: element with ID: '{}' doesn't exist",
                            element_ref.id
                        ));
                    }
                } else {
                    return CommandResult::Error(format!(
                        "set_element_property: {}.{} expected string value, but got: {}",
                        base,
                        property,
                        new_val.to_string()
                    ));
                }
                new_val.clone()
            }
            "active" => {
                let active = new_val.is_truthy();

                if let Some(element) = get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref) { element.set_active(active); } else {
                    return CommandResult::Error(format!(
                        "set_element_property: element with ID: '{}' doesn't exist",
                        element_ref.id
                    ));
                }
                Value::Bool(active)
            }
            "touchable" => {
                //println!("Setting touchable of element {:?}' to '{}'", element_ref, new_val.to_string());
                let touchable = new_val.is_truthy();

                if let Some(element) = get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref) { element.set_touchable(touchable); } else {
                    return CommandResult::Error(format!(
                        "set_element_property: element with ID: '{}' doesn't exist",
                        element_ref.id
                    ));
                }
                Value::Bool(touchable)
            }
            "texture" => {
                //println!("Trying to set template of element '{element_ref:?}' with '{new_val:?}'");
                if let Some(new_tex) = new_val.as_str() {
                    if let Some(element) = get_element_mut_dirty_layer(&mut ctx.ui.menus, &element_ref) {
                        if let Some(rect) = element.as_rect_mut() {
                            rect.texture = Some(new_tex.to_string());
                            rect.cached_texture = get_rect_texture(&rect.texture, &ctx.renderer.device, &ctx.renderer.queue, &ctx.game_state.mod_manager)
                        } else {
                            return CommandResult::Error(format!(
                                "set_element_property: element with ID: '{}' isn't Rect, only Rects have texture support!",
                                element_ref.id
                            ));
                        }
                    } else {
                        return CommandResult::Error(format!(
                            "set_element_property: element with ID: '{}' doesn't exist",
                            element_ref.id
                        ));
                    }
                } else {
                    return CommandResult::Error(format!(
                        "set_element_property: {}.{} expected string value, but got: {}",
                        base,
                        property,
                        new_val.to_string()
                    ));
                }
                new_val.clone()
            }
            _ => {
                return CommandResult::Error(format!(
                    "set_element_property: unknown property '{}'",
                    property
                ));
            }
        };
        ctx.ui.variables.set_var(format!("{}.{}", base, property).as_str(), after)
    }
    CommandResult::Ok
}

pub fn make_element(id: String, kind: ElementKind, center: [f32; 2]) -> Option<UiElement> {
    match kind {
        ElementKind::None => None,

        ElementKind::Text => Some(UiElement::Text({
            let mut el = UiButtonText::default();
            el.id = id;
            el.set_pos(center);
            el
        })),

        ElementKind::Circle => Some(UiElement::Circle({
            let mut el = UiButtonCircle::default();
            el.id = id;
            el.set_pos(center);
            el
        })),

        ElementKind::Outline => Some(UiElement::Outline({
            let mut el = UiButtonOutline::default();
            el.id = id;
            el.set_pos(center);
            el
        })),

        ElementKind::Handle => Some(UiElement::Handle({
            let mut el = UiButtonHandle::default();
            el.id = id;
            el.set_pos(center);
            el
        })),

        ElementKind::Polygon => Some(UiElement::Polygon({
            let mut el = UiButtonPolygon::default();
            el.id = id;
            el.set_pos(center);
            el
        })),

        ElementKind::Advanced => Some(UiElement::Advanced({
            let mut el = AdvancedPrimitive::default();
            el.id = id;
            el.set_pos(center);
            el
        })),

        ElementKind::Rect => Some(UiElement::Rect({
            let mut el = UiButtonRect::default();
            el.id = id;
            el.set_pos(center);
            el
        })),
    }
}

// pub fn send_element_properties_to_variables(
//     menus: &HashMap<String, Menu>,
//     variables: &mut Variables,
//     element_ctx: &ElementContext,
// ) {
//     return;
//     fn clear_properties(prefix: &str, variables: &mut Variables) {
//         for property in [
//             "menu",
//             "layer",
//             "idx",
//             "active",
//             "id",
//             "kind",
//             "center",
//             "radius",
//             "pt",
//             "size",
//             "color_components",
//         ] {
//             variables.set_var(&format!("{prefix}.{property}"), Value::None);
//         }
//
//         for component in ColorComponent::iter() {
//             variables.set_var(&format!("{prefix}.color.{component}"), Value::None);
//         }
//     }
//     fn send_properties(
//         prefix: &str,
//         menus: &HashMap<String, Menu>,
//         variables: &mut Variables,
//         element_ref: &ElementRef,
//     ) {
//         if let Some(menu) = menus.get(&element_ref.menu) {
//             variables.set_string(&format!("{prefix}.menu"), element_ref.menu.clone());
//
//             if let Some(layer) = menu.layers.iter().find(|l| l.name == element_ref.layer) {
//                 variables.set_string(&format!("{prefix}.layer"), element_ref.layer.clone());
//
//                 if let Some(element_idx) = layer.elements.iter().position(|e| e.id() == element_ref.id.as_str()) {
//                     let element = &layer.elements[element_idx];
//
//                     variables.set_i64(&format!("{prefix}.idx"), element_idx as i64);
//                     variables.set_bool(
//                         &format!("{prefix}.active"),
//                         menu.active && layer.active && element.is_active(),
//                     );
//                     variables.set_string(&format!("{prefix}.id"), element_ref.id.clone());
//                     variables.set_string(&format!("{prefix}.kind"), element_ref.kind.to_string());
//                     if let Some(text) = element.text() {
//                         //println!("{}", text);
//                         variables.set_string(&format!("{prefix}.text"), text);
//                     } else {
//                         variables.set_var(&format!("{prefix}.text"), Value::None);
//                     }
//                     if let Some(text) = element.template() {
//                         variables.set_string(&format!("{prefix}.template"), text);
//                     } else {
//                         variables.set_var(&format!("{prefix}.template"), Value::None);
//                     }
//
//                     variables.set_array(&format!("{prefix}.center"), element.center());
//                     // if let Some(text) = element.as_text() {
//                     //     variables.set_array(&format!("{prefix}.topleft"), anchor_to(text.anchor, text.center(), text.width, text.height));
//                     // }
//
//                     for prop in element.sizes() {
//                         match prop {
//                             SizeProperty::Radius(radius) => {
//                                 variables.set_f64(&format!("{prefix}.radius"), radius);
//                             }
//                             SizeProperty::Pt(pt) => {
//                                 variables.set_f64(&format!("{prefix}.pt"), pt);
//                             }
//                             SizeProperty::Rect(rect) => {
//                                 variables.set_array(&format!("{prefix}.size"), rect);
//                             }
//                             SizeProperty::PolygonScale(_) => {}
//                             SizeProperty::AdvancedPrimitiveScale(_) => {}
//                             SizeProperty::Border(border) => {
//                                 variables.set_f64(&format!("{prefix}.border"), border);
//                             }
//                             SizeProperty::InsideBorder(inside_border) => {
//                                 variables
//                                     .set_f64(&format!("{prefix}.inside_border"), inside_border);
//                             }
//                         }
//                     }
//
//                     let color_components = element.color_components();
//
//                     variables.set_array(
//                         &format!("{prefix}.color_components"),
//                         color_components
//                             .iter()
//                             .map(|c| Value::String(c.to_string()))
//                             .collect::<Vec<Value>>(),
//                     );
//
//                     for component in color_components {
//                         let name = format!("{prefix}.color.{}", component.to_string());
//                         if let Some(color) = element.color(&component) {
//                             variables.set_array(name.as_str(), color);
//                         }
//                     }
//                 }
//             }
//         }
//     }
//
//     if let Some(self_element_ref) = element_ctx.self_element.as_ref() {
//         send_properties("self", menus, variables, self_element_ref);
//     }
//     if let Some(as_element_ref) = element_ctx.as_element.as_ref() {
//         send_properties("as", menus, variables, as_element_ref);
//         variables.set_bool("as.exists", true);
//     } else {
//         clear_properties("as", variables);
//         variables.set_bool("as.exists", false);
//     }
// }

fn get_setting_key(name: &str) -> Option<SettingKey> {
    SettingKey::from_str(name)
}

fn set_variable_or_property(ctx: &mut CommandContext, name: &str, value: Value) -> CommandResult {
    match set_element_property(ctx, name, &value) {
        CommandResult::Error(e) => {
            error!("[UI Actions] In set_variable_or_property with Name: '{}' and value: '{}', Error: {}; {}", name, value.to_string(), e, ctx.element_ctx);
            return CommandResult::Ok
        },
        CommandResult::AnnoyingError(e) => {
            //error!("[UI Actions] In set_variable_or_property: ({}, {}): {}", name, value.to_string(), e);
        },
        _ => {}
    }

    ctx.ui.variables.set_var(name, value);

    CommandResult::Ok
}

#[test]
fn test_send_properties_to_variables() {
    use crate::ui::menu::Menu;
    use crate::ui::vertex::RuntimeLayer;
    use rand::RngExt;
    let with_as_element_percetage = 0.25;
    let variables = Variables::new();
    let mut menus = HashMap::new();
    let rect = UiButtonRect::default();
    let self_ref = ElementRef {
        menu: "Test_Menu".to_string(),
        layer: "test_layer".to_string(),
        id: rect.id.clone(),
        kind: ElementKind::Rect,
    };
    menus.insert(
        "Test_Menu".to_string(),
        Menu {
            layers: vec![RuntimeLayer {
                name: "test_layer".to_string(),
                ap_name: None,
                order: 0,
                string_actions: vec![],
                compiled_actions: vec![],
                elements: vec![UiElement::Rect(rect)],
                active: true,
                ap_vars: vec![],
                dirty: Default::default(),
                gpu: Default::default(),
                opaque: false,
                saveable: false,
                editing_tool: false,
                outline_poly_vertices: vec![],
            }],
            active: true,
        },
    );
    let rng = &mut rand::rngs::ThreadRng::default();
    let start = std::time::Instant::now();

    for _ in 0..150 {
        let element_ctx = ElementContext {
            self_element: Some(self_ref.clone()),
            as_element: if rng.random_bool(with_as_element_percetage) {
                Some(self_ref.clone())
            } else {
                None
            },
        };
        // send_element_properties_to_variables(&menus, &mut variables, &element_ctx);
    }

    println!("{:?}", start.elapsed());
    let fard = variables.get("mrbeastjusttomakethecompileracceptitall");
    println!("{:?}", fard);
}
