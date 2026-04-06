#![allow(dead_code, unused_variables)]
pub mod drag_hue_point;

use crate::data::{SettingKey, SettingOp, Settings};
use crate::helpers::paths::rusty_skylines_dir;
use crate::renderer::props::Props;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::ui::menu::Menu;
use crate::ui::parser::{Value, eval_expr};
use crate::ui::ui_edit_manager::{
    ChangeColorCommand, ColorComponent, CreateElementCommand, DeleteElementCommand,
    DuplicateElementCommand, MoveElementCommand, ResizeElementCommand,
};
use crate::ui::ui_editor::{Ui, get_element, get_element_position, get_layer_settings};
use crate::ui::ui_edits::{SizeProperty, create_element, delete_element};
use crate::ui::ui_text_editing::HitResult;
use crate::ui::ui_touch_manager::{ElementRef, UiTouchManager};
use crate::ui::variables::{Variables, initialize_value};
use crate::ui::vertex::{
    AdvancedPrimitive, ElementKind, UiButtonCircle, UiButtonHandle, UiButtonOutline,
    UiButtonPolygon, UiButtonRect, UiButtonText, UiElement,
};
use crate::world::camera::{Camera, CameraController};
use crate::world::game_state::{GameState, LoadResult, SaveResult};
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec2;
use std::cmp::PartialEq;
use std::collections::{HashMap, VecDeque};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
// ==================== COMMAND TYPE ENUM ====================

/// Canonical command type identifier for pattern matching and legacy conversion.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum UiCommandType {
    // Menus
    OpenMenu,
    CloseMenu,
    ToggleMenu,
    MenuActive,

    // Layers
    OpenLayer,
    CloseLayer,
    ToggleLayer,

    // Variables
    SetVar,
    IncVar,
    DecVar,
    MulVar,
    ToggleBool,
    Clamp,

    // Action state management
    StartAction,
    StopAction,
    RemoveAction,

    // World renderer
    SetPickRadius,
    GrowPickRadius,
    ShrinkPickRadius,

    // Flow control
    Delay,
    Halt,
    Skip,
    If,
    IfVarEq,

    // Debug
    Print,
    DebugVars,
    DebugMenus,
    DebugActions,

    // Events
    EmitEvent,

    // No-op
    Noop,
}

impl UiCommandType {
    /// Get the name for this command type.
    pub fn name(self) -> &'static str {
        match self {
            UiCommandType::OpenMenu => "open_menu",
            UiCommandType::CloseMenu => "close_menu",
            UiCommandType::ToggleMenu => "toggle_menu",
            UiCommandType::MenuActive => "menu_active",

            UiCommandType::OpenLayer => "open_layer",
            UiCommandType::CloseLayer => "close_layer",
            UiCommandType::ToggleLayer => "toggle_layer",

            UiCommandType::SetVar => "set_var",
            UiCommandType::IncVar => "inc",
            UiCommandType::DecVar => "dec",
            UiCommandType::MulVar => "mul",
            UiCommandType::ToggleBool => "toggle_bool",
            UiCommandType::Clamp => "clamp",

            UiCommandType::StartAction => "start_action",
            UiCommandType::StopAction => "stop_action",
            UiCommandType::RemoveAction => "remove_action",

            UiCommandType::SetPickRadius => "set_pick_radius",
            UiCommandType::GrowPickRadius => "grow_pick_radius",
            UiCommandType::ShrinkPickRadius => "shrink_pick_radius",

            UiCommandType::Delay => "delay",
            UiCommandType::Halt => "halt",
            UiCommandType::Skip => "skip",
            UiCommandType::If => "if",
            UiCommandType::IfVarEq => "if_var_eq",

            UiCommandType::Print => "print",
            UiCommandType::DebugVars => "debug_vars",
            UiCommandType::DebugMenus => "debug_menus",
            UiCommandType::DebugActions => "debug_actions",

            UiCommandType::EmitEvent => "emit_event",

            UiCommandType::Noop => "noop",
        }
    }
}

// ==================== COMMAND ENUM ====================

/// A fully-specified UI command with all data embedded.
/// Can be queued and executed without the original parsing context.
#[derive(Debug, Clone, PartialEq)]
pub enum UiCommand {
    // ===== MENU COMMANDS =====
    OpenMenu {
        menu_name: String,
    },
    CloseMenu {
        menu_name: String,
    },
    CloseAllMenus,
    ToggleMenu {
        menu_name: String,
    },
    MenuActive {
        menu_name: String,
    },

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
        element_ref: ElementRef,
        name: String,
        value: String,
    },
    IncVar {
        element_ref: ElementRef,
        name: String,
        amount: f64,
    },
    DecVar {
        element_ref: ElementRef,
        name: String,
        amount: f64,
    },
    MulVar {
        element_ref: ElementRef,
        name: String,
        factor: f64,
    },
    ToggleVar {
        element_ref: ElementRef,
        name: String,
    },
    Clamp {
        element_ref: ElementRef,
        name: String,
        min: f64,
        max: f64,
    },

    // ===== ACTION STATE COMMANDS =====
    StartAction {
        action_name: String,
    },
    StopAction {
        action_name: String,
    },
    RemoveAction {
        action_name: String,
    },

    // ===== FLOW CONTROL =====
    Delay {
        seconds: f64,
    },
    Halt,
    Skip {
        count: usize,
    },
    If {
        element_ref: ElementRef,
        condition: String,
        then: Vec<UiCommand>,
        else_branch: Vec<UiCommand>,
    },
    IfVarEq {
        element_ref: ElementRef,
        var_name: String,
        value: String,
        then: Vec<UiCommand>,
        else_branch: Vec<UiCommand>,
    },
    AddElement {
        element_ref: ElementRef,
        menu: String,
        layer: String,
        id: String,
        kind: String,
        center: String,
    },
    CloneElement {
        element_ref: ElementRef,
        from_menu: String,
        from_layer: String,
        from_id: String,
        to_menu: String,
        to_layer: String,
        to_id: String,
        center: String,
    },
    CloneElementUndoable {
        element_ref: ElementRef,
        from_menu: String,
        from_layer: String,
        from_id: String,
        to_menu: String,
        to_layer: String,
        to_id: String,
        center: String,
    },
    DeleteElement {
        element_ref: ElementRef,
        menu: String,
        layer: String,
        id: String,
    },
    DeleteElementUndoable {
        element_ref: ElementRef,
        menu: String,
        layer: String,
        id: String,
    },
    SaveGame,
    LoadSave {
        save_name: String,
        without_saving: bool,
    },
    ExitGame,
    // ===== DEBUG COMMANDS =====
    Print {
        element_ref: ElementRef,
        args: Vec<String>,
    },
    DebugVars,
    DebugMenus,
    DebugActions,

    // ===== EVENT COMMANDS =====
    EmitEvent {
        element_ref: ElementRef,
        event_name: String,
    },

    // ===== UTILITY =====
    Batch {
        commands: Vec<UiCommand>,
    },
    Noop,
    SetVarExpr {
        element_ref: ElementRef,
        name: String,
        expr: String,
    },
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
    pub ui: &'a mut Ui,
    pub input: &'a Input,
    pub time: &'a Time,
    pub terrain: &'a mut Terrain,
    pub hit: &'a Option<HitResult>,
    pub window_size: PhysicalSize<u32>,
    pub settings: &'a mut Settings,
    pub camera: &'a mut Camera,
    pub event_loop: &'a ActiveEventLoop,
    pub game_state: &'a mut GameState,
    pub roads: &'a mut Roads,
    pub props: &'a mut Props,
    pub camera_controller: &'a mut CameraController,
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
                            execute_at: ctx.time.total_time + seconds,
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
        let current_time = ctx.time.total_time;

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

    fn execute_one(&mut self, cmd: UiCommand, ctx: &mut CommandContext) -> CommandResult {
        match cmd {
            // ===== MENU COMMANDS =====
            UiCommand::OpenMenu { menu_name } => {
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
                    menu.active = true;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::CloseMenu { menu_name } => {
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
                    menu.active = false;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::CloseAllMenus => {
                for (_, menu) in ctx.ui.menus.iter_mut() {
                    menu.active = false;
                }
                CommandResult::Ok
            }

            UiCommand::ToggleMenu { menu_name } => {
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
                    menu.active = !menu.active;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::MenuActive { menu_name } => {
                let is_active = ctx
                    .ui
                    .menus
                    .get(&menu_name)
                    .map(|m| m.active)
                    .unwrap_or(false);
                ctx.ui.variables.set_bool("_result", is_active);
                CommandResult::Ok
            }

            // ===== LAYER COMMANDS =====
            UiCommand::OpenLayer {
                menu_name,
                layer_name,
            } => {
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        menu.active = true;
                        layer.active = true;
                        return CommandResult::Ok;
                    }
                    return CommandResult::Error(format!(
                        "Layer '{}' not found in '{}'",
                        layer_name, menu_name
                    ));
                }
                CommandResult::Error(format!("Menu '{}' not found", menu_name))
            }

            UiCommand::CloseLayer {
                menu_name,
                layer_name,
            } => {
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
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
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = !layer.active;
                        return CommandResult::Ok;
                    }
                    return CommandResult::Error(format!("Layer '{}' not found", layer_name));
                }
                CommandResult::Error(format!("Menu '{}' not found", menu_name))
            }

            // ===== VARIABLE COMMANDS =====
            UiCommand::SetVar {
                element_ref,
                name,
                value,
            } => {
                //println!("Value BEFORE: {} for {}", value, name);
                let value = string_to_value(ctx, &element_ref, value);
                //println!("Value AFTER: {} for {}", value, name);
                let (name, value) = initialize_value(name.as_str(), value);
                let mut result: CommandResult = CommandResult::Error(
                    "No settings nor vars found to set using UiCommand::SetVar".into(),
                );
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    if let Some(setting_value) = key.parse_command_arg(&value) {
                        ctx.settings
                            .apply_setting(key, SettingOp::Set(setting_value));
                        result = CommandResult::Ok
                    } else {
                        result = CommandResult::Error(format!(
                            "Cannot convert {:?} for setting '{}'",
                            value, name
                        ))
                    }
                }

                // Secondary: AP (layer settings)
                if name == "ap" && !matches!(result, CommandResult::Ok) {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        if let Some(setting_value) = setting.key.parse_command_arg(&value) {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::Set(setting_value.clone()));
                            result = CommandResult::Ok
                        } else {
                            result = CommandResult::Error(format!(
                                "Cannot convert {:?} for setting '{}'",
                                value, name
                            ))
                        }
                    }
                }

                // Ternary: Variables
                if !matches!(result, CommandResult::Ok) {
                    //println!("In setVAR: {} {}({})", name, value.type_name(), value);
                    match set_element_property(ctx, element_ref, name, &value) {
                        CommandResult::Error(e) => return CommandResult::Error(e),
                        _ => {}
                    }
                    ctx.ui.variables.set_var(&name, value);

                    result = CommandResult::Ok
                }
                result
            }

            UiCommand::IncVar {
                element_ref,
                name,
                amount,
            } => {
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    let current = ctx.settings.read_setting(key);
                    // Try numeric add first, fall back to cycle
                    if let Some(new_value) = current.add(amount) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    } else {
                        ctx.settings.apply_setting(key, SettingOp::CycleNext);
                    }
                    return CommandResult::Ok;
                }

                // Secondary: AP
                if name == "ap" {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        let current = ctx.settings.read_setting(setting.key);
                        // Try numeric add first, fall back to cycle
                        if let Some(new_value) = current.add(amount) {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::Set(new_value));
                        } else {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::CycleNext);
                        }
                        return CommandResult::Ok;
                    }
                }

                // Ternary: Variables
                let new_val = match ctx.ui.variables.get(&name) {
                    Some(Value::F64(f)) => Value::F64(f + amount),
                    Some(Value::I64(i)) => Value::F64(*i as f64 + amount),
                    _ => Value::F64(amount),
                };
                match set_element_property(ctx, element_ref, name.as_str(), &new_val) {
                    CommandResult::Error(e) => return CommandResult::Error(e),
                    _ => {}
                }
                ctx.ui.variables.set_var(name, new_val);
                CommandResult::Ok
            }

            UiCommand::DecVar {
                element_ref,
                name,
                amount,
            } => {
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    let current = ctx.settings.read_setting(key);
                    // Try numeric add first, fall back to cycle
                    if let Some(new_value) = current.subtract(amount) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    } else {
                        ctx.settings.apply_setting(key, SettingOp::CyclePrev);
                    }
                    return CommandResult::Ok;
                }

                // Secondary: AP
                if name == "ap" {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        let current = ctx.settings.read_setting(setting.key);
                        // Try numeric add first, fall back to cycle
                        if let Some(new_value) = current.subtract(amount) {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::Set(new_value));
                        } else {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::CyclePrev);
                        }
                        return CommandResult::Ok;
                    }
                }

                // Ternary: Variables
                let new_val = match ctx.ui.variables.get(&name) {
                    Some(Value::F64(f)) => Value::F64(f - amount),
                    Some(Value::I64(i)) => Value::F64(*i as f64 - amount),
                    _ => Value::F64(-amount),
                };
                match set_element_property(ctx, element_ref, name.as_str(), &new_val) {
                    CommandResult::Error(e) => return CommandResult::Error(e),
                    _ => {}
                }
                ctx.ui.variables.set_var(name, new_val);
                CommandResult::Ok
            }

            UiCommand::MulVar {
                element_ref,
                name,
                factor,
            } => {
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    let current = ctx.settings.read_setting(key);
                    if let Some(new_value) = current.multiply(factor) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    }
                    return CommandResult::Ok;
                }

                // Secondary: AP
                if name == "ap" {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        let current = ctx.settings.read_setting(setting.key);
                        if let Some(new_value) = current.multiply(factor) {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::Set(new_value));
                        }
                        return CommandResult::Ok;
                    }
                }

                // Ternary: Variables
                let new_val = match ctx.ui.variables.get(&name) {
                    Some(Value::F64(f)) => Value::F64(f * factor),
                    Some(Value::I64(i)) => Value::F64(*i as f64 * factor),
                    _ => Value::F64(factor),
                };
                match set_element_property(ctx, element_ref, name.as_str(), &new_val) {
                    CommandResult::Error(e) => return CommandResult::Error(e),
                    _ => {}
                }
                ctx.ui.variables.set_var(name, new_val);
                CommandResult::Ok
            }

            UiCommand::ToggleVar { element_ref, name } => {
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    ctx.settings.apply_setting(key, SettingOp::Toggle);
                    return CommandResult::Ok;
                }

                // Secondary: AP
                if name == "ap" {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        ctx.settings.apply_setting(setting.key, SettingOp::Toggle);
                        return CommandResult::Ok;
                    }
                }

                // Ternary: Variables
                let new_val = match ctx.ui.variables.get(&name) {
                    Some(Value::Bool(b)) => Value::Bool(!b),
                    _ => Value::Bool(false),
                };
                match set_element_property(ctx, element_ref, name.as_str(), &new_val) {
                    CommandResult::Error(e) => return CommandResult::Error(e),
                    _ => {}
                }

                ctx.ui.variables.set_var(name, new_val);
                CommandResult::Ok
            }

            UiCommand::Clamp {
                element_ref,
                name,
                min,
                max,
            } => {
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    let current = ctx.settings.read_setting(key);
                    if let Some(new_value) = current.clamp_range(min, max) {
                        ctx.settings.apply_setting(key, SettingOp::Set(new_value));
                    }
                    return CommandResult::Ok;
                }

                // Secondary: AP
                if name == "ap" {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        let current = ctx.settings.read_setting(setting.key);
                        if let Some(new_value) = current.clamp_range(min, max) {
                            ctx.settings
                                .apply_setting(setting.key, SettingOp::Set(new_value));
                        }
                        return CommandResult::Ok;
                    }
                }

                // Ternary: Variables
                let new_val = match ctx.ui.variables.get(&name) {
                    Some(Value::F64(f)) => Value::F64(f.clamp(min, max)),
                    Some(Value::I64(i)) => Value::F64((*i as f64).clamp(min, max)),
                    _ => Value::F64(min),
                };
                match set_element_property(ctx, element_ref, name.as_str(), &new_val) {
                    CommandResult::Error(e) => return CommandResult::Error(e),
                    _ => {}
                }
                ctx.ui.variables.set_var(name, new_val);
                CommandResult::Ok
            }

            UiCommand::SetVarExpr {
                element_ref,
                name,
                expr,
            } => {
                //println!("Evaluating: *{}*", expr);
                match eval_expr(&expr, &ctx.ui.variables) {
                    Some(value) => {
                        //println!("Result: *{}* to *{}*", value, name);
                        match set_element_property(ctx, element_ref, name.as_str(), &value) {
                            CommandResult::Error(e) => return CommandResult::Error(e),
                            _ => {}
                        }
                        ctx.ui.variables.set_var(name, value);
                        CommandResult::Ok
                    }
                    None => CommandResult::Error(format!("Failed to eval expr '{}'", expr)),
                }
            }
            // ===== ACTION STATE COMMANDS =====
            UiCommand::StartAction { action_name } => {
                let state = ActionState::with_time(&action_name, ctx.time.total_time);
                ctx.ui
                    .touch_manager
                    .runtimes
                    .action_states
                    .insert(action_name, state);
                CommandResult::Ok
            }

            UiCommand::StopAction { action_name } => {
                if let Some(state) = ctx
                    .ui
                    .touch_manager
                    .runtimes
                    .action_states
                    .get_mut(&action_name)
                {
                    state.active = false;
                }
                CommandResult::Ok
            }

            UiCommand::RemoveAction { action_name } => {
                ctx.ui
                    .touch_manager
                    .runtimes
                    .action_states
                    .remove(&action_name);
                CommandResult::Ok
            }

            // ===== FLOW CONTROL =====
            UiCommand::Delay { seconds } => {
                let remaining: Vec<UiCommand> = self.queue.drain(..).collect();
                CommandResult::Delay { seconds, remaining }
            }

            UiCommand::Halt => CommandResult::Stop,

            UiCommand::Skip { count } => CommandResult::Skip(count),

            UiCommand::If {
                element_ref,
                condition,
                then,
                else_branch,
            } => {
                //println!("{} {:?} {:?}", condition, then, else_branch);
                let condition = string_to_value(ctx, &element_ref, condition);
                if condition.is_truthy() {
                    for cmd in then.into_iter().rev() {
                        self.queue.push_front(cmd);
                    }
                } else {
                    for cmd in else_branch.into_iter().rev() {
                        self.queue.push_front(cmd);
                    }
                }
                CommandResult::Ok
            }

            UiCommand::IfVarEq {
                element_ref,
                var_name,
                value,
                then,
                else_branch,
            } => {
                let var_value = if var_name == "ap" {
                    get_layer_settings(&ctx.ui.menus, &element_ref)
                        .map(|setting| ctx.settings.read_setting(setting.key).to_value())
                } else {
                    ctx.ui.variables.get(&var_name).cloned()
                };

                let Some(var_value) = var_value else {
                    return CommandResult::Ok;
                };
                let compare_value = string_to_value(ctx, &element_ref, value);

                if var_value == compare_value {
                    for cmd in then.into_iter().rev() {
                        self.queue.push_front(cmd);
                    }
                } else {
                    for cmd in else_branch.into_iter().rev() {
                        self.queue.push_front(cmd);
                    }
                }
                CommandResult::Ok
            }
            UiCommand::AddElement {
                element_ref,
                menu,
                layer,
                id,
                kind,
                center,
            } => {
                let kind = ElementKind::from_string(kind.to_string().as_str());
                if kind == ElementKind::None {
                    return CommandResult::Error(
                        "Element Kind is None in AddElement kind argument".to_string(),
                    );
                }
                let center = string_to_value(ctx, &element_ref, center);
                let Some(center) = center.as_pos() else {
                    return CommandResult::Error(
                        "Couldn't unpack center pos from AddElement center argument".to_string(),
                    );
                };
                let Some(element) = make_element(id.to_string(), &kind, center) else {
                    return CommandResult::Error("Couldn't make element in AddElement".to_string());
                };
                ctx.ui.ui_edit_manager.execute_command(
                    CreateElementCommand {
                        affected_element: ElementRef::new(
                            menu.to_string().as_str(),
                            layer.to_string().as_str(),
                            id.to_string().as_str(),
                            kind,
                        ),
                        element,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.input.mouse,
                );
                CommandResult::Ok
            }
            UiCommand::CloneElement {
                element_ref,
                from_menu,
                from_layer,
                from_id,
                to_menu,
                to_layer,
                to_id,
                center,
            } => {
                let from_element = ElementRef::new(
                    from_menu.to_string().as_str(),
                    from_layer.to_string().as_str(),
                    from_id.to_string().as_str(),
                    ElementKind::None,
                );
                let Some(mut element) = get_element(&ctx.ui.menus, &from_element) else {
                    return CommandResult::Error(
                        "Couldn't unpack element in CloneElement".to_string(),
                    );
                };
                let to_element = ElementRef::new(
                    to_menu.to_string().as_str(),
                    to_layer.to_string().as_str(),
                    to_id.to_string().as_str(),
                    element.kind(),
                );

                element.set_id(&to_id.to_string());
                let center = string_to_value(ctx, &element_ref, center);
                if let Some(center) = center.as_pos() {
                    element.set_pos(center[0], center[1])
                };
                let result = create_element(
                    &mut ctx.ui.menus,
                    &to_element.menu,
                    &to_element.layer,
                    element,
                    &ctx.input.mouse,
                );
                match result {
                    Ok(ok) => CommandResult::Ok,
                    Err(err) => CommandResult::Error(err.to_string()),
                }
            }
            UiCommand::CloneElementUndoable {
                element_ref,
                from_menu,
                from_layer,
                from_id,
                to_menu,
                to_layer,
                to_id,
                center,
            } => {
                let from_element = ElementRef::new(
                    from_menu.to_string().as_str(),
                    from_layer.to_string().as_str(),
                    from_id.to_string().as_str(),
                    ElementKind::None,
                );
                let to_element = ElementRef::new(
                    to_menu.to_string().as_str(),
                    to_layer.to_string().as_str(),
                    to_id.to_string().as_str(),
                    ElementKind::None,
                );
                let center = string_to_value(ctx, &element_ref, center);
                ctx.ui.ui_edit_manager.execute_command(
                    DuplicateElementCommand {
                        from_element,
                        to_element,
                        cached_element: None,
                        optional_center: center.as_pos(),
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.input.mouse,
                );
                CommandResult::Ok
            }
            UiCommand::DeleteElement {
                element_ref,
                menu,
                layer,
                id,
            } => {
                let element = ElementRef::new(
                    menu.to_string().as_str(),
                    layer.to_string().as_str(),
                    id.to_string().as_str(),
                    ElementKind::None,
                );

                let result = delete_element(&mut ctx.ui.menus, &element);
                match result {
                    Ok(ok) => CommandResult::Ok,
                    Err(err) => CommandResult::Error(err.to_string()),
                }
            }
            UiCommand::DeleteElementUndoable {
                element_ref,
                menu,
                layer,
                id,
            } => {
                let element = ElementRef::new(
                    menu.to_string().as_str(),
                    layer.to_string().as_str(),
                    id.to_string().as_str(),
                    ElementKind::None,
                );
                ctx.ui.ui_edit_manager.execute_command(
                    DeleteElementCommand {
                        affected_element: element,
                        cached_element: None,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.input.mouse,
                );
                CommandResult::Ok
            }
            UiCommand::SaveGame => {
                save_game(
                    ctx.game_state,
                    ctx.camera,
                    ctx.roads,
                    ctx.terrain,
                    ctx.props,
                );
                CommandResult::Ok
            }
            UiCommand::LoadSave {
                save_name,
                without_saving,
            } => {
                if !without_saving {
                    save_game(
                        ctx.game_state,
                        ctx.camera,
                        ctx.roads,
                        ctx.terrain,
                        ctx.props,
                    );
                }
                load_save(
                    ctx.game_state,
                    ctx.camera,
                    ctx.camera_controller,
                    ctx.roads,
                    ctx.terrain,
                    ctx.props,
                    save_name.as_str(),
                );
                CommandResult::Ok
            }
            UiCommand::ExitGame => {
                exit_game(
                    ctx.game_state,
                    ctx.settings,
                    ctx.time,
                    ctx.camera,
                    ctx.roads,
                    ctx.terrain,
                    ctx.props,
                    ctx.event_loop,
                );
                CommandResult::Ok
            }

            // ===== DEBUG COMMANDS =====
            UiCommand::Print { element_ref, args } => {
                let msg: String = args
                    .into_iter()
                    .map(|s| string_to_value(ctx, &element_ref, s).to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("[UI] {}", msg);
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
                    }
                }
                CommandResult::Ok
            }

            UiCommand::DebugActions => {
                println!("[Debug] Active action states:");
                for (name, state) in &ctx.ui.touch_manager.runtimes.action_states {
                    println!("  '{}': active={}", name, state.active);
                }
                CommandResult::Ok
            }

            // ===== EVENT COMMANDS =====
            UiCommand::EmitEvent {
                element_ref,
                event_name,
            } => {
                println!("[Event] {} from element {:?}", event_name, element_ref.id);
                ctx.ui
                    .variables
                    .set_var("_last_event", Value::String(event_name));
                ctx.ui
                    .variables
                    .set_var("_last_event_element", Value::String(element_ref.id.clone()));
                CommandResult::Ok
            }

            // ===== UTILITY =====
            UiCommand::Batch { commands } => {
                for cmd in commands.into_iter().rev() {
                    self.queue.push_front(cmd);
                }
                CommandResult::Ok
            }

            UiCommand::Noop => CommandResult::Ok,
        }
    }

    // ==================== CONTINUOUS ACTIONS ====================

    /// Execute continuous/frame-based actions. Call every frame after drain().
    pub fn execute_continuous(&mut self, ctx: &mut CommandContext) {
        let active_actions: Vec<String> = ctx
            .ui
            .touch_manager
            .runtimes
            .action_states
            .iter()
            .filter(|(_, state)| state.active)
            .map(|(name, _)| name.clone())
            .collect();

        for action_name in active_actions {
            match action_name.as_str() {
                "Drag Hue Point" => {
                    // drag_hue_point(ctx.ui, &ctx.input.mouse, ctx.time);
                }
                _ => {}
            }
        }
    }
}

fn string_to_value(ctx: &mut CommandContext, self_element_ref: &ElementRef, s: String) -> Value {
    send_element_properties_to_variables(
        &ctx.ui.menus,
        &mut ctx.ui.variables,
        &ctx.ui.touch_manager,
        self_element_ref,
    );
    let val = Value::from_str(ctx.settings, &ctx.ui.variables, s.as_str());
    //println!("Parse arg for Value in action_parser input: {}: {}", s, val);
    val
}
// ==================== PARSER ====================

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
    hit: &Option<HitResult>,
    input: &Input,
    time: &Time,
    terrain: &mut Terrain,
    props: &mut Props,
    window_size: PhysicalSize<u32>,
    roads: &mut Roads,
    settings: &mut Settings,
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    event_loop: &ActiveEventLoop,
    game_state: &mut GameState,
) {
    let mut ctx = CommandContext {
        ui,
        input,
        time,
        terrain,
        hit,
        window_size,
        settings,
        camera,
        camera_controller,
        event_loop,
        game_state,
        roads,
        props,
    };

    command_queue.drain(&mut ctx);
    command_queue.execute_continuous(&mut ctx);
}

/// Deactivate a continuous action by name.
pub fn deactivate_action(loader: &mut Ui, action_name: &str) {
    if let Some(state) = loader
        .touch_manager
        .runtimes
        .action_states
        .get_mut(action_name)
    {
        state.active = false;
    }
}

pub fn exit_game(
    game_state: &mut GameState,
    settings: &mut Settings,
    time: &Time,
    camera: &Camera,
    roads: &Roads,
    terrain: &Terrain,
    props: &Props,
    event_loop: &ActiveEventLoop,
) {
    settings.total_game_time = time.total_game_time;
    match settings.save(rusty_skylines_dir("settings.toml")) {
        Ok(_) => println!("Settings saved"),
        Err(e) => eprintln!("Failed to save Settings: {e}"),
    }
    save_game(game_state, camera, roads, terrain, props);

    event_loop.exit();
    //std::process::exit(69); // Die.
}
pub fn save_game(
    game_state: &mut GameState,
    camera: &Camera,
    roads: &Roads,
    terrain: &Terrain,
    props: &Props,
) {
    match game_state.save(camera, roads, terrain, props) {
        SaveResult::Success => println!("World saved"),
        e => eprintln!("Failed to save World: {:#?}", e),
    }
}
pub fn load_save(
    game_state: &mut GameState,
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    roads: &mut Roads,
    terrain: &mut Terrain,
    props: &mut Props,
    save_name: &str,
) {
    match game_state.load(save_name, camera, camera_controller, roads, terrain, props) {
        LoadResult::Success => println!(
            "World '{}' loaded, {} Terrain Edited Chunks, {} Road Nodes",
            game_state.current_save.name,
            game_state.current_save.terrain_edits.len(),
            game_state.current_save.roads.nodes.len()
        ),
        e => eprintln!("Failed to load World: {:#?}", e),
    }
}
/// ONLY USE EXECUTE_COMMAND SO IT EXECUTES IMMEDIATELY!!
pub fn set_element_property(
    ctx: &mut CommandContext,
    element_ref: ElementRef,
    name: &str,
    new_val: &Value,
) -> CommandResult {
    // ONLY USE EXECUTE_COMMAND SO IT EXECUTES IMMEDIATELY!!
    let Some((base, suffix)) = name.split_once('.') else {
        return CommandResult::AnnoyingError(format!(
            "set_element_property: invalid property name '{}', expected '.' separator",
            name
        ));
    };
    let (property, component) = match suffix.split_once('.') {
        Some((property, component)) => (property, component),
        None => (suffix, ""),
    };

    let selections: Vec<ElementRef>;
    match base {
        "self" => selections = vec![element_ref],
        "editing" => {
            selections = ctx.ui.touch_manager.selection.selected.clone();
        }
        _ => {
            return CommandResult::AnnoyingError(format!(
                "set_element_property: unknown base '{}', expected 'self' or 'editing'",
                base
            ));
        }
    }

    for element_ref in selections {
        match property {
            "center" => {
                let Some(before) = get_element_position(&ctx.ui.menus, &element_ref) else {
                    return CommandResult::Error(
                        format!("get_element_position() failed in set_element_property for element {:?}", element_ref).to_string(),
                    );
                };

                let after = match Variables::component_index(component) {
                    // Setting full vector
                    None => {
                        let Some(arr) = new_val.as_array() else {
                            return CommandResult::Error("set_element_property: center requires an array value when no component specified".to_string());
                        };
                        if arr.len() != 2 {
                            return CommandResult::Error(format!(
                                "set_element_property: center array must have exactly 2 elements, got {}",
                                arr.len()
                            ));
                        }

                        let Some(x) = arr[0].as_f64() else {
                            return CommandResult::Error(
                                "set_element_property: center array[0] is not a valid number"
                                    .to_string(),
                            );
                        };
                        let Some(y) = arr[1].as_f64() else {
                            return CommandResult::Error(
                                "set_element_property: center array[1] is not a valid number"
                                    .to_string(),
                            );
                        };

                        [x as f32, y as f32]
                    }

                    // Setting single component
                    Some(idx) => {
                        let Some(val) = new_val.as_f64() else {
                            return CommandResult::Error("set_element_property: center component value is not a valid number".to_string());
                        };

                        let mut after = before;
                        match idx {
                            0 => after[0] = val as f32,
                            1 => after[1] = val as f32,
                            _ => {
                                return CommandResult::Error(format!(
                                    "set_element_property: invalid center component index {}",
                                    idx
                                ));
                            }
                        }
                        after
                    }
                };
                //println!("Setting CENTER to: {:?}, with {:?} and {:?}", after, name, new_val);
                ctx.ui.ui_edit_manager.execute_command(
                    MoveElementCommand {
                        affected_element: element_ref,
                        before: None,
                        after,
                    },
                    &mut ctx.ui.touch_manager,
                    &mut ctx.ui.menus,
                    &mut ctx.ui.variables,
                    &ctx.input.mouse,
                );
            }

            "radius" => {
                let Some(radius) = new_val.as_f64() else {
                    return CommandResult::Error(
                        "set_element_property: radius value is not a valid number".to_string(),
                    );
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
                    &ctx.input.mouse,
                );
            }

            "color" => {
                let color_property = ColorComponent::from_str(component); // MSRV!!
                //println!("{}", new_val);
                let Some(new_color) = new_val.as_color4() else {
                    return CommandResult::Error(
                        ".as_color4() failed in set_element_property".to_string(),
                    );
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
                    &ctx.input.mouse,
                );
            }

            _ => {
                return CommandResult::Error(format!(
                    "set_element_property: unknown property '{}'",
                    property
                ));
            }
        }
    }
    CommandResult::Ok
}

pub fn make_element(id: String, kind: &ElementKind, center: [f32; 2]) -> Option<UiElement> {
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

pub fn send_element_properties_to_variables(
    menus: &HashMap<String, Menu>,
    variables: &mut Variables,
    touch_manager: &UiTouchManager,
    self_element_ref: &ElementRef,
) {
    if let Some(menu) = menus.get(&self_element_ref.menu) {
        variables.set_string("self.menu", self_element_ref.menu.clone());
        if let Some(layer) = menu
            .layers
            .iter()
            .find(|l| l.name == self_element_ref.layer)
        {
            variables.set_string("self.layer", self_element_ref.layer.clone());
            if let Some(element) = layer.find_element(self_element_ref.id.as_str()) {
                variables.set_string("self.id", self_element_ref.id.clone());
                variables.set_string("self.kind", self_element_ref.kind.to_string());
                //println!("Center: {:?}, Mouse: {:?}, Radius: {}", element.center(), ctx.input.mouse.pos.to_array(), element.size());
                variables.set_array("self.center", element.center()); // Vec2
                variables.set_var(
                    "self.radius",
                    element
                        .size()
                        .radius()
                        .map(|r| Value::F64(r as f64))
                        .unwrap_or(Value::Null),
                ); // f64
                variables.set_var(
                    "self.pt",
                    element
                        .size()
                        .pt()
                        .map(|r| Value::F64(r as f64))
                        .unwrap_or(Value::Null),
                ); // f64
                let size = element.size().value_size2().unwrap_or(Value::Null);
                //println!("{:?}", element.size());
                variables.set_var("self.size", size);
                variables.set_array(
                    "self.color_components",
                    element
                        .color_components()
                        .iter()
                        .map(|c| Value::String(c.to_string()))
                        .collect::<Vec<Value>>(),
                );
            }
        }
    }
    //println!("{:?}", variables.get("self.layer"));
}
