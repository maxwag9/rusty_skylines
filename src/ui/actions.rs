#![allow(dead_code, unused_variables)]
pub mod drag_hue_point;

use crate::data::{SettingKey, SettingOp, Settings};
use crate::resources::Time;
use crate::ui::actions::drag_hue_point::drag_hue_point;
use crate::ui::input::Input;
use crate::ui::ui_editor::{Ui, get_layer_settings};
use crate::ui::ui_text_editing::HitResult;
use crate::ui::ui_touch_manager::ElementRef;
use crate::world::roads::road_structs::RoadStyleParams;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec2;
use std::collections::{HashMap, VecDeque};
use winit::dpi::PhysicalSize;
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

    // Legacy/Special
    DragHuePoint,
    SetRoadsFourLanes,

    // No-op
    Noop,
}

impl UiCommandType {
    /// Convert legacy string action names to explicit command types.
    pub fn from_legacy_name(name: &str) -> Option<Self> {
        let n = canonicalize_action_name(name);

        Some(match n.as_str() {
            // ===== MENU =====
            "open_menu" | "show_menu" => UiCommandType::OpenMenu,
            "close_menu" | "hide_menu" => UiCommandType::CloseMenu,
            "toggle_menu" => UiCommandType::ToggleMenu,
            "menu_active" => UiCommandType::MenuActive,

            // ===== LAYERS =====
            "open_layer" | "show_layer" => UiCommandType::OpenLayer,
            "close_layer" | "hide_layer" => UiCommandType::CloseLayer,
            "toggle_layer" => UiCommandType::ToggleLayer,

            // ===== VARIABLES =====
            "set" | "set_var" => UiCommandType::SetVar,
            "inc" | "increment" | "add" => UiCommandType::IncVar,
            "dec" | "decrement" | "sub" => UiCommandType::DecVar,
            "mul" | "multiply" => UiCommandType::MulVar,
            "toggle_bool" => UiCommandType::ToggleBool,
            "clamp" => UiCommandType::Clamp,

            // ===== ACTION STATE =====
            "start" | "activate" | "start_action" => UiCommandType::StartAction,
            "stop" | "deactivate" | "stop_action" => UiCommandType::StopAction,
            "remove_action" => UiCommandType::RemoveAction,

            // ===== WORLD RENDERER =====
            "set_pick_radius" => UiCommandType::SetPickRadius,
            "grow_pick_radius" => UiCommandType::GrowPickRadius,
            "shrink_pick_radius" => UiCommandType::ShrinkPickRadius,

            // ===== FLOW CONTROL =====
            "delay" | "wait" => UiCommandType::Delay,
            "halt" | "break" => UiCommandType::Halt,
            "skip" => UiCommandType::Skip,
            "if" => UiCommandType::If,
            "if_var_eq" => UiCommandType::IfVarEq,

            // ===== DEBUG =====
            "print" | "log" => UiCommandType::Print,
            "debug_vars" => UiCommandType::DebugVars,
            "debug_menus" => UiCommandType::DebugMenus,
            "debug_actions" => UiCommandType::DebugActions,

            // ===== EVENTS =====
            "emit" | "emit_event" => UiCommandType::EmitEvent,

            // ===== SPECIAL =====
            "drag_hue_point" | "drag_hue" | "drag_huepoint" => UiCommandType::DragHuePoint,
            "set_roads_four_lanes" => UiCommandType::SetRoadsFourLanes,

            // ===== NO-OP =====
            "" | "none" | "noop" => UiCommandType::Noop,

            _ => return None,
        })
    }

    /// Get the canonical name for this command type.
    pub fn canonical_name(self) -> &'static str {
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

            UiCommandType::DragHuePoint => "drag_hue_point",
            UiCommandType::SetRoadsFourLanes => "set_roads_four_lanes",

            UiCommandType::Noop => "noop",
        }
    }
}

// ==================== COMMAND ENUM ====================

/// A fully-specified UI command with all data embedded.
/// Can be queued and executed without the original parsing context.
#[derive(Debug, Clone)]
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
        value: CommandArg,
    },
    IncVar {
        element_ref: ElementRef,
        name: String,
        amount: f32,
    },
    DecVar {
        element_ref: ElementRef,
        name: String,
        amount: f32,
    },
    MulVar {
        element_ref: ElementRef,
        name: String,
        factor: f32,
    },
    ToggleVar {
        element_ref: ElementRef,
        name: String,
    },
    Clamp {
        element_ref: ElementRef,
        name: String,
        min: f32,
        max: f32,
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
        seconds: f32,
    },
    Halt,
    Skip {
        count: usize,
    },
    If {
        condition: CommandArg,
        then_branch: Vec<UiCommand>,
        else_branch: Vec<UiCommand>,
    },
    IfVarEq {
        var_name: String,
        value: CommandArg,
        then_branch: Vec<UiCommand>,
    },

    // ===== DEBUG COMMANDS =====
    Print {
        args: Vec<CommandArg>,
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
}

// ==================== COMMAND ARGUMENTS ====================

#[derive(Debug, Clone, PartialEq)]
pub enum CommandArg {
    String(String),
    Float(f32),
    Int(i64),
    Bool(bool),
    Var(String),
}

impl CommandArg {
    pub fn from_str(s: &str) -> Self {
        let s = s.trim();

        if s.is_empty() {
            return CommandArg::String(String::new());
        }

        // =========================
        // Explicit type prefix
        // =========================
        if let Some((ty, value)) = s.split_once(':') {
            match ty.to_ascii_lowercase().as_str() {
                "int" => {
                    if let Ok(i) = value.parse::<i64>() {
                        return CommandArg::Int(i);
                    }
                }
                "float" | "f32" => {
                    if let Ok(f) = value.parse::<f32>() {
                        return CommandArg::Float(f);
                    }
                }
                "bool" => match value.to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => return CommandArg::Bool(true),
                    "false" | "0" | "no" | "off" => return CommandArg::Bool(false),
                    _ => {}
                },
                "string" | "str" => {
                    return CommandArg::String(value.to_string());
                }
                "var" => {
                    return CommandArg::Var(value.to_string());
                }
                _ => {}
            }
        }

        // =========================
        // Auto-detect bool
        // =========================
        match s.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => return CommandArg::Bool(true),
            "false" | "no" | "off" => return CommandArg::Bool(false),
            _ => {}
        }

        // =========================
        // Auto-detect integer
        // =========================
        if let Ok(i) = s.parse::<i64>() {
            return CommandArg::Int(i);
        }

        // =========================
        // Auto-detect float
        // =========================
        if let Ok(f) = s.parse::<f32>() {
            return CommandArg::Float(f);
        }

        // =========================
        // Default: String
        // =========================
        CommandArg::String(s.to_string())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            CommandArg::String(s) | CommandArg::Var(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f32> {
        match self {
            CommandArg::Float(f) => Some(*f),
            CommandArg::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            CommandArg::Int(i) => Some(*i),
            CommandArg::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            CommandArg::Bool(b) => Some(*b),
            CommandArg::Int(i) => Some(*i != 0),
            _ => None,
        }
    }

    pub fn to_string_value(&self) -> String {
        match self {
            CommandArg::String(s) => s.clone(),
            CommandArg::Float(f) => f.to_string(),
            CommandArg::Int(i) => i.to_string(),
            CommandArg::Bool(b) => b.to_string(),
            CommandArg::Var(v) => format!("${}", v),
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            CommandArg::Bool(b) => *b,
            CommandArg::Int(i) => *i != 0,
            CommandArg::Float(f) => *f != 0.0,
            CommandArg::String(s) => !s.is_empty() && s != "false" && s != "0",
            CommandArg::Var(_) => true,
        }
    }
}

// ==================== ACTION STATE ====================

#[derive(Debug, Clone)]
pub struct ActionState {
    pub action_name: String,
    pub active: bool,
    pub started_at: f64,
    pub position: Option<Vec2>,
    pub last_pos: Option<Vec2>,
    pub custom_data: HashMap<String, CommandArg>,
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

    pub fn set_data(&mut self, key: &str, value: CommandArg) {
        self.custom_data.insert(key.to_string(), value);
    }

    pub fn get_data(&self, key: &str) -> Option<&CommandArg> {
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

#[derive(Debug, Clone)]
pub enum CommandResult {
    Ok,
    Stop,
    Delay {
        seconds: f32,
        remaining: Vec<UiCommand>,
    },
    Skip(usize),
    Error(String),
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
    pub road_style_params: &'a mut RoadStyleParams,
    pub settings: &'a mut Settings,
}

// ==================== COMMAND QUEUE ====================

/// The central command queue that processes commands.
/// Commands are queued without context, then drained with context each frame.
pub struct CommandQueue {
    queue: VecDeque<UiCommand>,
    delayed: Vec<DelayedCommands>,
    pub variables: HashMap<String, CommandArg>,
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
            variables: HashMap::new(),
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

    // ==================== VARIABLE MANAGEMENT ====================

    pub fn set_var(&mut self, name: &str, value: CommandArg) {
        self.variables.insert(name.to_string(), value);
    }

    pub fn get_var(&self, name: &str) -> Option<&CommandArg> {
        self.variables.get(name)
    }

    pub fn get_var_float(&self, name: &str) -> Option<f32> {
        self.variables.get(name).and_then(|v| v.as_float())
    }

    pub fn get_var_string(&self, name: &str) -> Option<&str> {
        self.variables.get(name).and_then(|v| v.as_str())
    }

    pub fn resolve_arg(&self, arg: &CommandArg) -> CommandArg {
        match arg {
            CommandArg::Var(name) => self
                .variables
                .get(name)
                .cloned()
                .unwrap_or(CommandArg::String(String::new())),
            other => other.clone(),
        }
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
                            execute_at: ctx.time.total_time + seconds as f64,
                        });
                    }
                    break;
                }
                CommandResult::Error(msg) => {
                    eprintln!("[CommandQueue] Error: {}", msg);
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
                self.set_var("_result", CommandArg::Bool(is_active));
                CommandResult::Ok
            }

            // ===== LAYER COMMANDS =====
            UiCommand::OpenLayer {
                menu_name,
                layer_name,
            } => {
                if let Some(menu) = ctx.ui.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
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
                    let resolved = self.resolve_arg(&value);
                    self.variables.insert(name, resolved);
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

                // Secondary: AV
                if name == "av" {
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
                let new_val = match self.variables.get(&name) {
                    Some(CommandArg::Float(f)) => CommandArg::Float(f + amount),
                    Some(CommandArg::Int(i)) => CommandArg::Float(*i as f32 + amount),
                    _ => CommandArg::Float(amount),
                };
                self.variables.insert(name, new_val);
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

                // Secondary: AV
                if name == "av" {
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
                let new_val = match self.variables.get(&name) {
                    Some(CommandArg::Float(f)) => CommandArg::Float(f - amount),
                    Some(CommandArg::Int(i)) => CommandArg::Float(*i as f32 - amount),
                    _ => CommandArg::Float(-amount),
                };
                self.variables.insert(name, new_val);
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

                // Secondary: AV
                if name == "av" {
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
                if let Some(CommandArg::Float(f)) = self.variables.get(&name) {
                    self.variables.insert(name, CommandArg::Float(f * factor));
                }
                CommandResult::Ok
            }

            UiCommand::ToggleVar { element_ref, name } => {
                // Primary: Settings
                if let Some(key) = SettingKey::from_str(&name) {
                    ctx.settings.apply_setting(key, SettingOp::Toggle);
                    return CommandResult::Ok;
                }

                // Secondary: AV
                if name == "av" {
                    if let Some(setting) = get_layer_settings(&ctx.ui.menus, &element_ref) {
                        ctx.settings.apply_setting(setting.key, SettingOp::Toggle);
                        return CommandResult::Ok;
                    }
                }

                // Ternary: Variables
                let new_val = match self.variables.get(&name) {
                    Some(CommandArg::Bool(b)) => CommandArg::Bool(!b),
                    _ => CommandArg::Bool(true),
                };
                self.variables.insert(name, new_val);
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

                // Secondary: AV
                if name == "av" {
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
                if let Some(CommandArg::Float(f)) = self.variables.get(&name).cloned() {
                    self.variables
                        .insert(name, CommandArg::Float(f.clamp(min, max)));
                }
                CommandResult::Ok
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
                condition,
                then_branch,
                else_branch,
            } => {
                let resolved = self.resolve_arg(&condition);
                if resolved.is_truthy() {
                    for cmd in then_branch.into_iter().rev() {
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
                var_name,
                value,
                then_branch,
            } => {
                let current = self.variables.get(&var_name).cloned();
                let compare_to = Some(self.resolve_arg(&value));
                if current == compare_to {
                    for cmd in then_branch.into_iter().rev() {
                        self.queue.push_front(cmd);
                    }
                }
                CommandResult::Ok
            }

            // ===== DEBUG COMMANDS =====
            UiCommand::Print { args } => {
                let msg: String = args
                    .iter()
                    .map(|a| self.resolve_arg(a).to_string_value())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("[UI] {}", msg);
                CommandResult::Ok
            }

            UiCommand::DebugVars => {
                println!("[Debug] Variables: {:?}", self.variables);
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
                self.set_var("_last_event", CommandArg::String(event_name));
                self.set_var(
                    "_last_event_element",
                    CommandArg::String(element_ref.id.clone()),
                );
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
                    drag_hue_point(ctx.ui, &ctx.input.mouse, ctx.time);
                }
                _ => {}
            }
        }
    }
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
    loader: &mut Ui,
    top_hit: &Option<HitResult>,
    input_state: &Input,
    time: &Time,
    world_renderer: &mut Terrain,
    window_size: PhysicalSize<u32>,
    road_style_params: &mut RoadStyleParams,
    settings: &mut Settings,
) {
    let mut ctx = CommandContext {
        ui: loader,
        input: input_state,
        time,
        terrain: world_renderer,
        hit: top_hit,
        window_size,
        road_style_params,
        settings,
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
