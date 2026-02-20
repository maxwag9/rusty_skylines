#![allow(dead_code, unused_variables)]
pub mod drag_hue_point;
use crate::resources::Time;
use crate::ui::actions::drag_hue_point::drag_hue_point;
use crate::ui::input::Input;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_text_editing::HitResult;
use crate::world::roads::road_structs::RoadStyleParams;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
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
        name: String,
        value: CommandArg,
    },
    IncVar {
        name: String,
        amount: f32,
    },
    DecVar {
        name: String,
        amount: f32,
    },
    MulVar {
        name: String,
        factor: f32,
    },
    ToggleBool {
        name: String,
    },
    Clamp {
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

    // ===== WORLD RENDERER COMMANDS =====
    SetPickRadius {
        radius: f32,
    },
    GrowPickRadius {
        amount: f32,
    },
    ShrinkPickRadius {
        amount: f32,
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
        event_name: String,
    },

    // ===== LEGACY/SPECIAL COMMANDS =====
    DragHuePoint,
    SetRoadsFourLanes {
        forward: usize,
        backward: usize,
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
    pub loader: &'a mut UiButtonLoader,
    pub input_state: &'a Input,
    pub time: &'a Time,
    pub world_renderer: &'a mut TerrainSubsystem,
    pub hit: &'a Option<HitResult>,
    pub window_size: PhysicalSize<u32>,
    pub road_style_params: &'a mut RoadStyleParams,
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

    /// Queue multiple commands.
    pub fn push_many(&mut self, cmds: impl IntoIterator<Item = UiCommand>) {
        for cmd in cmds {
            self.queue.push_back(cmd);
        }
    }

    /// Parse and queue commands from an action string.
    pub fn push_str(&mut self, action_str: &str) {
        let commands = parse_command_string(action_str);
        self.push_many(commands);
    }

    /// Queue commands from a UI hit result.
    pub(crate) fn push_from_hit(&mut self, hit: &HitResult) {
        if let Some(action_str) = &hit.action {
            if !action_str.is_empty() && action_str != "None" {
                self.push_str(action_str);
            }
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
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    menu.active = true;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::CloseMenu { menu_name } => {
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    menu.active = false;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::CloseAllMenus => {
                for (_, menu) in ctx.loader.menus.iter_mut() {
                    menu.active = false;
                }
                CommandResult::Ok
            }

            UiCommand::ToggleMenu { menu_name } => {
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    menu.active = !menu.active;
                    CommandResult::Ok
                } else {
                    CommandResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            UiCommand::MenuActive { menu_name } => {
                let is_active = ctx
                    .loader
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
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
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
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
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
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = !layer.active;
                        return CommandResult::Ok;
                    }
                    return CommandResult::Error(format!("Layer '{}' not found", layer_name));
                }
                CommandResult::Error(format!("Menu '{}' not found", menu_name))
            }

            // ===== VARIABLE COMMANDS =====
            UiCommand::SetVar { name, value } => {
                let resolved = self.resolve_arg(&value);
                self.variables.insert(name, resolved);
                CommandResult::Ok
            }

            UiCommand::IncVar { name, amount } => {
                let new_val = match self.variables.get(&name) {
                    Some(CommandArg::Float(f)) => CommandArg::Float(f + amount),
                    Some(CommandArg::Int(i)) => CommandArg::Float(*i as f32 + amount),
                    _ => CommandArg::Float(amount),
                };
                self.variables.insert(name, new_val);
                CommandResult::Ok
            }

            UiCommand::DecVar { name, amount } => {
                let new_val = match self.variables.get(&name) {
                    Some(CommandArg::Float(f)) => CommandArg::Float(f - amount),
                    Some(CommandArg::Int(i)) => CommandArg::Float(*i as f32 - amount),
                    _ => CommandArg::Float(-amount),
                };
                self.variables.insert(name, new_val);
                CommandResult::Ok
            }

            UiCommand::MulVar { name, factor } => {
                if let Some(CommandArg::Float(f)) = self.variables.get(&name) {
                    self.variables.insert(name, CommandArg::Float(f * factor));
                }
                CommandResult::Ok
            }

            UiCommand::ToggleBool { name } => {
                let new_val = match self.variables.get(&name) {
                    Some(CommandArg::Bool(b)) => CommandArg::Bool(!b),
                    _ => CommandArg::Bool(true),
                };
                self.variables.insert(name, new_val);
                CommandResult::Ok
            }

            UiCommand::Clamp { name, min, max } => {
                if let Some(CommandArg::Float(f)) = self.variables.get(&name).cloned() {
                    self.variables
                        .insert(name, CommandArg::Float(f.clamp(min, max)));
                }
                CommandResult::Ok
            }

            // ===== ACTION STATE COMMANDS =====
            UiCommand::StartAction { action_name } => {
                let state = ActionState::with_time(&action_name, ctx.time.total_time);
                ctx.loader
                    .touch_manager
                    .runtimes
                    .action_states
                    .insert(action_name, state);
                CommandResult::Ok
            }

            UiCommand::StopAction { action_name } => {
                if let Some(state) = ctx
                    .loader
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
                ctx.loader
                    .touch_manager
                    .runtimes
                    .action_states
                    .remove(&action_name);
                CommandResult::Ok
            }

            // ===== WORLD RENDERER COMMANDS =====
            UiCommand::SetPickRadius { radius } => {
                ctx.world_renderer.pick_radius_m = radius;
                CommandResult::Ok
            }

            UiCommand::GrowPickRadius { amount } => {
                ctx.world_renderer.pick_radius_m += amount;
                CommandResult::Ok
            }

            UiCommand::ShrinkPickRadius { amount } => {
                ctx.world_renderer.pick_radius_m =
                    (ctx.world_renderer.pick_radius_m - amount).max(0.1);
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
                for (name, menu) in &ctx.loader.menus {
                    println!("[Debug] Menu '{}': active={}", name, menu.active);
                    for layer in &menu.layers {
                        println!("  Layer '{}': active={}", layer.name, layer.active);
                    }
                }
                CommandResult::Ok
            }

            UiCommand::DebugActions => {
                println!("[Debug] Active action states:");
                for (name, state) in &ctx.loader.touch_manager.runtimes.action_states {
                    println!("  '{}': active={}", name, state.active);
                }
                CommandResult::Ok
            }

            // ===== EVENT COMMANDS =====
            UiCommand::EmitEvent { event_name } => {
                println!("[Event] {}", event_name);
                self.set_var("_last_event", CommandArg::String(event_name));
                CommandResult::Ok
            }

            // ===== LEGACY/SPECIAL COMMANDS =====
            UiCommand::DragHuePoint => {
                let state = ActionState::with_time("Drag Hue Point", ctx.time.total_time);
                ctx.loader
                    .touch_manager
                    .runtimes
                    .action_states
                    .insert("Drag Hue Point".to_string(), state);
                CommandResult::Ok
            }

            UiCommand::SetRoadsFourLanes { forward, backward } => {
                let mut road_type = ctx.road_style_params.road_type().clone();
                road_type.lanes_each_direction = (forward, backward);
                ctx.road_style_params.set_road_type(road_type);
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
            .loader
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
                    drag_hue_point(ctx.loader, &ctx.input_state.mouse, ctx.time);
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

/// Parsed raw action for intermediate representation.
#[derive(Debug, Clone)]
struct RawParsedAction {
    name: String,
    args: Vec<CommandArg>,
}

impl RawParsedAction {
    fn arg_str(&self, index: usize) -> Option<&str> {
        self.args.get(index).and_then(|a| a.as_str())
    }

    fn arg_string(&self, index: usize) -> String {
        self.arg_str(index).unwrap_or_default().to_string()
    }

    fn arg_float(&self, index: usize) -> Option<f32> {
        self.args.get(index).and_then(|a| a.as_float())
    }

    fn arg_float_or(&self, index: usize, default: f32) -> f32 {
        self.arg_float(index).unwrap_or(default)
    }

    fn arg_int(&self, index: usize) -> Option<i64> {
        self.args.get(index).and_then(|a| a.as_int())
    }

    fn arg_int_or(&self, index: usize, default: i64) -> i64 {
        self.arg_int(index).unwrap_or(default)
    }
}

/// Parse an action string into a list of commands.
pub fn parse_command_string(input: &str) -> Vec<UiCommand> {
    let raw_actions = parse_raw_action_chain(input);
    raw_actions
        .into_iter()
        .filter_map(convert_raw_to_command)
        .collect()
}

fn parse_raw_action_chain(input: &str) -> Vec<RawParsedAction> {
    let parts = split_by_delimiter(input, ';');
    parts
        .into_iter()
        .filter_map(|part| {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                parse_single_raw_action(trimmed)
            } else {
                None
            }
        })
        .collect()
}

fn parse_single_raw_action(input: &str) -> Option<RawParsedAction> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    if let Some(paren_start) = input.find('(') {
        let name = input[..paren_start].trim().to_string();
        if let Some(paren_end) = input.rfind(')') {
            let args_str = &input[paren_start + 1..paren_end];
            let args = parse_arguments(args_str);
            return Some(RawParsedAction { name, args });
        }
    }

    Some(RawParsedAction {
        name: input.to_string(),
        args: Vec::new(),
    })
}

fn parse_arguments(input: &str) -> Vec<CommandArg> {
    if input.trim().is_empty() {
        return Vec::new();
    }

    let parts = split_by_delimiter(input, ',');
    parts.iter().map(|s| parse_single_arg(s.trim())).collect()
}

fn parse_single_arg(input: &str) -> CommandArg {
    let s = input.trim();

    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return CommandArg::String(s[1..s.len() - 1].to_string());
    }

    if s.starts_with('$') {
        return CommandArg::Var(s[1..].to_string());
    }

    if s == "true" {
        return CommandArg::Bool(true);
    }
    if s == "false" {
        return CommandArg::Bool(false);
    }

    if let Ok(i) = s.parse::<i64>() {
        return CommandArg::Int(i);
    }

    if let Ok(f) = s.parse::<f32>() {
        return CommandArg::Float(f);
    }

    CommandArg::String(s.to_string())
}

fn split_by_delimiter(input: &str, delim: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut paren_depth = 0;

    for c in input.chars() {
        match c {
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = c;
                current.push(c);
            }
            c if c == quote_char && in_quotes => {
                in_quotes = false;
                current.push(c);
            }
            '(' if !in_quotes => {
                paren_depth += 1;
                current.push(c);
            }
            ')' if !in_quotes => {
                paren_depth = (paren_depth - 1).max(0);
                current.push(c);
            }
            c if c == delim && !in_quotes && paren_depth == 0 => {
                parts.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Convert a raw parsed action to a typed UiCommand.
fn convert_raw_to_command(raw: RawParsedAction) -> Option<UiCommand> {
    let cmd_type = UiCommandType::from_legacy_name(&raw.name)?;

    let cmd = match cmd_type {
        // ===== MENU =====
        UiCommandType::OpenMenu => UiCommand::OpenMenu {
            menu_name: raw.arg_string(0),
        },
        UiCommandType::CloseMenu => UiCommand::CloseMenu {
            menu_name: raw.arg_string(0),
        },
        UiCommandType::ToggleMenu => UiCommand::ToggleMenu {
            menu_name: raw.arg_string(0),
        },
        UiCommandType::MenuActive => UiCommand::MenuActive {
            menu_name: raw.arg_string(0),
        },

        // ===== LAYERS =====
        UiCommandType::OpenLayer => UiCommand::OpenLayer {
            menu_name: raw.arg_string(0),
            layer_name: raw.arg_string(1),
        },
        UiCommandType::CloseLayer => UiCommand::CloseLayer {
            menu_name: raw.arg_string(0),
            layer_name: raw.arg_string(1),
        },
        UiCommandType::ToggleLayer => UiCommand::ToggleLayer {
            menu_name: raw.arg_string(0),
            layer_name: raw.arg_string(1),
        },

        // ===== VARIABLES =====
        UiCommandType::SetVar => {
            let name = raw.arg_string(0);
            let value = raw
                .args
                .get(1)
                .cloned()
                .unwrap_or(CommandArg::String(String::new()));
            UiCommand::SetVar { name, value }
        }
        UiCommandType::IncVar => UiCommand::IncVar {
            name: raw.arg_string(0),
            amount: raw.arg_float_or(1, 1.0),
        },
        UiCommandType::DecVar => UiCommand::DecVar {
            name: raw.arg_string(0),
            amount: raw.arg_float_or(1, 1.0),
        },
        UiCommandType::MulVar => UiCommand::MulVar {
            name: raw.arg_string(0),
            factor: raw.arg_float_or(1, 1.0),
        },
        UiCommandType::ToggleBool => UiCommand::ToggleBool {
            name: raw.arg_string(0),
        },
        UiCommandType::Clamp => UiCommand::Clamp {
            name: raw.arg_string(0),
            min: raw.arg_float_or(1, 0.0),
            max: raw.arg_float_or(2, 1.0),
        },

        // ===== ACTION STATE =====
        UiCommandType::StartAction => UiCommand::StartAction {
            action_name: raw.arg_string(0),
        },
        UiCommandType::StopAction => UiCommand::StopAction {
            action_name: raw.arg_string(0),
        },
        UiCommandType::RemoveAction => UiCommand::RemoveAction {
            action_name: raw.arg_string(0),
        },

        // ===== WORLD RENDERER =====
        UiCommandType::SetPickRadius => UiCommand::SetPickRadius {
            radius: raw.arg_float_or(0, 10.0),
        },
        UiCommandType::GrowPickRadius => UiCommand::GrowPickRadius {
            amount: raw.arg_float_or(0, 10.0),
        },
        UiCommandType::ShrinkPickRadius => UiCommand::ShrinkPickRadius {
            amount: raw.arg_float_or(0, 10.0),
        },

        // ===== FLOW CONTROL =====
        UiCommandType::Delay => UiCommand::Delay {
            seconds: raw.arg_float_or(0, 0.0),
        },
        UiCommandType::Halt => UiCommand::Halt,
        UiCommandType::Skip => UiCommand::Skip {
            count: raw.arg_int_or(0, 1) as usize,
        },
        UiCommandType::If => {
            let condition = raw.args.get(0).cloned().unwrap_or(CommandArg::Bool(false));
            let then_str = raw.arg_string(1);
            let else_str = raw.arg_string(2);
            UiCommand::If {
                condition,
                then_branch: parse_command_string(&then_str),
                else_branch: parse_command_string(&else_str),
            }
        }
        UiCommandType::IfVarEq => {
            let var_name = raw.arg_string(0);
            let value = raw
                .args
                .get(1)
                .cloned()
                .unwrap_or(CommandArg::String(String::new()));
            let then_str = raw.arg_string(2);
            UiCommand::IfVarEq {
                var_name,
                value,
                then_branch: parse_command_string(&then_str),
            }
        }

        // ===== DEBUG =====
        UiCommandType::Print => UiCommand::Print {
            args: raw.args.clone(),
        },
        UiCommandType::DebugVars => UiCommand::DebugVars,
        UiCommandType::DebugMenus => UiCommand::DebugMenus,
        UiCommandType::DebugActions => UiCommand::DebugActions,

        // ===== EVENTS =====
        UiCommandType::EmitEvent => UiCommand::EmitEvent {
            event_name: raw.arg_string(0),
        },

        // ===== SPECIAL =====
        UiCommandType::DragHuePoint => UiCommand::DragHuePoint,
        UiCommandType::SetRoadsFourLanes => UiCommand::SetRoadsFourLanes {
            forward: raw.arg_int_or(0, 1) as usize,
            backward: raw.arg_int_or(1, 1) as usize,
        },

        // ===== NO-OP =====
        UiCommandType::Noop => UiCommand::Noop,
    };

    Some(cmd)
}

// ==================== BUILDER API ====================

impl UiCommand {
    /// Create an OpenMenu command.
    pub fn open_menu(name: impl Into<String>) -> Self {
        UiCommand::OpenMenu {
            menu_name: name.into(),
        }
    }

    /// Create a CloseMenu command.
    pub fn close_menu(name: impl Into<String>) -> Self {
        UiCommand::CloseMenu {
            menu_name: name.into(),
        }
    }

    /// Create a ToggleMenu command.
    pub fn toggle_menu(name: impl Into<String>) -> Self {
        UiCommand::ToggleMenu {
            menu_name: name.into(),
        }
    }

    /// Create an OpenLayer command.
    pub fn open_layer(menu: impl Into<String>, layer: impl Into<String>) -> Self {
        UiCommand::OpenLayer {
            menu_name: menu.into(),
            layer_name: layer.into(),
        }
    }

    /// Create a CloseLayer command.
    pub fn close_layer(menu: impl Into<String>, layer: impl Into<String>) -> Self {
        UiCommand::CloseLayer {
            menu_name: menu.into(),
            layer_name: layer.into(),
        }
    }

    /// Create a ToggleLayer command.
    pub fn toggle_layer(menu: impl Into<String>, layer: impl Into<String>) -> Self {
        UiCommand::ToggleLayer {
            menu_name: menu.into(),
            layer_name: layer.into(),
        }
    }

    /// Create a SetVar command.
    pub fn set_var(name: impl Into<String>, value: CommandArg) -> Self {
        UiCommand::SetVar {
            name: name.into(),
            value,
        }
    }

    /// Create an IncVar command.
    pub fn inc_var(name: impl Into<String>, amount: f32) -> Self {
        UiCommand::IncVar {
            name: name.into(),
            amount,
        }
    }

    /// Create a DecVar command.
    pub fn dec_var(name: impl Into<String>, amount: f32) -> Self {
        UiCommand::DecVar {
            name: name.into(),
            amount,
        }
    }

    /// Create a StartAction command.
    pub fn start_action(name: impl Into<String>) -> Self {
        UiCommand::StartAction {
            action_name: name.into(),
        }
    }

    /// Create a StopAction command.
    pub fn stop_action(name: impl Into<String>) -> Self {
        UiCommand::StopAction {
            action_name: name.into(),
        }
    }

    /// Create a Print command.
    pub fn print(args: Vec<CommandArg>) -> Self {
        UiCommand::Print { args }
    }

    /// Create a Delay command.
    pub fn delay(seconds: f32) -> Self {
        UiCommand::Delay { seconds }
    }

    /// Create a batch of commands.
    pub fn batch(commands: Vec<UiCommand>) -> Self {
        UiCommand::Batch { commands }
    }
}

// ==================== HELPER FUNCTIONS ====================

pub fn style_to_u32(style: &str) -> u32 {
    match style {
        "Hue Circle" => 1,
        _ => 0,
    }
}

// ==================== MAIN ENTRY POINTS ====================

/// Process commands and continuous actions. Call once per frame.
pub fn process_commands(
    command_queue: &mut CommandQueue,
    loader: &mut UiButtonLoader,
    top_hit: &Option<HitResult>,
    input_state: &Input,
    time: &Time,
    world_renderer: &mut TerrainSubsystem,
    window_size: PhysicalSize<u32>,
    road_style_params: &mut RoadStyleParams,
) {
    let mut ctx = CommandContext {
        loader,
        input_state,
        time,
        world_renderer,
        hit: top_hit,
        window_size,
        road_style_params,
    };

    command_queue.drain(&mut ctx);
    command_queue.execute_continuous(&mut ctx);
}

/// Queue commands from a UI hit (when element is clicked).
pub fn queue_from_hit(queue: &mut CommandQueue, hit: &Option<HitResult>) {
    if let Some(h) = hit {
        queue.push_from_hit(h);
    }
}

/// Deactivate a continuous action by name.
pub fn deactivate_action(loader: &mut UiButtonLoader, action_name: &str) {
    if let Some(state) = loader
        .touch_manager
        .runtimes
        .action_states
        .get_mut(action_name)
    {
        state.active = false;
    }
}
