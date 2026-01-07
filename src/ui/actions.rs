pub mod drag_hue_point;

use crate::data::BendMode;
use crate::hsv::{HSV, hsv_to_rgb, rgb_to_hsv};
use crate::renderer::world_renderer::WorldRenderer;
use crate::resources::{InputState, TimeSystem};
use crate::ui::actions::drag_hue_point::drag_hue_point;
use crate::ui::input::MouseState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_loader::load_menus_from_directory;
use crate::ui::ui_text_editing::HitResult;
use glam::Vec2;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use winit::dpi::PhysicalSize;
// ==================== ACTION ARGUMENTS ====================

#[derive(Debug, Clone, PartialEq)]
pub enum ActionArg {
    String(String),
    Float(f32),
    Int(i64),
    Bool(bool),
    Var(String), // Variable reference: $my_var
}

impl ActionArg {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ActionArg::String(s) | ActionArg::Var(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f32> {
        match self {
            ActionArg::Float(f) => Some(*f),
            ActionArg::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            ActionArg::Int(i) => Some(*i),
            ActionArg::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ActionArg::Bool(b) => Some(*b),
            ActionArg::Int(i) => Some(*i != 0),
            _ => None,
        }
    }

    pub fn to_string_value(&self) -> String {
        match self {
            ActionArg::String(s) => s.clone(),
            ActionArg::Float(f) => f.to_string(),
            ActionArg::Int(i) => i.to_string(),
            ActionArg::Bool(b) => b.to_string(),
            ActionArg::Var(v) => format!("${}", v),
        }
    }
}

// ==================== PARSED ACTION ====================

#[derive(Debug, Clone)]
pub struct ParsedAction {
    pub name: String,
    pub args: Vec<ActionArg>,
}

impl ParsedAction {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            args: Vec::new(),
        }
    }

    pub fn with_args(name: &str, args: Vec<ActionArg>) -> Self {
        Self {
            name: name.to_string(),
            args,
        }
    }

    pub fn arg_str(&self, index: usize) -> Option<&str> {
        self.args.get(index).and_then(|a| a.as_str())
    }

    pub fn arg_string(&self, index: usize) -> String {
        self.arg_str(index).unwrap_or_default().to_string()
    }

    pub fn arg_float(&self, index: usize) -> Option<f32> {
        self.args.get(index).and_then(|a| a.as_float())
    }

    pub fn arg_float_or(&self, index: usize, default: f32) -> f32 {
        self.arg_float(index).unwrap_or(default)
    }

    pub fn arg_int(&self, index: usize) -> Option<i64> {
        self.args.get(index).and_then(|a| a.as_int())
    }

    pub fn arg_bool(&self, index: usize) -> Option<bool> {
        self.args.get(index).and_then(|a| a.as_bool())
    }

    pub fn arg_bool_or(&self, index: usize, default: bool) -> bool {
        self.arg_bool(index).unwrap_or(default)
    }
}

// ==================== ACTION RESULT ====================

#[derive(Debug, Clone)]
pub enum ActionResult {
    /// Action completed successfully, continue chain
    Ok,
    /// Stop executing the action chain
    Stop,
    /// Delay remaining actions by specified seconds
    Delay(f32),
    /// Skip the next N actions in the chain
    Skip(usize),
    /// Error occurred
    Error(String),
}

// ==================== ACTION STATE ====================

#[derive(Debug, Clone)]
pub struct ActionState {
    pub action_name: String,
    pub active: bool,
    pub started_at: f64,
    pub position: Option<Vec2>,
    pub last_pos: Option<Vec2>,
    pub custom_data: HashMap<String, ActionArg>,
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

    pub fn set_data(&mut self, key: &str, value: ActionArg) {
        self.custom_data.insert(key.to_string(), value);
    }

    pub fn get_data(&self, key: &str) -> Option<&ActionArg> {
        self.custom_data.get(key)
    }
}

// ==================== DELAYED ACTION ====================

#[derive(Debug, Clone)]
struct DelayedAction {
    actions: Vec<ParsedAction>,
    execute_at: f64,
    source: String, // For debugging
}

// ==================== ACTION CONTEXT ====================

pub struct ActionContext<'a> {
    pub loader: &'a mut UiButtonLoader,
    pub mouse_state: &'a MouseState,
    pub input_state: &'a InputState,
    pub time: &'a TimeSystem,
    pub world_renderer: &'a mut WorldRenderer,
    pub hit: &'a Option<HitResult>,
    pub window_size: PhysicalSize<u32>,
}

// ==================== ACTION HANDLER TYPE ====================

pub type ActionHandler =
    Arc<dyn Fn(&ParsedAction, &mut ActionContext, &mut ActionSystem) -> ActionResult + Send + Sync>;

// ==================== ACTION SYSTEM ====================

pub struct ActionSystem {
    handlers: HashMap<String, ActionHandler>,
    delayed: Vec<DelayedAction>,
    pub variables: HashMap<String, ActionArg>,
    pub last_result: Option<ActionResult>,
}

impl Default for ActionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionSystem {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            delayed: Vec::new(),
            variables: HashMap::new(),
            last_result: None,
        }
    }

    // ==================== HANDLER REGISTRATION ====================

    /// Register a custom action handler
    pub fn register<F>(&mut self, name: &str, handler: F)
    where
        F: Fn(&ParsedAction, &mut ActionContext, &mut ActionSystem) -> ActionResult
            + Send
            + Sync
            + 'static,
    {
        self.handlers.insert(name.to_lowercase(), Arc::new(handler));
    }

    /// Register a simple action (no access to ActionSystem)
    pub fn register_simple<F>(&mut self, name: &str, handler: F)
    where
        F: Fn(&ParsedAction, &mut ActionContext) -> ActionResult + Send + Sync + 'static,
    {
        self.handlers.insert(
            name.to_lowercase(),
            Arc::new(move |action, ctx, _sys| handler(action, ctx)),
        );
    }

    // ==================== VARIABLE MANAGEMENT ====================

    pub fn set_var(&mut self, name: &str, value: ActionArg) {
        self.variables.insert(name.to_string(), value);
    }

    pub fn get_var(&self, name: &str) -> Option<&ActionArg> {
        self.variables.get(name)
    }

    pub fn get_var_float(&self, name: &str) -> Option<f32> {
        self.variables.get(name).and_then(|v| v.as_float())
    }

    pub fn get_var_string(&self, name: &str) -> Option<&str> {
        self.variables.get(name).and_then(|v| v.as_str())
    }

    /// Resolve an argument, replacing variable references with their values
    pub fn resolve_arg(&self, arg: &ActionArg) -> ActionArg {
        match arg {
            ActionArg::Var(name) => self
                .variables
                .get(name)
                .cloned()
                .unwrap_or(ActionArg::String(String::new())),
            other => other.clone(),
        }
    }

    // ==================== EXECUTION ====================

    /// Parse and execute an action string (supports chaining with `;`)
    pub fn run(&mut self, action_str: &str, ctx: &mut ActionContext) {
        let actions = parse_action_chain(action_str);
        self.execute_chain(actions, ctx, action_str);
    }

    /// Execute a chain of parsed actions
    fn execute_chain(&mut self, actions: Vec<ParsedAction>, ctx: &mut ActionContext, source: &str) {
        let mut skip_count = 0;

        for (i, action) in actions.iter().enumerate() {
            if skip_count > 0 {
                skip_count -= 1;
                continue;
            }

            let result = self.execute_single(action, ctx);
            self.last_result = Some(result.clone());

            match result {
                ActionResult::Ok => continue,
                ActionResult::Stop => break,
                ActionResult::Skip(n) => {
                    skip_count = n;
                }
                ActionResult::Delay(seconds) => {
                    let remaining: Vec<ParsedAction> = actions[i + 1..].to_vec();
                    if !remaining.is_empty() {
                        self.delayed.push(DelayedAction {
                            actions: remaining,
                            execute_at: ctx.time.total_time + seconds as f64,
                            source: source.to_string(),
                        });
                    }
                    break;
                }
                ActionResult::Error(msg) => {
                    eprintln!("[ActionSystem] Error in '{}': {}", action.name, msg);
                }
            }
        }
    }

    /// Execute a single action
    fn execute_single(&mut self, action: &ParsedAction, ctx: &mut ActionContext) -> ActionResult {
        let name = action.name.to_lowercase();

        // Check custom handlers first
        if let Some(handler) = self.handlers.get(&name).cloned() {
            return handler(action, ctx, self); // now mutable borrow is fine
        }

        // Otherwise use built-in handlers
        self.execute_builtin(&name, action, ctx)
    }

    /// Process delayed actions (call each frame)
    pub fn update(&mut self, ctx: &mut ActionContext) {
        let current_time = ctx.time.total_time;

        let ready: Vec<DelayedAction> = self
            .delayed
            .iter()
            .filter(|d| d.execute_at <= current_time)
            .cloned()
            .collect();

        self.delayed.retain(|d| d.execute_at > current_time);

        for delayed in ready {
            self.execute_chain(delayed.actions, ctx, &delayed.source);
        }
    }

    // ==================== BUILT-IN ACTIONS ====================

    fn execute_builtin(
        &mut self,
        name: &str,
        action: &ParsedAction,
        ctx: &mut ActionContext,
    ) -> ActionResult {
        match name {
            // ===== MENU ACTIONS =====
            "open_menu" | "show_menu" => {
                let menu_name = action.arg_string(0);
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    menu.active = true;
                    ActionResult::Ok
                } else {
                    ActionResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            "close_menu" | "hide_menu" => {
                let menu_name = action.arg_string(0);
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    menu.active = false;
                    ActionResult::Ok
                } else {
                    ActionResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            "toggle_menu" => {
                let menu_name = action.arg_string(0);
                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    menu.active = !menu.active;
                    ActionResult::Ok
                } else {
                    ActionResult::Error(format!("Menu '{}' not found", menu_name))
                }
            }

            "menu_active" => {
                let menu_name = action.arg_string(0);
                let is_active = ctx
                    .loader
                    .menus
                    .get(&menu_name)
                    .map(|m| m.active)
                    .unwrap_or(false);
                self.set_var("_result", ActionArg::Bool(is_active));
                ActionResult::Ok
            }

            // ===== LAYER ACTIONS =====
            "open_layer" | "show_layer" => {
                let menu_name = action.arg_string(0);
                let layer_name = action.arg_string(1);

                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = true;
                        return ActionResult::Ok;
                    }
                    return ActionResult::Error(format!(
                        "Layer '{}' not found in '{}'",
                        layer_name, menu_name
                    ));
                }
                ActionResult::Error(format!("Menu '{}' not found", menu_name))
            }

            "close_layer" | "hide_layer" => {
                let menu_name = action.arg_string(0);
                let layer_name = action.arg_string(1);

                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = false;
                        return ActionResult::Ok;
                    }
                    return ActionResult::Error(format!("Layer '{}' not found", layer_name));
                }
                ActionResult::Error(format!("Menu '{}' not found", menu_name))
            }

            "toggle_layer" => {
                let menu_name = action.arg_string(0);
                let layer_name = action.arg_string(1);

                if let Some(menu) = ctx.loader.menus.get_mut(&menu_name) {
                    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
                        layer.active = !layer.active;
                        return ActionResult::Ok;
                    }
                    return ActionResult::Error(format!("Layer '{}' not found", layer_name));
                }
                ActionResult::Error(format!("Menu '{}' not found", menu_name))
            }

            // ===== VARIABLE ACTIONS =====
            "set" | "set_var" => {
                let var_name = action.arg_string(0);
                if let Some(value) = action.args.get(1) {
                    self.variables.insert(var_name, self.resolve_arg(value));
                    ActionResult::Ok
                } else {
                    ActionResult::Error("set requires 2 arguments: name, value".to_string())
                }
            }

            "inc" | "increment" | "add" => {
                let var_name = action.arg_string(0);
                let amount = action.arg_float_or(1, 1.0);

                let new_val = match self.variables.get(&var_name) {
                    Some(ActionArg::Float(f)) => ActionArg::Float(f + amount),
                    Some(ActionArg::Int(i)) => ActionArg::Float(*i as f32 + amount),
                    _ => ActionArg::Float(amount),
                };
                self.variables.insert(var_name, new_val);
                ActionResult::Ok
            }

            "dec" | "decrement" | "sub" => {
                let var_name = action.arg_string(0);
                let amount = action.arg_float_or(1, 1.0);

                let new_val = match self.variables.get(&var_name) {
                    Some(ActionArg::Float(f)) => ActionArg::Float(f - amount),
                    Some(ActionArg::Int(i)) => ActionArg::Float(*i as f32 - amount),
                    _ => ActionArg::Float(-amount),
                };
                self.variables.insert(var_name, new_val);
                ActionResult::Ok
            }

            "mul" | "multiply" => {
                let var_name = action.arg_string(0);
                let amount = action.arg_float_or(1, 1.0);

                if let Some(ActionArg::Float(f)) = self.variables.get(&var_name) {
                    self.variables
                        .insert(var_name, ActionArg::Float(f * amount));
                }
                ActionResult::Ok
            }

            "toggle_bool" => {
                let var_name = action.arg_string(0);
                let new_val = match self.variables.get(&var_name) {
                    Some(ActionArg::Bool(b)) => ActionArg::Bool(!b),
                    _ => ActionArg::Bool(true),
                };
                self.variables.insert(var_name, new_val);
                ActionResult::Ok
            }

            "clamp" => {
                let var_name = action.arg_string(0);
                let min = action.arg_float_or(1, 0.0);
                let max = action.arg_float_or(2, 1.0);

                if let Some(ActionArg::Float(f)) = self.variables.get(&var_name).cloned() {
                    self.variables
                        .insert(var_name, ActionArg::Float(f.clamp(min, max)));
                }
                ActionResult::Ok
            }

            // ===== ACTION STATE MANAGEMENT =====
            "start" | "activate" | "start_action" => {
                let action_name = action.arg_string(0);
                let state = ActionState::with_time(&action_name, ctx.time.total_time);
                ctx.loader
                    .ui_runtime
                    .action_states
                    .insert(action_name, state);
                ActionResult::Ok
            }

            "stop" | "deactivate" | "stop_action" => {
                let action_name = action.arg_string(0);
                if let Some(state) = ctx.loader.ui_runtime.action_states.get_mut(&action_name) {
                    state.active = false;
                }
                ActionResult::Ok
            }

            "remove_action" => {
                let action_name = action.arg_string(0);
                ctx.loader.ui_runtime.action_states.remove(&action_name);
                ActionResult::Ok
            }

            // ===== WORLD RENDERER ACTIONS =====
            "set_pick_radius" => {
                if let Some(radius) = action.arg_float(0) {
                    ctx.world_renderer.pick_radius_m = radius;
                    ActionResult::Ok
                } else {
                    ActionResult::Error("set_pick_radius requires a number".to_string())
                }
            }

            "grow_pick_radius" => {
                let amount = action.arg_float_or(0, 10.0);
                ctx.world_renderer.pick_radius_m += amount;
                ActionResult::Ok
            }

            "shrink_pick_radius" => {
                let amount = action.arg_float_or(0, 10.0);
                ctx.world_renderer.pick_radius_m =
                    (ctx.world_renderer.pick_radius_m - amount).max(0.1);
                ActionResult::Ok
            }

            // ===== FLOW CONTROL =====
            "delay" | "wait" => {
                let seconds = action.arg_float_or(0, 0.0);
                ActionResult::Delay(seconds)
            }

            "halt" | "break" => ActionResult::Stop,

            "skip" => {
                let count = action.arg_int(0).unwrap_or(1) as usize;
                ActionResult::Skip(count)
            }

            "if" => {
                // if($var, action_if_true, action_if_false)
                let condition = action.args.get(0).map(|a| self.resolve_arg(a));
                let is_true = match condition {
                    Some(ActionArg::Bool(b)) => b,
                    Some(ActionArg::Int(i)) => i != 0,
                    Some(ActionArg::Float(f)) => f != 0.0,
                    Some(ActionArg::String(s)) => !s.is_empty() && s != "false" && s != "0",
                    _ => false,
                };

                if is_true {
                    if let Some(ActionArg::String(then_action)) = action.args.get(1) {
                        let parsed = parse_action_chain(then_action);
                        self.execute_chain(parsed, ctx, then_action);
                    }
                } else {
                    if let Some(ActionArg::String(else_action)) = action.args.get(2) {
                        let parsed = parse_action_chain(else_action);
                        self.execute_chain(parsed, ctx, else_action);
                    }
                }
                ActionResult::Ok
            }

            "if_var_eq" => {
                // if_var_eq(var_name, value, action_if_true)
                let var_name = action.arg_string(0);
                let compare_to = action.args.get(1).map(|a| self.resolve_arg(a));
                let current = self.variables.get(&var_name).cloned();

                if current == compare_to {
                    if let Some(ActionArg::String(then_action)) = action.args.get(2) {
                        let parsed = parse_action_chain(then_action);
                        self.execute_chain(parsed, ctx, then_action);
                    }
                }
                ActionResult::Ok
            }

            // ===== DEBUG ACTIONS =====
            "print" | "log" => {
                let msg: String = action
                    .args
                    .iter()
                    .map(|a| {
                        let resolved = self.resolve_arg(a);
                        resolved.to_string_value()
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("[UI] {}", msg);
                ActionResult::Ok
            }

            "debug_vars" => {
                println!("[Debug] Variables: {:?}", self.variables);
                ActionResult::Ok
            }

            "debug_menus" => {
                for (name, menu) in &ctx.loader.menus {
                    println!("[Debug] Menu '{}': active={}", name, menu.active);
                    for layer in &menu.layers {
                        println!("  Layer '{}': active={}", layer.name, layer.active);
                    }
                }
                ActionResult::Ok
            }

            "debug_actions" => {
                println!("[Debug] Active action states:");
                for (name, state) in &ctx.loader.ui_runtime.action_states {
                    println!("  '{}': active={}", name, state.active);
                }
                ActionResult::Ok
            }

            // ===== EVENT SYSTEM =====
            "emit" | "emit_event" => {
                let event = action.arg_string(0);
                // You can extend this with an actual event system
                println!("[Event] {}", event);
                self.set_var("_last_event", ActionArg::String(event));
                ActionResult::Ok
            }

            // ===== SPECIAL LEGACY ACTIONS =====
            "drag_hue_point" => {
                let state = ActionState::with_time("Drag Hue Point", ctx.time.total_time);
                ctx.loader
                    .ui_runtime
                    .action_states
                    .insert("Drag Hue Point".to_string(), state);
                ActionResult::Ok
            }

            // ===== NO-OP =====
            "none" | "noop" | "" => ActionResult::Ok,

            _ => ActionResult::Error(format!("Unknown action: '{}'", name)),
        }
    }
}

// ==================== PARSER ====================

/// Parse an action chain: `open_menu("Editor"); toggle_layer("Editor", "Tools"); delay(0.5)`
pub fn parse_action_chain(input: &str) -> Vec<ParsedAction> {
    let mut actions = Vec::new();
    let parts = split_by_delimiter(input, ';');

    for part in parts {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            if let Some(action) = parse_single_action(trimmed) {
                actions.push(action);
            }
        }
    }

    actions
}

fn parse_single_action(input: &str) -> Option<ParsedAction> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    // Find opening parenthesis
    if let Some(paren_start) = input.find('(') {
        let name = input[..paren_start].trim().to_string();

        // Find matching close
        if let Some(paren_end) = input.rfind(')') {
            let args_str = &input[paren_start + 1..paren_end];
            let args = parse_arguments(args_str);
            return Some(ParsedAction::with_args(&name, args));
        }
    }

    // No parens = just a command name
    Some(ParsedAction::new(input))
}

fn parse_arguments(input: &str) -> Vec<ActionArg> {
    if input.trim().is_empty() {
        return Vec::new();
    }

    let parts = split_by_delimiter(input, ',');
    parts.iter().map(|s| parse_single_arg(s.trim())).collect()
}

fn parse_single_arg(input: &str) -> ActionArg {
    let s = input.trim();

    // Quoted string
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return ActionArg::String(s[1..s.len() - 1].to_string());
    }

    // Variable reference
    if s.starts_with('$') {
        return ActionArg::Var(s[1..].to_string());
    }

    // Booleans
    if s == "true" {
        return ActionArg::Bool(true);
    }
    if s == "false" {
        return ActionArg::Bool(false);
    }

    // Integer
    if let Ok(i) = s.parse::<i64>() {
        return ActionArg::Int(i);
    }

    // Float
    if let Ok(f) = s.parse::<f32>() {
        return ActionArg::Float(f);
    }

    // Unquoted string (identifier)
    ActionArg::String(s.to_string())
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

// ==================== HELPER FUNCTION ====================

pub fn style_to_u32(style: &str) -> u32 {
    match style {
        "Hue Circle" => 1,
        _ => 0,
    }
}

// ==================== MAIN ENTRY POINTS ====================

/// Execute continuous actions each frame
pub fn execute_action(
    action_system: &mut ActionSystem,
    loader: &mut UiButtonLoader,
    top_hit: &Option<HitResult>,
    mouse_state: &MouseState,
    input_state: &InputState,
    time: &TimeSystem,
    world_renderer: &mut WorldRenderer,
    window_size: PhysicalSize<u32>,
) {
    let mut ctx = ActionContext {
        loader,
        mouse_state,
        input_state,
        time,
        world_renderer,
        hit: top_hit,
        window_size,
    };

    // Process delayed actions
    action_system.update(&mut ctx);

    // Execute continuous/dragging actions
    let active_actions: Vec<String> = ctx
        .loader
        .ui_runtime
        .action_states
        .iter()
        .filter(|(_, state)| state.active)
        .map(|(name, _)| name.clone())
        .collect();

    for action_name in active_actions {
        match action_name.as_str() {
            "Drag Hue Point" => {
                drag_hue_point(ctx.loader, ctx.mouse_state, ctx.time);
            }
            _ => {}
        }
    }
}

/// Handle action when UI element is clicked
pub fn activate_action(
    action_system: &mut ActionSystem,
    loader: &mut UiButtonLoader,
    top_hit: &Option<HitResult>,
    mouse_state: &MouseState,
    input_state: &InputState,
    time: &TimeSystem,
    world_renderer: &mut WorldRenderer,
    window_size: PhysicalSize<u32>,
) {
    if let Some(hit) = top_hit {
        let action_str = hit.action.clone().unwrap_or_default();
        if action_str.is_empty() || action_str == "None" {
            return;
        }

        let mut ctx = ActionContext {
            loader,
            mouse_state,
            input_state,
            time,
            world_renderer,
            hit: top_hit,
            window_size,
        };

        action_system.run(&action_str, &mut ctx);
    }
}

pub fn deactivate_action(loader: &mut UiButtonLoader, action_name: &str) {
    if let Some(state) = loader.ui_runtime.action_states.get_mut(action_name) {
        state.active = false;
    }
}

pub fn selection_needed(_loader: &UiButtonLoader, action_name: &str) -> bool {
    matches!(action_name, "Drag Hue Point")
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let actions = parse_action_chain("open_menu(\"Editor\")");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].name, "open_menu");
        assert_eq!(actions[0].arg_str(0), Some("Editor"));
    }

    #[test]
    fn test_parse_chain() {
        let actions = parse_action_chain(
            "open_menu(\"Menu1\"); toggle_layer(\"Menu1\", \"Layer1\"); print(\"done\")",
        );
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0].name, "open_menu");
        assert_eq!(actions[1].name, "toggle_layer");
        assert_eq!(actions[1].arg_str(0), Some("Menu1"));
        assert_eq!(actions[1].arg_str(1), Some("Layer1"));
        assert_eq!(actions[2].name, "print");
    }

    #[test]
    fn test_parse_numbers() {
        let actions = parse_action_chain("delay(0.5); set_pick_radius(100)");
        assert_eq!(actions[0].arg_float(0), Some(0.5));
        assert_eq!(actions[1].arg_float(0), Some(100.0));
    }

    #[test]
    fn test_parse_no_args() {
        let actions = parse_action_chain("halt; debug_vars");
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].name, "halt");
        assert_eq!(actions[1].name, "debug_vars");
    }

    #[test]
    fn test_parse_variables() {
        let actions = parse_action_chain("set(my_var, 42); inc($my_var, 10)");
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].arg_str(0), Some("my_var"));
        assert!(matches!(actions[1].args.get(0), Some(ActionArg::Var(_))));
    }

    #[test]
    fn test_nested_action_string() {
        let actions = parse_action_chain("if(true, \"print(hello)\", \"print(bye)\")");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].name, "if");
        assert_eq!(actions[0].arg_str(1), Some("print(hello)"));
    }
}

pub fn make_actions() -> ActionSystem {
    let mut sys = ActionSystem::new();

    register_color_picker_actions(&mut sys);
    register_editor_actions(&mut sys);
    register_tool_actions(&mut sys);
    register_debug_actions(&mut sys);

    sys
}

fn register_color_picker_actions(sys: &mut ActionSystem) {
    sys.register_simple("drag_hue_point", |_action, ctx| {
        let state = ActionState::with_time("Drag Hue Point", ctx.time.total_time);
        ctx.loader
            .ui_runtime
            .action_states
            .insert("Drag Hue Point".to_string(), state);
        ActionResult::Ok
    });

    sys.register_simple("stop_hue_drag", |_action, ctx| {
        if let Some(state) = ctx
            .loader
            .ui_runtime
            .action_states
            .get_mut("Drag Hue Point")
        {
            state.active = false;
        }
        ActionResult::Ok
    });

    sys.register_simple("set_color", |action, ctx| {
        let r = action.arg_float_or(0, 1.0);
        let g = action.arg_float_or(1, 1.0);
        let b = action.arg_float_or(2, 1.0);

        ctx.loader.variables.set_f32("color_picker.r", r);
        ctx.loader.variables.set_f32("color_picker.g", g);
        ctx.loader.variables.set_f32("color_picker.b", b);

        let HSV { h, s, v } = rgb_to_hsv([r, g, b, 1.0]);

        ctx.loader.variables.set_f32("color_picker.h", h);
        ctx.loader.variables.set_f32("color_picker.s", s);
        ctx.loader.variables.set_f32("color_picker.v", v);

        ActionResult::Ok
    });

    sys.register_simple("set_hsv", |action, ctx| {
        let h = action.arg_float_or(0, 0.0);
        let s = action.arg_float_or(1, 1.0);
        let v = action.arg_float_or(2, 1.0);

        ctx.loader.variables.set_f32("color_picker.h", h);
        ctx.loader.variables.set_f32("color_picker.s", s);
        ctx.loader.variables.set_f32("color_picker.v", v);

        let rgb = hsv_to_rgb(HSV { h, s, v });
        ctx.loader.variables.set_f32("color_picker.r", rgb[0]);
        ctx.loader.variables.set_f32("color_picker.g", rgb[1]);
        ctx.loader.variables.set_f32("color_picker.b", rgb[2]);

        ActionResult::Ok
    });
}

fn register_editor_actions(sys: &mut ActionSystem) {
    sys.register_simple("toggle_editor", |_action, ctx| {
        ctx.loader.touch_manager.options.editor_mode =
            !ctx.loader.touch_manager.options.editor_mode;
        ActionResult::Ok
    });

    sys.register_simple("enable_editor", |_action, ctx| {
        ctx.loader.touch_manager.options.editor_mode = true;
        ActionResult::Ok
    });

    sys.register_simple("disable_editor", |_action, ctx| {
        ctx.loader.touch_manager.options.editor_mode = false;
        ActionResult::Ok
    });

    sys.register_simple("toggle_gui", |_action, ctx| {
        ctx.loader.touch_manager.options.show_gui = !ctx.loader.touch_manager.options.show_gui;
        ActionResult::Ok
    });

    sys.register_simple("deselect_all", |_action, ctx| {
        ctx.loader
            .touch_manager
            .selection
            .deselect_all(&mut ctx.loader.menus); // Todo!
        ActionResult::Ok
    });

    sys.register_simple("save_ui", |action, ctx| {
        let filename = action.arg_str(0).unwrap_or("ui_layout.yaml");
        let path: PathBuf = PathBuf::from(filename);
        match ctx.loader.save_gui_to_file(path, ctx.window_size) {
            Ok(_) => ActionResult::Ok,
            Err(e) => ActionResult::Error(format!("Failed to save: {}", e)),
        }
    });

    sys.register_simple("load_ui", |action, _| {
        let filename = action.arg_str(0).unwrap_or("ui_layout.yaml");
        match load_menus_from_directory(&PathBuf::from(filename), &BendMode::Strict) {
            Ok(_) => ActionResult::Ok,
            Err(e) => ActionResult::Error(format!("Failed to load: {}", e)),
        }
    });
}

fn register_tool_actions(sys: &mut ActionSystem) {
    sys.register_simple("set_tool", |action, ctx| {
        let tool = action.arg_string(0);
        ctx.loader.variables.set_string("current_tool", &tool);
        ActionResult::Ok
    });

    sys.register_simple("set_brush_size", |action, ctx| {
        let size = action.arg_float_or(0, 10.0);
        ctx.world_renderer.pick_radius_m = size;
        ctx.loader.variables.set_f32("brush_size", size);
        ActionResult::Ok
    });

    sys.register_simple("adjust_brush_size", |action, ctx| {
        let delta = action.arg_float_or(0, 1.0);
        let min = action.arg_float_or(1, 1.0);
        let max = action.arg_float_or(2, 500.0);

        let new_size = (ctx.world_renderer.pick_radius_m + delta).clamp(min, max);
        ctx.world_renderer.pick_radius_m = new_size;
        ctx.loader.variables.set_f32("brush_size", new_size);
        ActionResult::Ok
    });
}

fn register_debug_actions(sys: &mut ActionSystem) {
    sys.register_simple("dump_variables", |_action, ctx| {
        println!("=== UI Variables ===");
        ctx.loader.variables.dump();
        ActionResult::Ok
    });

    sys.register_simple("dump_selection", |_action, ctx| {
        let sel = &ctx.loader.ui_runtime.selected_ui_element_primary;
        println!("=== Selection ===");
        println!("  Menu: {}", sel.menu_name);
        println!("  Layer: {}", sel.layer_name);
        println!("  Element: {}", sel.element_id);
        println!("  Active: {}", sel.active);
        println!("  Dragging: {}", sel.dragging);
        ActionResult::Ok
    });

    sys.register("list_actions", |_action, _ctx, sys| {
        println!("=== Action Variables ===");
        for (name, value) in &sys.variables {
            println!("  {}: {:?}", name, value);
        }
        ActionResult::Ok
    });
}
