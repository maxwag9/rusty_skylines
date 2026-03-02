use crate::ui::actions::{CommandArg, UiCommand};
use crate::ui::ui_editor::Ui;
use crate::ui::ui_touch_manager::{ElementRef, TouchEvent};

/// Helper trait for parsing argument types
trait ParseArg: Sized {
    fn parse_arg(s: &str) -> Option<Self>;
}

impl ParseArg for String {
    fn parse_arg(s: &str) -> Option<Self> {
        Some(s.to_string())
    }
}

impl ParseArg for f32 {
    fn parse_arg(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for f64 {
    fn parse_arg(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for i32 {
    fn parse_arg(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for u32 {
    fn parse_arg(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for usize {
    fn parse_arg(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for bool {
    fn parse_arg(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" | "enabled" => Some(true),
            "false" | "0" | "no" | "off" | "disabled" => Some(false),
            _ => None,
        }
    }
}

impl ParseArg for CommandArg {
    fn parse_arg(s: &str) -> Option<Self> {
        Some(CommandArg::from_str(s))
    }
}

// For Vec<CommandArg> - consumes ALL remaining args
impl ParseArg for Vec<CommandArg> {
    fn parse_arg(_s: &str) -> Option<Self> {
        // This is handled specially in the macro
        None
    }
}

/// Macro to define command mappings - just add lines when you add commands!
macro_rules! define_commands {
    (
        $(
            $( $name:literal )|+ => $variant:ident
            $( {
                $( $field:ident : $ftype:ty ),* $(,)?
            } )?
        ),* $(,)?
    ) => {
        fn make_ui_command(func_name: &str, args: Vec<String>, element: &ElementRef) -> Option<UiCommand> {
            let name_lower = func_name.to_ascii_lowercase();

            match name_lower.as_str() {
                $(
                    $( $name )|+ => {
                        define_commands!(
                            @dispatch
                            args,
                            element,
                            $variant
                            $( { $( $field : $ftype ),* } )?
                        )
                    }
                ),*
                ,
                _ => {
                    eprintln!("[Warning] Unknown UI command: {}", func_name);
                    None
                }
            }
        }
    };

    // ========================
    // Dispatch
    // ========================

    // Unit variant
    (@dispatch $args:ident, $element:ident, $variant:ident) => {
        Some(UiCommand::$variant)
    };

    // Struct variant
    (@dispatch $args:ident, $element:ident, $variant:ident { $( $field:ident : $ftype:ty ),* }) => {{
        #[allow(unused_mut)]
        let mut idx = 0usize;

        $(
            let $field = define_commands!(@parse_field $args, idx, $element, $field, $ftype)?;
        )*

        Some(UiCommand::$variant {
            $(
                $field,
            )*
        })
    }};

    // ========================
    // Field Parsing - match by FIELD NAME, not type
    // ========================

    // Special case: field named 'element_ref' gets injected from context (no idx increment!)
    (@parse_field $args:ident, $idx:ident, $element:ident, element_ref, $ftype:ty) => {{
        Some($element.clone())
    }};

    // All other fields: delegate to type-based parsing
    (@parse_field $args:ident, $idx:ident, $element:ident, $field:ident, $ftype:ty) => {{
        define_commands!(@parse_type $args, $idx, $element, $ftype)
    }};

    // ========================
    // Type-based Parsing
    // ========================

    // Vec<CommandArg> - consumes all remaining args
    (@parse_type $args:ident, $idx:ident, $element:ident, Vec<CommandArg>) => {{
        let result: Vec<CommandArg> = $args
            .iter()
            .skip($idx)
            .map(|s| CommandArg::from_str(s))
            .collect();

        $idx = $args.len();
        Some(result)
    }};

    // Vec<UiCommand>
    (@parse_type $args:ident, $idx:ident, $element:ident, Vec<UiCommand>) => {{
        let result: Vec<UiCommand> = $args
            .iter()
            .skip($idx)
            .filter_map(|s| make_ui_command(s, Vec::new(), $element))
            .collect();

        $idx = $args.len();
        Some(result)
    }};

    // Normal field - uses ParseArg trait
    (@parse_type $args:ident, $idx:ident, $element:ident, $ftype:ty) => {{
        let val = <$ftype as ParseArg>::parse_arg($args.get($idx)?)?;
        $idx += 1;
        Some(val)
    }};
}

// ============================================================================
// COMMAND DEFINITIONS - Just add a line here when you add a new UiCommand!
// ============================================================================

define_commands! {
    // ===== MENU COMMANDS =====
    "open_menu" | "openmenu" | "menu_open"
        => OpenMenu { menu_name: String },

    "close_menu" | "closemenu" | "menu_close"
        => CloseMenu { menu_name: String },

    "close_all_menus" | "closeallmenus"
        => CloseAllMenus,

    "toggle_menu" | "togglemenu" | "menu_toggle"
        => ToggleMenu { menu_name: String },

    "menu_active" | "menuactive" | "is_menu_active"
        => MenuActive { menu_name: String },

    // ===== LAYER COMMANDS =====
    "open_layer" | "openlayer" | "layer_open"
        => OpenLayer { menu_name: String, layer_name: String },

    "close_layer" | "closelayer" | "layer_close"
        => CloseLayer { menu_name: String, layer_name: String },

    "toggle_layer" | "togglelayer" | "layer_toggle"
        => ToggleLayer { menu_name: String, layer_name: String },

    // ===== VARIABLE COMMANDS =====
    "set_var" | "setvar" | "set"
        => SetVar { name: String, value: CommandArg },

    "inc_var" | "incvar" | "inc" | "increment" | "add"
        => IncVar { name: String, amount: f32 },

    "dec_var" | "decvar" | "dec" | "decrement" | "sub" | "subtract"
        => DecVar { name: String, amount: f32 },

    "mul_var" | "mulvar" | "mul" | "multiply"
        => MulVar { name: String, factor: f32 },

    "toggle_bool" | "togglebool" | "toggle" | "flip"
        => ToggleBool { name: String },

    "toggle_bool_setting" | "togglebool_setting" | "toggle_setting" | "flip_setting"
        => ToggleSettingBool { element_ref: ElementRef },

    "set_bool_setting" | "setbool_setting" | "set_setting" | "set_setting_to"
        => SetSettingBool { element_ref: ElementRef, state: bool },

    "clamp" | "clamp_var"
        => Clamp { name: String, min: f32, max: f32 },

    // ===== ACTION STATE COMMANDS =====
    "start_action" | "startaction" | "action_start"
        => StartAction { action_name: String },

    "stop_action" | "stopaction" | "action_stop"
        => StopAction { action_name: String },

    "remove_action" | "removeaction" | "action_remove" | "delete_action"
        => RemoveAction { action_name: String },

    // ===== WORLD RENDERER COMMANDS =====
    "set_pick_radius" | "setpickradius" | "pick_radius"
        => SetPickRadius { radius: f32 },

    "grow_pick_radius" | "growpickradius"
        => GrowPickRadius { amount: f32 },

    "shrink_pick_radius" | "shrinkpickradius"
        => ShrinkPickRadius { amount: f32 },

    // ===== FLOW CONTROL =====
    "delay" | "wait" | "sleep" | "pause"
        => Delay { seconds: f32 },

    "halt" | "stop" | "break"
        => Halt,

    "skip"
        => Skip { count: usize },

    // ===== DEBUG COMMANDS =====
    "print" | "log" | "echo"
        => Print { args: Vec<CommandArg> },

    "debug_vars" | "debugvars" | "vars"
        => DebugVars,

    "debug_menus" | "debugmenus" | "menus"
        => DebugMenus,

    "debug_actions" | "debugactions" | "actions"
        => DebugActions,

    // ===== EVENT COMMANDS =====
    "emit_event" | "emitevent" | "emit" | "event" | "fire"
        => EmitEvent { element_ref: ElementRef, event_name: String },

    // ===== LEGACY/SPECIAL COMMANDS =====
    "drag_hue_point" | "draghuepoint"
        => DragHuePoint { element_ref: ElementRef },

    "set_roads_four_lanes" | "setroadsfourlanes" | "four_lanes"
        => SetRoadsFourLanes { forward: usize, backward: usize },

    // ===== UTILITY =====
    "noop" | "no_op" | "nothing" | "none"
        => Noop,
}

pub fn action_to_uicommand(ui: &mut Ui, event: &TouchEvent) -> Option<UiCommand> {
    let (event_kind, action_opt, element) = match event {
        TouchEvent::HoverEnter { action, element } => ("hover_enter", action.as_deref(), element),
        TouchEvent::HoverExit { action, element } => ("hover_exit", action.as_deref(), element),
        TouchEvent::Press {
            action, element, ..
        } => ("press", action.as_deref(), element),
        TouchEvent::Release {
            action, element, ..
        } => ("release", action.as_deref(), element),
        TouchEvent::Click {
            action, element, ..
        } => ("click", action.as_deref(), element),
        TouchEvent::DoubleClick {
            action, element, ..
        } => ("double_click", action.as_deref(), element),
        TouchEvent::ScrollOnElement {
            action, element, ..
        } => ("scroll", action.as_deref(), element),
        _ => return None,
    };

    let action = match action_opt {
        Some(a) => a.trim(),
        None => return None,
    };

    handle_action_str_recursive(ui, event_kind, action, element)
}

fn handle_action_str_recursive(
    ui: &mut Ui,
    event_kind: &str,
    action: &str,
    element: &ElementRef,
) -> Option<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return None;
    }

    let bytes = s.as_bytes();
    let len = s.len();
    let mut pos = 0;

    // Iterate through ALL top-level action expressions
    // Supports: "a()b()", "a() b()", "a(), b()", "a() , b()" etc.
    while pos < len {
        // Skip separators: whitespace and commas
        while pos < len && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
            pos += 1;
        }

        if pos >= len {
            break;
        }

        // Parse identifier (wrapper/function name)
        let ident_start = pos;
        while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
            pos += 1;
        }

        // Must have identifier followed by '('
        if pos == ident_start || pos >= len || bytes[pos] != b'(' {
            pos += 1;
            continue;
        }

        let wrapper = &s[ident_start..pos];
        let open_paren = pos;

        // Find matching close paren (handles nested parens)
        let mut depth = 0isize;
        let mut close_paren = None;
        for i in open_paren..len {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        close_paren = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        let close_paren = match close_paren {
            Some(cp) => cp,
            None => break, // Unbalanced parens
        };

        let inner = s[open_paren + 1..close_paren].trim();
        let wrapper_lower = wrapper.to_ascii_lowercase();

        // Check if this wrapper matches the current event
        if wrapper_matches_event(&wrapper_lower, event_kind) {
            // Process inner content (may have nested wrappers or primitive actions)
            if let Some(cmd) = process_inner_content(ui, event_kind, inner, element) {
                return Some(cmd);
            }
        }

        // Move past this action, continue to next
        pos = close_paren + 1;
    }

    None
}

/// Process the inner content of a matched event wrapper
fn process_inner_content(
    ui: &mut Ui,
    event_kind: &str,
    inner: &str,
    element: &ElementRef,
) -> Option<UiCommand> {
    let s = inner.trim();

    if s.is_empty() {
        return None;
    }

    // Split by top-level commas and process each part
    let bytes = s.as_bytes();
    let len = s.len();
    let mut part_start = 0;
    let mut depth = 0isize;

    for i in 0..len {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b',' if depth == 0 => {
                let part = s[part_start..i].trim();
                if !part.is_empty() {
                    // Try as nested event wrapper first
                    if let Some(cmd) = handle_action_str_recursive(ui, event_kind, part, element) {
                        return Some(cmd);
                    }
                    // Try as primitive action
                    if let Some(cmd) = parse_primitive_action(ui, part, element) {
                        return Some(cmd);
                    }
                }
                part_start = i + 1;
            }
            _ => {}
        }
    }

    // Handle last (or only) part
    let last_part = s[part_start..].trim();
    if !last_part.is_empty() {
        // Try as nested event wrapper
        if let Some(cmd) = handle_action_str_recursive(ui, event_kind, last_part, element) {
            return Some(cmd);
        }
        // Try as primitive action
        if let Some(cmd) = parse_primitive_action(ui, last_part, element) {
            return Some(cmd);
        }
    }

    None
}

/// Parse arguments from a string into a Vec<String>
/// Supports: (arg1, arg2), (arg1,arg2), (arg1 arg2), (arg1   ,   arg2)
/// Handles nested parentheses: (func(a, b), other_arg) -> ["func(a, b)", "other_arg"]
fn parse_arguments(args_str: &str) -> Vec<String> {
    let s = args_str.trim();
    if s.is_empty() {
        return Vec::new();
    }

    let bytes = s.as_bytes();
    let len = s.len();
    let mut args = Vec::new();
    let mut pos = 0;

    while pos < len {
        // Skip separators: whitespace and commas
        while pos < len && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
            pos += 1;
        }

        if pos >= len {
            break;
        }

        // Start of an argument
        let arg_start = pos;
        let mut depth = 0isize;

        // Read until we hit a separator at depth 0
        while pos < len {
            let b = bytes[pos];
            if b == b'(' {
                depth += 1;
                pos += 1;
            } else if b == b')' {
                depth -= 1;
                pos += 1;
            } else if depth == 0 && (b.is_ascii_whitespace() || b == b',') {
                // Separator at top level - end of argument
                break;
            } else {
                pos += 1;
            }
        }

        let arg = s[arg_start..pos].trim();
        if !arg.is_empty() {
            args.push(arg.to_string());
        }
    }

    args
}

/// Parse a primitive action and create a UiCommand
/// Handles: "action_name" or "action_name(arg1, arg2, ...)"
fn parse_primitive_action(_ui: &mut Ui, action: &str, element: &ElementRef) -> Option<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return None;
    }

    // Check for function-call style: name(args)
    if let Some(open_paren) = s.find('(') {
        let func_name = s[..open_paren].trim();

        // Find matching close paren
        let bytes = s.as_bytes();
        let mut depth = 0isize;
        let mut close_paren = None;
        for i in open_paren..s.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        close_paren = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        if let Some(close_paren) = close_paren {
            let args_str = s[open_paren + 1..close_paren].trim();
            let args: Vec<String> = parse_arguments(args_str);

            println!("Primitive action: {}({:#?})", func_name, args); // Debug

            return make_ui_command(func_name, args, element);
        }
    }

    println!("Simple action: {}", s); // Debug
    make_ui_command(s, Vec::new(), element)
}

fn wrapper_matches_event(wrapper: &str, event_kind: &str) -> bool {
    match wrapper {
        // hover synonyms
        "on_hover_enter" | "hover_enter" => event_kind == "hover_enter",
        "on_hover" | "hover" | "while_hovering" | "hovering" => event_kind == "hovering",
        "on_hover_exit" | "hover_exit" | "hoverleave" | "hover_leave" => event_kind == "hover_exit",
        // press / release / click / double_click / scroll
        "on_press" | "press" => event_kind == "press",
        "on_release" | "release" => event_kind == "release",
        "on_click" | "click" => event_kind == "click",
        "on_double_click" | "double_click" | "doubleclick" => event_kind == "double_click",
        "on_scroll" | "scroll" => event_kind == "scroll",
        _ => false,
    }
}
