#[allow(unused_mut, unused_assignments)]
use crate::ui::actions::{CommandArg, UiCommand};
use crate::ui::ui_editor::Ui;
use crate::ui::ui_touch_manager::{ElementRef, TouchEvent};
use std::cmp::PartialEq;

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

/// Macro to define command mappings - just add lines when you add commands!
macro_rules! define_commands {
    // ============================================================================
    // Main entry point - generates the make_ui_command function
    // ============================================================================
    (
        $(
            $( $name:literal )|+ => $variant:ident
            $( {
                $( $field:ident : $ftype:ty ),* $(,)?
            } )?
        ),* $(,)?
    ) => {
        pub fn make_ui_command(func_name: &str, args: Vec<String>, element: &ElementRef) -> Option<UiCommand> {
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

    // ============================================================================
    // Dispatch - creates the UiCommand variant
    // ============================================================================

    // Unit variant (no fields)
    (@dispatch $args:ident, $element:ident, $variant:ident) => {
        Some(UiCommand::$variant)
    };

    // Struct variant (with fields)
    (@dispatch $args:ident, $element:ident, $variant:ident { $( $field:ident : $ftype:ty ),* }) => {{
        #[allow(unused_mut, unused_assignments)]
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

    // ============================================================================
    // Field Parsing - match by FIELD NAME for special cases
    // ============================================================================

    // Special case: field named 'element_ref' gets injected from context (no idx increment!)
    (@parse_field $args:ident, $idx:ident, $element:ident, element_ref, $ftype:ty) => {{
        Some($element.clone())
    }};

    // Special case: field named 'args' consumes ALL remaining arguments as Vec<CommandArg>
    (@parse_field $args:ident, $idx:ident, $element:ident, args, $ftype:ty) => {{
        let result: Vec<CommandArg> = $args
            .iter()
            .skip($idx)
            .map(|s| CommandArg::from_str(s))
            .collect();
        #[allow(unused_assignments)]
        {
            $idx = $args.len();
        }
        Some(result)
    }};

    // Special case: field named 'commands' consumes ALL remaining arguments as Vec<UiCommand>
    (@parse_field $args:ident, $idx:ident, $element:ident, commands, $ftype:ty) => {{
        let result: Vec<UiCommand> = $args
            .iter()
            .skip($idx)
            .filter_map(|s| make_ui_command(s, Vec::new(), $element))
            .collect();
        #[allow(unused_assignments)]
        {
            $idx = $args.len();
        }
        Some(result)
    }};

    // All other fields: use ParseArg trait for parsing
    (@parse_field $args:ident, $idx:ident, $element:ident, $field:ident, $ftype:ty) => {{
        let val = <$ftype as ParseArg>::parse_arg($args.get($idx)?)?;
        #[allow(unused_assignments)]
        {
            $idx += 1;
        }
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
    "set_var" | "setvar" | "set" | "set_var_to" | "set_setting" | "setsetting" | "set_settings" | "setsettings" | "set_setting_to" | "set_av_setting_to" | "set_av_setting"
        => SetVar { element_ref: ElementRef, name: String, value: CommandArg },

    "inc_var" | "incvar" | "inc" | "increment" | "add" | "cycle_setting" | "cyclesetting" | "cycle_settings" | "cyclesettings"
        => IncVar { element_ref: ElementRef, name: String, amount: f32 },

    "dec_var" | "decvar" | "dec" | "decrement" | "sub" | "subtract"
        => DecVar { element_ref: ElementRef, name: String, amount: f32 },

    "mul_var" | "mulvar" | "mul" | "multiply"
        => MulVar { element_ref: ElementRef, name: String, factor: f32 },

    "toggle_var" | "togglevar" | "toggle_variable" | "flip_var" | "toggle_setting" | "togglesetting" | "toggle_settings" | "flip_setting"
        => ToggleVar { element_ref: ElementRef, name: String },

    "clamp" | "clamp_var"
        => Clamp { element_ref: ElementRef, name: String, min: f32, max: f32 },

    // ===== ACTION STATE COMMANDS =====
    "start_action" | "startaction" | "action_start"
        => StartAction { action_name: String },

    "stop_action" | "stopaction" | "action_stop"
        => StopAction { action_name: String },

    "remove_action" | "removeaction" | "action_remove" | "delete_action"
        => RemoveAction { action_name: String },


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

    // ===== UTILITY =====
    "noop" | "no_op" | "nothing" | "none"
        => Noop,
}
#[derive(PartialEq, Debug)]
enum TouchEventKind {
    HoverEnter,
    Hovering,
    HoverExit,
    Press,
    Release,
    Click,
    DoubleClick,
    ScrollOnElement,
}
pub fn actions_to_uicommands(ui: &mut Ui, event: &TouchEvent) -> Vec<UiCommand> {
    let (event_kind, actions, element) = match event {
        TouchEvent::HoverEnter { actions, element } => {
            (TouchEventKind::HoverEnter, actions, element)
        }
        TouchEvent::Hovering { actions, element } => (TouchEventKind::Hovering, actions, element),
        TouchEvent::HoverExit { actions, element } => (TouchEventKind::HoverExit, actions, element),
        TouchEvent::Press {
            actions, element, ..
        } => (TouchEventKind::Press, actions, element),
        TouchEvent::Release {
            actions, element, ..
        } => (TouchEventKind::Release, actions, element),
        TouchEvent::Click {
            actions, element, ..
        } => (TouchEventKind::Click, actions, element),
        TouchEvent::DoubleClick {
            actions, element, ..
        } => (TouchEventKind::DoubleClick, actions, element),
        TouchEvent::ScrollOnElement {
            actions, element, ..
        } => (TouchEventKind::ScrollOnElement, actions, element),
        _ => return vec![],
    };

    let mut cmds = Vec::new();

    // Process each action string in the vec
    for action in actions {
        if let Some(cmd) = handle_action_str(ui, &event_kind, action.trim(), element) {
            cmds.push(cmd);
        }
    }
    cmds
}

/// Handle a single action string that may be an event wrapper
fn handle_action_str(
    ui: &mut Ui,
    event_kind: &TouchEventKind,
    action: &str,
    element: &ElementRef,
) -> Option<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return None;
    }

    let bytes = s.as_bytes();
    let len = s.len();

    // Parse identifier (wrapper/function name)
    let mut pos = 0;
    while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
        pos += 1;
    }

    // Must have identifier followed by '(' for it to be a wrapper
    if pos == 0 || pos >= len || bytes[pos] != b'(' {
        return None;
    }

    let wrapper = &s[..pos];
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
        None => return None, // Unbalanced parens
    };

    let inner = s[open_paren + 1..close_paren].trim();
    let wrapper_lower = wrapper.to_ascii_lowercase();

    // Check if this wrapper matches the current event
    if wrapper_matches_event(&wrapper_lower, event_kind) {
        return process_inner_content(ui, event_kind, inner, element);
    }

    None
}

/// Process the inner content of a matched event wrapper
fn process_inner_content(
    ui: &mut Ui,
    event_kind: &TouchEventKind,
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
                let last_part = s[part_start..i].trim();
                if !last_part.is_empty() {
                    // Try as nested event wrapper first
                    if let Some(cmd) = handle_action_str(ui, event_kind, last_part, element) {
                        return Some(cmd);
                    }
                    // Try as primitive action
                    if let Some(cmd) = parse_primitive_action(ui, last_part, element) {
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
        if let Some(cmd) = handle_action_str(ui, event_kind, last_part, element) {
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

            //println!("Primitive action: {}({:#?})", func_name, args); // Debug

            return make_ui_command(func_name, args, element);
        }
    }

    //println!("Simple action: {}", s); // Debug
    make_ui_command(s, Vec::new(), element)
}

fn wrapper_matches_event(wrapper: &str, event_kind: &TouchEventKind) -> bool {
    let wrapper_normalized = wrapper.to_ascii_lowercase().replace('_', "");

    match wrapper_normalized.as_str() {
        // hover synonyms
        "onhoverenter" | "hoverenter" => *event_kind == TouchEventKind::HoverEnter,
        "onhover" | "hover" | "whilehovering" | "hovering" => {
            *event_kind == TouchEventKind::Hovering
        }
        "onhoverexit" | "hoverexit" | "hoverleave" => *event_kind == TouchEventKind::HoverExit,
        // press / release / click / double_click / scroll
        "onpress" | "press" => *event_kind == TouchEventKind::Press,
        "onrelease" | "release" => *event_kind == TouchEventKind::Release,
        "onclick" | "click" => *event_kind == TouchEventKind::Click,
        "ondoubleclick" | "doubleclick" => *event_kind == TouchEventKind::DoubleClick,
        "onscroll" | "scroll" => *event_kind == TouchEventKind::ScrollOnElement,
        _ => false,
    }
}
