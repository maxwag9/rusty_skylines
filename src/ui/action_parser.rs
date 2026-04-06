use crate::data::Settings;
#[allow(unused_mut, unused_assignments)]
use crate::ui::actions::UiCommand;
use crate::ui::input::Input;
use crate::ui::menu::Menu;
use crate::ui::ui_editor::Ui;
use crate::ui::ui_touch_manager::UiTouchManager;
use crate::ui::ui_touch_manager::{ElementRef, MouseButtons, TouchEvent};
use crate::ui::variables::Variables;
use std::cmp::PartialEq;
use std::collections::HashMap;

/// Helper trait for parsing argument types
trait ParseArg: Sized {
    fn parse_arg(
        settings: &Settings,
        variables: &mut Variables,
        menus: &HashMap<String, Menu>,
        touch_manager: &UiTouchManager,
        element: &ElementRef,
        s: &str,
    ) -> Option<Self>;
}

impl ParseArg for String {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        Some(s.to_string())
    }
}

impl ParseArg for f32 {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for f64 {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for i32 {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for u32 {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for usize {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        s.parse().ok()
    }
}

impl ParseArg for bool {
    fn parse_arg(
        _settings: &Settings,
        _variables: &mut Variables,
        _menus: &HashMap<String, Menu>,
        _touch_manager: &UiTouchManager,
        _element: &ElementRef,
        s: &str,
    ) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" | "enabled" => Some(true),
            "false" | "0" | "no" | "off" | "disabled" => Some(false),
            _ => None,
        }
    }
}

/// Macro to define command mappings - just add lines when you add commands!
macro_rules! define_commands {
    (
        $(
            $( $name:literal )|+ => $variant:ident
            $( { $( $field:ident : $ftype:ty ),* $(,)? } )?
        ),* $(,)?
    ) => {

        pub fn make_ui_command(
            settings: &Settings,
            variables: &mut Variables,
            menus: &HashMap<String, Menu>,
            touch_manager: &UiTouchManager,
            func_name: &str,
            mut args: Vec<String>,
            element: &ElementRef
        ) -> Option<UiCommand> {
            let name = func_name.to_ascii_lowercase();

            match name.as_str() {
                $(
                    $( $name )|+ => {
                        define_commands!(@build settings, variables, menus, touch_manager, args, element, $variant $( { $( $field : $ftype ),* } )?)
                    }
                ),*,
                _ => {
                    eprintln!("[Warning] Unknown UI command: {}", func_name);
                    None
                }
            }
        }
    };

    // unit variant
    (@build $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $element:ident, $variant:ident) => {
    Some(UiCommand::$variant)
    };

    // struct variant
    (@build $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $element:ident, $variant:ident { $( $field:ident : $ftype:ty ),* }) => {{
        let mut idx = 0usize;

        $(
            let $field = define_commands!(
                @parse $settings, $vars, $menus, $tm, $args, idx, $element, $field, $ftype
            )?;
        )*

        Some(UiCommand::$variant { $( $field ),* })
    }};

    // -----------------------------
    // SPECIAL FIELD: element_ref
    // -----------------------------

    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element:ident, element_ref, $ftype:ty) => {{
        Some($element.clone())
    }};

    // -----------------------------
    // SPECIAL FIELD: args
    // -----------------------------

    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element:ident, args, $ftype:ty) => {{
        let out = $args.split_off($idx);
        #[allow(unused_assignments)]
        {
            $idx = $args.len();
        }
        Some(out)
    }};

    // -----------------------------
    // SPECIAL FIELD: commands
    // -----------------------------
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element:ident, commands, $ftype:ty) => {{
        let out = $args.iter()
            .skip($idx)
            .filter_map(|s| make_ui_command($settings, $vars, s, Vec::new(), $element))
            .collect();
        #[allow(unused_assignments)]
        {
            $idx = $args.len();
        }
        Some(out)
    }};

    // SPECIAL FIELD: then / else_branch
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element:ident, then, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            #[allow(unused_assignments)]
            {
                $idx += 1;
            }

            let cmds = raw.split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .filter_map(|s| parse_primitive_action($settings, $vars, $menus, $tm, s, $element))
                .collect();

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element:ident, else_branch, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            #[allow(unused_assignments)]
            {
                $idx += 1;
            }

            let cmds = raw.split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .filter_map(|s| parse_primitive_action($settings, $vars, $menus, $tm, s, $element))
                .collect();

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    (@branch $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident, $element:ident) => {{
        let raw = $args.get($idx)?;
        #[allow(unused_assignments)]
        {
            $idx += 1;
        }

        let cmds = raw.split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter_map(|s| parse_primitive_action($settings, $vars, $menus, $tm, s, $element))
            .collect();

        Some(cmds)
    }};

    // GENERIC FIELD PARSER
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element:ident, $field:ident, $ftype:ty) => {{

        let val = <$ftype as ParseArg>::parse_arg(
            $settings,
            $vars,
            $menus,
            $tm,
            $element,
            $args.get($idx)?
        )?;
        #[allow(unused_assignments)]
        {
            $idx += 1;
        }
        Some(val)
    }};
}

// COMMAND DEFINITIONS - I Just add a line here when I add a new UiCommand!
define_commands! {
    // ===== MENU COMMANDS =====
    "open_menu" | "openmenu"
        => OpenMenu { menu_name: String },

    "close_menu" | "closemenu"
        => CloseMenu { menu_name: String },

    "close_all_menus" | "closeall"
        => CloseAllMenus,

    "toggle_menu" | "togglemenu"
        => ToggleMenu { menu_name: String },

    "menu_active" | "menuactive"
        => MenuActive { menu_name: String },

    // ===== LAYER COMMANDS =====
    "open_layer" | "openlayer"
        => OpenLayer { menu_name: String, layer_name: String },

    "close_layer" | "closelayer"
        => CloseLayer { menu_name: String, layer_name: String },

    "toggle_layer" | "togglelayer"
        => ToggleLayer { menu_name: String, layer_name: String },

    // ===== VARIABLE COMMANDS =====
    "set_var" | "setvar" | "set"
        => SetVar { element_ref: ElementRef, name: String, value: String },

    "inc_var" | "incvar" | "inc"
        => IncVar { element_ref: ElementRef, name: String, amount: f64 },

    "dec_var" | "decvar" | "dec"
        => DecVar { element_ref: ElementRef, name: String, amount: f64 },

    "mul_var" | "mulvar" | "mul"
        => MulVar { element_ref: ElementRef, name: String, factor: f64 },

    "toggle_var" | "togglevar" | "toggle"
        => ToggleVar { element_ref: ElementRef, name: String },

    "clamp" | "clampvar"
        => Clamp { element_ref: ElementRef, name: String, min: f64, max: f64 },

    "set_var_expr" | "setexpr" | "set_expr"
        => SetVarExpr { element_ref: ElementRef, name: String, expr: String },

    // ===== ACTION STATE COMMANDS =====
    "start_action" | "startaction"
        => StartAction { action_name: String },

    "stop_action" | "stopaction"
        => StopAction { action_name: String },

    "remove_action" | "removeaction"
        => RemoveAction { action_name: String },


    // ===== FLOW CONTROL =====
    "delay" | "wait" | "sleep"
        => Delay { seconds: f64 },

    "halt" | "break"
        => Halt,

    "skip"
        => Skip { count: usize },

    "if" => If { element_ref: ElementRef, condition: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand> },

    "ifvareq"
        => IfVarEq { element_ref: ElementRef, var_name: String, value: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand>},

    "add_element" | "addelem" | "add"
        => AddElement { element_ref: ElementRef, menu: String, layer: String, id: String, kind: String, center: String},

    "clone_element" | "cloneelem" | "clone"
        => CloneElement { element_ref: ElementRef,
        from_menu: String,
        from_layer: String,
        from_id: String,
        to_menu: String,
        to_layer: String,
        to_id: String,
        center: String,},

    "clone_element_undoable" | "cloneu" | "copyu"
        => CloneElementUndoable {
        element_ref: ElementRef,
        from_menu: String,
        from_layer: String,
        from_id: String,
        to_menu: String,
        to_layer: String,
        to_id: String,
        center: String,},

    "delete_element" | "delelem" | "delete"
        => DeleteElement {element_ref: ElementRef, menu: String, layer: String, id: String},

    "delete_element_undoable" | "deleteu" | "removeu"
        => DeleteElement {element_ref: ElementRef, menu: String, layer: String, id: String},

    "save" | "savegame"
        => SaveGame,

    "load" | "loadgame" | "load_save"
        => LoadSave {save_name: String, without_saving: bool  },

    "exit" | "quit"
        => ExitGame,

    // ===== DEBUG COMMANDS =====
    "print" | "log" | "echo"
        => Print { element_ref: ElementRef, args: Vec<String> },

    "debug_vars" | "debugvars"
        => DebugVars,

    "debug_menus" | "debugmenus"
        => DebugMenus,

    "debug_actions" | "debugactions"
        => DebugActions,

    // ===== EVENT COMMANDS =====
    "emit_event" | "emitevent" | "emit"
        => EmitEvent { element_ref: ElementRef, event_name: String },

    // ===== UTILITY =====
    "noop" | "no_op" | "none"
        => Noop,
}
#[derive(PartialEq, Debug, Copy, Clone)]
enum TouchEventKind {
    HoverEnter,
    Hovering,
    HoverExit,
    Press,
    Down,
    Release,
    Click,
    DoubleClick,
    ScrollOnElement,
    Select,
    DeSelect,
    DragMove,
    Nothing,
}
pub fn actions_to_uicommands(
    ui: &mut Ui,
    event: &TouchEvent,
    settings: &Settings,
    input: &mut Input,
) -> Vec<UiCommand> {
    let (event_kind, actions, element, buttons) = match event {
        TouchEvent::HoverEnter { actions, element } => (
            TouchEventKind::HoverEnter,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::Hovering { actions, element } => (
            TouchEventKind::Hovering,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::HoverExit { actions, element } => (
            TouchEventKind::HoverExit,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::Press {
            actions,
            element,
            buttons,
            ..
        } => (TouchEventKind::Press, actions, element, *buttons),
        TouchEvent::Down {
            actions,
            element,
            buttons,
            ..
        } => (TouchEventKind::Down, actions, element, *buttons),
        TouchEvent::Release {
            actions,
            element,
            buttons,
            ..
        } => (TouchEventKind::Release, actions, element, *buttons),
        TouchEvent::Click {
            actions,
            element,
            buttons,
            ..
        } => (TouchEventKind::Click, actions, element, *buttons),
        TouchEvent::DoubleClick {
            actions,
            element,
            buttons,
            ..
        } => (TouchEventKind::DoubleClick, actions, element, *buttons),
        TouchEvent::DragMove {
            element,
            actions,
            buttons,
            ..
        } => (TouchEventKind::DragMove, actions, element, *buttons),
        TouchEvent::ScrollOnElement {
            actions, element, ..
        } => (
            TouchEventKind::ScrollOnElement,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::SelectionRequested { element, .. } => (
            TouchEventKind::Select,
            &vec![],
            element,
            MouseButtons::default(),
        ),
        TouchEvent::DeselectAllRequested {} => (
            TouchEventKind::DeSelect,
            &vec![],
            &ElementRef::default(),
            MouseButtons::default(),
        ),
        TouchEvent::Nothing { element, actions } => (
            TouchEventKind::Nothing,
            actions,
            element,
            MouseButtons::default(),
        ),
        _ => return vec![],
    };

    let mut cmds = Vec::new();

    for action in actions.iter().chain(ui.global_actions.actions.iter()) {
        let mut action_owned = action.clone();

        let filters = parse_action_filters(&mut action_owned);

        if filters_match(input, settings, &filters, &event_kind, &buttons) {
            // Now action_owned only contains the actual command
            if let Some(cmd) = parse_primitive_action(
                settings,
                &mut ui.variables,
                &ui.menus,
                &ui.touch_manager,
                action_owned.trim(),
                element,
            ) {
                cmds.push(cmd);
            }
        }
    }
    cmds
}

/// Handle a single action string that may be an event wrapper
fn handle_action_str(
    settings: &Settings,
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
    process_inner_content(settings, ui, event_kind, inner, element)
}

/// Process the inner content of a matched event wrapper
fn process_inner_content(
    settings: &Settings,
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
                    if let Some(cmd) =
                        handle_action_str(settings, ui, event_kind, last_part, element)
                    {
                        return Some(cmd);
                    }
                    // Try as primitive action
                    if let Some(cmd) = parse_primitive_action(
                        settings,
                        &mut ui.variables,
                        &ui.menus,
                        &ui.touch_manager,
                        last_part,
                        element,
                    ) {
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
        if let Some(cmd) = handle_action_str(settings, ui, event_kind, last_part, element) {
            return Some(cmd);
        }
        // Try as primitive action
        if let Some(cmd) = parse_primitive_action(
            settings,
            &mut ui.variables,
            &ui.menus,
            &ui.touch_manager,
            last_part,
            element,
        ) {
            return Some(cmd);
        }
    }

    None
}

/// Parse arguments from a string into a Vec<String>
///
/// Behavior:
/// - If there is at least one TOP-LEVEL comma, we split ONLY on top-level commas.
///   This preserves spaces inside an argument, e.g.:  BLALABLA DON'T LISTEN TO THIS AI'S BS!! MAYBE
///     "Editor_Menu, Color Picker" -> ["Editor_Menu", "Color Picker"]
/// - If there are NO top-level commas, we fall back to the old behavior:
///   split on whitespace (still respecting nested parentheses).
///
/// Nested parentheses are always respected:
///   "func(a, b), other arg" -> ["func(a, b)", "other arg"]
fn parse_arguments(args_str: &str) -> Vec<String> {
    let s = args_str.trim();
    if s.is_empty() {
        return Vec::new();
    }

    let bytes = s.as_bytes();
    let len = s.len();

    let mut args = Vec::new();
    let mut start = 0usize;

    let mut paren_depth = 0isize;
    let mut bracket_depth = 0isize;
    let mut brace_depth = 0isize;
    let mut in_string = false;
    let mut escape = false;

    let mut saw_top_level_comma = false;

    for i in 0..len {
        let b = bytes[i];

        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match b {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match b {
            b'"' => in_string = true,
            b'(' => paren_depth += 1,
            b')' => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' => bracket_depth -= 1,
            b'{' => brace_depth += 1,
            b'}' => brace_depth -= 1,
            b',' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                saw_top_level_comma = true;
                let part = s[start..i].trim();
                if !part.is_empty() {
                    args.push(part.to_string());
                }
                start = i + 1;
            }
            _ => {}
        }
    }

    let part = s[start..].trim();
    if !part.is_empty() {
        args.push(part.to_string());
    }

    if saw_top_level_comma {
        return args;
    }

    let mut args = Vec::new();
    let mut pos = 0usize;

    while pos < len {
        while pos < len && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= len {
            break;
        }

        let arg_start = pos;

        let mut paren_depth = 0isize;
        let mut bracket_depth = 0isize;
        let mut brace_depth = 0isize;
        let mut in_string = false;
        let mut escape = false;

        while pos < len {
            let b = bytes[pos];

            if in_string {
                if escape {
                    escape = false;
                    pos += 1;
                    continue;
                }
                match b {
                    b'\\' => escape = true,
                    b'"' => in_string = false,
                    _ => {}
                }
                pos += 1;
                continue;
            }

            match b {
                b'"' => {
                    in_string = true;
                    pos += 1;
                }
                b'(' => {
                    paren_depth += 1;
                    pos += 1;
                }
                b')' => {
                    paren_depth -= 1;
                    pos += 1;
                }
                b'[' => {
                    bracket_depth += 1;
                    pos += 1;
                }
                b']' => {
                    bracket_depth -= 1;
                    pos += 1;
                }
                b'{' => {
                    brace_depth += 1;
                    pos += 1;
                }
                b'}' => {
                    brace_depth -= 1;
                    pos += 1;
                }
                b if b.is_ascii_whitespace()
                    && paren_depth == 0
                    && bracket_depth == 0
                    && brace_depth == 0 =>
                {
                    break;
                }
                _ => pos += 1,
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
fn parse_primitive_action(
    settings: &Settings,
    variables: &mut Variables,
    menus: &HashMap<String, Menu>,
    touch_manager: &UiTouchManager,
    action: &str,
    element: &ElementRef,
) -> Option<UiCommand> {
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

            let command = make_ui_command(
                settings,
                variables,
                menus,
                touch_manager,
                func_name,
                args,
                element,
            );
            //println!("Command: {:?}", command);
            return command;
        }
    }

    //println!("Simple action: {}", s); // Debug
    make_ui_command(
        settings,
        variables,
        menus,
        touch_manager,
        s,
        Vec::new(),
        element,
    )
}

fn button_matches(input: &mut Input, button: ParsedButton, buttons: &MouseButtons) -> bool {
    let state = match button {
        ParsedButton::Any => return true,
        ParsedButton::Left => &buttons.left,
        ParsedButton::Right => &buttons.right,
        ParsedButton::Middle => &buttons.middle,
        ParsedButton::Back => &buttons.back,
        ParsedButton::Forward => &buttons.forward,
        ParsedButton::Key(s) => {
            let s = s.as_str();
            return if input.ensure_known_action(s) {
                input.action_down(s)
            } else {
                input.combo_down(s)
            };
        }
    };
    state.pressed || state.just_released
}
#[derive(Default, Debug)]
struct ActionFilters {
    buttons: Vec<ParsedButton>,
    events: Vec<TouchEventKind>,
    modes: Vec<String>,
}

struct ParsedAction {
    filters: ActionFilters,
    command: String,
}

fn parse_action_filters(action: &mut String) -> ActionFilters {
    // ALWAYS remove the found filter!!!!
    let mut filters = ActionFilters::default();

    // Keep removing filters until none are found
    loop {
        let mut found_something = false;
        let button_prefix = "button:";
        // Try to extract a button: filter
        if let Some(button) = try_extract_filter(
            action,
            button_prefix,
            &[
                ("any", ParsedButton::Any),
                ("right", ParsedButton::Right),
                ("middle", ParsedButton::Middle),
                ("back", ParsedButton::Back),
                ("forward", ParsedButton::Forward),
                ("left", ParsedButton::Left),
                ("a", ParsedButton::Any),
                ("l", ParsedButton::Left),
                ("r", ParsedButton::Right),
                ("m", ParsedButton::Middle),
                ("b", ParsedButton::Back),
                ("f", ParsedButton::Forward),
            ],
        ) {
            filters.buttons.push(button);
            found_something = true;
        } else if let Some(start) = action.find(button_prefix) {
            let rest = &action[start + button_prefix.len()..];

            // Handle quoted string
            if rest.starts_with('"') {
                if let Some(end_quote) = rest[1..].find('"') {
                    let key = rest[1..=end_quote].to_string();
                    let total_len = button_prefix.len() + end_quote + 2; // +2 for both quotes
                    remove_filter_span(action, start, start + total_len);
                    filters.buttons.push(ParsedButton::Key(key));
                    found_something = true;
                }
            }
        }

        // Try to extract an on: filter (event type)
        if let Some(event) = try_extract_filter(
            action,
            "on:",
            &[
                ("n", TouchEventKind::Nothing),
                ("hover_enter", TouchEventKind::HoverEnter),
                ("hoverenter", TouchEventKind::HoverEnter),
                ("hovering", TouchEventKind::Hovering),
                ("hover", TouchEventKind::Hovering),
                ("hover_exit", TouchEventKind::HoverExit),
                ("hoverexit", TouchEventKind::HoverExit),
                ("press", TouchEventKind::Press),
                ("release", TouchEventKind::Release),
                ("click", TouchEventKind::Click),
                ("double_click", TouchEventKind::DoubleClick),
                ("doubleclick", TouchEventKind::DoubleClick),
                ("drag_move", TouchEventKind::DragMove),
                ("dragging", TouchEventKind::DragMove),
                ("drag", TouchEventKind::DragMove),
                ("down", TouchEventKind::Down),
                ("d", TouchEventKind::Down),
                ("hold", TouchEventKind::Down),
                ("scroll", TouchEventKind::ScrollOnElement),
                ("h_enter", TouchEventKind::HoverEnter),
                ("h", TouchEventKind::Hovering),
                ("h_exit", TouchEventKind::HoverExit),
                ("p", TouchEventKind::Press),
                ("r", TouchEventKind::Release),
                ("c", TouchEventKind::Click),
                ("dc", TouchEventKind::DoubleClick),
                ("dr", TouchEventKind::DragMove),
                ("s", TouchEventKind::ScrollOnElement),
                ("sel", TouchEventKind::Select),
                ("desel", TouchEventKind::DeSelect),
            ],
        ) {
            filters.events.push(event);
            found_something = true;
        }

        // Try to extract an in: filter (mode)
        if let Some(mode) = try_extract_string_filter(action, "in:") {
            filters.modes.push(mode);
            found_something = true;
        }

        // If nothing was extracted, we're done
        if !found_something {
            break;
        }
    }
    if filters.buttons.is_empty() {
        filters.buttons.push(ParsedButton::Left);
    }
    filters
}

fn try_extract_filter<T: Clone>(
    action: &mut String,
    prefix: &str,
    options: &[(&str, T)],
) -> Option<T> {
    if let Some(start) = action.find(prefix) {
        let rest = &action[start + prefix.len()..];
        if rest.starts_with('"') {
            return None;
        }
        for (name, value) in options {
            if rest.starts_with(name) {
                // Verify it's a complete token (followed by separator or end)
                let after = start + prefix.len() + name.len();
                if after >= action.len() || is_separator(action.as_bytes()[after]) {
                    remove_filter_span(action, start, after);
                    return Some(value.clone());
                }
            }
        }

        // Unrecognized value after prefix
        let end = rest
            .find(|c: char| is_separator(c as u8))
            .unwrap_or(rest.len());
        let invalid = &rest[..end];
        let valid_options: Vec<_> = options.iter().map(|(n, _)| *n).collect();
        println!(
            "Invalid filter: '{}{}'. Valid options: {}",
            prefix,
            invalid,
            valid_options.join(", ")
        );
    }

    None
}

fn try_extract_string_filter(action: &mut String, prefix: &str) -> Option<String> {
    if let Some(start) = action.find(prefix) {
        let rest = &action[start + prefix.len()..];

        // Extract until separator
        let end = rest
            .find(|c: char| is_separator(c as u8))
            .unwrap_or(rest.len());

        if end > 0 {
            let value = rest[..end].to_string();
            remove_filter_span(action, start, start + prefix.len() + end);
            return Some(value);
        }
    }

    None
}

fn is_separator(b: u8) -> bool {
    b.is_ascii_whitespace() || b == b',' || b == b')' || b == b'('
}

fn remove_filter_span(action: &mut String, start: usize, end: usize) {
    let bytes = action.as_bytes();
    let mut actual_start = start;
    let mut actual_end = end;

    // Trim trailing comma/whitespace
    while actual_end < bytes.len()
        && (bytes[actual_end] == b',' || bytes[actual_end].is_ascii_whitespace())
    {
        actual_end += 1;
    }

    // Trim leading comma/whitespace
    while actual_start > 0 {
        let prev = bytes[actual_start - 1];
        if prev == b',' || prev.is_ascii_whitespace() {
            actual_start -= 1;
        } else {
            break;
        }
    }

    // If we trimmed leading, don't also trim trailing comma
    if actual_start < start {
        actual_end = end;
    }

    action.drain(actual_start..actual_end);
}

fn filters_match(
    input: &mut Input,
    settings: &Settings,
    filters: &ActionFilters,
    event_kind: &TouchEventKind,
    buttons: &MouseButtons,
) -> bool {
    // Check button filters (if any specified, at least one must match)
    if !filters.buttons.is_empty() {
        let any_button_matches = filters
            .buttons
            .iter()
            .any(|b| button_matches(input, b.clone(), buttons));
        if !any_button_matches {
            return false;
        }
    }

    // Check event filters (if any specified, at least one must match)
    if !filters.events.is_empty() {
        let any_event_matches = filters.events.iter().any(|e| e == event_kind);
        if !any_event_matches {
            return false;
        }
    }
    // If in editor mode, require the action to explicitly allow editor_mode.
    // Actions with no in: filter will be rejected while editor_mode is true.
    if settings.editor_mode && filters.modes.is_empty() {
        return false;
    }
    // Check mode filters
    if !filters.modes.is_empty() {
        let any_mode_matches = filters.modes.iter().any(|m| match m.as_str() {
            "editor_mode" => settings.editor_mode,
            "play_mode" => !settings.editor_mode,
            // Add more modes here as needed
            other => {
                println!("Unknown mode filter: '{}'", other);
                false
            }
        });

        if !any_mode_matches {
            return false;
        }
    }

    true
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum ParsedButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Any,
    Key(String),
}
