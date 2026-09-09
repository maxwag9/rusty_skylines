use crate::data::Settings;
#[allow(unused_mut, unused_assignments)]
use crate::ui::actions::UiCommand;
use crate::ui::actions::{CommandContext, CommandQueue, ElementContext, string_to_value};
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::ui_editor::{Menus, get_element_active, get_element_kind, get_layer_actions};
use crate::ui::ui_touch_manager::GlobalEvent;
use crate::ui::ui_touch_manager::{ElementEvent, ElementRef};
use std::cmp::PartialEq;

/// Helper trait for parsing argument types
pub trait ParseArg: Sized {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self>;
}
impl ParseArg for String {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let s = args.get(*idx)?.clone();
        *idx += 1;
        Some(s)
    }
}

impl ParseArg for Option<String> {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        if let Some(s) = args.get(*idx) {
            *idx += 1;
            Some(Some(s.clone()))
        } else {
            Some(None)
        }
    }
}
impl ParseArg for f32 {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let value = args.get(*idx)?.parse().ok()?;
        *idx += 1;
        Some(value)
    }
}

impl ParseArg for f64 {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let value = args.get(*idx)?.parse().ok()?;
        *idx += 1;
        Some(value)
    }
}

impl ParseArg for i32 {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let value = args.get(*idx)?.parse().ok()?;
        *idx += 1;
        Some(value)
    }
}

impl ParseArg for u32 {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let value = args.get(*idx)?.parse().ok()?;
        *idx += 1;
        Some(value)
    }
}

impl ParseArg for usize {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let value = args.get(*idx)?.parse().ok()?;
        *idx += 1;
        Some(value)
    }
}

impl ParseArg for bool {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        let value = args.get(*idx)?.parse().ok()?;
        *idx += 1;
        Some(value)
    }
}
impl ParseArg for Option<bool> {
    fn parse_arg(args: &[String], idx: &mut usize) -> Option<Self> {
        if let Some(value) = args.get(*idx) {
            *idx += 1;
            Some(Some(value.parse().ok()?))
        } else {
            Some(None)
        }
    }
}

/// Macro to define command mappings - just add lines when I add commands!
macro_rules! define_commands {
    (
        $(
            $( $name:literal )|+ => $variant:ident
            $( { $( $field:ident : $ftype:ty ),* $(,)? } )?
        ),* $(,)?
    ) => {

        pub fn make_ui_command(
            func_name: &str,
            args: Vec<String>
        ) -> Option<UiCommand> {
            let name = func_name.to_ascii_lowercase();

            match name.as_str() {
                $(
                    $( $name )|+ => {
                        define_commands!(@build args, $variant $( { $( $field : $ftype ),* } )?)
                    }
                ),*,
                _ => {
                    eprintln!("[Warning] Unknown UI command: '{}'", func_name);
                    None
                }
            }
        }
    };

    // unit variant
    (@build $args:ident, $variant:ident) => {
    Some(UiCommand::$variant)
    };

    // struct variant
    (@build $args:ident, $variant:ident { $( $field:ident : $ftype:ty ),* }) => {{
        let mut idx = 0usize;

        $(
            let $field = define_commands!(
                @parse $args, idx, $field, $ftype
            )?;
        )*

        Some(UiCommand::$variant { $( $field ),* })
    }};

    // SPECIAL FIELD: commands
    (@parse $args:ident, $idx:ident, commands, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            #[allow(unused_assignments)]
            {
                $idx += 1;
            }

            let mut cmds = Vec::new();
            let mut start = 0;
            let mut depth = 0isize; // Track () [] {}
            let bytes = raw.as_bytes();

            for i in 0..bytes.len() {
                match bytes[i] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b';' if depth == 0 => {
                        // Only split on ';' if we are not inside brackets/parens
                        let part = raw[start..i].trim();
                        if !part.is_empty() {
                            cmds.extend(parse_primitive_action(part))
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }

            // Don't forget the last command after the final ';'
            let part = raw[start..].trim();
            if !part.is_empty() {
                cmds.extend(parse_primitive_action(part))
            }

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    // SPECIAL FIELD: then / else_branch
    (@parse $args:ident, $idx:ident, then, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            #[allow(unused_assignments)]
            {
                $idx += 1;
            }

            let mut cmds = Vec::new();
            let mut start = 0;
            let mut depth = 0isize; // Track () [] {}
            let bytes = raw.as_bytes();

            for i in 0..bytes.len() {
                match bytes[i] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b';' if depth == 0 => {
                        // Only split on ';' if we are not inside brackets/parens
                        let part = raw[start..i].trim();
                        if !part.is_empty() {
                            cmds.extend(parse_primitive_action(part))
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }

            // Don't forget the last command after the final ';'
            let part = raw[start..].trim();
            if !part.is_empty() {
                cmds.extend(parse_primitive_action(part))
            }

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    (@parse $args:ident, $idx:ident, else_branch, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            #[allow(unused_assignments)]
            {
                $idx += 1;
            }

            let mut cmds = Vec::new();
            let mut start = 0;
            let mut depth = 0isize; // Track () [] {}
            let bytes = raw.as_bytes();

            for i in 0..bytes.len() {
                match bytes[i] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b';' if depth == 0 => {
                        // Only split on ';' if we are not inside brackets/parens
                        let part = raw[start..i].trim();
                        if !part.is_empty() {
                            cmds.extend(parse_primitive_action(part))
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }

            // Don't forget the last command after the final ';'
            let part = raw[start..].trim();
            if !part.is_empty() {
                cmds.extend(parse_primitive_action(part))
            }

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    (@branch $args:ident, $idx:ident) => {{
        let raw = $args.get($idx)?;
        #[allow(unused_assignments)]
        {
            $idx += 1;
        }

        let cmds = raw.split(';').map(str::trim).filter(|s| !s.is_empty()).map(|s| parse_primitive_action(s)).flatten().collect();

        Some(cmds)
    }};
    // SPECIAL FIELD: raw String
    (@parse $args:ident, $idx:ident, $field:ident, String) => {{
        let val = $args.get($idx).cloned().unwrap_or_default();
        $idx += 1;
        Some(val)
    }};
    // GENERIC FIELD PARSER
    (@parse $args:ident, $idx:ident, $field:ident, $ftype:ty) => {{
        let val = <$ftype as ParseArg>::parse_arg(&$args, &mut $idx)?;
        Some(val)
    }};
}

// COMMAND DEFINITIONS - I Just add a line here when I add a new UiCommand!
define_commands! {
    // ===== MENU COMMANDS =====
    "open_menu" | "openmenu"
        => OpenMenu {menu_name: String },

    "close_menu" | "closemenu"
        => CloseMenu {menu_name: String },

    "close_all_menus" | "closeall"
        => CloseAllMenus,
    "close_all_layers"
        => CloseAllLayers {menu_name: String },
    "toggle_menu" | "togglemenu"
        => ToggleMenu {menu_name: String },

    "menu_active" | "menuactive"
        => MenuActive {menu_name: String },

    // ===== LAYER COMMANDS =====
    "open_layer" | "openlayer"
        => OpenLayer {menu_name: String, layer_name: String },

    "close_layer" | "closelayer"
        => CloseLayer {menu_name: String, layer_name: String },

    "toggle_layer" | "togglelayer"
        => ToggleLayer {menu_name: String, layer_name: String },

    // ===== VARIABLE COMMANDS =====
    "set_var" | "setvar" | "set"
        => SetVar {name: String, value: String },

    "inc_var" | "incvar" | "inc"
        => IncVar {name: String, amount: String },

    "dec_var" | "decvar" | "dec"
        => DecVar {name: String, amount: String },

    "mul_var" | "mulvar" | "mul"
        => MulVar {name: String, factor: String },

    "toggle_var" | "togglevar" | "toggle"
        => ToggleVar {name: String },

    "clamp" | "clampvar"
        => Clamp {name: String, min: String, max: String },

    // ===== FLOW CONTROL =====
    "delay" | "wait" | "sleep"
        => Delay {seconds: String },

    "halt" | "break"
        => Halt,

    "skip"
        => Skip { count: usize },
    "for" | "forin" => For {value: String, commands: Vec<UiCommand> },
    "if" => If {condition: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand> },

    "ifvareq"
        => IfVarEq {var_name: String, value: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand>},

    "add_element" | "addelem" | "add"
        => AddElement {menu: String, layer: String, id: String, kind: String, center: String, actions: String, undoable: bool},

    "add_ap" | "addap"
        => AddAP {menu: String, name: String, ap_name: String, ap_var: String, center: String, scale: String, is_temporary: bool},

    "del_ap" | "delap"
        => DeleteAP {menu: String, layer: String, reference_id: String},

    "clone_element" | "cloneelem" | "clone"
        => CloneElement {
        from_menu: String,
        from_layer: String,
        from_id: String,
        to_menu: String,
        to_layer: String,
        to_id: String,
        center: String,
        actions: String,
        undoable: bool
    },

    "clone_layer" | "clonelayer"
        => CloneLayer {
        from_menu: String,
        from_layer: String,
        to_menu: String,
        to_layer: String,
        undoable: bool},

    "delete_element" | "delelem" | "delete"
        => DeleteElement {menu: String, layer: String, id: String, undoable: bool},

    "delete_layer" | "dellayer"
        => DeleteLayer {menu: String, layer: String, undoable: bool},

    "save" | "savegame" => SaveGame {and_exit: String },

    "load" | "loadgame" | "load_save"
        => LoadSave {save_name: String, without_saving: bool  },

    "exit_game" | "leave_game"
        => ExitGame,
    // ===== DEBUG COMMANDS =====
    "print" | "log" | "echo"
        => Print {statement: String },

    "debug_vars" | "debugvars"
        => DebugVars,

    "debug_menus" | "debugmenus"
        => DebugMenus,

    "call" => Call {function_name: String, args: Option<String> },

    // ===== UTILITY =====
    "noop" | "no_op" | "none"
        => Noop
}
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum ActionEvent {
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
    Always,
    Activated,
    Deactivated,
    StartUp,
    ScreenResize,
    DragStart,
    DragEnd,
}
impl ActionEvent {
    pub fn from_element_event(element_event: &ElementEvent) -> ActionEvent {
        match element_event {
            ElementEvent::HoverEnter { .. } => ActionEvent::HoverEnter,

            ElementEvent::Hovering { .. } => ActionEvent::Hovering,

            ElementEvent::HoverExit { .. } => ActionEvent::HoverExit,

            ElementEvent::Press { .. } => ActionEvent::Press,

            ElementEvent::Down { .. } => ActionEvent::Down,

            ElementEvent::Release { .. } => ActionEvent::Release,

            ElementEvent::Click { .. } => ActionEvent::Click,

            ElementEvent::DoubleClick { .. } => ActionEvent::DoubleClick,

            ElementEvent::DragStart { .. } => ActionEvent::DragStart,

            ElementEvent::DragMove { .. } => ActionEvent::DragMove,

            ElementEvent::DragEnd { .. } => ActionEvent::DragEnd,

            ElementEvent::ScrollOnElement { delta, .. } => ActionEvent::ScrollOnElement,
            ElementEvent::SelectionRequested { .. } => ActionEvent::Select,

            ElementEvent::Nothing { .. } => ActionEvent::Nothing,

            ElementEvent::Activated { .. } => ActionEvent::Activated,

            ElementEvent::Deactivated { .. } => ActionEvent::Deactivated,

            ElementEvent::TextEditRequested => ActionEvent::Nothing,

            ElementEvent::TextEditEnded => ActionEvent::Nothing,
        }
    }
    pub fn from_global_event(global_event: &GlobalEvent) -> ActionEvent {
        match global_event {
            GlobalEvent::DeselectAllRequested {} => ActionEvent::DeSelect,
            GlobalEvent::StartUp { .. } => ActionEvent::StartUp,

            GlobalEvent::ScreenResize { .. } => ActionEvent::ScreenResize,

            GlobalEvent::BoxSelectStart { .. } => ActionEvent::Nothing,
            GlobalEvent::BoxSelectMove { .. } => ActionEvent::Nothing,
            GlobalEvent::BoxSelectEnd { .. } => ActionEvent::Nothing,
            GlobalEvent::NavigateDirection { .. } => ActionEvent::Nothing,
        }
    }
    pub fn is_always_event(&self) -> bool {
        match self {
            ActionEvent::Select => true,
            ActionEvent::DeSelect => true,
            ActionEvent::Activated => true,
            ActionEvent::Deactivated => true,
            ActionEvent::StartUp => true,
            ActionEvent::ScreenResize => true,
            _ => false,
        }
    }
}
/// ONLY for elements/layer/global_element actions, NOT GLOBAL global ACTIONS!
pub fn run_actions(
    command_queue: &mut CommandQueue,
    ctx: &mut CommandContext,
    element: &ElementRef,
    element_actions: Vec<CompiledAction>,
) {
    let layer_actions: Vec<CompiledAction> = get_layer_actions(&ctx.ui.menus, element);
    for action in element_actions {
        run_action(command_queue, action, ctx);
    }
    for action in layer_actions {
        run_action_with_element(command_queue, action, ctx, element.clone());
    }
    for action in ctx
        .ui
        .global_actions
        .element_compiled_actions
        .clone()
        .into_iter()
    {
        // Thanks maxim MAxim Maxim
        run_action_with_element(command_queue, action, ctx, element.clone());
    }
}

/// Handle a single action string that may be an event wrapper
fn handle_action_str(action: &str) -> Vec<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return vec![];
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
        return vec![];
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
        None => return vec![], // Unbalanced parens
    };

    let inner = s[open_paren + 1..close_paren].trim();
    process_inner_content(inner)
}

/// Process the inner content of a matched event wrapper
fn process_inner_content(inner: &str) -> Vec<UiCommand> {
    for part in split_top_level(inner, b',') {
        let cmds = handle_action_str(part);

        if !cmds.is_empty() {
            return cmds;
        }

        let cmds = parse_primitive_action(part);

        if !cmds.is_empty() {
            return cmds;
        }
    }

    vec![]
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

    let mut args = vec![s.to_string()];
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

fn split_top_level_semicolons(s: &str) -> Vec<&str> {
    split_top_level(s, b';')
}
fn split_top_level(s: &str, delimiter: u8) -> Vec<&str> {
    let bytes = s.as_bytes();

    let mut parts = Vec::new();
    let mut start = 0usize;

    let mut paren_depth = 0isize;
    let mut bracket_depth = 0isize;
    let mut brace_depth = 0isize;

    let mut in_string = false;
    let mut escape = false;

    for i in 0..bytes.len() {
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

            _ if b == delimiter && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                let part = s[start..i].trim();

                if !part.is_empty() {
                    parts.push(part);
                }

                start = i + 1;
            }

            _ => {}
        }
    }

    let part = s[start..].trim();

    if !part.is_empty() {
        parts.push(part);
    }

    parts
}

fn parse_primitive_action(action: &str) -> Vec<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return Vec::new();
    }

    let chained = split_top_level_semicolons(s);
    if chained.len() > 1 {
        return chained
            .into_iter()
            .flat_map(|part| parse_primitive_action(part))
            .collect();
    }

    let mut out = Vec::new();

    if let Some(open_paren) = s.find('(') {
        let func_name = s[..open_paren].trim();
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

            if let Some(cmd) = make_ui_command(func_name, args) {
                out.push(cmd);
            }

            let rest = s[close_paren + 1..].trim();
            if let Some(rest) = rest.strip_prefix(';') {
                out.extend(parse_primitive_action(rest));
            }

            return out;
        }
    }

    if let Some(cmd) = make_ui_command(s, Vec::new()) {
        out.push(cmd);
    }

    out
}

fn button_matches(
    input: &mut Input,
    button: ParsedButton,
    keybind_trigger: KeyBindTrigger,
) -> bool {
    let buttons = &input.mouse.buttons;
    let state = match button {
        ParsedButton::Any => return true,
        ParsedButton::Left => &buttons.left,
        ParsedButton::Right => &buttons.right,
        ParsedButton::Middle => &buttons.middle,
        ParsedButton::Back => &buttons.back,
        ParsedButton::Forward => &buttons.forward,
        ParsedButton::Key(key) => {
            return match keybind_trigger {
                KeyBindTrigger::Down => {
                    if input.action_known(&key) {
                        input.action_down(&key)
                    } else {
                        input.combo_down(&key)
                    }
                }

                KeyBindTrigger::Press => {
                    if input.action_known(&key) {
                        input.action_pressed_once(&key)
                    } else {
                        input.combo_pressed_once(&key)
                    }
                }

                KeyBindTrigger::Release => {
                    if input.action_known(&key) {
                        let r = input.action_released(&key);
                        r
                    } else {
                        input.combo_released(&key)
                    }
                }
                KeyBindTrigger::Repeat => {
                    if input.action_known(&key) {
                        let r = input.action_repeat(&key);
                        r
                    } else {
                        input.combo_repeat(&key)
                    }
                }
            };
        }
    };
    state.pressed || state.just_released
}
#[derive(Default, Debug, Clone)]
pub enum AsFilter {
    Compiled(ElementRef),
    Dynamic(Vec<String>),
    #[default]
    None,
}
impl AsFilter {
    pub fn dynamic(&self) -> Option<&[String]> {
        match self {
            AsFilter::Dynamic(arr) => Some(arr),
            _ => None,
        }
    }
}
#[derive(Default, Debug, Clone)]
struct ActionFilters {
    pub scope: ActionScope,
    pub buttons: Vec<ParsedButton>,
    pub events: Vec<ActionEvent>,
    pub keybind_trigger: KeyBindTrigger,
    pub modes: Vec<String>,
    pub as_filter: AsFilter,
}
#[derive(Default, Debug, Copy, Clone)]
enum KeyBindTrigger {
    #[default]
    Down,
    Press,
    Release,
    Repeat,
}
#[derive(Debug, Clone, Default)]
pub enum ActionScope {
    #[default]
    ActiveOnly,
    Always,
}
#[derive(Debug, Clone)]
pub struct CompiledAction {
    pub element_ctx: ElementContext,
    pub filters: ActionFilters,
    pub commands: Vec<UiCommand>,
}
pub fn run_action(
    command_queue: &mut CommandQueue,
    action: CompiledAction,
    ctx: &mut CommandContext,
) {
    let mut element_ctx = action.element_ctx;
    prepare_filters(ctx, &action.filters, &mut element_ctx);
    if filters_match(
        &ctx.ui.menus,
        &mut ctx.world.input,
        ctx.settings,
        &action.filters,
        ctx.ui.action_events.as_slice(),
        &element_ctx,
    ) {
        ctx.element_ctx = element_ctx;
        command_queue.execute_multiple(action.commands, ctx);
    };
}
pub fn run_action_with_element(
    command_queue: &mut CommandQueue,
    action: CompiledAction,
    ctx: &mut CommandContext,
    element: ElementRef,
) {
    let mut element_ctx = ElementContext {
        self_element: Some(element),
        as_element: None,
    };
    prepare_filters(ctx, &action.filters, &mut element_ctx);
    if filters_match(
        &ctx.ui.menus,
        &mut ctx.world.input,
        ctx.settings,
        &action.filters,
        ctx.ui.action_events.as_slice(),
        &element_ctx,
    ) {
        ctx.element_ctx = element_ctx;
        command_queue.execute_multiple(action.commands, ctx);
    };
}
pub fn run_action_with_events(
    command_queue: &mut CommandQueue,
    action: CompiledAction,
    ctx: &mut CommandContext,
    events: &[ActionEvent],
) {
    let mut element_ctx = action.element_ctx;
    prepare_filters(ctx, &action.filters, &mut element_ctx);
    if filters_match(
        &ctx.ui.menus,
        &mut ctx.world.input,
        ctx.settings,
        &action.filters,
        events,
        &element_ctx,
    ) {
        ctx.ui.action_events = events.to_vec();
        ctx.element_ctx = element_ctx;
        command_queue.execute_multiple(action.commands, ctx);
    };
}

pub fn compile_actions(
    menus: &Menus,
    self_element: Option<ElementRef>,
    string_actions: Vec<String>,
) -> Vec<CompiledAction> {
    // Command context for parsing as:, which parses an array, I should compile that too, but whatever.
    let mut compiled_actions = Vec::new();
    for mut action_string in string_actions {
        let mut element_ctx = ElementContext {
            self_element: self_element.clone(),
            as_element: None,
        };
        let filters = parse_action_filters(&mut element_ctx, &mut action_string, menus);

        let commands = parse_primitive_action(action_string.trim());
        let compiled_action = CompiledAction {
            element_ctx,
            filters,
            commands,
        };
        compiled_actions.push(compiled_action);
    }
    compiled_actions
}

fn parse_action_filters(
    element_ctx: &mut ElementContext,
    action: &mut String,
    menus: &Menus,
) -> ActionFilters {
    let mut filters = ActionFilters::default();
    let mut consumed = 0usize;

    loop {
        let (rest, ws) = trim_leading_whitespace(&action[consumed..]);
        consumed += ws;

        if rest.is_empty() {
            break;
        }

        if rest.starts_with(',') {
            consumed += 1;
            continue;
        }

        // Do not parse filters after entering a command/string.
        // A filter token can only exist before the first command '('.
        if let Some(paren) = rest.find('(') {
            let before = &rest[..paren];

            // "set(" / "if(" etc. means filters are finished.
            if !before.contains(':') {
                break;
            }
        }

        let parsed = try_parse_button(rest, &mut filters)
            .or_else(|| try_parse_scope(rest, &mut filters))
            .or_else(|| try_parse_trigger(rest, &mut filters))
            .or_else(|| try_parse_on(rest, &mut filters))
            .or_else(|| try_parse_in(rest, &mut filters))
            .or_else(|| try_parse_as(rest, &mut filters, element_ctx, menus));

        let Some(used) = parsed else {
            break;
        };

        consumed += used;
    }

    action.drain(..consumed);
    filters
}

fn try_parse_button(input: &str, filters: &mut ActionFilters) -> Option<usize> {
    let (value, consumed) = parse_prefixed_value(input, "button:")?;

    let button = match value.to_ascii_lowercase().as_str() {
        "any" => ParsedButton::Any,
        "right" => ParsedButton::Right,
        "middle" => ParsedButton::Middle,
        "back" => ParsedButton::Back,
        "forward" => ParsedButton::Forward,
        "left" => ParsedButton::Left,
        "a" => ParsedButton::Any,
        "l" => ParsedButton::Left,
        "r" => ParsedButton::Right,
        "m" => ParsedButton::Middle,
        "b" => ParsedButton::Back,
        "f" => ParsedButton::Forward,
        _ => ParsedButton::Key(value),
    };

    filters.buttons.push(button);
    Some(consumed)
}
fn try_parse_scope(input: &str, filters: &mut ActionFilters) -> Option<usize> {
    let (value, consumed) = parse_prefixed_value(input, "scope:")?;

    let scope = match value.to_ascii_lowercase().as_str() {
        "a" | "always" => ActionScope::Always,
        "active" => ActionScope::ActiveOnly,
        _ => {
            println!("Invalid on filter: scope:{}", value);
            return None;
        }
    };

    filters.scope = scope;
    Some(consumed)
}
fn try_parse_trigger(input: &str, filters: &mut ActionFilters) -> Option<usize> {
    let (value, consumed) = parse_prefixed_value(input, "trigger:")?;

    filters.keybind_trigger = match value.to_ascii_lowercase().as_str() {
        "down" | "d" => KeyBindTrigger::Down,
        "press" | "p" => KeyBindTrigger::Press,
        "release" | "r" => KeyBindTrigger::Release,
        "repeat" | "rp" => KeyBindTrigger::Repeat,
        _ => {
            println!(
                "Invalid Keybind trigger: {}, options are: down, d, press, p, release, r, repeat, rp",
                value
            );
            return None;
        }
    };

    Some(consumed)
}

fn try_parse_on(input: &str, filters: &mut ActionFilters) -> Option<usize> {
    let (value, consumed) = parse_prefixed_value(input, "on:")?;

    let event = match value.to_ascii_lowercase().as_str() {
        "a" | "always" => ActionEvent::Always,
        "n" | "nothing" => ActionEvent::Nothing,
        "hover_enter" | "hoverenter" | "h_enter" => ActionEvent::HoverEnter,
        "hovering" | "hover" | "h" => ActionEvent::Hovering,
        "hover_exit" | "hoverexit" | "h_exit" => ActionEvent::HoverExit,
        "press" | "p" => ActionEvent::Press,
        "release" | "r" => ActionEvent::Release,
        "click" | "c" => ActionEvent::Click,
        "double_click" | "doubleclick" | "dc" => ActionEvent::DoubleClick,
        "drag_start" => ActionEvent::DragStart,
        "drag_move" | "dragging" | "drag" | "dr" => ActionEvent::DragMove,
        "drag_end" => ActionEvent::DragEnd,
        "down" | "d" | "hold" => ActionEvent::Down,
        "scroll" | "s" => ActionEvent::ScrollOnElement,
        "sel" => ActionEvent::Select,
        "desel" => ActionEvent::DeSelect,
        "activated" => ActionEvent::Activated,
        "deactivated" => ActionEvent::Deactivated,
        "startup" => ActionEvent::StartUp,
        "screen_resize" => ActionEvent::ScreenResize,
        _ => {
            println!("Invalid on filter: on:{}", value);
            return None;
        }
    };

    filters.events.push(event);
    Some(consumed)
}

fn try_parse_in(input: &str, filters: &mut ActionFilters) -> Option<usize> {
    let (value, consumed) = parse_prefixed_value(input, "in:")?;
    filters.modes.push(value);
    Some(consumed)
}

fn try_parse_as(
    input: &str,
    filters: &mut ActionFilters,
    element_ctx: &mut ElementContext,
    menus: &Menus,
) -> Option<usize> {
    filters.as_filter = AsFilter::None;
    // Is asdyn:?
    if let Some((raw, consumed)) = parse_prefixed_value(input, "asdyn:") {
        let inner = if raw.starts_with('[') && raw.ends_with(']') {
            &raw[1..raw.len() - 1]
        } else {
            raw.as_str()
        };
        let arr = Value::split_array_elements(inner);
        filters.as_filter = AsFilter::Dynamic(arr.iter().map(|str| str.to_string()).collect());
        return Some(consumed);
    };
    // Is as:?
    let (raw, consumed) = parse_prefixed_value(input, "as:")?;

    let Some(arr) = Value::parse_array_pure(raw.as_str()) else {
        return Some(consumed);
    };

    let self_element = if let Some(current) = element_ctx.self_element.as_ref() {
        current
    } else if arr.len() == 3 {
        &ElementRef::default()
    } else {
        return Some(consumed);
    };
    //println!("{:?}", arr);
    let (menu, layer, id) = match arr.as_slice() {
        [id] => (
            self_element.menu.clone(),
            self_element.layer.clone(),
            id.to_string(),
        ),
        [layer, id] => (self_element.menu.clone(), layer.to_string(), id.to_string()),
        [menu, layer, id, ..] => (menu.to_string(), layer.to_string(), id.to_string()),
        [] => return Some(consumed),
    };
    let Some(kind) = get_element_kind(menus, menu.as_str(), layer.as_str(), id.as_str()) else {
        return Some(consumed);
    }; // grad sport gleich sport, hmm grad sport, deswegen schwitze ich mama
    let as_element = ElementRef {
        menu,
        layer,
        id,
        kind,
    };
    //println!("{:?}", as_element);
    filters.as_filter = AsFilter::Compiled(as_element.clone());
    element_ctx.as_element = Some(as_element);

    Some(consumed)
}
fn prepare_filters(
    ctx: &mut CommandContext,
    filters: &ActionFilters,
    element_ctx: &mut ElementContext,
) {
    let Some(arr) = filters.as_filter.dynamic() else {
        return;
    };
    let self_element = if let Some(current) = element_ctx.self_element.as_ref() {
        current
    } else if arr.len() == 3 {
        &ElementRef::default()
    } else {
        return;
    };
    ctx.element_ctx = element_ctx.clone(); // Very important for string_to_value()
    let mut resolve = |s: String| string_to_value(ctx, s).into_string();
    //println!("{:?}", arr);
    let (menu, layer, id) = match arr {
        [id] => (
            self_element.menu.clone(),
            self_element.layer.clone(),
            resolve(id.to_string()),
        ),
        [layer, id] => (
            self_element.menu.clone(),
            resolve(layer.to_string()),
            resolve(id.to_string()),
        ),
        [menu, layer, id, ..] => (
            resolve(menu.to_string()),
            resolve(layer.to_string()),
            resolve(id.to_string()),
        ),
        [] => return,
    };
    let Some(kind) = get_element_kind(&ctx.ui.menus, menu.as_str(), layer.as_str(), id.as_str())
    else {
        return;
    }; // grad sport gleich sport, hmm grad sport, deswegen schwitze ich mama
    let as_element = ElementRef {
        menu,
        layer,
        id,
        kind,
    };
    element_ctx.as_element = Some(as_element);
}
// fn try_parse_as(
//     input: &str,
//     filters: &mut ActionFilters,
//     element_ctx: &mut ElementContext,
//     ctx: &mut CommandContext
// ) -> Option<usize> {
//     let (raw, consumed) = parse_prefixed_value(input, "as:")?;
//     let arr = Value::parse_array(&ctx.settings, &ctx.ui.variables, raw.as_str())?;
//     let self_element = if let Some(current) = element_ctx.self_element.as_ref() {
//         current
//     } else if arr.len() == 3 {
//         &ElementRef::default()
//     } else { return Some(consumed) };
//     let mut resolve = |s: String| string_to_value(ctx, element_ctx, s).into_string();
//
//
//     let (menu, layer) = match arr.as_slice() {
//         [_id] => (
//             self_element.menu.clone(),
//             self_element.layer.clone()
//         ),
//         [layer, _id] => (
//             self_element.menu.clone(),
//             resolve(layer.to_string())
//         ),
//         [menu, layer, _id, ..] => (
//             resolve(menu.to_string()),
//             resolve(layer.to_string())
//         ),
//         [] => return Some(consumed),
//     };
//     let mut as_elements = Vec::new();
//     let mut resolve_ids = |id: &Value| {
//         if let Some(arr_id) = id.as_array() {
//             for id in arr_id.iter() {
//                 let id = id.to_string();
//                 let Some(kind) = get_element_kind(&ctx.ui.menus, menu.as_str(), layer.as_str(), id.as_str()) else { continue };
//                 as_elements.push(ElementRef {
//                     menu: menu.clone(),
//                     layer: layer.clone(),
//                     id,
//                     kind
//                 })
//             }
//         }
//     };
//     match arr.as_slice() {
//         [id] => resolve_ids(id),
//         [_layer, id] => resolve_ids(id),
//         [_menu, _layer, id, ..] => resolve_ids(id),
//         [] => return Some(consumed),
//     };
//     // grad sport gleich sport, hmm grad sport, deswegen schwitze ich mama
//     element_ctx.as_elements = as_elements;
//
//     Some(consumed)
// }
fn parse_prefixed_value(input: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = input.strip_prefix(prefix)?;

    if rest.is_empty() {
        return None;
    }

    let (value, value_len) = if rest.starts_with('"') {
        parse_quoted_value(rest)?
    } else {
        parse_bare_value(rest)?
    };

    let after = &rest[value_len..];
    if let Some(ch) = after.chars().next() {
        if !ch.is_whitespace() && ch != ',' {
            return None;
        }
    }

    Some((value, prefix.len() + value_len))
}

fn parse_bare_value(input: &str) -> Option<(String, usize)> {
    let mut square_depth = 0;
    let mut curly_depth = 0;
    let mut round_depth = 0;

    for (i, c) in input.char_indices() {
        match c {
            '[' => square_depth += 1,
            ']' => square_depth -= 1,
            '{' => curly_depth += 1,
            '}' => curly_depth -= 1,
            '(' => round_depth += 1,
            ')' => round_depth -= 1,

            c if c.is_whitespace() && square_depth == 0 && curly_depth == 0 && round_depth == 0 => {
                if i == 0 {
                    return None;
                }
                return Some((input[..i].to_string(), i));
            }

            ',' if square_depth == 0 && curly_depth == 0 && round_depth == 0 => {
                if i == 0 {
                    return None;
                }
                return Some((input[..i].to_string(), i));
            }

            _ => {}
        }
    }

    if input.is_empty() {
        None
    } else {
        Some((input.to_string(), input.len()))
    }
}

fn parse_quoted_value(input: &str) -> Option<(String, usize)> {
    let mut escaped = false;

    for (i, c) in input.char_indices().skip(1) {
        if escaped {
            escaped = false;
            continue;
        }

        match c {
            '\\' => escaped = true,
            '"' => {
                let value = input[1..i].to_string();
                return Some((value, i + 1));
            }
            _ => {}
        }
    }

    None
}

fn trim_leading_whitespace(s: &str) -> (&str, usize) {
    let trimmed = s.trim_start();
    let consumed = s.len() - trimmed.len();
    (trimmed, consumed)
}

fn filters_match(
    menus: &Menus,
    input: &mut Input,
    settings: &Settings,
    filters: &ActionFilters,
    events: &[ActionEvent],
    element_ctx: &ElementContext,
) -> bool {
    match filters.scope {
        ActionScope::ActiveOnly => {
            if !filters.events.is_empty() {
                let event_requires_active = filters
                    .events
                    .iter()
                    .filter(|filter_event| events.contains(filter_event))
                    .any(|event| !event.is_always_event());

                if event_requires_active {
                    let Some(element_ref) = element_ctx.self_element.as_ref() else {
                        return false;
                    };

                    if !get_element_active(menus, element_ref).unwrap_or(false) {
                        return false;
                    }
                }
            } else {
                let Some(element_ref) = element_ctx.self_element.as_ref() else {
                    return false;
                };

                if !get_element_active(menus, element_ref).unwrap_or(false) {
                    return false;
                }
            }
        }

        ActionScope::Always => {
            // Just let it pass
        }
    }

    // Check button filters
    if !filters.buttons.is_empty() {
        let any_button_matches = filters
            .buttons
            .iter()
            .any(|b| button_matches(input, b.clone(), filters.keybind_trigger));

        if !any_button_matches {
            return false;
        }
    }

    // Check event filters
    if !filters.events.is_empty() {
        let any_event_matches = filters.events.iter().any(|e| events.contains(e));

        if !any_event_matches {
            if !filters.events.iter().any(|e| e == &ActionEvent::Always) {
                return false;
            }
        }
    }

    // If in editor mode, require the action to explicitly allow editor_mode.
    if settings.editor_mode && filters.modes.is_empty() {
        return false;
    }

    // Check mode filters
    if !filters.modes.is_empty() {
        let any_mode_matches = filters.modes.iter().any(|m| match m.as_str() {
            "editor_mode" => settings.editor_mode,
            "play_mode" => !settings.editor_mode,
            "any" => true,
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
fn testing_slider() {
    let min: f64 = 1.0;
    let max: f64 = 10.0;
    let step = 1.0;

    let slider_bar_width = 140.0;
    let slider_bar_middle_x = 410.0;
    let knob_x = 334.0;
    let mouse_x = 362.0;

    let min_px = slider_bar_middle_x - slider_bar_width * 0.5;
    let max_px = slider_bar_middle_x + slider_bar_width * 0.5;
    let num_steps: f64 = (min - max).abs();
    let px_per_step: f64 = slider_bar_width / num_steps;

    let target_x = mouse_x;
    let left_step = target_x - px_per_step;
    let right_step = target_x + px_per_step;
    let abs_left = (target_x - left_step).abs();
    let abs_right = (target_x - right_step).abs();

    let knob_x = if abs_left < abs_right {
        left_step
    } else {
        right_step
    };
}
