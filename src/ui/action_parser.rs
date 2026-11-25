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
use tracing::error;

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
                    eprintln!("[Warning] Unknown UI command: '{}', Args: '{:?}'", func_name, args);
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
            Some(parse_action_sequence(raw))
        } else {
            Some(Vec::new())
        }
    }};

    // SPECIAL FIELD: then / else_branch
    (@parse $args:ident, $idx:ident, then, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            {
                $idx += 1;
            }
            Some(parse_action_sequence(raw))
        } else {
            Some(Vec::new())
        }
    }};

    (@parse $args:ident, $idx:ident, else_branch, $ftype:ty) => {{
        if let Some(raw) = $args.get($idx) {
            Some(parse_action_sequence(raw))
        } else {
            Some(Vec::new())
        }
    }};

    (@branch $args:ident, $idx:ident) => {{
        let raw = $args.get($idx)?;
        $idx += 1;
        Some(parse_action_sequence(raw))
    }};
    // SPECIAL FIELD: raw String
    (@parse $args:ident, $idx:ident, $field:ident, String) => {{
        let val = $args.get($idx).cloned().unwrap_or_default();
        {
            $idx += 1;
        }
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
    "text_utils" => TextUtils {menu: String, layer: String, id: String, action: String},
    "deselect_all" => DeselectAll,
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

    "layer_active" | "layeractive"
        => LayerActive {menu_name: String, layer_name: String },
    "undirty" => UnDirty {menu_name: String, layer_name: String },

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

    "toggle_var" | "togglevar" | "toggle"=> ToggleVar {name: String },

    "clamp" | "clampvar"=> Clamp {name: String, min: String, max: String },

    // ===== FLOW CONTROL =====
    "delay" | "wait" | "sleep"=> Delay {seconds: String },

    "halt" | "break"=> Halt,

    "skip"=> Skip { count: usize },
    "for" | "forin" => For {value: String, commands: Vec<UiCommand> },
    "if" => If {condition: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand> },

    "ifvareq"=> IfVarEq {var_name: String, value: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand>},

    "add_element" | "addelem" | "add"=> AddElement {menu: String, layer: String, id: String, kind: String, center: String, actions: String, undoable: bool},

    "add_ap" | "addap"=> AddAP {menu: String, name: String, ap_name: String, ap_var: String, center: String, scale: String, is_temporary: bool},

    "del_ap" | "delap"=> DeleteAP {menu: String, layer: String, reference_id: String},
    
    "del_aps" | "delaps"=> DeleteAPsInLayer {menu: String, layer: String},

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

    "delete_element" | "delelem" | "delete" => DeleteElement {menu: String, layer: String, id: String, undoable: bool},

    "delete_layer" | "dellayer" => DeleteLayer {menu: String, layer: String, undoable: bool},

    "save" | "savegame" => SaveGame {and_exit: String },

    "load" | "loadgame" | "load_save"
        => LoadSave {save_name: String, without_saving: bool  },

    "exit_game" | "leave_game"
        => ExitGame,
    // ===== DEBUG COMMANDS =====
    "print" | "log" | "echo"
        => Print {statement: String },
    "printnc"
        => PrintNoChat {statement: String },

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
    DeSelectAll,
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
    TextEditStart,
    TextEditStop,
    EditingText,
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

            ElementEvent::TextEditStart => ActionEvent::TextEditStart,

            ElementEvent::TextEditStop => ActionEvent::TextEditStop,
            ElementEvent::EditingText => ActionEvent::EditingText,
        }
    }
    pub fn from_global_event(global_event: &GlobalEvent) -> ActionEvent {
        match global_event {
            GlobalEvent::DeselectAllRequested => ActionEvent::DeSelectAll,
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
            ActionEvent::DeSelectAll => true,
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
/// Process the inner content of a matched event wrapper
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
//kkl/*kkl?*/

fn handle_action_str(action: &str) -> Vec<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return vec![];
    }

    let bytes = s.as_bytes();
    let len = s.len();

    let mut pos = 0;
    while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
        pos += 1;
    }

    if pos == 0 || pos >= len || bytes[pos] != b'(' {
        return vec![];
    }

    let open_paren = pos;

    let close_paren = match find_matching_close_paren(s, open_paren) {
        Some(cp) => cp,
        None => return vec![],
    };

    let inner = s[open_paren + 1..close_paren].trim();
    process_inner_content(inner)
}

fn process_inner_content(inner: &str) -> Vec<UiCommand> {
    for part in split_top_level(inner, b',') {
        let cmds = handle_action_str(&part);

        if !cmds.is_empty() {
            return cmds;
        }

        let cmds = parse_primitive_action(&part);

        if !cmds.is_empty() {
            return cmds;
        }
    }

    vec![]
}

pub fn parse_arguments(args_str: &str) -> Vec<String> {
    let s = args_str.trim();

    if s.is_empty() {
        return Vec::new();
    }

    let (args, saw_comma, _) =
        split_top_level_scoped(s, |ch| ch == ',');

    if saw_comma {
        args
    } else {
        vec![s.to_string()]
    }
}
fn parse_action_sequence(raw: &str) -> Vec<UiCommand> {
    split_top_level_semicolons(raw)
        .into_iter()
        .flat_map(|part| parse_primitive_action(&part))
        .collect()
}
fn split_top_level_semicolons(s: &str) -> Vec<String> {
    split_top_level(s, b';')
}

fn split_top_level(s: &str, delimiter: u8) -> Vec<String> {
    split_top_level_scoped(s, |ch| ch == delimiter as char).0
}

fn split_top_level_scoped<F>(
    s: &str,
    is_separator: F,
) -> (Vec<String>, bool, String)
where
    F: Fn(char) -> bool,
{
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut normalized = String::new();

    let mut paren_depth = 0isize;
    let mut bracket_depth = 0isize;
    let mut brace_depth = 0isize;

    let mut in_string = false;
    let mut saw_separator = false;

    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.peek().copied() {
                let at_top_level = !in_string
                    && paren_depth == 0
                    && bracket_depth == 0
                    && brace_depth == 0;

                if at_top_level && is_separator(next) {
                    chars.next();
                    current.push(next);
                    normalized.push(next);
                    continue;
                }

                if matches!(
                    next,
                    '"' | '\\' | '(' | ')' | '[' | ']' | '{' | '}'
                ) {
                    chars.next();

                    current.push('\\');
                    current.push(next);

                    normalized.push('\\');
                    normalized.push(next);

                    continue;
                }
            }

            current.push('\\');
            normalized.push('\\');
            continue;
        }

        if in_string {
            if ch == '"' {
                in_string = false;
            }

            current.push(ch);
            normalized.push(ch);
            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                current.push(ch);
                normalized.push(ch);
            }
            '(' => {
                paren_depth += 1;
                current.push(ch);
                normalized.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                current.push(ch);
                normalized.push(ch);
            }
            '[' => {
                bracket_depth += 1;
                current.push(ch);
                normalized.push(ch);
            }
            ']' => {
                bracket_depth -= 1;
                current.push(ch);
                normalized.push(ch);
            }
            '{' => {
                brace_depth += 1;
                current.push(ch);
                normalized.push(ch);
            }
            '}' => {
                brace_depth -= 1;
                current.push(ch);
                normalized.push(ch);
            }
            _ if paren_depth == 0
                && bracket_depth == 0
                && brace_depth == 0
                && is_separator(ch) =>
                {
                    saw_separator = true;

                    let part = current.trim();

                    if !part.is_empty() {
                        parts.push(part.to_string());
                    }

                    current.clear();
                    normalized.push(ch);
                }
            _ => {
                current.push(ch);
                normalized.push(ch);
            }
        }
    }

    let part = current.trim();

    if !part.is_empty() {
        parts.push(part.to_string());
    }

    (parts, saw_separator, normalized)
}

pub fn find_matching_close_paren(s: &str, open_paren: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0isize;
    let mut in_string = false;
    let mut i = open_paren;

    while i < bytes.len() {
        let b = bytes[i];

        if b == b'\\' {
            if i + 1 < bytes.len() {
                let next = bytes[i + 1];

                if in_string
                    || matches!(next, b'\\' | b'"' | b'(' | b')')
                {
                    i += 2;
                    continue;
                }
            }

            i += 1;
            continue;
        }

        if in_string {
            if b == b'"' {
                in_string = false;
            }

            i += 1;
            continue;
        }

        match b {
            b'"' => in_string = true,
            b'(' => depth += 1,
            b')' => {
                depth -= 1;

                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }

        i += 1;
    }

    None
}

// fn parse_primitive_action(action: &str) -> Vec<UiCommand> {
//     let s = action.trim();
//
//     if s.is_empty() {
//         return Vec::new();
//     }
//
//     let chained = split_top_level_semicolons(s);
//
//     if chained.len() > 1 {
//         return chained
//             .into_iter()
//             .flat_map(|part| parse_primitive_action(&part))
//             .collect();
//     }
//
//     let mut out = Vec::new();
//
//     if let Some(open_paren) = s.find('(') {
//         let func_name = s[..open_paren].trim();
//
//         if let Some(close_paren) = find_matching_close_paren(s, open_paren) {
//             let args_str = s[open_paren + 1..close_paren].trim();
//             let args = parse_arguments(args_str);
//
//             if let Some(cmd) = make_ui_command(func_name, args) {
//                 out.push(cmd);
//             }
//
//             let remainder = s[close_paren + 1..]
//                 .trim_start_matches(|c: char| c.is_whitespace() || c == ';');
//
//             if !remainder.is_empty() {
//                 out.extend(parse_primitive_action(remainder));
//             }
//
//             return out;
//         }
//     }
//
//     if let Some(cmd) = make_ui_command(s, Vec::new()) {
//         out.push(cmd);
//     }
//
//     out
// }
fn parse_primitive_action(action: &str) -> Vec<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return Vec::new();
    }

    let chained = split_top_level_semicolons(s);

    if chained.len() > 1 {
        return chained
            .into_iter()
            .flat_map(|part| parse_primitive_action(&part))
            .collect();
    }

    let mut out = Vec::new();

    if let Some(open_paren) = s.find('(') {
        let func_name = s[..open_paren].trim();

        if let Some(close_paren) = find_matching_close_paren(s, open_paren) {
            let args_str = s[open_paren + 1..close_paren].trim();
            let args = parse_arguments(args_str);

            if let Some(cmd) = make_ui_command(func_name, args) {
                out.push(cmd);
            }

            let remainder = s[close_paren + 1..]
                .trim_start_matches(|c: char| c.is_whitespace() || c == ';');

            if remainder.is_empty() || remainder.starts_with(')') {
                return out;
            }

            out.extend(parse_primitive_action(remainder));
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
            // if key == "Enter".to_string() {
            //     println!("{} {} {:?}", key, input.combo_down(&key), keybind_trigger);
            // }

            let p = match keybind_trigger {
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
            //println!("{}", p);
            return p;
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

        // println!(
        //     "[Action Compile] Filters: {:?}\nRemaining: {:?}",
        //     filters,
        //     action_string
        // );

        let commands = parse_primitive_action(action_string.trim());

        //println!("[Action Compile] Commands: {:#?}", commands);
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
        "desel" => ActionEvent::DeSelectAll,
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
        filters.as_filter = AsFilter::Dynamic(
            arr.iter().map(|str| str.to_string()).collect()
        );

        return Some(consumed);
    }
    // Is as:?
    let Some((raw, consumed)) = parse_prefixed_value(input, "as:") else {
        return None;
    };

    let inner = raw
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'));

    let Some(inner) = inner else {
        error!("Invalid `as:` filter: expected an array, got {:?}", raw);
        return Some(consumed);
    };

    let elements = Value::split_array_elements(inner);

    if let Some(element) = elements.iter().find(|element| {
        contains_unquoted_dynamic_reference(element)
    }) {
        error!(
        "Invalid `as:` filter: dynamic expression {:?} is not supported. Use `asdyn:` instead.",
        element
    );
        return Some(consumed);
    }

    let Some(arr) = Value::parse_array_pure(raw.as_str()) else {
        error!(
            "Failed to parse `as:` filter: expected an array, got {:?}",
            raw
        );
        return Some(consumed);
    };
    //println!("{:?}", arr);
    let (menu, layer, id) = match arr.as_slice() {
        [id] => {
            let Some(current) = element_ctx.self_element.as_ref() else {
                error!(
                    "Failed to resolve `as:[{}]`: no current self element context",
                    id
                );
                return Some(consumed);
            };

            (
                current.menu.clone(),
                current.layer.clone(),
                id.to_string(),
            )
        }
        [layer, id] => {
            let Some(current) = element_ctx.self_element.as_ref() else {
                error!(
                    "Failed to resolve `as:[{}, {}]`: no current self element context",
                    layer, id
                );
                return Some(consumed);
            };

            (
                current.menu.clone(),
                layer.to_string(),
                id.to_string(),
            )
        }
        [menu, layer, id] => (
            menu.to_string(),
            layer.to_string(),
            id.to_string(),
        ),
        _ => {
            error!(
                "Invalid `as:` filter: expected 1, 2, or 3 elements, got {} in {:?}",
                arr.len(),
                raw
            );
            return Some(consumed);
        }
    };

    let Some(kind) = get_element_kind(
        menus,
        menu.as_str(),
        layer.as_str(),
        id.as_str(),
    ) else {
        error!(
            "Failed to resolve `as:` filter: element does not exist: menu={:?}, layer={:?}, id={:?}. Might need 'asdyn:'",
            menu,
            layer,
            id
        );
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
        // let p = events.contains(&ActionEvent::DeSelectAll) && filters.events.contains(&ActionEvent::DeSelectAll);

        let any_event_matches = filters.events.iter().any(|e| events.contains(e));
        // if p {
        //     println!("{:?};;;;; {:?} {:?}, Any event matches: {}", events, filters.events, element_ctx, any_event_matches);
        // }
        if !any_event_matches {
            if !filters.events.iter().any(|e| e == &ActionEvent::Always) {
                // if p {
                //     println!("NOT PASSED");
                // }
                return false;
            }
        }
        // if p {
        //     println!("PASSED");
        // }
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
fn contains_unquoted_dynamic_reference(input: &str) -> bool {
    let input = input.trim();

    if input.starts_with("str:") {
        return false;
    }
    if input.starts_with("self.") || input.starts_with("as.") {
        return true;
    }

    let bytes = input.as_bytes();
    let mut quote = None;
    let mut escaped = false;

    for i in 0..bytes.len() {
        let byte = bytes[i];

        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == q {
                quote = None;
            }

            continue;
        }

        if byte == b'"' || byte == b'\'' {
            quote = Some(byte);
            continue;
        }
    }

    false
}