use crate::data::Settings;
#[allow(unused_mut, unused_assignments)]
use crate::ui::actions::UiCommand;
use crate::ui::actions::{CommandContext, ElementContext, string_to_value};
use crate::ui::input::Input;
use crate::ui::menu::Menu;
use crate::ui::parser::Value;
use crate::ui::ui_editor::{Ui, get_element_kind};
use crate::ui::ui_touch_manager::UiTouchManager;
use crate::ui::ui_touch_manager::{ElementRef, MouseButtons, TouchEvent};
use crate::ui::variables::Variables;
use std::cmp::PartialEq;
use std::collections::HashMap;

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
            variables: &Variables,
            menus: &HashMap<String, Menu>,
            touch_manager: &UiTouchManager,
            func_name: &str,
            args: Vec<String>,
            element_ctx: &ElementContext,
            event_kind: &ActionEvent,
            buttons: &MouseButtons
        ) -> Option<UiCommand> {
            let name = func_name.to_ascii_lowercase();

            match name.as_str() {
                $(
                    $( $name )|+ => {
                        define_commands!(@build settings, variables, menus, touch_manager, args, element_ctx, event_kind, buttons, $variant $( { $( $field : $ftype ),* } )?)
                    }
                ),*,
                _ => {
                    eprintln!("[Warning] Unknown UI command: '{}' in element: {}", func_name, element_ctx.self_element.as_ref().map(|s|s.id.clone()).unwrap_or("Unknown".to_string()));
                    None
                }
            }
        }
    };

    // unit variant
    (@build $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $element_ctx:ident, $event_kind:ident, $buttons:ident, $variant:ident) => {
    Some(UiCommand::$variant)
    };

    // struct variant
    (@build $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $element_ctx:ident, $event_kind:ident, $buttons:ident, $variant:ident { $( $field:ident : $ftype:ty ),* }) => {{
        let mut idx = 0usize;

        $(
            let $field = define_commands!(
                @parse $settings, $vars, $menus, $tm, $args, idx, $element_ctx, $event_kind, $buttons, $field, $ftype
            )?;
        )*

        Some(UiCommand::$variant { $( $field ),* })
    }};

    // -----------------------------
    // SPECIAL FIELD: element_ref
    // -----------------------------

    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, element_ctx, $ftype:ty) => {{
        Some($element_ctx.clone())
    }};

    // SPECIAL FIELD: event_kind
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, event_kind, $ftype:ty) => {{
        Some($event_kind.clone())
    }};

    // SPECIAL FIELD: buttons
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, buttons, $ftype:ty) => {{
        Some($buttons.clone())
    }};

    // SPECIAL FIELD: commands
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, commands, $ftype:ty) => {{
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
                            cmds.extend(parse_primitive_action($settings, $vars, $menus, $tm, part, $element_ctx, $event_kind, $buttons))
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }

            // Don't forget the last command after the final ';'
            let part = raw[start..].trim();
            if !part.is_empty() {
                cmds.extend(parse_primitive_action($settings, $vars, $menus, $tm, part, $element_ctx, $event_kind, $buttons))
            }

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    // SPECIAL FIELD: then / else_branch
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, then, $ftype:ty) => {{
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
                            cmds.extend(parse_primitive_action($settings, $vars, $menus, $tm, part, $element_ctx, $event_kind, $buttons))
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }

            // Don't forget the last command after the final ';'
            let part = raw[start..].trim();
            if !part.is_empty() {
                cmds.extend(parse_primitive_action($settings, $vars, $menus, $tm, part, $element_ctx, $event_kind, $buttons))
            }

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, else_branch, $ftype:ty) => {{
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
                            cmds.extend(parse_primitive_action($settings, $vars, $menus, $tm, part, $element_ctx, $event_kind, $buttons))
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }

            // Don't forget the last command after the final ';'
            let part = raw[start..].trim();
            if !part.is_empty() {
                cmds.extend(parse_primitive_action($settings, $vars, $menus, $tm, part, $element_ctx, $event_kind, $buttons))
            }

            Some(cmds)
        } else {
            Some(Vec::new())
        }
    }};

    (@branch $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident, $element_ctx:ident, $event_kind:ident, $buttons:ident) => {{
        let raw = $args.get($idx)?;
        #[allow(unused_assignments)]
        {
            $idx += 1;
        }

        let cmds = raw.split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| parse_primitive_action($settings, $vars, $menus, $tm, s, $element_ctx, $event_kind, $buttons)).flatten()
            .collect();

        Some(cmds)
    }};
    // SPECIAL FIELD: raw String
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident,
        $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident,
        $field:ident, String) => {{

        let val = $args.get($idx)
            .cloned()
            .unwrap_or_default();

        $idx += 1;

        Some(val)
    }};
    // GENERIC FIELD PARSER
    (@parse $settings:ident, $vars:ident, $menus:ident, $tm:ident, $args:ident, $idx:ident,
        $element_ctx:ident, $event_kind:ident, $buttons:ident, $field:ident, $ftype:ty) => {{

        let val = <$ftype as ParseArg>::parse_arg(&$args, &mut $idx)?;
        Some(val)
    }};
}

// COMMAND DEFINITIONS - I Just add a line here when I add a new UiCommand!
define_commands! {
    // ===== MENU COMMANDS =====
    "open_menu" | "openmenu"
        => OpenMenu { element_ctx: ElementContext, menu_name: String },

    "close_menu" | "closemenu"
        => CloseMenu { element_ctx: ElementContext, menu_name: String },

    "close_all_menus" | "closeall"
        => CloseAllMenus,
    "close_all_layers"
        => CloseAllLayers { element_ctx: ElementContext, menu_name: String },
    "toggle_menu" | "togglemenu"
        => ToggleMenu { element_ctx: ElementContext, menu_name: String },

    "menu_active" | "menuactive"
        => MenuActive { element_ctx: ElementContext, menu_name: String },

    // ===== LAYER COMMANDS =====
    "open_layer" | "openlayer"
        => OpenLayer { element_ctx: ElementContext, menu_name: String, layer_name: String },

    "close_layer" | "closelayer"
        => CloseLayer { element_ctx: ElementContext, menu_name: String, layer_name: String },

    "toggle_layer" | "togglelayer"
        => ToggleLayer { element_ctx: ElementContext, menu_name: String, layer_name: String },

    // ===== VARIABLE COMMANDS =====
    "set_var" | "setvar" | "set"
        => SetVar { element_ctx: ElementContext, name: String, value: String },

    "inc_var" | "incvar" | "inc"
        => IncVar { element_ctx: ElementContext, name: String, amount: String },

    "dec_var" | "decvar" | "dec"
        => DecVar { element_ctx: ElementContext, name: String, amount: String },

    "mul_var" | "mulvar" | "mul"
        => MulVar { element_ctx: ElementContext, name: String, factor: String },

    "toggle_var" | "togglevar" | "toggle"
        => ToggleVar { element_ctx: ElementContext, name: String },

    "clamp" | "clampvar"
        => Clamp { element_ctx: ElementContext, name: String, min: String, max: String },

    // ===== FLOW CONTROL =====
    "delay" | "wait" | "sleep"
        => Delay { element_ctx: ElementContext, seconds: String },

    "halt" | "break"
        => Halt,

    "skip"
        => Skip { count: usize },
    "for" | "forin" => For { element_ctx: ElementContext, value: String, commands: Vec<UiCommand> },
    "if" => If { element_ctx: ElementContext, condition: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand> },

    "ifvareq"
        => IfVarEq { element_ctx: ElementContext, var_name: String, value: String, then: Vec<UiCommand>, else_branch: Vec<UiCommand>},

    "add_element" | "addelem" | "add"
        => AddElement { element_ctx: ElementContext, menu: String, layer: String, id: String, kind: String, center: String, actions: String, undoable: bool},

    "add_ap" | "addap"
        => AddAP { element_ctx: ElementContext, menu: String, name: String, ap_name: String, ap_var: String, center: String, scale: String, is_temporary: bool},

    "del_ap" | "delap"
        => DeleteAP { element_ctx: ElementContext, menu: String, layer: String, reference_id: String},

    "clone_element" | "cloneelem" | "clone"
        => CloneElement { element_ctx: ElementContext,
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
        => CloneLayer { element_ctx: ElementContext,
        from_menu: String,
        from_layer: String,
        to_menu: String,
        to_layer: String,
        undoable: bool},

    "delete_element" | "delelem" | "delete"
        => DeleteElement { element_ctx: ElementContext, menu: String, layer: String, id: String, undoable: bool},

    "delete_layer" | "dellayer"
        => DeleteLayer { element_ctx: ElementContext, menu: String, layer: String, undoable: bool},

    "save" | "savegame" => SaveGame { element_ctx: ElementContext, and_exit: String },

    "load" | "loadgame" | "load_save"
        => LoadSave { element_ctx: ElementContext, save_name: String, without_saving: bool  },

    "exit_game" | "leave_game"
        => ExitGame,

    "show_interaction" => ShowInteraction { element_ctx: ElementContext, event_kind: TouchEventKind, buttons: MouseButtons, color: String, shadow: bool },
    // ===== DEBUG COMMANDS =====
    "print" | "log" | "echo"
        => Print { element_ctx: ElementContext, statement: String },

    "debug_vars" | "debugvars"
        => DebugVars,

    "debug_menus" | "debugmenus"
        => DebugMenus,

    "call" => Call { element_ctx: ElementContext, event_kind: TouchEventKind, buttons: MouseButtons, function_name: String, args: Option<String> },

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
pub fn actions_to_uicommands(ctx: &mut CommandContext, event: &TouchEvent) -> Vec<UiCommand> {
    let (event_kind, actions, element, buttons) = match event {
        TouchEvent::HoverEnter { actions, element } => (
            ActionEvent::HoverEnter,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::Hovering { actions, element } => (
            ActionEvent::Hovering,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::HoverExit { actions, element } => (
            ActionEvent::HoverExit,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::Press {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::Press, actions, element, *buttons),
        TouchEvent::Down {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::Down, actions, element, *buttons),
        TouchEvent::Release {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::Release, actions, element, *buttons),
        TouchEvent::Click {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::Click, actions, element, *buttons),
        TouchEvent::DoubleClick {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::DoubleClick, actions, element, *buttons),
        TouchEvent::DragStart {
            element,
            actions,
            buttons,
            ..
        } => (ActionEvent::DragStart, actions, element, *buttons),
        TouchEvent::DragMove {
            element,
            actions,
            buttons,
            ..
        } => (ActionEvent::DragMove, actions, element, *buttons),
        TouchEvent::DragEnd {
            element,
            actions,
            buttons,
            ..
        } => (ActionEvent::DragEnd, actions, element, *buttons),
        TouchEvent::ScrollOnElement {
            actions,
            element,
            delta,
        } => {
            //println!("SCROLLED!!");
            ctx.ui.variables.set_f64("scroll_delta", *delta);
            (
                ActionEvent::ScrollOnElement,
                actions,
                element,
                MouseButtons::default(),
            )
        }
        TouchEvent::SelectionRequested { element, .. } => {
            //println!("{:?}", element);
            (
                ActionEvent::Select,
                &vec![],
                element,
                MouseButtons::default(),
            )
        }
        TouchEvent::DeselectAllRequested {} => (
            ActionEvent::DeSelect,
            &vec![],
            &ElementRef::default(),
            MouseButtons::default(),
        ),
        TouchEvent::Nothing { element, actions } => (
            ActionEvent::Nothing,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::Activated {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::Activated, actions, element, *buttons),
        TouchEvent::Deactivated {
            actions,
            element,
            buttons,
            ..
        } => (ActionEvent::Deactivated, actions, element, *buttons),
        TouchEvent::StartUp { actions, element } => (
            ActionEvent::StartUp,
            actions,
            element,
            MouseButtons::default(),
        ),
        TouchEvent::ScreenResize { actions, element } => (
            ActionEvent::ScreenResize,
            actions,
            element,
            MouseButtons::default(),
        ),
        _ => return vec![],
    };

    let mut cmds = Vec::new();
    let layer_actions = ctx.ui.menus
        .get(element.menu.as_str())
        .and_then(|m| m.layers.iter().find(|l| l.name == element.layer))
        .map(|l| l.actions.clone())
        .unwrap_or_default();

    for action in actions.iter()
        .chain(ctx.ui.global_actions.element_actions.clone().iter())
        .chain(layer_actions.iter())
    {
        let element_ctx = ElementContext {
            self_element: Some(element.clone()),
            as_element: None,
        };
        cmds.extend(parse_action(
            action,
            ctx,
            &event_kind,
            &buttons,
            element_ctx,
        ))
    }
    cmds
}
pub fn parse_action(
    action: &String,
    ctx: &mut CommandContext,
    event_kind: &ActionEvent,
    buttons: &MouseButtons,
    mut element_ctx: ElementContext,
) -> Vec<UiCommand> {
    let mut action_owned = action.clone();

    let filters = parse_action_filters(ctx, &mut element_ctx, &mut action_owned);

    if filters_match(
        &mut ctx.world.input,
        ctx.settings,
        &filters,
        &event_kind,
        &buttons,
    ) {
        // Now action_owned only contains the actual command
        return parse_primitive_action(
            ctx.settings,
            &ctx.ui.variables,
            &ctx.ui.menus,
            &ctx.ui.touch_manager,
            action_owned.trim(),
            &element_ctx,
            event_kind,
            buttons,
        );
    };
    vec![]
}
/// Handle a single action string that may be an event wrapper
fn handle_action_str(
    settings: &Settings,
    ui: &mut Ui,
    event_kind: &ActionEvent,
    buttons: &MouseButtons,
    action: &str,
    element_ctx: &ElementContext,
) -> Vec<UiCommand> {
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
    process_inner_content(settings, ui, event_kind, buttons, inner, element_ctx)
}

/// Process the inner content of a matched event wrapper
fn process_inner_content(
    settings: &Settings,
    ui: &mut Ui,
    event_kind: &ActionEvent,
    buttons: &MouseButtons,
    inner: &str,
    element_ctx: &ElementContext,
) -> Vec<UiCommand> {
    for part in split_top_level(inner, b',') {
        let cmds = handle_action_str(
            settings,
            ui,
            event_kind,
            buttons,
            part,
            element_ctx,
        );

        if !cmds.is_empty() {
            return cmds;
        }

        let cmds = parse_primitive_action(
            settings,
            &ui.variables,
            &ui.menus,
            &ui.touch_manager,
            part,
            element_ctx,
            event_kind,
            buttons,
        );

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

            _ if b == delimiter
                && paren_depth == 0
                && bracket_depth == 0
                && brace_depth == 0 =>
                {
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

fn parse_primitive_action(
    settings: &Settings,
    variables: &Variables,
    menus: &HashMap<String, Menu>,
    touch_manager: &UiTouchManager,
    action: &str,
    element_ctx: &ElementContext,
    event_kind: &ActionEvent,
    buttons: &MouseButtons,
) -> Vec<UiCommand> {
    let s = action.trim();

    if s.is_empty() {
        return Vec::new();
    }

    let chained = split_top_level_semicolons(s);
    if chained.len() > 1 {
        return chained
            .into_iter()
            .flat_map(|part| {
                parse_primitive_action(
                    settings,
                    variables,
                    menus,
                    touch_manager,
                    part,
                    element_ctx,
                    event_kind,
                    buttons,
                )
            })
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

            if let Some(cmd) = make_ui_command(
                settings,
                variables,
                menus,
                touch_manager,
                func_name,
                args,
                element_ctx,
                event_kind,
                buttons,
            ) {
                out.push(cmd);
            }

            let rest = s[close_paren + 1..].trim();
            if let Some(rest) = rest.strip_prefix(';') {
                out.extend(parse_primitive_action(
                    settings,
                    variables,
                    menus,
                    touch_manager,
                    rest,
                    element_ctx,
                    event_kind,
                    buttons,
                ));
            }

            return out;
        }
    }

    if let Some(cmd) = make_ui_command(
        settings,
        variables,
        menus,
        touch_manager,
        s,
        Vec::new(),
        element_ctx,
        event_kind,
        buttons,
    ) {
        out.push(cmd);
    }

    out
}

fn button_matches(
    input: &mut Input,
    button: ParsedButton,
    buttons: &MouseButtons,
    keybind_trigger: KeyBindTrigger,
) -> bool {
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

#[derive(Default, Debug)]
struct ActionFilters {
    buttons: Vec<ParsedButton>,
    events: Vec<ActionEvent>,
    keybind_trigger: KeyBindTrigger,
    modes: Vec<String>,
}
#[derive(Default, Debug, Copy, Clone)]
enum KeyBindTrigger {
    #[default]
    Down,
    Press,
    Release,
    Repeat,
}

struct ParsedAction {
    filters: ActionFilters,
    command: String,
}

fn parse_action_filters(
    ctx: &mut CommandContext,
    element_ctx: &mut ElementContext,
    action: &mut String,
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
            .or_else(|| try_parse_trigger(rest, &mut filters))
            .or_else(|| try_parse_on(rest, &mut filters))
            .or_else(|| try_parse_in(rest, &mut filters))
            .or_else(|| try_parse_as(rest, &mut filters, element_ctx, ctx));

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
    ctx: &mut CommandContext,
) -> Option<usize> {
    let (raw, consumed) = parse_prefixed_value(input, "as:")?;
    let Some(arr) = Value::parse_array(&ctx.settings, &ctx.ui.variables, raw.as_str()) else {
        return Some(consumed);
    };
    let self_element = if let Some(current) = element_ctx.self_element.as_ref() {
        current
    } else if arr.len() == 3 {
        &ElementRef::default()
    } else {
        return Some(consumed);
    };
    let mut resolve = |s: String| string_to_value(ctx, element_ctx, s).into_string();
    //println!("{:?}", arr);
    let (menu, layer, id) = match arr.as_slice() {
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
        [] => return Some(consumed),
    };
    let Some(kind) = get_element_kind(&ctx.ui.menus, menu.as_str(), layer.as_str(), id.as_str())
    else {
        return Some(consumed);
    }; // grad sport gleich sport, hmm grad sport, deswegen schwitze ich mama
    let as_element = Some(ElementRef {
        menu,
        layer,
        id,
        kind,
    });
    //println!("{:?}", as_element);
    element_ctx.as_element = as_element;

    Some(consumed)
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
    input: &mut Input,
    settings: &Settings,
    filters: &ActionFilters,
    event_kind: &ActionEvent,
    buttons: &MouseButtons,
) -> bool {
    // Check button filters (if any specified, at least one must match)
    if !filters.buttons.is_empty() {
        let any_button_matches = filters
            .buttons
            .iter()
            .any(|b| button_matches(input, b.clone(), buttons, filters.keybind_trigger));
        if !any_button_matches {
            return false;
        }
    }

    // Check event filters (if any specified, at least one must match)
    if !filters.events.is_empty() {
        let any_event_matches = filters.events.iter().any(|e| e == event_kind);
        if !any_event_matches {
            if !filters.events.iter().any(|e| e == &ActionEvent::Always) {
                return false;
            }
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
            "any" => true,
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

    let knob_x = if abs_left < abs_right { left_step } else { right_step };
}