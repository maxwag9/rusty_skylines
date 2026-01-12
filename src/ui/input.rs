use crate::paths::data_dir;
use glam::Vec2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, NamedKey, PhysicalKey};

#[derive(Debug, Clone)]
pub struct RepeatTimer {
    first_press: f64,
    last_fire: f64,
    initial_delay: f32,
    warmup_time: f32,
    warmup_interval: f32,
    fast_interval: f32,
}

impl RepeatTimer {
    pub fn new() -> Self {
        Self {
            first_press: -1.0,
            last_fire: -1.0,
            initial_delay: 0.25,
            warmup_time: 0.08,
            warmup_interval: 0.07,
            fast_interval: 0.03,
        }
    }

    pub fn tick(&mut self, now: f64, is_down: bool) -> bool {
        if !is_down {
            self.first_press = -1.0;
            self.last_fire = -1.0;
            return false;
        }

        if self.first_press < 0.0 {
            self.first_press = now;
            self.last_fire = now;
            return true;
        }

        let held = now - self.first_press;
        let since_last = now - self.last_fire;

        if held < self.initial_delay as f64 {
            return false;
        }

        if held < (self.initial_delay + self.warmup_time) as f64 {
            if since_last >= self.warmup_interval as f64 {
                self.last_fire = now;
                return true;
            }
            return false;
        }

        if since_last >= self.fast_interval as f64 {
            self.last_fire = now;
            return true;
        }

        false
    }
}

#[derive(Debug, Clone)]
enum BindingKey {
    Physical(PhysicalKey),
    Logical(NamedKey),
    Mouse(MouseButton),
    Character(String),
    WheelUp,
    WheelDown,
    WheelLeft,
    WheelRight,
}

#[derive(Debug, Clone)]
struct ParsedKeyCombo {
    require_ctrl: bool,
    require_shift: bool,
    require_alt: bool,
    key: BindingKey,
}

impl ParsedKeyCombo {
    fn matches(&self, input: &InputState) -> bool {
        if input.ctrl != self.require_ctrl {
            return false;
        }
        if input.shift != self.require_shift {
            return false;
        }
        if input.alt != self.require_alt {
            return false;
        }

        match &self.key {
            BindingKey::Physical(p) => input.physical.get(p).copied().unwrap_or(false),
            BindingKey::Logical(n) => input.logical.get(n).copied().unwrap_or(false),
            BindingKey::Mouse(m) => input.mouse.is_button_down(*m),
            BindingKey::Character(ch) => input.text_chars.contains(ch),
            BindingKey::WheelUp => input.scroll_up_hit,
            BindingKey::WheelDown => input.scroll_down_hit,
            BindingKey::WheelLeft => input.scroll_left_hit,
            BindingKey::WheelRight => input.scroll_right_hit,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionBind {
    pub keys: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Keybinds {
    pub binds: HashMap<String, ActionBind>,
}

impl Keybinds {
    fn read_bindings(path: impl AsRef<Path>) -> Option<HashMap<String, ActionBind>> {
        fs::read_to_string(path)
            .ok()
            .and_then(|data| toml::from_str(&data).ok())
    }

    fn write_bindings(path: impl AsRef<Path>, data: &HashMap<String, ActionBind>) {
        if let Ok(serialized) = toml::to_string(data) {
            let _ = fs::write(path, serialized);
        }
    }

    pub fn load(path: impl AsRef<Path>, default_path: impl AsRef<Path>) -> Self {
        let default_map: HashMap<String, ActionBind> = match Self::read_bindings(&default_path) {
            Some(map) => map,
            None => {
                println!("default_keybinds.toml missing");

                match fs::read_to_string(&path) {
                    Ok(user_data) => {
                        println!("copying keybinds.toml to default_keybinds.toml");
                        let _ = fs::write(&default_path, &user_data);
                        toml::from_str(&user_data).unwrap_or_else(|_| HashMap::new())
                    }
                    Err(_) => {
                        println!("no keybinds.toml either, creating empty defaults");
                        let empty: HashMap<String, ActionBind> = HashMap::new();
                        Self::write_bindings(&default_path, &empty);
                        empty
                    }
                }
            }
        };

        let mut merged = default_map.clone();

        if let Some(user) = Self::read_bindings(&path) {
            merged.extend(user);
        } else {
            println!("keybinds.toml missing or invalid, rewriting from defaults");
            Self::write_bindings(&path, &merged);
        }

        Keybinds { binds: merged }
    }

    pub fn save(&self, path: impl AsRef<Path>) {
        Self::write_bindings(path, &self.binds);
    }
}

#[derive(Debug, Clone)]
pub struct MouseState {
    pub last_pos: Vec2,
    pub pos: Vec2,
    pub delta: Vec2,
    pub middle_pressed: bool,
    pub left_pressed: bool,
    pub right_pressed: bool,
    pub back_pressed: bool,
    pub forward_pressed: bool,

    pub left_just_pressed: bool,
    pub left_just_released: bool,
    pub right_just_pressed: bool,
    pub right_just_released: bool,

    pub scroll_delta: Vec2,
}

impl MouseState {
    pub fn new() -> Self {
        Self {
            last_pos: Vec2::ZERO,
            pos: Vec2::ZERO,
            delta: Vec2::ZERO,

            middle_pressed: false,
            left_pressed: false,
            right_pressed: false,
            back_pressed: false,
            forward_pressed: false,

            left_just_pressed: false,
            left_just_released: false,
            right_just_pressed: false,
            right_just_released: false,

            scroll_delta: Vec2::ZERO,
        }
    }

    pub fn update_just_states(&mut self) {
        // Reset per-frame flags
        self.left_just_pressed = false;
        self.left_just_released = false;
        self.right_just_pressed = false;
        self.right_just_released = false;
        self.scroll_delta = Vec2::ZERO;
    }

    pub fn is_button_down(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Left => self.left_pressed,
            MouseButton::Right => self.right_pressed,
            MouseButton::Middle => self.middle_pressed,
            MouseButton::Back => self.back_pressed,
            MouseButton::Forward => self.forward_pressed,
            MouseButton::Other(_) => false,
        }
    }
}

pub struct InputState {
    pub physical: HashMap<PhysicalKey, bool>,
    pub logical: HashMap<NamedKey, bool>,
    pub text_chars: HashSet<String>,

    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,

    pub _keybinds: Keybinds,
    pub mouse: MouseState,
    scroll_up_hit: bool,
    scroll_down_hit: bool,
    scroll_left_hit: bool,
    scroll_right_hit: bool,

    parsed: HashMap<String, Vec<ParsedKeyCombo>>,
    warned_missing: HashSet<String>,
    repeat_timers: HashMap<String, RepeatTimer>,
    action_last_down: HashMap<String, bool>,
    pub gameplay_last_down: HashMap<String, bool>,
    pub gameplay_repeat_timers: HashMap<String, RepeatTimer>,
    pub now: f64,
}

impl InputState {
    pub fn new() -> Self {
        let keybinds = Keybinds::load(data_dir("keybinds.toml"), data_dir("default_keybinds.toml"));
        let parsed = Self::parse_all(&keybinds);

        Self {
            physical: HashMap::new(),
            logical: HashMap::new(),
            text_chars: HashSet::new(),
            shift: false,
            ctrl: false,
            alt: false,
            _keybinds: keybinds,
            mouse: MouseState::new(),
            scroll_up_hit: false,
            scroll_down_hit: false,
            scroll_left_hit: false,
            scroll_right_hit: false,

            parsed,
            warned_missing: HashSet::new(),
            repeat_timers: HashMap::new(),
            action_last_down: HashMap::new(),
            gameplay_last_down: HashMap::new(),
            gameplay_repeat_timers: HashMap::new(),
            now: 0.0,
        }
    }

    pub fn begin_frame(&mut self, now: f64) {
        self.now = now;
        self.scroll_up_hit = false;
        self.scroll_down_hit = false;
        self.scroll_left_hit = false;
        self.scroll_right_hit = false;

        self.mouse.update_just_states();
        self.text_chars.clear();
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let down = state == ElementState::Pressed;
        self.set_mouse_button(button, down);
    }

    pub fn handle_mouse_move(&mut self, x: f64, y: f64) {
        self.mouse.last_pos = self.mouse.pos;

        let pos = Vec2::new(x as f32, y as f32);
        let last = self.mouse.last_pos;
        let delta = { pos - last };

        self.mouse.delta = delta;
        self.mouse.pos = pos;
    }

    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) -> Vec2 {
        let mut out = Vec2::ZERO;

        match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                out.x = x;
                out.y = y;
            }
            MouseScrollDelta::PixelDelta(p) => {
                out.x = p.x as f32 * 0.1;
                out.y = p.y as f32 * 0.1;
            }
        };

        self.add_scroll_delta(out);
        out
    }

    pub fn set_physical(&mut self, key: PhysicalKey, down: bool) {
        self.physical.insert(key, down);
    }

    pub fn set_logical(&mut self, key: NamedKey, down: bool) {
        self.logical.insert(key, down);
    }

    pub fn set_character(&mut self, ch: &str, down: bool) {
        if down {
            self.text_chars.insert(ch.to_string());
        } else {
            self.text_chars.remove(ch);
        }
    }

    pub fn set_mouse_button(&mut self, button: MouseButton, down: bool) {
        match button {
            MouseButton::Left => {
                if down && !self.mouse.left_pressed {
                    self.mouse.left_just_pressed = true;
                }
                if !down && self.mouse.left_pressed {
                    self.mouse.left_just_released = true;
                }
                self.mouse.left_pressed = down;
            }
            MouseButton::Right => {
                if down && !self.mouse.right_pressed {
                    self.mouse.right_just_pressed = true;
                }
                if !down && self.mouse.right_pressed {
                    self.mouse.right_just_released = true;
                }
                self.mouse.right_pressed = down;
            }
            MouseButton::Middle => {
                self.mouse.middle_pressed = down;
            }
            MouseButton::Back => {
                self.mouse.back_pressed = down;
            }
            MouseButton::Forward => {
                self.mouse.forward_pressed = down;
            }
            MouseButton::Other(_) => {}
        }
    }

    pub fn add_scroll_delta(&mut self, delta: Vec2) {
        self.mouse.scroll_delta += delta;

        // vertical
        if delta.y > 0.0 {
            self.scroll_up_hit = true;
        } else if delta.y < 0.0 {
            self.scroll_down_hit = true;
        }

        // horizontal
        if delta.x > 0.0 {
            self.scroll_right_hit = true;
        } else if delta.x < 0.0 {
            self.scroll_left_hit = true;
        }
    }

    fn parse_all(keybinds: &Keybinds) -> HashMap<String, Vec<ParsedKeyCombo>> {
        let mut out = HashMap::new();
        for (name, bind) in &keybinds.binds {
            let combos = bind
                .keys
                .iter()
                .filter_map(|s| parse_combo(s))
                .collect::<Vec<_>>();
            out.insert(name.clone(), combos);
        }
        out
    }

    fn combos_for(&self, action: &str) -> Option<&[ParsedKeyCombo]> {
        self.parsed.get(action).map(|v| v.as_slice())
    }

    fn action_down_raw(&self, action: &str) -> bool {
        let combos = match self.combos_for(action) {
            Some(c) => c,
            None => return false,
        };

        for combo in combos {
            if combo.matches(self) {
                return true;
            }
        }
        false
    }

    fn ensure_known_action(&mut self, action: &str) -> bool {
        if self.parsed.contains_key(action) {
            return true;
        }
        if !self.warned_missing.contains(action) {
            println!("Warning: action '{action}' has no keybind");
            self.warned_missing.insert(action.to_string());
        }
        false
    }

    pub fn action_down(&mut self, action: &str) -> bool {
        if !self.ensure_known_action(action) {
            return false;
        }
        self.action_down_raw(action)
    }

    pub fn action_pressed_once(&mut self, action: &str) -> bool {
        if !self.ensure_known_action(action) {
            return false;
        }
        let now = self.action_down_raw(action);
        let last = self
            .action_last_down
            .entry(action.to_string())
            .or_insert(false);
        let fired = now && !*last;
        *last = now;
        fired
    }

    pub fn action_released(&mut self, action: &str) -> bool {
        if !self.ensure_known_action(action) {
            return false;
        }
        let now_down = self.action_down_raw(action);
        let last_down = self
            .action_last_down
            .entry(action.to_string())
            .or_insert(false);
        let released = *last_down && !now_down;
        *last_down = now_down;
        released
    }

    pub fn action_repeat(&mut self, action: &str) -> bool {
        if !self.ensure_known_action(action) {
            return false;
        }
        let down = self.action_down_raw(action);
        let timer = self
            .repeat_timers
            .entry(action.to_string())
            .or_insert_with(RepeatTimer::new);
        timer.tick(self.now, down)
    }

    pub fn repeat(&mut self, id: &str, is_down: bool) -> bool {
        let timer = self
            .repeat_timers
            .entry(id.to_string())
            .or_insert_with(RepeatTimer::new);
        timer.tick(self.now, is_down)
    }

    fn key_active(&self, key: &BindingKey) -> bool {
        match key {
            BindingKey::Physical(p) => self.physical.get(p).copied().unwrap_or(false),
            BindingKey::Logical(l) => self.logical.get(l).copied().unwrap_or(false),
            BindingKey::Character(c) => self.text_chars.contains(c.as_str()),
            BindingKey::Mouse(m) => self.mouse.is_button_down(*m),
            BindingKey::WheelUp => self.scroll_up_hit,
            BindingKey::WheelDown => self.scroll_down_hit,
            BindingKey::WheelLeft => self.scroll_left_hit,
            BindingKey::WheelRight => self.scroll_right_hit,
        }
    }

    pub fn gameplay_down(&mut self, action: &str) -> bool {
        if !self.ensure_known_action(action) {
            return false;
        }

        if let Some(combos) = self.parsed.get(action) {
            for combo in combos {
                let modifiers_ok = (!combo.require_ctrl || self.ctrl)
                    && (!combo.require_shift || self.shift)
                    && (!combo.require_alt || self.alt);

                if modifiers_ok && self.key_active(&combo.key) {
                    return true;
                }
            }
        }

        false
    }

    pub fn gameplay_pressed_once(&mut self, action: &str) -> bool {
        let now = self.gameplay_down(action);
        let last = self
            .gameplay_last_down
            .entry(action.to_string())
            .or_insert(false);
        let fired = now && !*last;
        *last = now;
        fired
    }

    pub fn gameplay_released(&mut self, action: &str) -> bool {
        let now = self.gameplay_down(action);
        let last = self
            .gameplay_last_down
            .entry(action.to_string())
            .or_insert(false);
        let released = *last && !now;
        *last = now;
        released
    }

    pub fn gameplay_repeat(&mut self, action: &str) -> bool {
        let down = self.gameplay_down(action);
        let timer = self
            .gameplay_repeat_timers
            .entry(action.to_string())
            .or_insert_with(RepeatTimer::new);
        timer.tick(self.now, down)
    }
}

fn parse_combo(s: &str) -> Option<ParsedKeyCombo> {
    // First: try to interpret the whole string as a character binding.
    // This makes "+" / "-" / "=" etc. work with no modifiers.
    if let Some(ch) = map_to_char(s) {
        return Some(ParsedKeyCombo {
            require_ctrl: false,
            require_shift: false,
            require_alt: false,
            key: BindingKey::Character(ch),
        });
    }

    let mut require_ctrl = false;
    let mut require_shift = false;
    let mut require_alt = false;
    let mut key_part: Option<BindingKey> = None;

    for raw in s.split('+') {
        let t = raw.trim();
        if t.eq_ignore_ascii_case("ctrl") || t.eq_ignore_ascii_case("control") {
            require_ctrl = true;
        } else if t.eq_ignore_ascii_case("shift") {
            require_shift = true;
        } else if t.eq_ignore_ascii_case("alt") {
            require_alt = true;
        } else if let Some(k) = map_to_keycode(t) {
            key_part = Some(BindingKey::Physical(PhysicalKey::Code(k)));
        } else if let Some(n) = map_to_named(t) {
            key_part = Some(BindingKey::Logical(n));
        } else if let Some(m) = map_to_mouse(t) {
            key_part = Some(BindingKey::Mouse(m));
        } else if let Some(ch) = map_to_char(t) {
            key_part = Some(BindingKey::Character(ch));
        } else if t.eq_ignore_ascii_case("WheelUp") {
            key_part = Some(BindingKey::WheelUp);
        } else if t.eq_ignore_ascii_case("WheelDown") {
            key_part = Some(BindingKey::WheelDown);
        } else if t.eq_ignore_ascii_case("WheelLeft") {
            key_part = Some(BindingKey::WheelLeft);
        } else if t.eq_ignore_ascii_case("WheelRight") {
            key_part = Some(BindingKey::WheelRight);
        }
    }

    let key = key_part?;

    Some(ParsedKeyCombo {
        require_ctrl,
        require_shift,
        require_alt,
        key,
    })
}

fn map_to_mouse(token: &str) -> Option<MouseButton> {
    match token {
        "MouseLeft" => Some(MouseButton::Left),
        "MouseRight" => Some(MouseButton::Right),
        "MouseMiddle" => Some(MouseButton::Middle),
        "MouseBack" => Some(MouseButton::Back),
        "MouseForward" => Some(MouseButton::Forward),
        _ => None,
    }
}

fn map_to_char(token: &str) -> Option<String> {
    if let Some(stripped) = token
        .strip_prefix("Char(")
        .and_then(|s| s.strip_suffix(')'))
    {
        if !stripped.is_empty() {
            return Some(stripped.to_string());
        }
    }

    if token.chars().count() == 1 {
        return Some(token.to_string());
    }

    None
}

fn map_to_keycode(token: &str) -> Option<KeyCode> {
    if token.starts_with("Key") && token.len() == 4 {
        let ch = token.chars().nth(3)?;
        let upper = ch.to_ascii_uppercase();
        return match upper {
            'A' => Some(KeyCode::KeyA),
            'B' => Some(KeyCode::KeyB),
            'C' => Some(KeyCode::KeyC),
            'D' => Some(KeyCode::KeyD),
            'E' => Some(KeyCode::KeyE),
            'F' => Some(KeyCode::KeyF),
            'G' => Some(KeyCode::KeyG),
            'H' => Some(KeyCode::KeyH),
            'I' => Some(KeyCode::KeyI),
            'J' => Some(KeyCode::KeyJ),
            'K' => Some(KeyCode::KeyK),
            'L' => Some(KeyCode::KeyL),
            'M' => Some(KeyCode::KeyM),
            'N' => Some(KeyCode::KeyN),
            'O' => Some(KeyCode::KeyO),
            'P' => Some(KeyCode::KeyP),
            'Q' => Some(KeyCode::KeyQ),
            'R' => Some(KeyCode::KeyR),
            'S' => Some(KeyCode::KeyS),
            'T' => Some(KeyCode::KeyT),
            'U' => Some(KeyCode::KeyU),
            'V' => Some(KeyCode::KeyV),
            'W' => Some(KeyCode::KeyW),
            'X' => Some(KeyCode::KeyX),
            'Y' => Some(KeyCode::KeyY),
            'Z' => Some(KeyCode::KeyZ),
            _ => None,
        };
    }

    match token {
        "F1" => Some(KeyCode::F1),
        "F2" => Some(KeyCode::F2),
        "F3" => Some(KeyCode::F3),
        "F4" => Some(KeyCode::F4),
        "F5" => Some(KeyCode::F5),
        "F6" => Some(KeyCode::F6),
        "F7" => Some(KeyCode::F7),
        "F8" => Some(KeyCode::F8),
        "F9" => Some(KeyCode::F9),
        "F10" => Some(KeyCode::F10),
        "F11" => Some(KeyCode::F11),
        "F12" => Some(KeyCode::F12),
        "Plus" => Some(KeyCode::NumpadAdd),

        // universal minus
        "Minus" => Some(KeyCode::Minus),
        "NumpadSubtract" => Some(KeyCode::NumpadSubtract),
        "ESC" => Some(KeyCode::Escape),

        _ => None,
    }
}

fn map_to_named(token: &str) -> Option<NamedKey> {
    match token {
        "ArrowLeft" => Some(NamedKey::ArrowLeft),
        "ArrowRight" => Some(NamedKey::ArrowRight),
        "ArrowUp" => Some(NamedKey::ArrowUp),
        "ArrowDown" => Some(NamedKey::ArrowDown),
        "Space" => Some(NamedKey::Space),
        "Enter" => Some(NamedKey::Enter),
        "Backspace" => Some(NamedKey::Backspace),

        _ => None,
    }
}
