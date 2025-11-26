use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use winit::keyboard::{KeyCode, NamedKey, PhysicalKey};

#[derive(Debug, Clone)]
pub struct RepeatTimer {
    first_press: f32,
    last_fire: f32,
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
            warmup_time: 0.15,
            warmup_interval: 0.07,
            fast_interval: 0.03,
        }
    }

    pub fn tick(&mut self, now: f32, is_down: bool) -> bool {
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

        if held < self.initial_delay {
            return false;
        }

        if held < self.initial_delay + self.warmup_time {
            if since_last >= self.warmup_interval {
                self.last_fire = now;
                return true;
            }
            return false;
        }

        if since_last >= self.fast_interval {
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
        if self.require_ctrl && !input.ctrl {
            return false;
        }
        if self.require_shift && !input.shift {
            return false;
        }
        if self.require_alt && !input.alt {
            return false;
        }

        match &self.key {
            BindingKey::Physical(p) => input.physical.get(p).copied().unwrap_or(false),
            BindingKey::Logical(n) => input.logical.get(n).copied().unwrap_or(false),
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
    pub fn load(path: &str, default_path: &str) -> Self {
        let default_map: HashMap<String, ActionBind> = match std::fs::read_to_string(default_path) {
            Ok(d) => toml::from_str(&d).unwrap_or_else(|_| {
                println!("default_keybinds.toml invalid, using empty defaults");
                HashMap::new()
            }),
            Err(_) => {
                println!("default_keybinds.toml missing");
                match std::fs::read_to_string(path) {
                    Ok(user_data) => {
                        println!("copying keybinds.toml to default_keybinds.toml");
                        let _ = std::fs::write(default_path, &user_data);
                        toml::from_str(&user_data).unwrap_or_else(|_| HashMap::new())
                    }
                    Err(_) => {
                        println!("no keybinds.toml either, creating empty defaults");
                        let empty: HashMap<String, ActionBind> = HashMap::new();
                        if let Ok(s) = toml::to_string(&empty) {
                            let _ = std::fs::write(default_path, s);
                        }
                        empty
                    }
                }
            }
        };

        let user_map: Option<HashMap<String, ActionBind>> = std::fs::read_to_string(path)
            .ok()
            .and_then(|d| toml::from_str(&d).ok());

        let final_map = match user_map {
            Some(user) => {
                let mut merged = default_map.clone();
                for (k, v) in user {
                    merged.insert(k, v);
                }
                merged
            }
            None => {
                println!("keybinds.toml missing or invalid, rewriting from defaults");
                let mut merged = default_map.clone();
                let kb = Keybinds {
                    binds: merged.clone(),
                };
                kb.save(path);
                merged
            }
        };

        Keybinds { binds: final_map }
    }

    pub fn save(&self, path: &str) {
        if let Ok(data) = toml::to_string(&self.binds) {
            let _ = std::fs::write(path, data);
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

    pub keybinds: Keybinds,

    parsed: HashMap<String, Vec<ParsedKeyCombo>>,
    warned_missing: HashSet<String>,
    repeat_timers: HashMap<String, RepeatTimer>,
    action_last_down: HashMap<String, bool>,
}

impl InputState {
    pub fn new() -> Self {
        let keybinds = Keybinds::load("keybinds.toml", "default_keybinds.toml");
        let parsed = Self::parse_all(&keybinds);

        Self {
            physical: HashMap::new(),
            logical: HashMap::new(),
            text_chars: HashSet::new(),
            shift: false,
            ctrl: false,
            alt: false,
            keybinds,
            parsed,
            warned_missing: HashSet::new(),
            repeat_timers: HashMap::new(),
            action_last_down: HashMap::new(),
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

    pub fn action_repeat(&mut self, action: &str, now_time: f32) -> bool {
        if !self.ensure_known_action(action) {
            return false;
        }
        let down = self.action_down_raw(action);
        let timer = self
            .repeat_timers
            .entry(action.to_string())
            .or_insert_with(RepeatTimer::new);
        timer.tick(now_time, down)
    }

    pub fn repeat(&mut self, id: &str, now: f32, is_down: bool) -> bool {
        let timer = self
            .repeat_timers
            .entry(id.to_string())
            .or_insert_with(RepeatTimer::new);
        timer.tick(now, is_down)
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
}

fn parse_combo(s: &str) -> Option<ParsedKeyCombo> {
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
