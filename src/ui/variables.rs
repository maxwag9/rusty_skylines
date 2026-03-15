use crate::data::{SettingKey, Settings};
use crate::ui::ui_editor::Ui;
use crate::world::astronomy::{Astronomy, TimeScales};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum UiValue {
    F64(f64),
    I64(i64),
    Bool(bool),
    String(String),
}

impl UiValue {
    pub fn from_str(settings: &Settings, variables: &Variables, s: &str) -> Self {
        let s = s.trim();

        if s.is_empty() {
            return UiValue::String(String::new());
        }

        // Explicit type prefix
        if let Some((ty, value)) = s.split_once(':') {
            match ty.to_ascii_lowercase().as_str() {
                "int" => {
                    if let Ok(i) = value.parse::<i64>() {
                        return UiValue::I64(i);
                    }
                }
                "float" | "f32" | "f64" => {
                    if let Ok(f) = value.parse::<f64>() {
                        return UiValue::F64(f);
                    }
                }
                "bool" => match value.to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => return UiValue::Bool(true),
                    "false" | "0" | "no" | "off" => return UiValue::Bool(false),
                    _ => {}
                },
                "string" | "str" => {
                    return UiValue::String(value.to_string());
                }
                "setting" => {
                    let key = SettingKey::from_str(value);
                    if let Some(key) = key {
                        return settings.read_setting(key).to_ui_value();
                    }
                }
                "var" | "variable" => match Self::load_variable(variables, &s.to_string()) {
                    Some(value) => return value,
                    None => {}
                },
                _ => {}
            }
        }

        // =========================
        // Auto-detect bool
        // =========================
        match s.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => return UiValue::Bool(true),
            "false" | "no" | "off" => return UiValue::Bool(false),
            _ => {}
        }

        // =========================
        // Auto-detect integer
        // =========================
        if let Ok(i) = s.parse::<i64>() {
            return UiValue::I64(i);
        }

        // =========================
        // Auto-detect float
        // =========================
        if let Ok(f) = s.parse::<f64>() {
            return UiValue::F64(f);
        }

        // =========================
        // Default: String
        // =========================
        UiValue::String(s.to_string())
    }
    pub fn is_truthy(&self) -> bool {
        match self {
            UiValue::Bool(b) => *b,
            UiValue::I64(i) => *i != 0,
            UiValue::F64(f) => *f != 0.0,
            UiValue::String(s) => !s.is_empty() && s != "false" && s != "0",
        }
    }
    fn load_variable(variables: &Variables, name: &String) -> Option<UiValue> {
        variables.get(name).cloned()
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            UiValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            UiValue::F64(f) => Some(*f),
            UiValue::I64(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            UiValue::I64(i) => Some(*i),
            UiValue::F64(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> bool {
        self.is_truthy()
    }
}
impl fmt::Display for UiValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UiValue::F64(v) => write!(f, "{v}"),
            UiValue::I64(v) => write!(f, "{v}"),
            UiValue::Bool(v) => write!(f, "{v}"),
            UiValue::String(v) => write!(f, "{v}"),
        }
    }
}

impl From<f64> for UiValue {
    fn from(v: f64) -> Self {
        UiValue::F64(v)
    }
}

impl From<i64> for UiValue {
    fn from(v: i64) -> Self {
        UiValue::I64(v)
    }
}

impl From<bool> for UiValue {
    fn from(v: bool) -> Self {
        UiValue::Bool(v)
    }
}

impl From<&str> for UiValue {
    fn from(v: &str) -> Self {
        UiValue::String(v.to_string())
    }
}

impl From<String> for UiValue {
    fn from(v: String) -> Self {
        UiValue::String(v)
    }
}

pub struct Variables {
    vars: HashMap<String, UiValue>,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set_f64<T: Into<f64>>(&mut self, name: &str, value: T) {
        let value = UiValue::F64(value.into());
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_i64<T: Into<i64>>(&mut self, name: &str, value: T) {
        let value = UiValue::I64(value.into());
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        let value = UiValue::Bool(value);
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_string<S>(&mut self, name: &str, value: S)
    where
        S: Into<String>,
    {
        let value = UiValue::String(value.into());
        self.vars.insert(name.to_string(), value);
    }
    pub fn set_var_ui_value<S>(&mut self, name: S, value: UiValue)
    where
        S: AsRef<str>,
    {
        let name = name.as_ref();
        match value {
            UiValue::String(v) => self.set_string(name, v),
            UiValue::F64(v) => self.set_f64(name, v),
            UiValue::I64(v) => self.set_i64(name, v),
            UiValue::Bool(v) => self.set_bool(name, v),
        }
    }
    pub fn set_var<S, V>(&mut self, name: S, value: V)
    where
        S: AsRef<str>,
        V: Into<UiValue>,
    {
        self.vars.insert(name.as_ref().to_string(), value.into());
    }
    pub fn get(&self, name: &str) -> Option<&UiValue> {
        self.vars.get(name)
    }
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        match self.vars.get(name)? {
            UiValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_i64(&self, name: &str) -> Option<i64> {
        match self.vars.get(name)? {
            UiValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.vars.get(name)? {
            UiValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.vars.get(name)? {
            UiValue::String(v) => Some(v),
            _ => None,
        }
    }
    pub fn dump(&self) {
        let mut keys: Vec<_> = self.vars.keys().collect();
        keys.sort();

        let max_len = keys.iter().map(|k| k.len()).max().unwrap_or(0);

        for key in keys {
            let value = &self.vars[key];
            let type_tag = match value {
                UiValue::F64(_) => "f32",
                UiValue::I64(_) => "i32",
                UiValue::Bool(_) => "bool",
                UiValue::String(_) => "str",
            };
            println!(
                "  {:width$} : {} ({})",
                key,
                value,
                type_tag,
                width = max_len
            );
        }

        if self.vars.is_empty() {
            println!("  (empty)");
        }
    }

    pub fn _dump_filtered(&self, prefix: &str) {
        let mut keys: Vec<_> = self.vars.keys().filter(|k| k.starts_with(prefix)).collect();
        keys.sort();

        if keys.is_empty() {
            println!("  (no variables matching '{}')", prefix);
            return;
        }

        let max_len = keys.iter().map(|k| k.len()).max().unwrap_or(0);

        for key in keys {
            let value = &self.vars[key];
            println!("  {:width$} : {}", key, value, width = max_len);
        }
    }
}

pub fn update_ui_variables(
    ui_loader: &mut Ui,
    time_scales: &TimeScales,
    astronomy: &Astronomy,
    obliquity: f32,
    settings: &Settings,
) {
    ui_loader
        .variables
        .set_f64("day_length", time_scales.day_length);
    ui_loader
        .variables
        .set_f64("total_days", time_scales.total_days);
    ui_loader
        .variables
        .set_f64("base_year", time_scales.base_year);
    ui_loader
        .variables
        .set_f64("current_year", time_scales.current_year);
    ui_loader.variables.set_f64("earth_obliquity", obliquity);
    ui_loader
        .variables
        .set_f64("sun_declination", astronomy.sun_declination);
    ui_loader
        .variables
        .set_f64("moon_phase", astronomy.moon_phase);
    ui_loader
        .variables
        .set_bool("reversed_depth_z", settings.reversed_depth_z);
}
