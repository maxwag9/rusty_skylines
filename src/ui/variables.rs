use crate::data::Settings;
use crate::renderer::astronomy::{AstronomyState, TimeScales};
use crate::ui::ui_editor::UiButtonLoader;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub enum UiValue {
    F32(f32),
    I32(i32),
    Bool(bool),
    String(String),
}

impl fmt::Display for UiValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UiValue::F32(v) => write!(f, "{v}"),
            UiValue::I32(v) => write!(f, "{v}"),
            UiValue::Bool(v) => write!(f, "{v}"),
            UiValue::String(v) => write!(f, "{v}"),
        }
    }
}

impl From<f32> for UiValue {
    fn from(v: f32) -> Self {
        UiValue::F32(v)
    }
}

impl From<i32> for UiValue {
    fn from(v: i32) -> Self {
        UiValue::I32(v)
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

pub struct UiVariableRegistry {
    vars: HashMap<String, UiValue>,
}

impl UiVariableRegistry {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set_f32(&mut self, name: &str, value: f32) {
        let value = UiValue::F32(value);
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_i32(&mut self, name: &str, value: i32) {
        let value = UiValue::I32(value);
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

    pub fn get(&self, name: &str) -> Option<&UiValue> {
        self.vars.get(name)
    }
    pub fn get_f32(&self, name: &str) -> Option<f32> {
        match self.vars.get(name)? {
            UiValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_i32(&self, name: &str) -> Option<i32> {
        match self.vars.get(name)? {
            UiValue::I32(v) => Some(*v),
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
                UiValue::F32(_) => "f32",
                UiValue::I32(_) => "i32",
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
    ui_loader: &mut UiButtonLoader,
    time_scales: &TimeScales,
    astronomy: &AstronomyState,
    obliquity: f32,
    settings: &Settings,
) {
    ui_loader
        .variables
        .set_f32("day_length", time_scales.day_length);
    ui_loader
        .variables
        .set_f32("total_days", time_scales.total_days);
    ui_loader
        .variables
        .set_f32("base_year", time_scales.base_year);
    ui_loader
        .variables
        .set_f32("current_year", time_scales.current_year);
    ui_loader.variables.set_f32("earth_obliquity", obliquity);
    ui_loader
        .variables
        .set_f32("sun_declination", astronomy.sun_declination);
    ui_loader
        .variables
        .set_f32("moon_phase", astronomy.moon_phase);
    ui_loader
        .variables
        .set_bool("reversed_depth_z", settings.reversed_depth_z);
}
