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
}

pub fn update_ui_variables(
    ui_loader: &mut UiButtonLoader,
    time_scales: &TimeScales,
    astronomy: &AstronomyState,
    obliquity: f32,
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
}
