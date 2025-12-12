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
        let valu = UiValue::F32(value);
        println!("{name}: {valu:#?}");
        self.vars.insert(name.to_string(), valu);
    }

    pub fn set_i32(&mut self, name: &str, value: i32) {
        let valu = UiValue::I32(value);
        println!("{name}: {valu:#?}");
        self.vars.insert(name.to_string(), valu);
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        let valu = UiValue::Bool(value);
        println!("{name}: {valu:#?}");
        self.vars.insert(name.to_string(), valu);
    }

    pub fn set_string<S>(&mut self, name: &str, value: S)
    where
        S: Into<String>,
    {
        let valu = UiValue::String(value.into());
        println!("{name}: {valu:#?}");
        self.vars.insert(name.to_string(), valu);
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
