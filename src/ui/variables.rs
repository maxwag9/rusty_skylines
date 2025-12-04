use std::collections::HashMap;
use std::str::FromStr;

pub struct UiVariableRegistry {
    pub(crate) vars: HashMap<String, String>,
}

impl UiVariableRegistry {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        self.vars.insert(name.to_string(), value.into());
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(|s| s.as_str())
    }

    pub fn get_f32(&self, name: &str) -> Option<f32> {
        self.get(name)?.parse().ok()
    }

    pub fn get_i32(&self, name: &str) -> Option<i32> {
        self.get(name)?.parse().ok()
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.get(name)?.parse().ok()
    }

    pub fn get_parsed<T>(&self, name: &str) -> Option<T>
    where
        T: FromStr,
    {
        self.get(name)?.parse().ok()
    }
}
