use crate::data::Settings;
use crate::ui::parser::Value;
use crate::ui::ui_editor::Ui;
use crate::world::astronomy::{Astronomy, TimeScales};
use std::collections::HashMap;

pub struct Variables {
    vars: HashMap<String, Value>,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set_f64<T: Into<f64>>(&mut self, name: &str, value: T) {
        let value = Value::F64(value.into());
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_i64<T: Into<i64>>(&mut self, name: &str, value: T) {
        let value = Value::I64(value.into());
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        let value = Value::Bool(value);
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_string<S>(&mut self, name: &str, value: S)
    where
        S: Into<String>,
    {
        let value = Value::String(value.into());
        self.vars.insert(name.to_string(), value);
    }

    pub fn set_array<I, T>(&mut self, name: &str, iter: I)
    where
        I: IntoIterator<Item = T>,
        T: Into<Value>,
    {
        let array_value = Value::Array(iter.into_iter().map(|e| e.into()).collect());

        let (base, suffix_opt) = match name.rsplit_once('.') {
            Some((base, suffix)) => {
                if Self::component_index(suffix).is_some() {
                    (base, Some(suffix))
                } else {
                    (name, None)
                }
            }
            None => (name, None),
        };

        let suffix = match suffix_opt {
            Some(s) => s,
            None => {
                self.vars.insert(base.to_string(), array_value);
                return;
            }
        };

        let idx = Self::component_index(suffix).unwrap();

        if !matches!(self.vars.get(base), Some(Value::Array(_))) {
            self.vars.insert(base.to_string(), Value::Array(Vec::new()));
        }

        if let Some(Value::Array(a)) = self.vars.get_mut(base) {
            if idx >= a.len() {
                a.resize(idx + 1, Value::Null);
            }
            a[idx] = array_value;
        }
    }

    pub fn set_var<S, V>(&mut self, name: S, value: V)
    where
        S: AsRef<str>,
        V: Into<Value>,
    {
        let name = name.as_ref();
        let value = value.into();

        let (base, suffix_opt) = match name.rsplit_once('.') {
            Some((base, suffix)) => {
                if Self::component_index(suffix).is_some() {
                    (base, Some(suffix))
                } else {
                    (name, None)
                }
            }
            None => (name, None),
        };

        let suffix = match suffix_opt {
            Some(s) => s,
            None => {
                self.vars.insert(base.to_string(), value);
                return;
            }
        };

        let idx = Self::component_index(suffix).unwrap();

        // Ensure an array exists at base (create if missing or not an array)
        if !matches!(self.vars.get(base), Some(Value::Array(_))) {
            self.vars.insert(base.to_string(), Value::Array(Vec::new()));
        }

        if let Some(Value::Array(a)) = self.vars.get_mut(base) {
            if idx >= a.len() {
                a.resize(idx + 1, Value::Null);
            }
            a[idx] = value;
        }
    }
    pub fn component_index(s: &str) -> Option<usize> {
        match s {
            "x" | "r" | "h" => Some(0),
            "y" | "g" | "s" => Some(1),
            "z" | "b" | "v" => Some(2),
            "w" => Some(3),
            _ => s.parse::<usize>().ok(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        let (base, suffix_opt) = match name.rsplit_once('.') {
            Some((base, suffix)) => {
                if Self::component_index(suffix).is_some() {
                    (base, Some(suffix))
                } else {
                    (name, None)
                }
            }
            None => (name, None),
        };

        let v = self.vars.get(base)?;

        let suffix = match suffix_opt {
            Some(s) => s,
            None => return Some(v),
        };

        let idx = Self::component_index(suffix)?;

        match v {
            Value::Array(a) => a.get(idx),
            _ => None,
        }
    }
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        match self.vars.get(name)? {
            Value::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_i64(&self, name: &str) -> Option<i64> {
        match self.vars.get(name)? {
            Value::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.vars.get(name)? {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.vars.get(name)? {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_array(&self, name: &str) -> Option<&Vec<Value>> {
        match self.get(name)? {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    pub fn dump(&self) {
        let mut keys: Vec<_> = self.vars.keys().collect();
        keys.sort();

        let max_len = keys.iter().map(|k| k.len()).max().unwrap_or(0);

        for key in keys {
            let value = &self.vars[key];
            let type_tag = value.type_name();
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
impl From<glam::Vec3> for Value {
    fn from(v: glam::Vec3) -> Self {
        Value::Array(vec![
            Value::F64(v.x as f64),
            Value::F64(v.y as f64),
            Value::F64(v.z as f64),
        ])
    }
}

// Bonus: Vec2 and Vec4 if you need them
impl From<glam::Vec2> for Value {
    fn from(v: glam::Vec2) -> Self {
        Value::Array(vec![Value::F64(v.x as f64), Value::F64(v.y as f64)])
    }
}

impl From<glam::Vec4> for Value {
    fn from(v: glam::Vec4) -> Self {
        Value::Array(vec![
            Value::F64(v.x as f64),
            Value::F64(v.y as f64),
            Value::F64(v.z as f64),
            Value::F64(v.w as f64),
        ])
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
