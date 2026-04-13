use crate::data::Settings;
use crate::ui::parser::Value;
use crate::ui::ui_editor::Ui;
use crate::world::astronomy::{Astronomy, TimeScales};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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
        // println!("IN set_array(): {}, {}", name, array_value);
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
        S: AsRef<str> + std::fmt::Debug,
        V: Into<Value> + std::fmt::Display,
    {
        //println!("In variables set(): {:?} to {}", name, value);
        let name = name.as_ref();

        let (init, value) = initialize_value(name, value.into());
        let (base, suffix_opt) = match init.rsplit_once('.') {
            Some((base, suffix)) => {
                if Self::component_index(suffix).is_some() {
                    (base, Some(suffix))
                } else {
                    (init, None)
                }
            }
            None => (init, None),
        };
        //println!("In variables set() base: {:?}, suffix: {:?}", base, suffix_opt);
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
            //println!("Array: {:?}", a);
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
        //println!("In Variables::get(): Name: {}, Value: {:?}", base, v);
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

/// Output is (base, value)
pub fn initialize_value(name: &str, value: Value) -> (&str, Value) {
    let (field_type, base) = match name.split_once(':') {
        Some((field_type, base)) => (field_type, base),
        None => ("None", name),
    };

    let field_type = field_type.to_ascii_lowercase();

    let value = match field_type.as_str() {
        "f" | "f64" | "f32" => coerce_to_f64(value),
        "i" | "i64" | "i32" => coerce_to_i64(value),
        "str" | "string" | "s" => coerce_to_string(value),
        "bool" | "b" => coerce_to_bool(value),
        _ => value,
    };

    (base, value)
}

fn coerce_to_f64(value: Value) -> Value {
    match value {
        Value::F64(v) => Value::F64(v),
        Value::I64(v) => Value::F64(v as f64),
        Value::Bool(v) => Value::F64(if v { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().map(Value::F64).unwrap_or(Value::F64(1.0)),
        _ => value,
    }
}

fn coerce_to_i64(value: Value) -> Value {
    match value {
        Value::I64(v) => Value::I64(v),
        Value::F64(v) => {
            if v.is_finite() && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                Value::I64(v as i64)
            } else {
                Value::I64(1)
            }
        }
        Value::Bool(v) => Value::I64(if v { 1 } else { 0 }),
        Value::String(s) => s.parse::<i64>().map(Value::I64).unwrap_or(Value::I64(1)),
        _ => value,
    }
}

fn coerce_to_string(value: Value) -> Value {
    match value {
        Value::String(s) => Value::String(s),
        Value::F64(v) => Value::String(v.to_string()),
        Value::I64(v) => Value::String(v.to_string()),
        Value::Bool(v) => Value::String(v.to_string()),
        _ => value,
    }
}

fn coerce_to_bool(value: Value) -> Value {
    match value {
        Value::Bool(v) => Value::Bool(v),
        Value::I64(v) => Value::Bool(v != 0),
        Value::F64(v) => Value::Bool(v != 0.0),
        Value::String(s) => {
            let t = s.trim().to_ascii_lowercase();
            match t.as_str() {
                "true" | "1" | "yes" | "y" | "on" => Value::Bool(true),
                "false" | "0" | "no" | "n" | "off" => Value::Bool(false),
                _ => Value::Bool(true),
            }
        }
        _ => value,
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

pub fn load_colors(path: PathBuf, settings: &Settings, vars: &mut Variables) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[colors] Failed to read file: {}", e);
            return;
        }
    };
    let mut keys: Vec<&str> = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();

        // skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (key, value_str) = match line.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => {
                eprintln!("[colors] Invalid line {}: '{}'", line_idx + 1, line);
                continue;
            }
        };
        keys.push(key);
        let value = Value::from_str(settings, vars, value_str);
        match &value {
            Value::Array(_) => {}
            _ => {
                eprintln!(
                    "[colors] Failed to parse value for '{}': {}  From: {}",
                    key, value, value_str
                );
                continue;
            }
        };
        let out = match value.as_color4() {
            Some(c) => c,
            None => {
                eprintln!(
                    "[colors] '{}' must have 3 or 4 elements in F64 format (WITH a DOT '.' and decimal like 0.0!)",
                    key
                );
                continue;
            }
        };

        vars.set_array(key, out);
    }
    vars.set_array("color_keys", keys)
}

pub fn save_colors(path: PathBuf, vars: &Variables) {
    let mut out = String::new();
    for key in vars.get_array("color_keys").into_iter().flatten() {
        let key = match key {
            Value::String(key) => key.as_str(),
            _ => continue,
        };
        let color = match vars.get(key) {
            None => continue,
            Some(value) => value.as_color4().unwrap_or([1.0, 0.0, 1.0, 1.0]),
        };
        if (color[3] - 1.0).abs() < f32::EPSILON {
            out.push_str(&format!(
                "{}: [{}, {}, {}]\n",
                key, color[0], color[1], color[2]
            ));
        } else {
            out.push_str(&format!(
                "{}: [{}, {}, {}, {}]\n",
                key, color[0], color[1], color[2], color[3]
            ));
        }
    }

    if let Err(e) = fs::write(path, out) {
        eprintln!("[colors] Failed to write file: {}", e);
    }
}
