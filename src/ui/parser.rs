use crate::data::{SettingKey, Settings};
use crate::helpers::hsv::{HSV, hsv_to_rgb};
use crate::ui::variables::{Variables, initialize_value};
use rand::RngExt;
use rand::rngs::ThreadRng;
use std::fmt;

// ------------------------------------------------------------
// Value type
// ------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    F64(f64),
    I64(i64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_value().fmt(f)
    }
}

impl Value {
    pub fn from_str(
        settings: &Settings,
        variables: &Variables,
        s: &str,
        with_expr: bool,
        with_post_expr: bool,
    ) -> Self {
        let s = Self::replace_inline_variables(variables, s);
        let s = s.trim();

        if s.is_empty() {
            return Value::String(String::new());
        }
        // Explicit type prefix
        if let Some((ty, value)) = s.split_once(':') {
            match ty.to_ascii_lowercase().as_str() {
                "int" => {
                    if let Ok(i) = value.parse::<i64>() {
                        return Value::I64(i);
                    }
                }
                "float" | "f32" | "f64" => {
                    if let Ok(f) = value.parse::<f64>() {
                        return Value::F64(f);
                    }
                }
                "bool" => match value.to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => return Value::Bool(true),
                    "false" | "0" | "no" | "off" => return Value::Bool(false),
                    _ => {}
                },
                "string" | "str" => {
                    return Value::String(value.to_string());
                }
                "strexpr" => {
                    //println!("strexpr was given: {}", value);
                    let s = Value::String(
                        Value::from_str(settings, variables, value, true, false)
                            .into_string_value(),
                    );
                    //println!("strexpr gave: {}", s);
                    return s;
                }
                // "strexprvar" => {
                //     println!("strexprvar was given: {}", value);
                //     let s = Value::String(Value::from_str(settings, variables, value, true, true).into_string_value());
                //     println!("strexprvar gave: {}", s);
                //     return s;
                // }
                "setting" => {
                    match value.split_once(".") {
                        None => {
                            let key = SettingKey::from_str(value);
                            if let Some(key) = key {
                                let value = settings.read_setting(key).to_value();
                                //println!("{:?} {:?}", key, value);
                                return value;
                            }
                        }
                        Some((l, r)) => {
                            let key = SettingKey::from_str(l);
                            if let Some(key) = key {
                                match r {
                                    "options" => {
                                        return key.options();
                                    }
                                    _ => {}
                                }
                                let value = settings.read_setting(key).to_value();
                                //println!("{:?} {:?}", key, value);
                                return value;
                            }
                        }
                    }
                }
                "var" | "variable" => match Self::load_variable(variables, &s.to_string()) {
                    Some(value) => return value,
                    None => {}
                },
                "array" | "list" | "vec" | "slice" => {
                    if let Some(arr) = Self::parse_array(settings, variables, value) {
                        return Value::Array(arr);
                    }
                }
                _ => {}
            }
        }

        // Auto-detect array (bracket notation)
        if s.starts_with('[') && s.ends_with(']') {
            if let Some(arr) = Self::parse_array(settings, variables, s) {
                //println!("from_str color: {:?}", arr);
                return Value::Array(arr);
            }
        }

        // Auto-detect bool
        match s.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => return Value::Bool(true),
            "false" | "no" | "off" => return Value::Bool(false),
            _ => {}
        }

        // Auto-detect integer
        if let Ok(i) = s.parse::<i64>() {
            return Value::I64(i);
        }

        // Auto-detect float
        if let Ok(f) = s.parse::<f64>() {
            return Value::F64(f);
        }

        // Might be setting key?
        match s.split_once(".") {
            None => {
                let key = SettingKey::from_str(s);
                if let Some(key) = key {
                    let value = settings.read_setting(key).to_value();
                    //println!("{:?} {:?}", key, value);
                    return value;
                }
            }
            Some((l, r)) => {
                let key = SettingKey::from_str(l);
                if let Some(key) = key {
                    match r {
                        "options" => {
                            return key.options();
                        }
                        _ => {
                            match r.split_once(".") {
                                None => {
                                    return key.options();
                                }
                                Some((l, r)) => {
                                    if let Ok(idx) = r.parse::<usize>() {
                                        if let Some(array) = key.options().as_array() {
                                            if let Some(val) = array.get(idx) {
                                                //println!("HOLY JESUS0, {:?}, {}", array, idx);
                                                return val.to_owned();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let value = settings.read_setting(key).to_value();
                    //println!("{:?} {:?}", key, value);
                    return value;
                }
            }
        }

        match Self::load_variable(variables, &s.to_string()) {
            Some(value) => return value,
            None => {}
        };

        //println!("In from_str() Evaluating Expression... Input: {} ", s);
        if with_expr {
            let expr_value = match eval_expr(s, variables, settings) {
                Some(value) => match value {
                    Value::Null => {
                        //println!("Input: {}, Output: Null!!!", s);
                        Value::String(s.to_string())
                    }
                    _ => {
                        //println!("Input: {}, Output: {}({})", s, value.type_name(), value);
                        value
                    }
                },
                None => {
                    //println!("NONE!! in from_str() NONE!!: {}", s);
                    Value::String(s.to_string())
                }
            };
            if with_post_expr {
                match expr_value {
                    Value::String(s) => {
                        //println!("{s}");
                        Value::from_str(settings, variables, &s, false, false)
                    }
                    _ => expr_value,
                }
            } else {
                expr_value
            }
            //expr_value
        } else {
            Value::String(s.to_string())
        }
    }
    pub fn from_vec<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<f64>,
    {
        Value::Array(iter.into_iter().map(|e| Value::F64(e.into())).collect())
    }
    fn load_variable(variables: &Variables, name: &String) -> Option<Value> {
        variables.get(name).map(|v| v.into_owned())
    }
    pub fn is_f64(&self) -> Option<Value> {
        match self {
            Value::F64(n) => Some(self.clone()),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(n) => Some(*n),
            Value::I64(n) => Some(*n as f64),
            Value::String(s) => s.parse().ok(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Array(arr) => Some(arr.len() as f64),
            Value::Null => Some(0.0),
        }
    }

    pub fn to_f64(self) -> Value {
        match self {
            Value::F64(v) => Value::F64(v),
            Value::I64(v) => Value::F64(v as f64),
            Value::Bool(v) => Value::F64(if v { 1.0 } else { 0.0 }),
            Value::String(s) => s.parse::<f64>().map(Value::F64).unwrap_or(Value::F64(1.0)),
            Value::Null => Value::F64(0.0),
            Value::Array(_) => Value::F64(0.0),
        }
    }

    pub fn is_i64(&self) -> Option<Value> {
        match self {
            Value::I64(n) => Some(self.clone()),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::F64(n) => Some(*n as i64),
            Value::I64(n) => Some(*n),
            Value::String(s) => s.parse().ok(),
            Value::Bool(b) => Some(if *b { 1 } else { 0 }),
            Value::Array(arr) => Some(arr.len() as i64),
            Value::Null => Some(0),
        }
    }

    pub fn to_i64(self) -> Value {
        match self {
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
            Value::Null => Value::I64(0),
            Value::Array(_) => Value::I64(0),
        }
    }
    pub fn is_string(&self) -> Option<Value> {
        match self {
            Value::String(n) => Some(self.clone()),
            _ => None,
        }
    }
    pub fn is_bool(&self) -> Option<Value> {
        match self {
            Value::Bool(n) => Some(self.clone()),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<Vec<Value>> {
        match self {
            Value::Array(arr) => Some(arr.clone()),
            _ => None,
        }
    }
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(str) => Some(str),
            _ => None,
        }
    }
    pub fn as_pos(&self) -> Option<[f32; 2]> {
        let arr = self.as_array()?;
        if arr.len() != 2 {
            return None;
        }
        Some([arr[0].as_f64()? as f32, arr[1].as_f64()? as f32])
    }
    pub fn as_color3(&self) -> Option<[f32; 3]> {
        let arr = self.as_array()?;
        if arr.len() != 4 {
            return None;
        }
        Some([
            arr[0].as_f64()? as f32,
            arr[1].as_f64()? as f32,
            arr[2].as_f64()? as f32,
        ])
    }

    pub fn as_color4(&self) -> Option<[f32; 4]> {
        let arr = self.as_array()?;

        let alpha = match arr.len() {
            3 => 1.0,
            4 => arr[3].as_f64()? as f32,
            _ => return None,
        };

        Some([
            arr[0].as_f64()? as f32,
            arr[1].as_f64()? as f32,
            arr[2].as_f64()? as f32,
            alpha,
        ])
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::I64(i) => *i != 0,
            Value::F64(f) => *f != 0.0,
            Value::String(s) => {
                if s.is_empty() {
                    return false;
                };
                let t = s.trim().to_ascii_lowercase();
                match t.as_str() {
                    "true" | "1" | "yes" | "y" | "on" => true,
                    "false" | "0" | "no" | "n" | "off" => false,
                    _ => true,
                }
            }
            Value::Array(arr) => !arr.is_empty(),
            Value::Null => false,
        }
    }
    pub fn to_string_value(&self) -> String {
        match self {
            Value::F64(n) => n.to_string(),
            Value::I64(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string_value()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Null => "null".to_string(),
        }
    }

    pub fn into_string_value(self) -> String {
        match self {
            Value::F64(n) => n.to_string(),
            Value::I64(n) => n.to_string(),
            Value::String(s) => s, // moved, no clone
            Value::Bool(b) => b.to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.into_iter().map(|v| v.into_string_value()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Null => "null".to_string(),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::I64(_) => "integer64",
            Value::F64(_) => "float64",
            Value::String(_) => "string",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::Null => "null",
        }
    }

    /// Split a string by commas, respecting nested brackets and quoted strings
    fn split_array_elements(s: &str) -> Vec<&str> {
        let mut elements = Vec::new();
        let mut start = 0;
        let mut bracket_depth: i32 = 0i32;
        let mut in_quotes = false;
        let mut chars = s.char_indices().peekable();
        let mut prev_char = '\0';

        while let Some((i, c)) = chars.next() {
            match c {
                '"' if prev_char != '\\' => {
                    in_quotes = !in_quotes;
                }
                '[' | '(' | '{' if !in_quotes => {
                    bracket_depth += 1;
                }
                ']' | ')' | '}' if !in_quotes => {
                    bracket_depth = bracket_depth.saturating_sub(1);
                }
                ',' if !in_quotes && bracket_depth == 0 => {
                    elements.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
            prev_char = c;
        }

        // Add the remaining element
        if start < s.len() {
            elements.push(&s[start..]);
        } else if start == s.len() && !s.is_empty() && s.ends_with(',') {
            // Handle trailing comma - add empty element
            elements.push("");
        }

        elements
    }

    /// Parse an array from a string like "[1, 2, 3]" or "1, 2, 3"
    fn parse_array(settings: &Settings, variables: &Variables, s: &str) -> Option<Vec<Value>> {
        let s = s.trim();

        // Handle empty input
        if s.is_empty() {
            return Some(Vec::new());
        }

        // Remove surrounding brackets if present
        let inner = if s.starts_with('[') && s.ends_with(']') {
            &s[1..s.len() - 1]
        } else {
            s
        };

        let inner = inner.trim();

        if inner.is_empty() {
            return Some(Vec::new());
        }

        // Split by commas, respecting nested brackets and quotes
        let elements = Self::split_array_elements(inner);

        let mut result = Vec::with_capacity(elements.len());
        for elem in elements {
            let elem = elem.trim();
            if !elem.is_empty() {
                result.push(Value::from_str(settings, variables, elem, true, true));
            }
        }

        Some(result)
    }
    fn replace_inline_variables(variables: &Variables, s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                let mut var_name = String::new();
                let mut found_end = false;

                // Consume characters until we find the closing '}'
                for inner_c in chars.by_ref() {
                    if inner_c == '}' {
                        found_end = true;
                        break;
                    }
                    var_name.push(inner_c);
                }

                if found_end && !var_name.is_empty() {
                    // Try to load the variable
                    if let Some(value) = Self::load_variable(variables, &var_name) {
                        // Replace with the variable's string representation
                        result.push_str(&value.to_string());
                    } else {
                        // Variable doesn't exist yet, leave it exactly as it was
                        result.push('{');
                        result.push_str(&var_name);
                        result.push('}');
                    }
                } else {
                    // Unclosed brace or empty {}, leave as is
                    result.push('{');
                    result.push_str(&var_name);
                    if found_end {
                        result.push('}');
                    }
                }
            } else {
                result.push(c);
            }
        }
        result
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}

impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F64(v as f64)
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::I64(v as i64)
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::I64(v as i64)
    }
}

impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::I64(v as i64)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::I64(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I64(v as i64)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

// Built-in functions registry
type BuiltinFn = fn(Vec<Value>) -> Option<Value>;

fn get_builtin(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "fix" => |args| {
            let val = args.first()?;
            let n = val.as_f64()?;
            //println!("'{}'", n);
            // defaults
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(3.0) as usize;
            let min_int = args
                .get(2)
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::INFINITY) as usize;

            let sign = if n < 0.0 { "-" } else { "" };
            let abs = n.abs();

            // format with exact decimals
            let formatted = format!("{:.*}", decimals, abs);

            let mut parts = formatted.split('.');
            let int_part = parts.next().unwrap_or("");
            let frac_part = parts.next().unwrap_or("");

            // left pad integers with spaces
            let padded_int = if min_int == usize::MAX {
                int_part.to_string()
            } else {
                if int_part.len() < min_int {
                    let pad = " ".repeat(min_int - int_part.len());
                    format!("{}{}", pad, int_part)
                } else {
                    int_part.to_string()
                }
            };

            let result = if decimals > 0 {
                format!("{}{}.{}", sign, padded_int, frac_part)
            } else {
                format!("{}{}", sign, padded_int)
            };
            //println!("'{}'", result);
            Some(Value::String(result))
        },
        // Math functions
        "abs" => |args| match args.first()? {
            Value::F64(n) => Some(Value::F64(n.abs())),
            Value::I64(n) => Some(Value::I64(n.abs())),
            _ => None,
        },
        "floor" => |args| args.first()?.as_f64().map(|n| Value::F64(n.floor())),
        "ceil" => |args| args.first()?.as_f64().map(|n| Value::F64(n.ceil())),
        "round" => |args| {
            let n = args.first()?.as_f64()?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
            let factor = 10f64.powi(decimals);
            Some(Value::F64((n * factor).round() / factor))
        },
        "trunc" => |args| args.first()?.as_f64().map(|n| Value::F64(n.trunc())),
        "sqrt" => |args| args.first()?.as_f64().map(|n| Value::F64(n.sqrt())),
        "cbrt" => |args| args.first()?.as_f64().map(|n| Value::F64(n.cbrt())),
        "pow" => |args| {
            let base = args.first()?.as_f64()?;
            let exp = args.get(1)?.as_f64()?;
            Some(Value::F64(base.powf(exp)))
        },
        "exp" => |args| args.first()?.as_f64().map(|n| Value::F64(n.exp())),
        "ln" => |args| args.first()?.as_f64().map(|n| Value::F64(n.ln())),
        "log" => |args| {
            let n = args.first()?.as_f64()?;
            let base = args.get(1).and_then(|v| v.as_f64()).unwrap_or(10.0);
            Some(Value::F64(n.log(base)))
        },
        "log2" => |args| args.first()?.as_f64().map(|n| Value::F64(n.log2())),
        "log10" => |args| args.first()?.as_f64().map(|n| Value::F64(n.log10())),
        "sin" => |args| {
            args.first()?.as_f64().map(|n| {
                //println!("Sin() in get_builtin() input: {n}");
                Value::F64(n.sin())
            })
        },
        "cos" => |args| args.first()?.as_f64().map(|n| Value::F64(n.cos())),
        "tan" => |args| args.first()?.as_f64().map(|n| Value::F64(n.tan())),
        "asin" => |args| args.first()?.as_f64().map(|n| Value::F64(n.asin())),
        "acos" => |args| args.first()?.as_f64().map(|n| Value::F64(n.acos())),
        "atan" => |args| args.first()?.as_f64().map(|n| Value::F64(n.atan())),
        "atan2" => |args| {
            let y = args.first()?.as_f64()?;
            let x = args.get(1)?.as_f64()?;
            Some(Value::F64(y.atan2(x)))
        },
        "sinh" => |args| args.first()?.as_f64().map(|n| Value::F64(n.sinh())),
        "cosh" => |args| args.first()?.as_f64().map(|n| Value::F64(n.cosh())),
        "tanh" => |args| args.first()?.as_f64().map(|n| Value::F64(n.tanh())),
        "degrees" => |args| args.first()?.as_f64().map(|n| Value::F64(n.to_degrees())),
        "radians" => |args| args.first()?.as_f64().map(|n| Value::F64(n.to_radians())),
        "min" => |args| {
            let mut min_val: Option<f64> = None;
            for arg in &args {
                if let Some(n) = arg.as_f64() {
                    min_val = Some(min_val.map_or(n, |m| m.min(n)));
                }
            }
            min_val.map(Value::F64)
        },
        "max" => |args| {
            let mut max_val: Option<f64> = None;
            for arg in &args {
                if let Some(n) = arg.as_f64() {
                    max_val = Some(max_val.map_or(n, |m| m.max(n)));
                }
            }
            max_val.map(Value::F64)
        },
        "clamp" => |args| {
            let val = args.first()?.as_f64()?;
            let min = args.get(1)?.as_f64()?;
            let max = args.get(2)?.as_f64()?;
            //println!("{}, {}, {}, {}", val, min, max, val.clamp(min, max));
            Some(Value::F64(val.clamp(min, max)))
        },
        "lerp" => |args| {
            let a = args.first()?.as_f64()?;
            let b = args.get(1)?.as_f64()?;
            let t = args.get(2)?.as_f64()?;
            Some(Value::F64(a + (b - a) * t))
        },
        "sign" => |args| {
            let n = args.first()?.as_f64()?;
            Some(Value::F64(if n > 0.0 {
                1.0
            } else if n < 0.0 {
                -1.0
            } else {
                0.0
            }))
        },
        "fract" => |args| args.first()?.as_f64().map(|n| Value::F64(n.fract())),
        "mod" => |args| {
            let a = args.first()?.as_f64()?;
            let b = args.get(1)?.as_f64()?;
            Some(Value::F64(a % b))
        },
        "hypot" => |args| {
            let a = args.first()?.as_f64()?;
            let b = args.get(1)?.as_f64()?;
            Some(Value::F64(a.hypot(b)))
        },

        // Constants
        "pi" => |_| Some(Value::F64(std::f64::consts::PI)),
        "e" => |_| Some(Value::F64(std::f64::consts::E)),
        "tau" => |_| Some(Value::F64(std::f64::consts::TAU)),
        "inf" => |_| Some(Value::F64(f64::INFINITY)),
        "nan" => |_| Some(Value::F64(f64::NAN)),

        // Number checks
        "isnan" => |args| args.first()?.as_f64().map(|n| Value::Bool(n.is_nan())),
        "isinf" => |args| args.first()?.as_f64().map(|n| Value::Bool(n.is_infinite())),
        "isfinite" => |args| args.first()?.as_f64().map(|n| Value::Bool(n.is_finite())),

        // String / array functions
        "len" => |args| match args.first()? {
            Value::String(s) => Some(Value::F64(s.len() as f64)),
            Value::Array(arr) => Some(Value::F64(arr.len() as f64)),
            _ => None,
        },
        "upper" => |args| match args.first()? {
            Value::String(s) => Some(Value::String(s.to_uppercase())),
            v => Some(Value::String(v.to_string().to_uppercase())),
        },
        "lower" => |args| match args.first()? {
            Value::String(s) => Some(Value::String(s.to_lowercase())),
            v => Some(Value::String(v.to_string().to_lowercase())),
        },
        "capitalize" => |args| match args.first()? {
            Value::String(s) => {
                let mut chars = s.chars();
                let result = match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    None => String::new(),
                };
                Some(Value::String(result))
            }
            _ => None,
        },
        "title" => |args| match args.first()? {
            Value::String(s) => {
                let result = s
                    .split_whitespace()
                    .map(|word| {
                        let mut chars = word.chars();
                        match chars.next() {
                            Some(first) => {
                                first.to_uppercase().collect::<String>()
                                    + &chars.as_str().to_lowercase()
                            }
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                Some(Value::String(result))
            }
            _ => None,
        },
        "trim" => |args| match args.first()? {
            Value::String(s) => Some(Value::String(s.trim().to_string())),
            _ => None,
        },
        "ltrim" => |args| match args.first()? {
            Value::String(s) => Some(Value::String(s.trim_start().to_string())),
            _ => None,
        },
        "rtrim" => |args| match args.first()? {
            Value::String(s) => Some(Value::String(s.trim_end().to_string())),
            _ => None,
        },
        "reverse" => |args| match args.first()? {
            Value::String(s) => Some(Value::String(s.chars().rev().collect())),
            Value::Array(arr) => Some(Value::Array(arr.iter().rev().cloned().collect())),
            _ => None,
        },
        "repeat" => |args| {
            let s = match args.first()? {
                Value::String(s) => s.clone(),
                v => v.to_string(),
            };
            let n = args.get(1)?.as_f64()? as usize;
            Some(Value::String(s.repeat(n)))
        },
        "replace" => |args| match args.first()? {
            Value::String(s) => {
                let from = match args.get(1)? {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                let to = match args.get(2)? {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                Some(Value::String(s.replace(&from, &to)))
            }
            _ => None,
        },
        "split" => |args| match args.first()? {
            Value::String(s) => {
                let delim = match args.get(1) {
                    Some(Value::String(d)) => d.clone(),
                    _ => " ".to_string(),
                };
                let parts: Vec<Value> = s
                    .split(&delim)
                    .map(|p| Value::String(p.to_string()))
                    .collect();
                Some(Value::Array(parts))
            }
            _ => None,
        },
        "join" => |args| match args.first()? {
            Value::Array(arr) => {
                let delim = match args.get(1) {
                    Some(Value::String(d)) => d.clone(),
                    _ => "".to_string(),
                };
                let result: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                Some(Value::String(result.join(&delim)))
            }
            _ => None,
        },
        "substr" => |args| match args.first()? {
            Value::String(s) => {
                let start = args.get(1)?.as_f64()? as usize;
                let len = args.get(2).and_then(|v| v.as_f64()).map(|n| n as usize);
                let chars: Vec<char> = s.chars().collect();
                let end = len
                    .map(|l| (start + l).min(chars.len()))
                    .unwrap_or(chars.len());
                if start >= chars.len() {
                    Some(Value::String(String::new()))
                } else {
                    Some(Value::String(chars[start..end].iter().collect()))
                }
            }
            _ => None,
        },
        "contains" => |args| match args.first()? {
            Value::String(s) => {
                let needle = match args.get(1)? {
                    Value::String(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Bool(s.contains(&needle)))
            }
            Value::Array(arr) => {
                let needle = args.get(1)?;
                Some(Value::Bool(arr.iter().any(|v| v == needle)))
            }
            _ => None,
        },
        "startswith" => |args| match args.first()? {
            Value::String(s) => {
                let prefix = match args.get(1)? {
                    Value::String(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Bool(s.starts_with(&prefix)))
            }
            _ => None,
        },
        "endswith" => |args| match args.first()? {
            Value::String(s) => {
                let suffix = match args.get(1)? {
                    Value::String(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Bool(s.ends_with(&suffix)))
            }
            _ => None,
        },
        "indexof" => |args| match args.first()? {
            Value::String(s) => {
                let needle = match args.get(1)? {
                    Value::String(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::F64(
                    s.find(&needle).map(|i| i as f64).unwrap_or(-1.0),
                ))
            }
            Value::Array(arr) => {
                let needle = args.get(1)?;
                for (i, v) in arr.iter().enumerate() {
                    if v == needle {
                        return Some(Value::F64(i as f64));
                    }
                }
                Some(Value::F64(-1.0))
            }
            _ => None,
        },
        "padleft" => |args| match args.first()? {
            Value::String(s) => {
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::String(s.clone()))
                } else {
                    let padding: String =
                        std::iter::repeat(pad_char).take(width - s.len()).collect();
                    Some(Value::String(padding + s))
                }
            }
            v => {
                let s = v.to_string();
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::String(s))
                } else {
                    let padding: String =
                        std::iter::repeat(pad_char).take(width - s.len()).collect();
                    Some(Value::String(padding + &s))
                }
            }
        },
        "padright" => |args| match args.first()? {
            Value::String(s) => {
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::String(s.clone()))
                } else {
                    let padding: String =
                        std::iter::repeat(pad_char).take(width - s.len()).collect();
                    Some(Value::String(s.clone() + &padding))
                }
            }
            _ => None,
        },
        "center" => |args| match args.first()? {
            Value::String(s) => {
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::String(s.clone()))
                } else {
                    let total_pad = width - s.len();
                    let left_pad = total_pad / 2;
                    let right_pad = total_pad - left_pad;
                    let left: String = std::iter::repeat(pad_char).take(left_pad).collect();
                    let right: String = std::iter::repeat(pad_char).take(right_pad).collect();
                    Some(Value::String(left + s + &right))
                }
            }
            _ => None,
        },
        "char" => |args| {
            let code = args.first()?.as_f64()? as u32;
            char::from_u32(code).map(|c| Value::String(c.to_string()))
        },
        "ord" => |args| match args.first()? {
            Value::String(s) => s.chars().next().map(|c| Value::I64(c as i64)),
            _ => None,
        },
        "hex" => |args| {
            let n = args.first()?.as_f64()? as i64;
            Some(Value::String(format!("{:x}", n)))
        },
        "bin" => |args| {
            let n = args.first()?.as_f64()? as i64;
            Some(Value::String(format!("{:b}", n)))
        },
        "oct" => |args| {
            let n = args.first()?.as_f64()? as i64;
            Some(Value::String(format!("{:o}", n)))
        },

        // Array functions
        "first" => |args| match args.first()? {
            Value::Array(arr) => arr.first().cloned(),
            Value::String(s) => s.chars().next().map(|c| Value::String(c.to_string())),
            _ => None,
        },
        "last" => |args| match args.first()? {
            Value::Array(arr) => arr.last().cloned(),
            Value::String(s) => s.chars().last().map(|c| Value::String(c.to_string())),
            _ => None,
        },
        "sum" => |args| match args.first()? {
            Value::Array(arr) => {
                let sum: f64 = arr.iter().filter_map(|v| v.as_f64()).sum();
                Some(Value::F64(sum))
            }
            _ => {
                let sum: f64 = args.iter().filter_map(|v| v.as_f64()).sum();
                Some(Value::F64(sum))
            }
        },

        "avg" => |args| match args.first()? {
            Value::Array(arr) => {
                let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                if nums.is_empty() {
                    None
                } else {
                    Some(Value::F64(nums.iter().sum::<f64>() / nums.len() as f64))
                }
            }
            _ => {
                let nums: Vec<f64> = args.iter().filter_map(|v| v.as_f64()).collect();
                if nums.is_empty() {
                    None
                } else {
                    Some(Value::F64(nums.iter().sum::<f64>() / nums.len() as f64))
                }
            }
        },

        "count" => |args| match args.first()? {
            Value::Array(arr) => Some(Value::I64(arr.len() as i64)),
            Value::String(s) => Some(Value::I64(s.chars().count() as i64)),
            _ => Some(Value::I64(args.len() as i64)),
        },
        "range" => |args| {
            let start = args.first()?.as_f64()? as i64;
            let end = args.get(1)?.as_f64()? as i64;
            let step = args.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as i64;

            if step == 0 {
                return None;
            }

            let mut result = Vec::new();
            let mut i = start;

            if step > 0 {
                while i < end {
                    result.push(Value::I64(i));
                    i += step;
                }
            } else {
                while i > end {
                    result.push(Value::I64(i));
                    i += step;
                }
            }

            Some(Value::Array(result))
        },

        "slice" => |args| match args.first()? {
            Value::Array(arr) => {
                let start = args.get(1)?.as_f64()? as i64;
                let end = args.get(2).and_then(|v| v.as_f64()).map(|n| n as i64);

                let len = arr.len() as i64;
                let start = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let end = match end {
                    Some(e) => (if e < 0 { (len + e).max(0) } else { e.min(len) }) as usize,
                    None => arr.len(),
                };

                if start >= end {
                    Some(Value::Array(vec![]))
                } else {
                    Some(Value::Array(arr[start..end].to_vec()))
                }
            }
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let start = args.get(1)?.as_f64()? as i64;
                let end = args.get(2).and_then(|v| v.as_f64()).map(|n| n as i64);

                let len = chars.len() as i64;
                let start = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let end = match end {
                    Some(e) => (if e < 0 { (len + e).max(0) } else { e.min(len) }) as usize,
                    None => chars.len(),
                };

                if start >= end {
                    Some(Value::String(String::new()))
                } else {
                    Some(Value::String(chars[start..end].iter().collect()))
                }
            }
            _ => None,
        },

        // Type functions
        "type" => |args| Some(Value::String(args.first()?.type_name().to_string())),
        "isnull" => |args| {
            Some(Value::Bool(
                args.first().map(|v| v.is_null()).unwrap_or(true),
            ))
        },
        "isfloat" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::F64(_))))),
        "isint" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::I64(_))))),
        "isstr" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::String(_))))),
        "isbool" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::Bool(_))))),
        "isarray" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::Array(_))))),

        // Conversion functions
        "str" => |args| Some(Value::String(args.first()?.clone().into_string_value())),
        "bool" => |args| Some(Value::Bool(args.first()?.is_truthy())),
        "int" => |args| Some(args.first()?.clone().to_i64()),
        "float" => |args| Some(args.first()?.clone().to_f64()),

        // Formatting functions
        "format" => |args| {
            let val = args.first()?;
            let precision = args.get(1).and_then(|v| v.as_f64()).map(|n| n as usize);

            if let (Some(n), Some(p)) = (val.as_f64(), precision) {
                Some(Value::String(format!("{:.*}", p, n)))
            } else {
                Some(Value::String(val.to_string()))
            }
        },
        "comma" => |args| {
            // Format number with thousands separator
            let n = args.first()?.as_f64()?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;

            let formatted = if decimals > 0 {
                format!("{:.*}", decimals, n)
            } else {
                format!("{}", n.trunc() as i64)
            };

            let parts: Vec<&str> = formatted.split('.').collect();
            let int_part = parts[0];
            let frac_part = parts.get(1);

            let negative = int_part.starts_with('-');
            let digits: String = int_part.chars().filter(|c| c.is_ascii_digit()).collect();

            let with_commas = insert_thousands_commas(&digits);

            let result = if negative {
                format!("-{}", with_commas)
            } else {
                with_commas
            };

            match frac_part {
                Some(frac) => Some(Value::String(format!("{}.{}", result, frac))),
                None => Some(Value::String(result)),
            }
        },
        "percent" => |args| {
            let n = args.first()?.as_f64()?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;
            Some(Value::String(format!("{:.*}%", decimals, n * 100.0)))
        },
        "currency" => |args| {
            let n = args.first()?.as_f64()?;
            let symbol = match args.get(1) {
                Some(Value::String(s)) => s.clone(),
                _ => "$".to_string(),
            };
            let decimals = args.get(2).and_then(|v| v.as_f64()).unwrap_or(2.0) as usize;

            // Use comma formatter
            let formatted = if decimals > 0 {
                format!("{:.*}", decimals, n.abs())
            } else {
                format!("{}", n.abs().trunc() as i64)
            };

            let parts: Vec<&str> = formatted.split('.').collect();
            let int_part = parts[0];
            let frac_part = parts.get(1);

            let digits: String = int_part.chars().filter(|c| c.is_ascii_digit()).collect();
            let with_commas = insert_thousands_commas(&digits);

            let num_str = match frac_part {
                Some(frac) => format!("{}.{}", with_commas, frac),
                None => with_commas,
            };

            if n < 0.0 {
                Some(Value::String(format!("-{}{}", symbol, num_str)))
            } else {
                Some(Value::String(format!("{}{}", symbol, num_str)))
            }
        },
        "ordinal" => |args| {
            let n = args.first()?.as_f64()? as i64;
            let suffix = match (n % 10, n % 100) {
                (1, 11) => "th",
                (2, 12) => "th",
                (3, 13) => "th",
                (1, _) => "st",
                (2, _) => "nd",
                (3, _) => "rd",
                _ => "th",
            };
            Some(Value::String(format!("{}{}", n, suffix)))
        },
        "bytes" => |args| {
            // Format bytes to human-readable
            let n = args.first()?.as_f64()?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(2.0) as usize;
            let units = ["B", "KB", "MB", "GB", "TB", "PB"];

            let mut value = n.abs();
            let mut unit_idx = 0;
            while value >= 1024.0 && unit_idx < units.len() - 1 {
                value /= 1024.0;
                unit_idx += 1;
            }

            let sign = if n < 0.0 { "-" } else { "" };
            Some(Value::String(format!(
                "{}{:.*} {}",
                sign, decimals, value, units[unit_idx]
            )))
        },

        // Conditional helpers
        "if" => |args| {
            let cond = args.first()?.is_truthy();
            let yes = args.get(1)?.clone();
            let no = args.get(2).cloned().unwrap_or(Value::Null);
            Some(if cond { yes } else { no })
        },
        "ifnull" => |args| {
            let val = args.first()?;
            if val.is_null() {
                args.get(1).cloned()
            } else {
                Some(val.clone())
            }
        },
        "default" => |args| {
            let val = args.first()?;
            if val.is_null() || matches!(val, Value::String(s) if s.is_empty()) {
                args.get(1).cloned()
            } else {
                Some(val.clone())
            }
        },
        "coalesce" => |args| {
            for arg in &args {
                if !arg.is_null() {
                    return Some(arg.clone());
                }
            }
            Some(Value::Null)
        },
        "choose" => |args| {
            let idx = args.first()?.as_f64()? as usize;
            args.get(idx + 1).cloned()
        },
        "switch" => |args| {
            // switch(value, case1, result1, case2, result2, ..., default)
            let val = args.first()?;
            let pairs = &args[1..];
            let mut i = 0;
            while i + 1 < pairs.len() {
                if val == &pairs[i] {
                    return Some(pairs[i + 1].clone());
                }
                i += 2;
            }
            // Return default if odd number of remaining args
            if pairs.len() % 2 == 1 {
                Some(pairs.last()?.clone())
            } else {
                Some(Value::Null)
            }
        },
        "map" => |args| {
            // map(value, inMin, inMax, outMin, outMax)
            let val = args.first()?.as_f64()?;
            let in_min = args.get(1)?.as_f64()?;
            let in_max = args.get(2)?.as_f64()?;
            let out_min = args.get(3)?.as_f64()?;
            let out_max = args.get(4)?.as_f64()?;

            let t = (val - in_min) / (in_max - in_min);
            Some(Value::F64(out_min + t * (out_max - out_min)))
        },

        // Time/Date helpers (basic, no actual time - uses numeric input)
        "duration" => |args| {
            // Format seconds as duration
            let mut secs = args.first()?.as_f64()?;
            let negative = secs < 0.0;
            secs = secs.abs();

            let hours = (secs / 3600.0).floor() as i64;
            let mins = ((secs % 3600.0) / 60.0).floor() as i64;
            let s = (secs % 60.0).floor() as i64;

            let result = if hours > 0 {
                format!("{}:{:02}:{:02}", hours, mins, s)
            } else {
                format!("{}:{:02}", mins, s)
            };

            Some(Value::String(if negative {
                format!("-{}", result)
            } else {
                result
            }))
        },
        "elapsed" => |args| {
            let secs = args.first()?.as_f64()?.abs();

            let result = if secs < 60.0 {
                let n = secs.round();
                format!("{:.0} second{}", n, if n == 1.0 { "" } else { "s" })
            } else if secs < 3600.0 {
                let n = (secs / 60.0).round();
                format!("{:.0} minute{}", n, if n == 1.0 { "" } else { "s" })
            } else if secs < 86400.0 {
                format!("{:.1} hours", secs / 3600.0)
            } else if secs < 604800.0 {
                format!("{:.1} days", secs / 86400.0)
            } else if secs < 2592000.0 {
                format!("{:.1} weeks", secs / 604800.0)
            } else if secs < 31536000.0 {
                format!("{:.1} months", secs / 2592000.0)
            } else {
                format!("{:.1} years", secs / 31536000.0)
            };

            Some(Value::String(result))
        },

        // Debug/utility
        "debug" => |args| Some(Value::String(format!("{:?}", args.first()?))),
        "typeof" => |args| Some(Value::String(args.first()?.type_name().to_string())),
        "defined" => |args| {
            Some(Value::Bool(
                !args.first().map(|v| v.is_null()).unwrap_or(true),
            ))
        },
        "empty" => |args| match args.first() {
            Some(Value::String(s)) => Some(Value::Bool(s.is_empty())),
            Some(Value::Array(arr)) => Some(Value::Bool(arr.is_empty())),
            Some(Value::Null) => Some(Value::Bool(true)),
            _ => Some(Value::Bool(false)),
        },
        "hsv_to_rgb" => |args| match args.first() {
            Some(Value::Array(hsv)) => {
                let hsv = HSV {
                    h: hsv[0].as_f64()? as f32,
                    s: hsv[1].as_f64()? as f32,
                    v: hsv[2].as_f64()? as f32,
                };
                Some(Value::from_vec(hsv_to_rgb(hsv)))
            }
            Some(Value::F64(hue)) => {
                let hsv = HSV {
                    h: *hue as f32,
                    s: args[1].as_f64()? as f32,
                    v: args[2].as_f64()? as f32,
                };
                Some(Value::from_vec(hsv_to_rgb(hsv)))
            }
            _ => Some(Value::Null),
        },
        "random" => |args| {
            //println!("{:?}", args);
            let mut rng = ThreadRng::default();
            match args.first() {
                Some(Value::Array(array)) => {
                    // Array as [min, max] range, or pick random element
                    if array.len() == 2 {
                        match (&array[0], &array[1]) {
                            (Value::F64(min), Value::F64(max)) => {
                                Some(Value::F64(rng.random_range(*min..=*max)))
                            }
                            (Value::I64(min), Value::I64(max)) => {
                                Some(Value::I64(rng.random_range(*min..=*max)))
                            }
                            _ => None,
                        }
                    } else if !array.is_empty() {
                        // Pick random element from array
                        let idx = rng.random_range(0..array.len());
                        Some(array[idx].clone())
                    } else {
                        None
                    }
                }
                Some(Value::F64(first)) => {
                    // Check for second argument to form range
                    match args.get(1) {
                        Some(Value::F64(second)) => {
                            Some(Value::F64(rng.random_range(*first..=*second)))
                        }
                        _ => {
                            // Single value: random from 0.0 to first
                            Some(Value::F64(rng.random_range(0.0..=*first)))
                        }
                    }
                }
                Some(Value::I64(first)) => {
                    // Check for second argument to form range
                    match args.get(1) {
                        Some(Value::I64(second)) => {
                            Some(Value::I64(rng.random_range(*first..=*second)))
                        }
                        _ => {
                            // Single value: random from 0 to first
                            Some(Value::I64(rng.random_range(0..=*first)))
                        }
                    }
                }
                Some(Value::String(string)) => {
                    // Hash the string for deterministic-ish random value
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    string.hash(&mut hasher);
                    Some(Value::F64((hasher.finish() % 1000000) as f64 / 1000000.0))
                }
                Some(Value::Bool(_)) => {
                    // Return random boolean
                    Some(Value::Bool(rng.random()))
                }
                _ => {
                    // Default: random f64 between 0.0 and 1.0
                    Some(Value::F64(rng.random()))
                }
            }
        },

        _ => return None,
    })
}

#[derive(Debug, Clone)]
pub enum ParseError {
    UnexpectedToken {
        expected: String,
        found: Token,
        pos: usize,
    },
    UnexpectedEnd {
        expected: String,
    },
    TypeMismatch {
        operation: String,
        expected: String,
        found: String,
        pos: usize,
    },
    UndefinedVariable {
        name: String,
        pos: usize,
    },
    UndefinedFunction {
        name: String,
        pos: usize,
    },
    InvalidPropertyAccess {
        property: String,
        on_type: String,
        pos: usize,
    },
    InvalidIndexAccess {
        index_type: String,
        on_type: String,
        pos: usize,
    },
}

#[derive(Debug, Clone)]
pub enum AnnoyingError {
    DivisionByZero { pos: usize },
    NullValue { pos: usize },
    OverflowWarning { pos: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::UnexpectedToken {
                expected,
                found,
                pos,
            } => {
                write!(
                    f,
                    "Parse error at position {}: expected {}, but found {:?}",
                    pos, expected, found
                )
            }
            Self::UnexpectedEnd { expected } => {
                write!(
                    f,
                    "Parse error: unexpected end of input, expected {}",
                    expected
                )
            }
            Self::TypeMismatch {
                operation,
                expected,
                found,
                pos,
            } => {
                write!(
                    f,
                    "Type error at position {}: {} requires {}, but got {}",
                    pos, operation, expected, found
                )
            }
            Self::UndefinedVariable { name, pos } => {
                write!(
                    f,
                    "Reference error at position {}: undefined variable '{}'",
                    pos, name
                )
            }
            Self::UndefinedFunction { name, pos } => {
                write!(
                    f,
                    "Reference error at position {}: undefined function '{}'",
                    pos, name
                )
            }
            Self::InvalidPropertyAccess {
                property,
                on_type,
                pos,
            } => {
                write!(
                    f,
                    "Property error at position {}: type '{}' has no property '{}'",
                    pos, on_type, property
                )
            }
            Self::InvalidIndexAccess {
                index_type,
                on_type,
                pos,
            } => {
                write!(
                    f,
                    "Index error at position {}: cannot index '{}' with '{}'",
                    pos, on_type, index_type
                )
            }
        }
    }
}

impl fmt::Display for AnnoyingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::DivisionByZero { pos } => {
                write!(
                    f,
                    "Warning at position {}: division by zero (result is Infinity)",
                    pos
                )
            }
            Self::NullValue { pos } => {
                write!(
                    f,
                    "Warning at position {}: null value used in operation",
                    pos
                )
            }
            Self::OverflowWarning { pos } => {
                write!(
                    f,
                    "Warning at position {}: potential overflow in operation",
                    pos
                )
            }
        }
    }
}

type ParseResult<T> = Result<T, ParseError>;

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    StrLit(String),
    True,
    False,
    Null,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Power,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    StrictEq,
    StrictNeq,
    And,
    Or,
    Not,
    NullCoalesce,
    OptChain,
    Question,
    Colon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    DotDot,
    DotDotEq,
    Dollar,
    Pipe,
    End,
    RBrace,
    LBrace,
}

fn tokenize_expr(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '"' || c == '\'' || c == '`' {
            let quote = c;
            chars.next();
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                chars.next();
                if d == quote {
                    break;
                } else if d == '\\' {
                    if let Some(&e) = chars.peek() {
                        chars.next();
                        match e {
                            'n' => s.push('\n'),
                            't' => s.push('\t'),
                            'r' => s.push('\r'),
                            '\\' => s.push('\\'),
                            '"' => s.push('"'),
                            '\'' => s.push('\''),
                            '0' => s.push('\0'),
                            _ => {
                                s.push('\\');
                                s.push(e);
                            }
                        }
                    }
                } else {
                    s.push(d);
                }
            }
            tokens.push(Token::StrLit(s));
        } else {
            let prev_is_operand = matches!(
                tokens.last(),
                Some(Token::Number(_))
                    | Some(Token::Ident(_))
                    | Some(Token::RParen)
                    | Some(Token::RBracket)
                    | Some(Token::StrLit(_))
            );

            if c.is_ascii_digit()
                || (c == '.'
                    && !prev_is_operand
                    && !matches!(tokens.last(), Some(Token::Colon))
                    && chars.clone().nth(1).map_or(false, |n| n.is_ascii_digit()))
            {
                let mut s = String::new();
                let mut has_dot = false;
                let mut has_exp = false;

                if c == '0' {
                    s.push(c);
                    chars.next();
                    if let Some(&next) = chars.peek() {
                        match next {
                            'x' | 'X' => {
                                s.push(next);
                                chars.next();
                                while let Some(&d) = chars.peek() {
                                    if d.is_ascii_hexdigit() || d == '_' {
                                        if d != '_' {
                                            s.push(d);
                                        }
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                                if let Ok(v) = i64::from_str_radix(&s[2..], 16) {
                                    tokens.push(Token::Number(v as f64));
                                }
                                continue;
                            }
                            'b' | 'B' => {
                                s.push(next);
                                chars.next();
                                while let Some(&d) = chars.peek() {
                                    if d == '0' || d == '1' || d == '_' {
                                        if d != '_' {
                                            s.push(d);
                                        }
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                                if let Ok(v) = i64::from_str_radix(&s[2..], 2) {
                                    tokens.push(Token::Number(v as f64));
                                }
                                continue;
                            }
                            'o' | 'O' => {
                                s.push(next);
                                chars.next();
                                while let Some(&d) = chars.peek() {
                                    if ('0'..='7').contains(&d) || d == '_' {
                                        if d != '_' {
                                            s.push(d);
                                        }
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                                if let Ok(v) = i64::from_str_radix(&s[2..], 8) {
                                    tokens.push(Token::Number(v as f64));
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }
                } else {
                    s.push(c);
                    chars.next();
                }

                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '_' {
                        if d != '_' {
                            s.push(d);
                        }
                        chars.next();
                    } else if d == '.' && !has_dot && !has_exp {
                        let mut peek_chars = chars.clone();
                        peek_chars.next();
                        if peek_chars.peek() == Some(&'.') {
                            break;
                        }
                        has_dot = true;
                        s.push(d);
                        chars.next();
                    } else if (d == 'e' || d == 'E') && !has_exp {
                        has_exp = true;
                        s.push(d);
                        chars.next();
                        if let Some(&sign) = chars.peek() {
                            if sign == '+' || sign == '-' {
                                s.push(sign);
                                chars.next();
                            }
                        }
                    } else {
                        break;
                    }
                }
                if let Ok(v) = s.parse() {
                    tokens.push(Token::Number(v));
                }
            } else if c.is_alphabetic() || c == '_' {
                let mut s = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_alphanumeric() || d == '_' {
                        s.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }

                match s.as_str() {
                    "true" => tokens.push(Token::True),
                    "false" => tokens.push(Token::False),
                    "null" | "nil" | "none" => tokens.push(Token::Null),
                    _ => tokens.push(Token::Ident(s)),
                }
            } else {
                chars.next();
                match c {
                    '+' => tokens.push(Token::Plus),
                    '-' => tokens.push(Token::Minus),
                    '*' => {
                        if chars.peek() == Some(&'*') {
                            chars.next();
                            tokens.push(Token::Power);
                        } else {
                            tokens.push(Token::Star);
                        }
                    }
                    '/' => tokens.push(Token::Slash),
                    '%' => tokens.push(Token::Percent),
                    '(' => tokens.push(Token::LParen),
                    ')' => tokens.push(Token::RParen),
                    '[' => tokens.push(Token::LBracket),
                    ']' => tokens.push(Token::RBracket),
                    '{' => tokens.push(Token::LBrace),
                    '}' => tokens.push(Token::RBrace),
                    ',' => tokens.push(Token::Comma),
                    '$' => tokens.push(Token::Dollar),
                    '~' => tokens.push(Token::BitNot),
                    '?' => {
                        if chars.peek() == Some(&'?') {
                            chars.next();
                            tokens.push(Token::NullCoalesce);
                        } else if chars.peek() == Some(&'.') {
                            chars.next();
                            tokens.push(Token::OptChain);
                        } else {
                            tokens.push(Token::Question);
                        }
                    }
                    ':' => {
                        tokens.push(Token::Colon);

                        if chars.peek() == Some(&'.') {
                            chars.next();
                            tokens.push(Token::Dot);
                        }
                    }
                    '.' => {
                        if chars.peek() == Some(&'.') {
                            chars.next();
                            if chars.peek() == Some(&'=') {
                                chars.next();
                                tokens.push(Token::DotDotEq);
                            } else {
                                tokens.push(Token::DotDot);
                            }
                        } else {
                            tokens.push(Token::Dot);
                        }
                    }
                    '!' => {
                        if chars.peek() == Some(&'=') {
                            chars.next();
                            if chars.peek() == Some(&'=') {
                                chars.next();
                                tokens.push(Token::StrictNeq);
                            } else {
                                tokens.push(Token::Neq);
                            }
                        } else {
                            tokens.push(Token::Not);
                        }
                    }
                    '=' => {
                        if chars.peek() == Some(&'=') {
                            chars.next();
                            if chars.peek() == Some(&'=') {
                                chars.next();
                                tokens.push(Token::StrictEq);
                            } else {
                                tokens.push(Token::Eq);
                            }
                        }
                    }
                    '<' => {
                        if chars.peek() == Some(&'=') {
                            chars.next();
                            tokens.push(Token::Le);
                        } else if chars.peek() == Some(&'<') {
                            chars.next();
                            tokens.push(Token::Shl);
                        } else {
                            tokens.push(Token::Lt);
                        }
                    }
                    '>' => {
                        if chars.peek() == Some(&'=') {
                            chars.next();
                            tokens.push(Token::Ge);
                        } else if chars.peek() == Some(&'>') {
                            chars.next();
                            tokens.push(Token::Shr);
                        } else {
                            tokens.push(Token::Gt);
                        }
                    }
                    '&' => {
                        if chars.peek() == Some(&'&') {
                            chars.next();
                            tokens.push(Token::And);
                        } else {
                            tokens.push(Token::BitAnd);
                        }
                    }
                    '|' => {
                        if chars.peek() == Some(&'|') {
                            chars.next();
                            tokens.push(Token::Or);
                        } else if chars.peek() == Some(&'>') {
                            chars.next();
                            tokens.push(Token::Pipe);
                        } else {
                            tokens.push(Token::BitOr);
                        }
                    }
                    '^' => tokens.push(Token::BitXor),
                    _ => {}
                }
            }
        }
    }

    tokens.push(Token::End);
    tokens
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    vars: &'a Variables,
    settings: &'a Settings,
    is_inside_braces: bool,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], vars: &'a Variables, settings: &'a Settings) -> Self {
        Self {
            tokens,
            pos: 0,
            vars,
            settings,
            is_inside_braces: false,
        }
    }

    fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token::End)
    }

    fn peek_n(&self, n: usize) -> Token {
        self.tokens.get(self.pos + n).cloned().unwrap_or(Token::End)
    }

    pub fn parse(&mut self) -> ParseResult<Value> {
        let value = self.parse_expr_bp(0, false)?;
        if self.peek() != Token::End {
            return Err(ParseError::UnexpectedToken {
                expected: "end of expression".to_string(),
                found: self.peek(),
                pos: self.pos,
            });
        }
        Ok(value)
    }

    fn parse_expr(&mut self, min_bp: u8) -> ParseResult<Value> {
        self.parse_expr_bp(min_bp, false)
    }

    fn parse_expr_bp(&mut self, min_bp: u8, stop_at_ternary_colon: bool) -> ParseResult<Value> {
        let mut left = self.parse_primary()?;

        loop {
            let op = self.peek();

            if stop_at_ternary_colon && op == Token::Colon && self.peek_n(1) != Token::Dot {
                break;
            }

            if let Value::String(ref l) = left {
                if matches!(op, Token::Ident(_) | Token::StrLit(_) | Token::LBrace) {
                    let right = self.parse_primary()?;
                    left = Value::String(format!("{}{}", l, value_to_text(&right)));
                    continue;
                }
            }

            let (l_bp, r_bp) = match op {
                Token::Pipe => (1, 2),
                Token::Question => (2, 0),
                Token::Colon => (16, 17),
                Token::NullCoalesce => (3, 4),
                Token::Or => (4, 5),
                Token::And => (5, 6),
                Token::BitOr => (6, 7),
                Token::BitXor => (7, 8),
                Token::BitAnd => (8, 9),
                Token::Eq | Token::Neq | Token::StrictEq | Token::StrictNeq => (9, 10),
                Token::Lt | Token::Gt | Token::Le | Token::Ge => (10, 11),
                Token::Shl | Token::Shr => (11, 12),
                Token::Plus | Token::Minus => (12, 13),
                Token::Star | Token::Slash | Token::Percent => (13, 14),
                Token::Power => (15, 14),
                Token::Dot | Token::LBracket => (16, 17),
                _ => break,
            };

            if l_bp < min_bp {
                break;
            }

            self.pos += 1;

            if op == Token::Question {
                let yes = self.parse_expr_bp(0, true)?;
                if self.peek() != Token::Colon {
                    return Err(ParseError::UnexpectedToken {
                        expected: ":".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }
                self.pos += 1;

                let no = self.parse_expr_bp(2, false)?;
                left = if left.is_truthy() { yes } else { no };
                continue;
            }

            if op == Token::Pipe {
                let name = match self.peek() {
                    Token::Ident(s) => {
                        self.pos += 1;
                        s
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "identifier".to_string(),
                            found: self.peek(),
                            pos: self.pos,
                        });
                    }
                };

                let mut args = vec![left.clone()];

                if self.peek() == Token::LParen {
                    self.pos += 1;
                    let mut more_args = Vec::new();

                    if self.peek() != Token::RParen {
                        loop {
                            more_args.push(self.parse_expr_bp(0, false)?);
                            match self.peek() {
                                Token::Comma => {
                                    self.pos += 1;
                                }
                                Token::RParen => break,
                                tok => {
                                    return Err(ParseError::UnexpectedToken {
                                        expected: "',' or ')'".to_string(),
                                        found: tok.clone(),
                                        pos: self.pos,
                                    });
                                }
                            }
                        }
                    }

                    self.pos += 1;
                    args.append(&mut more_args);
                }

                left = call_builtin(&name, args).ok_or(ParseError::UndefinedFunction {
                    name: name.clone(),
                    pos: self.pos - 1,
                })?;
                continue;
            }

            if op == Token::Colon {
                if self.peek() != Token::Dot {
                    return Err(ParseError::UnexpectedToken {
                        expected: ".".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }

                self.pos += 1;

                let precision = match self.peek() {
                    Token::Number(n) => {
                        self.pos += 1;
                        n as usize
                    }
                    Token::Ident(_) | Token::StrLit(_) | Token::LBrace => {
                        let precision_val = self.parse_expr_bp(17, false)?;
                        match precision_val.as_f64() {
                            Some(v) => v as usize,
                            None => {
                                return Err(ParseError::TypeMismatch {
                                    operation: "precision".to_string(),
                                    expected: "number".to_string(),
                                    found: precision_val.type_name().to_string(),
                                    pos: self.pos,
                                });
                            }
                        }
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "precision value".to_string(),
                            found: self.peek(),
                            pos: self.pos,
                        });
                    }
                };

                let Some(n) = left.as_f64() else {
                    return Err(ParseError::TypeMismatch {
                        operation: "format".to_string(),
                        expected: "number".to_string(),
                        found: left.type_name().to_string(),
                        pos: self.pos,
                    });
                };

                left = Value::String(format!("{:.*}", precision, n));
                continue;
            }

            if op == Token::Dot {
                let prop_tok = self.peek();
                self.pos += 1;

                match prop_tok {
                    Token::Ident(prop) => {
                        left = match get_property(&left, &prop) {
                            Some(v) => v,
                            None => Value::String(format!("{}.{}", value_to_text(&left), prop)),
                        };
                    }
                    Token::Number(n) => {
                        if n.fract() == 0.0 && n >= 0.0 {
                            let idx = Value::I64(n as i64);
                            left = match get_index(&left, &idx) {
                                Some(v) => v,
                                None => {
                                    let full_name =
                                        format!("{}.{}", value_to_text(&left), n as i64);
                                    match self.vars.get(full_name.as_str()) {
                                        None => Value::String(full_name),
                                        Some(val) => val.into_owned(),
                                    }
                                }
                            };
                        } else {
                            let full_name = format!("{}.{}", value_to_text(&left), n as i64);
                            left = match self.vars.get(full_name.as_str()) {
                                None => Value::String(full_name),
                                Some(val) => val.into_owned(),
                            };
                        }
                    }
                    Token::LBrace => {
                        let prev_inside = self.is_inside_braces;
                        self.is_inside_braces = true;
                        let key = self.parse_expr_bp(0, false)?;
                        if self.peek() != Token::RBrace {
                            return Err(ParseError::UnexpectedToken {
                                expected: "}".to_string(),
                                found: self.peek(),
                                pos: self.pos,
                            });
                        }
                        self.pos += 1;
                        self.is_inside_braces = prev_inside;

                        let resolved = match &key {
                            Value::I64(_) | Value::F64(_) => get_index(&left, &key),
                            _ => get_property(&left, &value_to_text(&key)),
                        };

                        left = resolved.unwrap_or_else(|| {
                            Value::String(format!(
                                "{}.{}",
                                value_to_text(&left),
                                value_to_text(&key)
                            ))
                        });
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "identifier or numeric index".to_string(),
                            found: prop_tok.clone(),
                            pos: self.pos - 1,
                        });
                    }
                }
            } else if op == Token::LBracket {
                let idx = self.parse_expr_bp(0, false)?;
                if self.peek() != Token::RBracket {
                    return Err(ParseError::UnexpectedToken {
                        expected: "]".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }
                self.pos += 1;
                left = get_index(&left, &idx).ok_or(ParseError::InvalidIndexAccess {
                    index_type: idx.type_name().to_string(),
                    on_type: left.type_name().to_string(),
                    pos: self.pos - 1,
                })?;
            } else {
                let right = self.parse_expr_bp(r_bp, false)?;
                left = eval_binary(left, right, &op, self.pos - 1)?;
            }
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> ParseResult<Value> {
        let tok = self.peek();
        self.pos += 1;
        let pos = self.pos - 1;

        match tok {
            Token::Number(n) => Ok(Value::F64(n)),
            Token::StrLit(s) => Ok(Value::String(s)),
            Token::True => Ok(Value::Bool(true)),
            Token::False => Ok(Value::Bool(false)),
            Token::Null => Ok(Value::Null),

            Token::Minus => Ok(Value::F64(
                -(self
                    .parse_expr_bp(15, false)?
                    .as_f64()
                    .ok_or(ParseError::TypeMismatch {
                        operation: "negation".to_string(),
                        expected: "number".to_string(),
                        found: "value".to_string(),
                        pos,
                    })?),
            )),

            Token::Not => Ok(Value::Bool(!self.parse_expr_bp(15, false)?.is_truthy())),

            Token::BitNot => Ok(Value::I64(
                !(self
                    .parse_expr_bp(15, false)?
                    .as_i64()
                    .ok_or(ParseError::TypeMismatch {
                        operation: "bitnot".to_string(),
                        expected: "integer".to_string(),
                        found: "value".to_string(),
                        pos,
                    })?),
            )),

            Token::LBrace => {
                let prev_inside = self.is_inside_braces;
                self.is_inside_braces = true;
                let v = self.parse_expr_bp(0, false)?;
                if self.peek() != Token::RBrace {
                    return Err(ParseError::UnexpectedToken {
                        expected: "}".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }
                self.pos += 1;
                self.is_inside_braces = prev_inside;
                Ok(v)
            }

            Token::LParen => {
                let v = self.parse_expr_bp(0, false)?;
                if self.peek() != Token::RParen {
                    return Err(ParseError::UnexpectedToken {
                        expected: ")".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }
                self.pos += 1;
                Ok(v)
            }

            Token::LBracket => {
                let mut items = Vec::new();

                if self.peek() != Token::RBracket {
                    loop {
                        let item = self.parse_expr_bp(0, false)?;

                        if self.peek() == Token::DotDot || self.peek() == Token::DotDotEq {
                            let inclusive = self.peek() == Token::DotDotEq;
                            self.pos += 1;

                            let end_val = self.parse_expr_bp(0, false)?;
                            let start = item.as_i64().ok_or(ParseError::TypeMismatch {
                                operation: "range start".to_string(),
                                expected: "integer".to_string(),
                                found: item.type_name().to_string(),
                                pos,
                            })?;
                            let end = end_val.as_i64().ok_or(ParseError::TypeMismatch {
                                operation: "range end".to_string(),
                                expected: "integer".to_string(),
                                found: end_val.type_name().to_string(),
                                pos,
                            })?;

                            let range: Box<dyn Iterator<Item = i64>> = if inclusive {
                                Box::new(start..=end)
                            } else {
                                Box::new(start..end)
                            };

                            items.extend(range.map(|i| Value::F64(i as f64)));
                        } else {
                            items.push(item);
                        }

                        match self.peek() {
                            Token::Comma => {
                                self.pos += 1;
                            }
                            Token::RBracket => break,
                            tok => {
                                return Err(ParseError::UnexpectedToken {
                                    expected: "',' or ']'".to_string(),
                                    found: tok.clone(),
                                    pos: self.pos,
                                });
                            }
                        }
                    }
                }

                self.pos += 1;
                Ok(Value::Array(items))
            }

            Token::Ident(name) => {
                if self.peek() == Token::LParen {
                    self.pos += 1;
                    let mut args = Vec::new();

                    if self.peek() != Token::RParen {
                        loop {
                            args.push(self.parse_expr_bp(0, false)?);
                            match self.peek() {
                                Token::Comma => {
                                    self.pos += 1;
                                }
                                Token::RParen => break,
                                tok => {
                                    return Err(ParseError::UnexpectedToken {
                                        expected: "',' or ')'".to_string(),
                                        found: tok.clone(),
                                        pos: self.pos,
                                    });
                                }
                            }
                        }
                    }

                    self.pos += 1;
                    call_builtin(&name, args).ok_or(ParseError::UndefinedFunction {
                        name: name.clone(),
                        pos,
                    })
                } else if self.is_inside_braces {
                    //println!("{}", name);
                    match name.as_str() {
                        "PI" | "pi" => Ok(Value::F64(std::f64::consts::PI)),
                        "TAU" | "tau" => Ok(Value::F64(std::f64::consts::TAU)),
                        "E" | "e" => Ok(Value::F64(std::f64::consts::E)),
                        "INF" | "inf" => Ok(Value::F64(f64::INFINITY)),
                        "NAN" | "NaN" => Ok(Value::F64(f64::NAN)),
                        _ => {
                            if let Some(key) = SettingKey::from_str(name.as_str()) {
                                Ok(self.settings.read_setting(key).to_value())
                            } else {
                                let val = self.vars.get(&name).as_deref().cloned();
                                let (field_type, _) = match name.split_once('=') {
                                    Some((field_type, base)) => (field_type, base),
                                    None => ("None", name.as_str()),
                                };
                                Ok(initialize_value(field_type, val))
                            }
                        }
                    }
                } else {
                    Ok(Value::String(name))
                }
            }

            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: tok,
                pos,
            }),
        }
    }
}

fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    }
}

fn eval_binary(left: Value, right: Value, op: &Token, pos: usize) -> ParseResult<Value> {
    match op {
        Token::Plus => add_values(left.clone(), right.clone()).ok_or(ParseError::TypeMismatch {
            operation: "addition".to_string(),
            expected: "number or string".to_string(),
            found: format!("{} + {}", left.type_name(), right.type_name()),
            pos,
        }),
        Token::Minus => {
            let a = left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "subtraction".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let b = right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "subtraction".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            Ok(Value::F64(a - b))
        }
        Token::Star => multiply_values(left, right).ok_or(ParseError::TypeMismatch {
            operation: "multiplication".to_string(),
            expected: "number".to_string(),
            found: "incompatible types".to_string(),
            pos,
        }),
        Token::Slash | Token::Percent => {
            let a = left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "div/mod".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let b = right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "div/mod".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            Ok(Value::F64(if matches!(op, Token::Slash) {
                a / b
            } else {
                a % b
            }))
        }
        Token::Power => {
            let a = left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "power".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let b = right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "power".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            Ok(Value::F64(a.powf(b)))
        }
        Token::Eq | Token::StrictEq => Ok(Value::Bool(left == right)),
        Token::Neq | Token::StrictNeq => Ok(Value::Bool(left != right)),
        Token::Lt => Ok(Value::Bool(
            left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "lt".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? < right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "lt".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::Gt => Ok(Value::Bool(
            left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "gt".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? > right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "gt".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::Le => Ok(Value::Bool(
            left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "le".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? <= right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "le".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::Ge => Ok(Value::Bool(
            left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "ge".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? >= right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "ge".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::Shl => Ok(Value::I64(
            left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "shl".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? << right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "shl".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })? as u32,
        )),
        Token::Shr => Ok(Value::I64(
            left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "shr".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? >> right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "shr".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })? as u32,
        )),
        Token::BitAnd => Ok(Value::I64(
            left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitand".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? & right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitand".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::BitOr => Ok(Value::I64(
            left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitor".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? | right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitor".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::BitXor => Ok(Value::I64(
            left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitxor".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })? ^ right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitxor".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?,
        )),
        Token::And => Ok(Value::Bool(left.is_truthy() && right.is_truthy())),
        Token::Or => Ok(Value::Bool(left.is_truthy() || right.is_truthy())),
        Token::NullCoalesce => Ok(if left.is_truthy() { left } else { right }),
        _ => unreachable!(),
    }
}

fn call_builtin(name: &str, args: Vec<Value>) -> Option<Value> {
    get_builtin(name)?(args)
}

fn add_values(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => Some(Value::F64(x + y)),
        (Value::I64(x), Value::F64(y)) => Some(Value::F64(x as f64 + y)),
        (Value::F64(x), Value::I64(y)) => Some(Value::F64(x + y as f64)),
        (Value::String(s), Value::F64(n)) => Some(Value::String(format!("{}{}", s, n))),
        (Value::F64(n), Value::String(s)) => Some(Value::String(format!("{}{}", n, s))),
        (Value::I64(x), Value::I64(y)) => Some(Value::I64(x + y)),
        (Value::String(s), Value::I64(n)) => Some(Value::String(format!("{}{}", s, n))),
        (Value::I64(n), Value::String(s)) => Some(Value::String(format!("{}{}", n, s))),
        (Value::String(x), Value::String(y)) => Some(Value::String(x + &y)),
        (Value::Bool(x), Value::String(y)) => Some(Value::String(format!("{}{}", x, y))),
        (Value::String(x), Value::Bool(y)) => Some(Value::String(format!("{}{}", x, y))),
        (Value::Array(mut x), Value::Array(y)) => {
            x.extend(y);
            Some(Value::Array(x))
        }
        (Value::Array(mut x), v) => {
            x.push(v);
            Some(Value::Array(x))
        }
        (v, Value::Array(mut y)) => {
            y.insert(0, v);
            Some(Value::Array(y))
        }
        (Value::String(s), Value::Null) => Some(Value::String(s)),
        (Value::Null, Value::String(s)) => Some(Value::String(s)),
        _ => None,
    }
}

fn multiply_values(l: Value, r: Value) -> Option<Value> {
    match (&l, &r) {
        (Value::String(s), Value::I64(n)) | (Value::I64(n), Value::String(s)) => {
            Some(Value::String(s.repeat(*n as usize)))
        }
        _ => Some(Value::F64(l.as_f64()? * r.as_f64()?)),
    }
}

fn get_property(value: &Value, prop: &str) -> Option<Value> {
    match value {
        Value::String(s) => string_property(s, prop),
        Value::Array(arr) => array_property(arr, prop),
        Value::F64(n) => float_property(*n, prop),
        Value::I64(n) => int_property(*n, prop),
        _ => None,
    }
}

fn string_property(s: &str, prop: &str) -> Option<Value> {
    Some(match prop {
        "length" | "len" => Value::I64(s.len() as i64),
        "upper" => Value::String(s.to_uppercase()),
        "lower" => Value::String(s.to_lowercase()),
        "trim" => Value::String(s.trim().to_string()),
        "reverse" => Value::String(s.chars().rev().collect()),
        "first" => Value::String(s.chars().next()?.to_string()),
        "last" => Value::String(s.chars().last()?.to_string()),
        "empty" => Value::Bool(s.is_empty()),
        "chars" => Value::Array(s.chars().map(|c| Value::String(c.to_string())).collect()),
        "lines" => Value::Array(s.lines().map(|l| Value::String(l.to_string())).collect()),
        "words" => Value::Array(
            s.split_whitespace()
                .map(|w| Value::String(w.to_string()))
                .collect(),
        ),
        _ => return None,
    })
}

fn array_property(arr: &[Value], prop: &str) -> Option<Value> {
    Some(match prop {
        "length" | "len" | "count" => Value::I64(arr.len() as i64),
        "first" => arr.first()?.clone(),
        "last" => arr.last()?.clone(),
        "empty" => Value::Bool(arr.is_empty()),
        "sum" => Value::F64(arr.iter().filter_map(|v| v.as_f64()).sum()),
        "avg" => {
            let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
            Value::F64(nums.iter().sum::<f64>() / nums.len().max(1) as f64)
        }
        "min" => Value::F64(
            arr.iter()
                .filter_map(|v| v.as_f64())
                .fold(f64::INFINITY, f64::min),
        ),
        "max" => Value::F64(
            arr.iter()
                .filter_map(|v| v.as_f64())
                .fold(f64::NEG_INFINITY, f64::max),
        ),
        "reverse" => Value::Array(arr.iter().rev().cloned().collect()),
        _ => return None,
    })
}

fn float_property(n: f64, prop: &str) -> Option<Value> {
    Some(match prop {
        "abs" => Value::F64(n.abs()),
        "floor" => Value::F64(n.floor()),
        "ceil" => Value::F64(n.ceil()),
        "round" => Value::F64(n.round()),
        "trunc" => Value::F64(n.trunc()),
        "sqrt" => Value::F64(n.sqrt()),
        "sign" => Value::F64(n.signum()),
        "fract" => Value::F64(n.fract()),
        "int" => Value::I64(n as i64),
        "neg" => Value::F64(-n),
        "isnan" => Value::Bool(n.is_nan()),
        "isinf" => Value::Bool(n.is_infinite()),
        "isfinite" => Value::Bool(n.is_finite()),
        "hex" => Value::String(format!("{:x}", n as i64)),
        "bin" => Value::String(format!("{:b}", n as i64)),
        "oct" => Value::String(format!("{:o}", n as i64)),
        _ => return None,
    })
}

fn int_property(n: i64, prop: &str) -> Option<Value> {
    Some(match prop {
        "abs" => Value::I64(n.abs()),
        "neg" => Value::I64(-n),
        "sign" => Value::I64(n.signum()),
        "tofloat" => Value::F64(n as f64),
        "isnan" | "isinf" => Value::Bool(false),
        "isfinite" => Value::Bool(true),
        "hex" => Value::String(format!("{:x}", n)),
        "bin" => Value::String(format!("{:b}", n)),
        "oct" => Value::String(format!("{:o}", n)),
        _ => return None,
    })
}

fn get_index(value: &Value, index: &Value) -> Option<Value> {
    let idx = index.as_i64()?;
    match value {
        // Value::String(s) => {
        //     let chars: Vec<char> = s.chars().collect();
        //     let i = normalize_index(idx, chars.len())?;
        //     Some(Value::String(chars[i].to_string()))
        // }
        Value::Array(arr) => {
            let i = normalize_index(idx, arr.len())?;
            // println!("Get index thingie: {} {} {}", value, index, idx);
            Some(arr[i].clone())
        }
        _ => None,
    }
}

fn normalize_index(idx: i64, len: usize) -> Option<usize> {
    let i = if idx < 0 { idx + len as i64 } else { idx };
    if i >= 0 && (i as usize) < len {
        Some(i as usize)
    } else {
        None
    }
}

// pub fn evaluate_placeholder(src: &str, vars: &Variables) -> String {
//     format_slot(src.trim(), vars)
// }

fn format_slot(src: &str, vars: &Variables, settings: &Settings) -> String {
    let (expr, modifiers) = split_format_chain(src);
    let mut value = match eval_expr(expr, vars, settings) {
        Some(v) => v,
        None => return src.to_string(),
    };
    for modifier in modifiers {
        value = match apply_modifier(value, modifier) {
            Some(v) => v,
            None => return src.to_string(),
        };
    }
    value.into_string_value()
}

fn split_format_chain(src: &str) -> (&str, Vec<&str>) {
    let mut paren_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;
    let mut brace_depth: i32 = 0;
    let mut ternary_depth = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    for (idx, ch) in src.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_single {
            match ch {
                '\\' => escape = true,
                '\'' => in_single = false,
                _ => {}
            }
            continue;
        }
        if in_double {
            match ch {
                '\\' => escape = true,
                '"' => in_double = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '\'' => in_single = true,
            '"' => in_double = true,
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '?' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => ternary_depth += 1,
            ':' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                if ternary_depth > 0 {
                    ternary_depth -= 1;
                } else {
                    let expr = src[..idx].trim();
                    let mods = src[idx + 1..]
                        .split(':')
                        .map(|m| m.trim())
                        .filter(|m| !m.is_empty())
                        .collect();
                    return (expr, mods);
                }
            }
            _ => {}
        }
    }
    (src.trim(), Vec::new())
}

fn apply_modifier(value: Value, modifier: &str) -> Option<Value> {
    let modifier = modifier.trim();
    if modifier.is_empty() {
        return Some(value);
    }
    if let Some(precision_str) = modifier.strip_prefix('.') {
        let precision = precision_str.trim().parse::<usize>().ok()?;
        let n = value.as_f64()?;
        return Some(Value::String(format!("{:.*}", precision, n)));
    }
    if let Some(func) = get_builtin(modifier) {
        return func(vec![value]);
    }
    None
}

pub fn eval_expr(expr: &str, vars: &Variables, settings: &Settings) -> Option<Value> {
    let tokens = tokenize_expr(expr);
    //println!("{:?}", expr);
    let result = Parser::new(&tokens, vars, settings).parse();
    match result {
        Ok(result) => Some(result),
        Err(e) => {
            //println!("ParseError for '{}': {}", expr, e);
            None
        }
    }
}

pub fn resolve_template(template: &str, vars: &Variables, settings: &Settings) -> String {
    let mut out = String::new();
    let mut chars = template.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if c == '{' {
            let start = i + 1;
            let mut end_opt = None;
            let mut brace_depth = 1;
            while let Some(&(j, cj)) = chars.peek() {
                chars.next();
                if cj == '{' {
                    brace_depth += 1;
                } else if cj == '}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        end_opt = Some(j);
                        break;
                    }
                }
            }
            if let Some(end) = end_opt {
                let inside = &template[start..end];
                let val = if let Some(key) = SettingKey::from_str(inside) {
                    settings.read_setting(key).to_string()
                } else {
                    Value::from_str(settings, vars, inside.trim(), true, true).into_string_value()
                };
                out.push_str(&val);
            } else {
                out.push('{');
            }
        } else if c == '}' {
            out.push(c);
        } else {
            out.push(c);
        }
    }
    out
}

fn trim_float(n: f64) -> String {
    if !n.is_finite() {
        return n.to_string();
    }

    let mut s = if n.fract() == 0.0 {
        format!("{:.0}", n)
    } else {
        format!("{n}")
    };

    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }

    if s == "-0" { "0".to_string() } else { s }
}

pub fn set_input_box(template: &str, current_text: &str, _vars: &mut Variables) -> String {
    let start = template.find('{');
    let end = template.find('}');

    if start.is_none() || end.is_none() || end.unwrap() <= start.unwrap() {
        return current_text.to_string();
    }

    let start = start.unwrap();
    let end = end.unwrap();
    let _var_name = &template[start + 1..end];

    let prefix = &template[..start];
    let suffix = &template[end + 1..];

    if !current_text.starts_with(prefix) {
        return current_text.to_string();
    }

    let after_prefix = &current_text[prefix.len()..];

    let _var_value = if suffix.is_empty() {
        after_prefix
    } else if let Some(pos) = after_prefix.find(suffix) {
        &after_prefix[..pos]
    } else {
        after_prefix
    };

    // vars.set(var_name, var_value.trim());

    // Visible text keeps what the user typed, nothing blanked
    current_text.to_string()
}
fn insert_thousands_commas(digits: &str) -> String {
    digits
        .chars()
        .rev()
        .enumerate()
        .fold(String::new(), |mut acc, (i, c)| {
            if i > 0 && i % 3 == 0 {
                acc.push(',');
            }
            acc.push(c);
            acc
        })
        .chars()
        .rev()
        .collect()
}
fn is_component_suffix(s: &str) -> bool {
    matches!(s, "x" | "y" | "z" | "w" | "r" | "g" | "b" | "h" | "s" | "v")
        || s.parse::<usize>().is_ok()
}
