use crate::data::{SettingKey, Settings};
use crate::helpers::hsv::{HSV, hsv_to_rgb};
use crate::ui::variables::Variables;
use chrono::{DateTime, Utc};
use rand::RngExt;
use rand::rngs::ThreadRng;
use std::collections::HashSet;
use std::fmt;
use std::hash::{DefaultHasher, Hasher};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    F64(f64),
    I64(i64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string().fmt(f)
    }
}

impl Value {
    pub fn from_str(
        settings: &Settings,
        variables: &Variables,
        menus: &Menus,
        element_ctx: &ElementContext,
        s: &str,
    ) -> Self {
        //println!("Before Replace: '{}'", s);
        //let s = Self::replace_inline_variables(variables, settings, s);
        //println!("After replace: '{}'", s);
        let s = s.trim();
        //println!("In from_str: {}", s);
        if s.is_empty() {
            return Value::String(String::new());
        }
        let (s, precision) = match s.rsplit_once(':') {
            Some((base, suffix)) if suffix.starts_with('.') => {
                let decimals = &suffix[1..];

                if !decimals.is_empty() && decimals.chars().all(|c| c.is_ascii_digit()) {
                    match decimals.parse::<usize>() {
                        Ok(n) => (base.trim(), Some(n)),
                        Err(_) => (s, None),
                    }
                } else {
                    (s, None)
                }
            }
            _ => (s, None),
        };
        let value = Self::from_str_unrounded(settings, variables, menus, element_ctx, s);

        match precision {
            Some(decimals) => Self::apply_precision(value, decimals),
            None => value,
        }
    }
    fn from_str_unrounded(
        settings: &Settings,
        variables: &Variables,
        menus: &Menus,
        element_ctx: &ElementContext,
        s: &str,
    ) -> Value {
        // Auto-detect array (bracket notation)
        if s.starts_with('[') && s.ends_with(']') {
            if let Some(arr) = Self::parse_array(settings, variables, menus, element_ctx, s) {
                //println!("from_str color: {:?}", arr);
                return Value::Array(arr);
            }
        }

        // Auto-detect string
        if let Some(stripped) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            return Value::String(stripped.to_string());
        }
        // Explicit type prefix
        if let Some((ty, value)) = s.split_once(':') {
            if !ty.contains(char::is_whitespace) {
                match ty.to_ascii_lowercase().as_str() {
                    "int" => {
                        return if let Ok(i) = value.parse::<i64>() {
                            Value::I64(i)
                        } else {
                            Value::String(format!("'{}' couldn't get converted to int!", s))
                        };
                    }
                    "float" | "f32" | "f64" => {
                        return if let Ok(f) = value.parse::<f64>() {
                            Value::F64(f)
                        } else {
                            Value::String(format!("'{}' couldn't get converted to float!", s))
                        };
                    }
                    "bool" => {
                        return match value.to_ascii_lowercase().as_str() {
                            "true" | "1" | "yes" | "on" | "some" => Value::Bool(true),
                            "false" | "0" | "no" | "off" | "none" | "null" => Value::Bool(false),
                            _ => {
                                Value::String(format!("'{}' couldn't get converted to boolean!", s))
                            }
                        };
                    }
                    "string" | "str" => {
                        return Value::String(value.to_string());
                    }
                    // "strexpr" => {
                    //     //println!("strexpr was given: {}", value);
                    //     let s = Value::String(
                    //         Value::from_str(settings, variables, value, true, false).into_string(),
                    //     );
                    //     //println!("strexpr gave: {}", s);
                    //     return s;
                    // }
                    "expr" => {
                        let expr_value =
                            match eval_expr(value, variables, settings, menus, element_ctx) {
                                Some(value) => match value {
                                    Value::None => Value::String(value.to_string()),
                                    _ => value,
                                },
                                None => Value::String(value.to_string()),
                            };
                        //println!("expr gets: '{value}' and returns: '{:?}'", expr_value);
                        return expr_value;
                    }
                    "exprvar" => {
                        let expr_value =
                            match eval_expr(value, variables, settings, menus, element_ctx) {
                                Some(value) => match value {
                                    Value::None => Value::String(value.to_string()),
                                    _ => value,
                                },
                                None => Value::String(value.to_string()),
                            };
                        let expr_value = expr_value.into_string();
                        let post_value = Value::from_str(
                            settings,
                            variables,
                            menus,
                            element_ctx,
                            expr_value.as_str(),
                        );
                        println!(
                            "exprvar gets: '{value}' and returns: '{expr_value}', then returns '{:?}'",
                            post_value
                        );
                        return post_value;
                    }
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
                    "var" | "variable" => {
                        return match Self::load_variable(
                            variables,
                            menus,
                            element_ctx,
                            &value.to_string(),
                        ) {
                            Some(value) => value,
                            None => Value::String(format!("'{}' variable doesn't exist yet!", s)),
                        };
                    }
                    "array" | "list" | "vec" | "slice" => {
                        if let Some(arr) =
                            Self::parse_array(settings, variables, menus, element_ctx, value)
                        {
                            return Value::Array(arr);
                        }
                    }
                    "none" => {
                        return Value::None;
                    }
                    _ => {
                        return Value::String(format!(
                            "'{}', the type '{}' doesn't exist. Use 'expr:xyz' for Expressions.",
                            s, ty
                        ));
                    }
                }
            }
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

        match Self::load_variable(variables, menus, element_ctx, &s.to_string()) {
            Some(value) => return value,
            None => {}
        };

        // Auto-detect integer
        if let Ok(i) = s.parse::<i64>() {
            return Value::I64(i);
        }

        // Auto-detect float
        if let Ok(f) = s.parse::<f64>() {
            return Value::F64(f);
        }

        Value::String(s.to_string())
    }
    pub fn from_str_pure(s: &str) -> Self {
        let s = s.trim();
        if s.is_empty() {
            return Value::String(String::new());
        }
        let (s, precision) = match s.rsplit_once(':') {
            Some((base, suffix)) if suffix.starts_with('.') => {
                let decimals = &suffix[1..];

                if !decimals.is_empty() && decimals.chars().all(|c| c.is_ascii_digit()) {
                    match decimals.parse::<usize>() {
                        Ok(n) => (base.trim(), Some(n)),
                        Err(_) => (s, None),
                    }
                } else {
                    (s, None)
                }
            }
            _ => (s, None),
        };
        let value = Self::from_str_unrounded_pure(s);

        match precision {
            Some(decimals) => Self::apply_precision(value, decimals),
            None => value,
        }
    }
    fn from_str_unrounded_pure(s: &str) -> Value {
        // Auto-detect array (bracket notation)
        if s.starts_with('[') && s.ends_with(']') {
            if let Some(arr) = Self::parse_array_pure(s) {
                //println!("from_str color: {:?}", arr);
                return Value::Array(arr);
            }
        }

        // Auto-detect string
        if let Some(stripped) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            return Value::String(stripped.to_string());
        }
        // Explicit type prefix
        if let Some((ty, value)) = s.split_once(':') {
            if !ty.contains(char::is_whitespace) {
                match ty.to_ascii_lowercase().as_str() {
                    "int" => {
                        return if let Ok(i) = value.parse::<i64>() {
                            Value::I64(i)
                        } else {
                            Value::String(format!("'{}' couldn't get converted to int!", s))
                        };
                    }
                    "float" | "f32" | "f64" => {
                        return if let Ok(f) = value.parse::<f64>() {
                            Value::F64(f)
                        } else {
                            Value::String(format!("'{}' couldn't get converted to float!", s))
                        };
                    }
                    "bool" => {
                        return match value.to_ascii_lowercase().as_str() {
                            "true" | "1" | "yes" | "on" | "some" => Value::Bool(true),
                            "false" | "0" | "no" | "off" | "none" | "null" => Value::Bool(false),
                            _ => {
                                Value::String(format!("'{}' couldn't get converted to boolean!", s))
                            }
                        };
                    }
                    "string" | "str" => {
                        return Value::String(value.to_string());
                    }
                    "array" | "list" | "vec" | "slice" => {
                        if let Some(arr) = Self::parse_array_pure(value) {
                            return Value::Array(arr);
                        }
                    }
                    "none" => {
                        return Value::None;
                    }
                    _ => {
                        return Value::String(format!(
                            "'{}', the type '{}' doesn't exist. Use 'expr:xyz' for Expressions.",
                            s, ty
                        ));
                    }
                }
            }
        }

        // Auto-detect integer
        if let Ok(i) = s.parse::<i64>() {
            return Value::I64(i);
        }

        // Auto-detect float
        if let Ok(f) = s.parse::<f64>() {
            return Value::F64(f);
        }

        Value::String(s.to_string())
    }
    fn apply_precision(value: Value, decimals: usize) -> Value {
        match value {
            Value::F64(f) => {
                let factor = 10_f64.powi(decimals as i32);
                Value::F64((f * factor).round() / factor)
            }

            // Integers are already exact.
            Value::I64(i) => Value::I64(i),

            // Everything else stays unchanged.
            other => other,
        }
    }
    pub fn from_vec<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<f64>,
    {
        Value::Array(iter.into_iter().map(|e| Value::F64(e.into())).collect())
    }
    pub fn load_variable(
        variables: &Variables,
        menus: &Menus,
        element_ctx: &ElementContext,
        name: &str,
    ) -> Option<Value> {
        if let Some(var) = Self::get_element_property(menus, element_ctx, name) {
            // Maybe optimize here? But low priority, because it just checks sef/as anyway.
            return Some(var);
        }
        //println!("Name: '{}', Element Context: {:?}", name, element_ctx);
        let val = variables.get(name).map(|v| v.into_owned());
        //println!("Became: '{:?}'", val.as_ref().map(|v| v.to_string()));
        val

        // if let Some(var) = variables.get(name) {
        //     return Some(var.into_owned())
        // }
        // //println!("Name: '{}', Element Context: {:?}", name, element_ctx);
        // let val = Self::get_element_property(menus, element_ctx, name);
        // //println!("Became: '{:?}'", val.as_ref().map(|v| v.to_string()));
        // val
    }
    fn get_element_property(
        menus: &Menus,
        element_ctx: &ElementContext,
        name: &str,
    ) -> Option<Value> {
        let mut parts = name.split('.');

        let element_ref = match parts.next()? {
            "self" => element_ctx.self_element.as_ref(),
            "as" => element_ctx.as_element.as_ref(),
            _ => return None,
        }?;

        let property = parts.next()?;

        let first = parts.next();
        let (component, index) = match first {
            None => (None, None),
            Some(part) => {
                if let Some(index) = Variables::component_index(part) {
                    (None, Some(index))
                } else {
                    let index = parts.next().and_then(Variables::component_index);
                    (Some(part), index)
                }
            }
        };

        if parts.next().is_some() {
            return None;
        }

        let value = Self::get_element_property2(menus, element_ref, property, component)?;

        match index {
            Some(index) => value.as_array().and_then(|array| array.get(index).cloned()),
            None => Some(value),
        }
    }
    fn get_element_property2(
        menus: &Menus,
        element_ref: &ElementRef,
        property: &str,
        component: Option<&str>,
    ) -> Option<Value> {
        match property {
            "menu" => {
                return Some(Value::String(element_ref.menu.clone()));
            }
            "layer" => {
                return Some(Value::String(element_ref.layer.clone()));
            }
            "id" => {
                return Some(Value::String(element_ref.id.clone()));
            }
            "kind" => {
                return Some(Value::String(element_ref.kind.to_string()));
            }
            _ => {}
        };

        let Some(menu) = menus.get(&element_ref.menu) else {
            return None;
        };
        let Some(layer) = menu.layers.iter().find(|l| l.name == element_ref.layer) else {
            return None;
        };
        let Some(element_idx) = layer
            .elements
            .iter()
            .position(|e| e.id() == element_ref.id.as_str())
        else {
            return None;
        };
        let element = &layer.elements[element_idx];
        match property {
            "idx" => Some(Value::I64(element_idx as i64)),
            "active" => Some(Value::Bool(
                menu.active && layer.active && element.is_active(),
            )),
            "center" => Some(Value::Array(vec![
                Value::F64(element.center()[0] as f64),
                Value::F64(element.center()[1] as f64),
            ])),
            "anchor_center" => {
                if let Some(text) = element.as_text() {
                    let center = anchor_to(text.anchor, text.center(), text.width, text.height);
                    Some(Value::Array(vec![
                        Value::F64(center[0] as f64),
                        Value::F64(center[1] as f64),
                    ]))
                } else {
                    None
                }
            }
            "radius" => {
                if let Some(radius) = element.radius() {
                    Some(Value::F64(radius as f64))
                } else {
                    None
                }
            }
            "pt" => {
                if let Some(pt) = element.pt() {
                    Some(Value::F64(pt as f64))
                } else {
                    None
                }
            }
            "size" => {
                let prop = element.main_size();
                Some(prop.to_value())
            }
            "rect" => {
                if let Some(prop) = element.size2() {
                    Some(Value::from_vec(prop))
                } else {
                    None
                }
            }
            "color_components" => {
                let color_components = element.color_components();
                let comps = color_components
                    .iter()
                    .map(|c| Value::String(c.to_string()))
                    .collect::<Vec<Value>>();
                Some(Value::Array(comps))
            }
            "color" => {
                if let Some(component) = component {
                    if let Some(color) = element.color(ColorComponent::from_str(component)) {
                        Some(Value::from_vec(color))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            "text" => {
                if let Some(text) = element.text() {
                    Some(Value::String(text.to_string()))
                } else {
                    None
                }
            }
            "template" => {
                if let Some(template) = element.template() {
                    Some(Value::String(template.to_string()))
                } else {
                    None
                }
            }
            _ => None,
        }
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
            Value::None => None,
        }
    }

    pub fn to_f64(self) -> Value {
        match self {
            Value::F64(v) => Value::F64(v),
            Value::I64(v) => Value::F64(v as f64),
            Value::Bool(v) => Value::F64(if v { 1.0 } else { 0.0 }),
            Value::String(s) => s.parse::<f64>().map(Value::F64).unwrap_or(Value::F64(1.0)),
            Value::None => Value::F64(0.0),
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
            Value::None => Some(0),
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
            Value::None => Value::I64(0),
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
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(n) => Some(*n),
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
            Value::None => false,
        }
    }
    pub fn to_string(&self) -> String {
        match self {
            Value::F64(n) => n.to_string(),
            Value::I64(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::None => "None".to_string(),
        }
    }

    pub fn into_string(self) -> String {
        match self {
            Value::F64(n) => n.to_string(),
            Value::I64(n) => n.to_string(),
            Value::String(s) => s, // moved, no clone
            Value::Bool(b) => b.to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.into_iter().map(|v| v.into_string()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::None => "none".to_string(),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Value::None)
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::I64(_) => "integer64",
            Value::F64(_) => "float64",
            Value::String(_) => "string",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::None => "none",
        }
    }

    /// Split a string by commas, respecting nested brackets and quoted strings
    pub fn split_array_elements(s: &str) -> Vec<&str> {
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
    pub fn parse_array(
        settings: &Settings,
        variables: &Variables,
        menus: &Menus,
        element_ctx: &ElementContext,
        s: &str,
    ) -> Option<Vec<Value>> {
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
                result.push(Value::from_str(
                    settings,
                    variables,
                    menus,
                    element_ctx,
                    elem,
                ));
            }
        }

        Some(result)
    }
    /// Parse an array from a string like "[1, 2, 3]" or "1, 2, 3", but without dynamic variables or expressions.
    pub fn parse_array_pure(s: &str) -> Option<Vec<Value>> {
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
                result.push(Value::from_str_pure(elem));
            }
        }

        Some(result)
    }
    // fn replace_inline_variables(variables: &Variables, settings: &Settings, menus: &Menus, element_ctx: &ElementContext, s: &str) -> String {
    //     let mut result = String::with_capacity(s.len());
    //     let mut chars = s.chars().peekable();
    //
    //     while let Some(c) = chars.next() {
    //         if c == '{' {
    //             let mut var_name = String::new();
    //             let mut found_end = false;
    //
    //             // Consume characters until we find the closing '}'
    //             for inner_c in chars.by_ref() {
    //                 if inner_c == '}' {
    //                     found_end = true;
    //                     break;
    //                 }
    //                 var_name.push(inner_c);
    //             }
    //
    //             if found_end && !var_name.is_empty() {
    //                 // Try to load the variable
    //                 if let Some(value) = get_var_opt(variables, settings, var_name.as_str()) {
    //                     // Replace with the variable's string representation
    //                     result.push_str(&value.to_string());
    //                 } else {
    //                     // Variable doesn't exist yet, leave it exactly as it was
    //                     result.push('{');
    //                     result.push_str(&var_name);
    //                     result.push('}');
    //                 }
    //             } else {
    //                 // Unclosed brace or empty {}, leave as is
    //                 result.push('{');
    //                 result.push_str(&var_name);
    //                 if found_end {
    //                     result.push('}');
    //                 }
    //             }
    //         } else {
    //             result.push(c);
    //         }
    //     }
    //     result
    // }
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

#[derive(Debug, Clone)]
pub enum FunctionError {
    MissingArgument {
        index: usize,
        function: &'static str,
    },
    InvalidType {
        expected: &'static str,
        got: &'static str,
        function: &'static str,
    },
    InvalidValue {
        message: String,
        function: &'static str,
    },
    InvalidArgumentCount {
        expected: usize,
        got: usize,
        function: &'static str,
    },
    ConversionFailed {
        from: &'static str,
        to: &'static str,
        function: &'static str,
    },
    IndexOutOfBounds {
        index: usize,
        len: usize,
        function: &'static str,
    },
    EmptyInput {
        function: &'static str,
    },
    Other {
        message: String,
        function: &'static str,
    },
}

fn get_arg<'a>(
    args: &'a [Value],
    idx: usize,
    func: &'static str,
) -> Result<&'a Value, FunctionError> {
    args.get(idx).ok_or(FunctionError::MissingArgument {
        index: idx,
        function: func,
    })
}

fn arg_f64(args: &[Value], idx: usize, func: &'static str) -> Result<f64, FunctionError> {
    let v = get_arg(args, idx, func)?;
    v.as_f64().ok_or(FunctionError::ConversionFailed {
        from: v.type_name(),
        to: "f64",
        function: func,
    })
}
fn arg_bool(args: &[Value], idx: usize, func: &'static str) -> Result<bool, FunctionError> {
    let v = get_arg(args, idx, func)?;
    Ok(v.is_truthy()) //.ok_or(FunctionError::ConversionFailed { from: v.type_name(), to: "bool", function: func })
}
fn arg_i64(args: &[Value], idx: usize, func: &'static str) -> Result<i64, FunctionError> {
    let v = get_arg(args, idx, func)?;
    v.as_i64().ok_or(FunctionError::ConversionFailed {
        from: v.type_name(),
        to: "i64",
        function: func,
    })
}

fn arg_time_seconds(args: &[Value], idx: usize, func: &'static str) -> Result<f64, FunctionError> {
    let mut value = arg_f64(args, idx, func)?;
    let from_millis = arg_bool(args, idx + 1, func)?;

    if from_millis {
        value /= 1000.0;
    }

    Ok(value)
}
fn format_duration(secs: f64) -> String {
    let negative = secs < 0.0;
    let secs = secs.abs();

    let hours = (secs / 3600.0).floor() as i64;
    let mins = ((secs % 3600.0) / 60.0).floor() as i64;
    let seconds = (secs % 60.0).floor() as i64;

    let result = if hours > 0 {
        format!("{hours}:{mins:02}:{seconds:02}")
    } else {
        format!("{mins}:{seconds:02}")
    };

    if negative {
        format!("-{result}")
    } else {
        result
    }
}
fn format_elapsed(secs: f64) -> String {
    let secs = secs.abs();

    if secs < 60.0 {
        let n = secs.round();
        format!("{n:.0} second{}", if n == 1.0 { "" } else { "s" })
    } else if secs < 3600.0 {
        let n = (secs / 60.0).round();
        format!("{n:.0} minute{}", if n == 1.0 { "" } else { "s" })
    } else if secs < 86400.0 {
        let n = secs / 3600.0;
        format!("{n:.1} hours")
    } else if secs < 604800.0 {
        let n = secs / 86400.0;
        format!("{n:.1} days")
    } else if secs < 2_592_000.0 {
        let n = secs / 604800.0;
        format!("{n:.1} weeks")
    } else if secs < 31_536_000.0 {
        let n = secs / 2_592_000.0;
        format!("{n:.1} months")
    } else {
        let n = secs / 31_536_000.0;
        format!("{n:.1} years")
    }
}

fn format_custom_time(
    secs: f64,
    day_seconds: f64,
    max_parts: usize,
    use_day_seconds: bool,
) -> String {
    if day_seconds <= 0.0 {
        return "invalid day length".to_string();
    }

    let negative = secs < 0.0;
    let mut secs = secs.abs();

    let (second, minute, hour, day) = if use_day_seconds {
        let second = day_seconds / 86_400.0;
        let minute = second * 60.0;
        let hour = minute * 60.0;

        (second, minute, hour, day_seconds)
    } else {
        let second = 1.0;
        let minute = 60.0;
        let hour = 3_600.0;
        let day = day_seconds;

        (second, minute, hour, day)
    };

    let week = 7.0 * day;
    let month = 30.0 * day;
    let year = 12.0 * month;
    let decade = 10.0 * year;
    let century = 10.0 * decade;

    let units = [
        ("century", century),
        ("decade", decade),
        ("year", year),
        ("month", month),
        ("week", week),
        ("day", day),
        ("hour", hour),
        ("minute", minute),
        ("second", second),
    ];

    let mut parts = Vec::new();

    for (name, unit) in units {
        if parts.len() >= max_parts {
            break;
        }

        if secs >= unit {
            let amount = (secs / unit).floor() as i64;
            secs -= amount as f64 * unit;

            let name = if amount == 1 {
                name.to_string()
            } else {
                format!("{name}s")
            };

            parts.push(format!("{amount} {name}"));
        }
    }

    if parts.is_empty() {
        parts.push("0 seconds".to_string());
    }

    let result = parts.join(", ");

    if negative {
        format!("-{result}")
    } else {
        result
    }
}
type BuiltinFn = fn(Vec<Value>) -> Result<Value, FunctionError>;

pub fn get_builtin(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "fix" => |args| {
            let val = get_arg(&args, 0, "fix")?;
            let n = val.as_f64().ok_or(FunctionError::ConversionFailed {
                from: val.type_name(),
                to: "f64",
                function: "fix",
            })?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(3.0) as usize;
            let min_int = args
                .get(2)
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::INFINITY) as usize;
            let sign = if n < 0.0 { "-" } else { "" };
            let abs = n.abs();
            let formatted = format!("{:.*}", decimals, abs);
            let mut parts = formatted.split('.');
            let int_part = parts.next().unwrap_or("");
            let frac_part = parts.next().unwrap_or("");
            let padded_int = if min_int == usize::MAX {
                int_part.to_string()
            } else if int_part.len() < min_int {
                let pad = " ".repeat(min_int - int_part.len());
                format!("{}{}", pad, int_part)
            } else {
                int_part.to_string()
            };
            let result = if decimals > 0 {
                format!("{}{}.{}", sign, padded_int, frac_part)
            } else {
                format!("{}{}", sign, padded_int)
            };
            Ok(Value::String(result))
        },
        "abs" => |args| {
            let val = get_arg(&args, 0, "abs")?;
            match val {
                Value::F64(n) => Ok(Value::F64(n.abs())),
                Value::I64(n) => Ok(Value::I64(n.abs())),
                v => Err(FunctionError::InvalidType {
                    expected: "number",
                    got: v.type_name(),
                    function: "abs",
                }),
            }
        },
        "floor" => |args| {
            let n = arg_f64(&args, 0, "floor")?;
            Ok(Value::F64(n.floor()))
        },
        "ceil" => |args| {
            let n = arg_f64(&args, 0, "ceil")?;
            Ok(Value::F64(n.ceil()))
        },
        "round" => |args| {
            let n = arg_f64(&args, 0, "round")?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
            let factor = 10f64.powi(decimals);
            Ok(Value::F64((n * factor).round() / factor))
        },
        "trunc" => |args| {
            let n = arg_f64(&args, 0, "trunc")?;
            Ok(Value::F64(n.trunc()))
        },
        "sqrt" => |args| {
            let n = arg_f64(&args, 0, "sqrt")?;
            Ok(Value::F64(n.sqrt()))
        },
        "cbrt" => |args| {
            let n = arg_f64(&args, 0, "cbrt")?;
            Ok(Value::F64(n.cbrt()))
        },
        "pow" => |args| {
            let base = arg_f64(&args, 0, "pow")?;
            let exp = arg_f64(&args, 1, "pow")?;
            Ok(Value::F64(base.powf(exp)))
        },
        "exp" => |args| {
            let n = arg_f64(&args, 0, "exp")?;
            Ok(Value::F64(n.exp()))
        },
        "ln" => |args| {
            let n = arg_f64(&args, 0, "ln")?;
            Ok(Value::F64(n.ln()))
        },
        "log" => |args| {
            let n = arg_f64(&args, 0, "log")?;
            let base = args.get(1).and_then(|v| v.as_f64()).unwrap_or(10.0);
            Ok(Value::F64(n.log(base)))
        },
        "log2" => |args| {
            let n = arg_f64(&args, 0, "log2")?;
            Ok(Value::F64(n.log2()))
        },
        "log10" => |args| {
            let n = arg_f64(&args, 0, "log10")?;
            Ok(Value::F64(n.log10()))
        },
        "sin" => |args| {
            let n = arg_f64(&args, 0, "sin")?;
            Ok(Value::F64(n.sin()))
        },
        "cos" => |args| {
            let n = arg_f64(&args, 0, "cos")?;
            Ok(Value::F64(n.cos()))
        },
        "tan" => |args| {
            let n = arg_f64(&args, 0, "tan")?;
            Ok(Value::F64(n.tan()))
        },
        "asin" => |args| {
            let n = arg_f64(&args, 0, "asin")?;
            Ok(Value::F64(n.asin()))
        },
        "acos" => |args| {
            let n = arg_f64(&args, 0, "acos")?;
            Ok(Value::F64(n.acos()))
        },
        "atan" => |args| {
            let n = arg_f64(&args, 0, "atan")?;
            Ok(Value::F64(n.atan()))
        },
        "atan2" => |args| {
            let y = arg_f64(&args, 0, "atan2")?;
            let x = arg_f64(&args, 1, "atan2")?;
            Ok(Value::F64(y.atan2(x)))
        },
        "sinh" => |args| {
            let n = arg_f64(&args, 0, "sinh")?;
            Ok(Value::F64(n.sinh()))
        },
        "cosh" => |args| {
            let n = arg_f64(&args, 0, "cosh")?;
            Ok(Value::F64(n.cosh()))
        },
        "tanh" => |args| {
            let n = arg_f64(&args, 0, "tanh")?;
            Ok(Value::F64(n.tanh()))
        },
        "degrees" => |args| {
            let n = arg_f64(&args, 0, "degrees")?;
            Ok(Value::F64(n.to_degrees()))
        },
        "radians" => |args| {
            let n = arg_f64(&args, 0, "radians")?;
            Ok(Value::F64(n.to_radians()))
        },
        "min" => |args| {
            let mut min_val: Option<f64> = None;
            for arg in &args {
                if let Some(n) = arg.as_f64() {
                    min_val = Some(min_val.map_or(n, |m| m.min(n)));
                }
            }
            min_val.map(Value::F64).ok_or(FunctionError::Other {
                message: "no numeric arguments provided".to_string(),
                function: "min",
            })
        },
        "max" => |args| {
            let mut max_val: Option<f64> = None;
            for arg in &args {
                if let Some(n) = arg.as_f64() {
                    max_val = Some(max_val.map_or(n, |m| m.max(n)));
                }
            }
            max_val.map(Value::F64).ok_or(FunctionError::Other {
                message: "no numeric arguments provided".to_string(),
                function: "max",
            })
        },
        "clamp" => |args| {
            let val = arg_f64(&args, 0, "clamp")?;
            let min = arg_f64(&args, 1, "clamp")?;
            let max = arg_f64(&args, 2, "clamp")?;
            Ok(Value::F64(val.clamp(min, max)))
        },
        "saturate" => |args| {
            let val = arg_f64(&args, 0, "saturate")?;
            Ok(Value::F64(val.clamp(0.0, 1.0)))
        },
        "lerp" => |args| {
            let a = arg_f64(&args, 0, "lerp")?;
            let b = arg_f64(&args, 1, "lerp")?;
            let t = arg_f64(&args, 2, "lerp")?;
            Ok(Value::F64(a + (b - a) * t))
        },
        "sign" => |args| {
            let n = arg_f64(&args, 0, "sign")?;
            Ok(Value::F64(if n > 0.0 {
                1.0
            } else if n < 0.0 {
                -1.0
            } else {
                0.0
            }))
        },
        "fract" => |args| {
            let n = arg_f64(&args, 0, "fract")?;
            Ok(Value::F64(n.fract()))
        },
        "mod" => |args| {
            let a = arg_f64(&args, 0, "mod")?;
            let b = arg_f64(&args, 1, "mod")?;
            if b == 0.0 {
                return Err(FunctionError::InvalidValue {
                    message: "modulo by zero".to_string(),
                    function: "mod",
                });
            }
            Ok(Value::F64(a % b))
        },
        "hypot" => |args| {
            let a = arg_f64(&args, 0, "hypot")?;
            let b = arg_f64(&args, 1, "hypot")?;
            Ok(Value::F64(a.hypot(b)))
        },
        "pi" => |_| Ok(Value::F64(std::f64::consts::PI)),
        "e" => |_| Ok(Value::F64(std::f64::consts::E)),
        "tau" => |_| Ok(Value::F64(std::f64::consts::TAU)),
        "inf" => |_| Ok(Value::F64(f64::INFINITY)),
        "nan" => |_| Ok(Value::F64(f64::NAN)),
        "isnan" => |args| {
            let n = arg_f64(&args, 0, "isnan")?;
            Ok(Value::Bool(n.is_nan()))
        },
        "isinf" => |args| {
            let n = arg_f64(&args, 0, "isinf")?;
            Ok(Value::Bool(n.is_infinite()))
        },
        "isfinite" => |args| {
            let n = arg_f64(&args, 0, "isfinite")?;
            Ok(Value::Bool(n.is_finite()))
        },
        "len" => |args| {
            let val = get_arg(&args, 0, "len")?;
            match val {
                Value::String(s) => Ok(Value::F64(s.len() as f64)),
                Value::Array(arr) => Ok(Value::F64(arr.len() as f64)),
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "len",
                }),
            }
        },
        "upper" => |args| {
            let val = get_arg(&args, 0, "upper")?;
            Ok(Value::String(val.to_string().to_uppercase()))
        },
        "lower" => |args| {
            let val = get_arg(&args, 0, "lower")?;
            Ok(Value::String(val.to_string().to_lowercase()))
        },
        "capitalize" => |args| {
            let val = get_arg(&args, 0, "capitalize")?;
            match val {
                Value::String(s) => {
                    let mut chars = s.chars();
                    let result = match chars.next() {
                        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                        None => String::new(),
                    };
                    Ok(Value::String(result))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "capitalize",
                }),
            }
        },
        "title" => |args| {
            let val = get_arg(&args, 0, "title")?;
            match val {
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
                    Ok(Value::String(result))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "title",
                }),
            }
        },
        "trim" => |args| {
            let val = get_arg(&args, 0, "trim")?;
            match val {
                Value::String(s) => Ok(Value::String(s.trim().to_string())),
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "trim",
                }),
            }
        },
        "ltrim" => |args| {
            let val = get_arg(&args, 0, "ltrim")?;
            match val {
                Value::String(s) => Ok(Value::String(s.trim_start().to_string())),
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "ltrim",
                }),
            }
        },
        "rtrim" => |args| {
            let val = get_arg(&args, 0, "rtrim")?;
            match val {
                Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "rtrim",
                }),
            }
        },
        "reverse" => |args| {
            let val = get_arg(&args, 0, "reverse")?;
            match val {
                Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                Value::Array(arr) => Ok(Value::Array(arr.iter().rev().cloned().collect())),
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "reverse",
                }),
            }
        },
        "repeat" => |args| {
            let val = get_arg(&args, 0, "repeat")?;
            let s = match val {
                Value::String(s) => s.clone(),
                v => v.to_string(),
            };
            let n = arg_f64(&args, 1, "repeat")? as usize;
            Ok(Value::String(s.repeat(n)))
        },
        "replace" => |args| {
            let val = get_arg(&args, 0, "replace")?;
            match val {
                Value::String(s) => {
                    let from = match get_arg(&args, 1, "replace")? {
                        Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    let to = match get_arg(&args, 2, "replace")? {
                        Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    Ok(Value::String(s.replace(&from, &to)))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "replace",
                }),
            }
        },
        "split" => |args| {
            let val = get_arg(&args, 0, "split")?;
            match val {
                Value::String(s) => {
                    let delim = match args.get(1) {
                        Some(Value::String(d)) => d.clone(),
                        _ => " ".to_string(),
                    };
                    let parts: Vec<Value> = s
                        .split(&delim)
                        .map(|p| Value::String(p.to_string()))
                        .collect();
                    Ok(Value::Array(parts))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "split",
                }),
            }
        },
        "join" => |args| {
            let val = get_arg(&args, 0, "join")?;
            match val {
                Value::Array(arr) => {
                    let delim = match args.get(1) {
                        Some(Value::String(d)) => d.clone(),
                        _ => "".to_string(),
                    };
                    let result: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                    Ok(Value::String(result.join(&delim)))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "array",
                    got: v.type_name(),
                    function: "join",
                }),
            }
        },
        "substr" => |args| {
            let val = get_arg(&args, 0, "substr")?;
            match val {
                Value::String(s) => {
                    let start = arg_f64(&args, 1, "substr")? as usize;
                    let len = args.get(2).and_then(|v| v.as_f64()).map(|n| n as usize);
                    let chars: Vec<char> = s.chars().collect();
                    let end = len
                        .map(|l| (start + l).min(chars.len()))
                        .unwrap_or(chars.len());
                    if start >= chars.len() {
                        Ok(Value::String(String::new()))
                    } else {
                        Ok(Value::String(chars[start..end].iter().collect()))
                    }
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "substr",
                }),
            }
        },
        "contains" => |args| {
            let val = get_arg(&args, 0, "contains")?;
            match val {
                Value::String(s) => {
                    let needle = match get_arg(&args, 1, "contains")? {
                        Value::String(n) => n.clone(),
                        v => v.to_string(),
                    };
                    Ok(Value::Bool(s.contains(&needle)))
                }
                Value::Array(arr) => {
                    let needle = get_arg(&args, 1, "contains")?;
                    Ok(Value::Bool(arr.iter().any(|v| v == needle)))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "contains",
                }),
            }
        },
        "startswith" => |args| {
            let val = get_arg(&args, 0, "startswith")?;
            match val {
                Value::String(s) => {
                    let prefix = match get_arg(&args, 1, "startswith")? {
                        Value::String(n) => n.clone(),
                        v => v.to_string(),
                    };
                    Ok(Value::Bool(s.starts_with(&prefix)))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "startswith",
                }),
            }
        },
        "endswith" => |args| {
            let val = get_arg(&args, 0, "endswith")?;
            match val {
                Value::String(s) => {
                    let suffix = match get_arg(&args, 1, "endswith")? {
                        Value::String(n) => n.clone(),
                        v => v.to_string(),
                    };
                    Ok(Value::Bool(s.ends_with(&suffix)))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "endswith",
                }),
            }
        },
        "indexof" => |args| {
            let val = get_arg(&args, 0, "indexof")?;
            match val {
                Value::String(s) => {
                    let needle = match get_arg(&args, 1, "indexof")? {
                        Value::String(n) => n.clone(),
                        v => v.to_string(),
                    };
                    Ok(Value::F64(
                        s.find(&needle).map(|i| i as f64).unwrap_or(-1.0),
                    ))
                }
                Value::Array(arr) => {
                    let needle = get_arg(&args, 1, "indexof")?;
                    for (i, v) in arr.iter().enumerate() {
                        if v == needle {
                            return Ok(Value::F64(i as f64));
                        }
                    }
                    Ok(Value::F64(-1.0))
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "indexof",
                }),
            }
        },
        "get" => |args| {
            let arr = match args.get(0) {
                Some(Value::Array(arr)) => arr,
                _ => {
                    return Err(FunctionError::InvalidType {
                        expected: "array",
                        got: "value",
                        function: "get",
                    });
                }
            };
            let index = arg_i64(&args, 1, "get")? as usize;
            arr.get(index)
                .cloned()
                .ok_or(FunctionError::IndexOutOfBounds {
                    index,
                    len: arr.len(),
                    function: "get",
                })
        },
        "padleft" => |args| {
            let val = get_arg(&args, 0, "padleft")?;
            let s = val.to_string();
            let width = arg_f64(&args, 1, "padleft")? as usize;
            let pad_char = match args.get(2) {
                Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                _ => ' ',
            };
            if s.len() >= width {
                Ok(Value::String(s))
            } else {
                let padding: String = std::iter::repeat(pad_char).take(width - s.len()).collect();
                Ok(Value::String(padding + &s))
            }
        },
        "padright" => |args| {
            let val = get_arg(&args, 0, "padright")?;
            match val {
                Value::String(s) => {
                    let width = arg_f64(&args, 1, "padright")? as usize;
                    let pad_char = match args.get(2) {
                        Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                        _ => ' ',
                    };
                    if s.len() >= width {
                        Ok(Value::String(s.clone()))
                    } else {
                        let padding: String =
                            std::iter::repeat(pad_char).take(width - s.len()).collect();
                        Ok(Value::String(s.clone() + &padding))
                    }
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "padright",
                }),
            }
        },
        "center" => |args| {
            let val = get_arg(&args, 0, "center")?;
            match val {
                Value::String(s) => {
                    let width = arg_f64(&args, 1, "center")? as usize;
                    let pad_char = match args.get(2) {
                        Some(Value::String(p)) if !p.is_empty() => p.chars().next().unwrap(),
                        _ => ' ',
                    };
                    if s.len() >= width {
                        Ok(Value::String(s.clone()))
                    } else {
                        let total_pad = width - s.len();
                        let left_pad = total_pad / 2;
                        let right_pad = total_pad - left_pad;
                        let left: String = std::iter::repeat(pad_char).take(left_pad).collect();
                        let right: String = std::iter::repeat(pad_char).take(right_pad).collect();
                        Ok(Value::String(left + s + &right))
                    }
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "center",
                }),
            }
        },
        "char" => |args| {
            let code = arg_f64(&args, 0, "char")? as u32;
            char::from_u32(code)
                .map(|c| Value::String(c.to_string()))
                .ok_or_else(|| FunctionError::InvalidValue {
                    message: format!("invalid unicode codepoint: {}", code),
                    function: "char",
                })
        },
        "ord" => |args| {
            let val = get_arg(&args, 0, "ord")?;
            match val {
                Value::String(s) => {
                    s.chars()
                        .next()
                        .map(|c| Value::I64(c as i64))
                        .ok_or_else(|| FunctionError::InvalidValue {
                            message: "empty string".to_string(),
                            function: "ord",
                        })
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string",
                    got: v.type_name(),
                    function: "ord",
                }),
            }
        },
        "hex" => |args| {
            let n = arg_f64(&args, 0, "hex")? as i64;
            Ok(Value::String(format!("{:x}", n)))
        },
        "bin" => |args| {
            let n = arg_f64(&args, 0, "bin")? as i64;
            Ok(Value::String(format!("{:b}", n)))
        },
        "oct" => |args| {
            let n = arg_f64(&args, 0, "oct")? as i64;
            Ok(Value::String(format!("{:o}", n)))
        },
        "first" => |args| {
            let val = get_arg(&args, 0, "first")?;
            match val {
                Value::Array(arr) => arr
                    .first()
                    .cloned()
                    .ok_or(FunctionError::EmptyInput { function: "first" }),
                Value::String(s) => s
                    .chars()
                    .next()
                    .map(|c| Value::String(c.to_string()))
                    .ok_or(FunctionError::EmptyInput { function: "first" }),
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "first",
                }),
            }
        },
        "last" => |args| {
            let val = get_arg(&args, 0, "last")?;
            match val {
                Value::Array(arr) => arr
                    .last()
                    .cloned()
                    .ok_or(FunctionError::EmptyInput { function: "last" }),
                Value::String(s) => s
                    .chars()
                    .last()
                    .map(|c| Value::String(c.to_string()))
                    .ok_or(FunctionError::EmptyInput { function: "last" }),
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "last",
                }),
            }
        },
        "sum" => |args| {
            let val = get_arg(&args, 0, "sum")?;
            match val {
                Value::Array(arr) => {
                    let sum: f64 = arr.iter().filter_map(|v| v.as_f64()).sum();
                    Ok(Value::F64(sum))
                }
                _ => {
                    let sum: f64 = args.iter().filter_map(|v| v.as_f64()).sum();
                    Ok(Value::F64(sum))
                }
            }
        },
        "avg" => |args| {
            let val = get_arg(&args, 0, "avg")?;
            match val {
                Value::Array(arr) => {
                    let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    if nums.is_empty() {
                        Err(FunctionError::EmptyInput { function: "avg" })
                    } else {
                        Ok(Value::F64(nums.iter().sum::<f64>() / nums.len() as f64))
                    }
                }
                _ => {
                    let nums: Vec<f64> = args.iter().filter_map(|v| v.as_f64()).collect();
                    if nums.is_empty() {
                        Err(FunctionError::EmptyInput { function: "avg" })
                    } else {
                        Ok(Value::F64(nums.iter().sum::<f64>() / nums.len() as f64))
                    }
                }
            }
        },
        "nearest" => |args| {
            let choices_arr = match args.get(1) {
                Some(Value::Array(arr)) => arr,
                _ => return Ok(args.get(0).cloned().unwrap_or(Value::None)),
            };
            let value = arg_f64(&args, 0, "nearest")?;
            let choices: Vec<f64> = choices_arr.iter().filter_map(|v| v.as_f64()).collect();
            if choices.is_empty() {
                return Err(FunctionError::EmptyInput {
                    function: "nearest",
                });
            }
            let nearest = choices
                .into_iter()
                .min_by(|a, b| {
                    (value - a)
                        .abs()
                        .partial_cmp(&(value - b).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or(FunctionError::EmptyInput {
                    function: "nearest",
                })?;
            Ok(Value::F64(nearest))
        },
        "nearest_index" => |args| {
            let choices_arr = match args.get(0) {
                Some(Value::Array(arr)) => arr,
                _ => {
                    return Err(FunctionError::InvalidType {
                        expected: "array",
                        got: "value",
                        function: "nearest_index",
                    });
                }
            };
            let choices: Vec<f64> = choices_arr.iter().filter_map(|v| v.as_f64()).collect();
            if choices.is_empty() {
                return Err(FunctionError::EmptyInput {
                    function: "nearest_index",
                });
            }
            let value = arg_f64(&args, 1, "nearest_index")?;
            let index = choices
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (value - **a)
                        .abs()
                        .partial_cmp(&(value - **b).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .ok_or(FunctionError::EmptyInput {
                    function: "nearest_index",
                })?;
            Ok(Value::I64(index as i64))
        },
        "array_get" => |args| {
            let arr = match args.get(0) {
                Some(Value::Array(arr)) => arr,
                _ => {
                    return Err(FunctionError::InvalidType {
                        expected: "array",
                        got: "value",
                        function: "array_get",
                    });
                }
            };
            let index = arg_i64(&args, 1, "array_get")? as usize;
            arr.get(index)
                .cloned()
                .ok_or(FunctionError::IndexOutOfBounds {
                    index,
                    len: arr.len(),
                    function: "array_get",
                })
        },
        "index_of" => |args| {
            let array = match args.get(0) {
                Some(Value::Array(arr)) => arr,
                _ => {
                    return Err(FunctionError::InvalidType {
                        expected: "array",
                        got: "value",
                        function: "index_of",
                    });
                }
            };
            let value = get_arg(&args, 1, "index_of")?;

            let index = array
                .iter()
                .position(|v| {
                    if v == value {
                        return true;
                    }
                    match (v.as_f64(), value.as_f64()) {
                        (Some(a), Some(b)) => a == b,
                        _ => false,
                    }
                })
                .ok_or_else(|| FunctionError::Other {
                    message: "value not found in array".to_string(),
                    function: "index_of",
                })?;

            Ok(Value::I64(index as i64))
        },
        "count" => |args| {
            let val = get_arg(&args, 0, "count")?;
            match val {
                Value::Array(arr) => Ok(Value::I64(arr.len() as i64)),
                Value::String(s) => Ok(Value::I64(s.chars().count() as i64)),
                _ => Ok(Value::I64(args.len() as i64)),
            }
        },
        "range" => |args| {
            let start = arg_f64(&args, 0, "range")? as i64;
            let end = arg_f64(&args, 1, "range")? as i64;
            let step = args.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as i64;
            if step == 0 {
                return Err(FunctionError::InvalidValue {
                    message: "step cannot be zero".to_string(),
                    function: "range",
                });
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
            Ok(Value::Array(result))
        },
        "slice" => |args| {
            let val = get_arg(&args, 0, "slice")?;
            match val {
                Value::Array(arr) => {
                    let start = arg_f64(&args, 1, "slice")? as i64;
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
                        Ok(Value::Array(vec![]))
                    } else {
                        Ok(Value::Array(arr[start..end].to_vec()))
                    }
                }
                Value::String(s) => {
                    let chars: Vec<char> = s.chars().collect();
                    let start = arg_f64(&args, 1, "slice")? as i64;
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
                        Ok(Value::String(String::new()))
                    } else {
                        Ok(Value::String(chars[start..end].iter().collect()))
                    }
                }
                v => Err(FunctionError::InvalidType {
                    expected: "string or array",
                    got: v.type_name(),
                    function: "slice",
                }),
            }
        },
        "type" => |args| {
            let val = get_arg(&args, 0, "type")?;
            Ok(Value::String(val.type_name().to_string()))
        },
        "isnone" => |args| {
            Ok(Value::Bool(
                args.first().map(|v| v.is_none()).unwrap_or(true),
            ))
        },
        "isfloat" => |args| Ok(Value::Bool(matches!(args.first(), Some(Value::F64(_))))),
        "isint" => |args| Ok(Value::Bool(matches!(args.first(), Some(Value::I64(_))))),
        "isstr" => |args| Ok(Value::Bool(matches!(args.first(), Some(Value::String(_))))),
        "isbool" => |args| Ok(Value::Bool(matches!(args.first(), Some(Value::Bool(_))))),
        "isarray" => |args| Ok(Value::Bool(matches!(args.first(), Some(Value::Array(_))))),
        "str" => |args| {
            let val = get_arg(&args, 0, "str")?;
            Ok(Value::String(val.clone().into_string()))
        },
        "bool" => |args| {
            let val = get_arg(&args, 0, "bool")?;
            Ok(Value::Bool(val.is_truthy()))
        },
        "int" => |args| {
            let val = get_arg(&args, 0, "int")?;
            Ok(val.clone().to_i64())
        },
        "float" => |args| {
            let val = get_arg(&args, 0, "float")?;
            Ok(val.clone().to_f64())
        },
        "format" => |args| {
            let val = get_arg(&args, 0, "format")?;
            let precision = args.get(1).and_then(|v| v.as_f64()).map(|n| n as usize);
            if let (Some(n), Some(p)) = (val.as_f64(), precision) {
                Ok(Value::String(format!("{:.*}", p, n)))
            } else {
                Ok(Value::String(val.to_string()))
            }
        },
        "comma" => |args| {
            let n = arg_f64(&args, 0, "comma")?;
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
                Some(frac) => Ok(Value::String(format!("{}.{}", result, frac))),
                None => Ok(Value::String(result)),
            }
        },
        "percent" => |args| {
            let n = arg_f64(&args, 0, "percent")?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;
            Ok(Value::String(format!("{:.*}%", decimals, n * 100.0)))
        },
        "currency" => |args| {
            let n = arg_f64(&args, 0, "currency")?;
            let symbol = match args.get(1) {
                Some(Value::String(s)) => s.clone(),
                _ => "$".to_string(),
            };
            let decimals = args.get(2).and_then(|v| v.as_f64()).unwrap_or(2.0) as usize;
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
                Ok(Value::String(format!("-{}{}", symbol, num_str)))
            } else {
                Ok(Value::String(format!("{}{}", symbol, num_str)))
            }
        },
        "ordinal" => |args| {
            let n = arg_f64(&args, 0, "ordinal")? as i64;
            let suffix = match (n % 10, n % 100) {
                (1, 11) => "th",
                (2, 12) => "th",
                (3, 13) => "th",
                (1, _) => "st",
                (2, _) => "nd",
                (3, _) => "rd",
                _ => "th",
            };
            Ok(Value::String(format!("{}{}", n, suffix)))
        },
        "bytes" => |args| {
            let n = arg_f64(&args, 0, "bytes")?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(2.0) as usize;
            let units = ["B", "KB", "MB", "GB", "TB", "PB"];
            let mut value = n.abs();
            let mut unit_idx = 0;
            while value >= 1024.0 && unit_idx < units.len() - 1 {
                value /= 1024.0;
                unit_idx += 1;
            }
            let sign = if n < 0.0 { "-" } else { "" };
            Ok(Value::String(format!(
                "{}{:.*} {}",
                sign, decimals, value, units[unit_idx]
            )))
        },
        "if" => |args| {
            let cond = get_arg(&args, 0, "if")?.is_truthy();
            let yes = get_arg(&args, 1, "if")?.clone();
            let no = args.get(2).cloned().unwrap_or(Value::None);
            Ok(if cond { yes } else { no })
        },
        "ifnone" => |args| {
            let val = get_arg(&args, 0, "ifnone")?;
            if val.is_none() {
                Ok(args.get(1).cloned().unwrap_or(Value::None))
            } else {
                Ok(val.clone())
            }
        },
        "default" => |args| {
            let val = get_arg(&args, 0, "default")?;
            if val.is_none() || matches!(val, Value::String(s) if s.is_empty()) {
                Ok(args.get(1).cloned().unwrap_or(Value::None))
            } else {
                Ok(val.clone())
            }
        },
        "coalesce" => |args| {
            for arg in &args {
                if !arg.is_none() {
                    return Ok(arg.clone());
                }
            }
            Ok(Value::None)
        },
        "choose" => |args| {
            let idx = arg_f64(&args, 0, "choose")? as usize;
            args.get(idx + 1)
                .cloned()
                .ok_or(FunctionError::IndexOutOfBounds {
                    index: idx,
                    len: args.len().saturating_sub(1),
                    function: "choose",
                })
        },
        "switch" => |args| {
            let val = get_arg(&args, 0, "switch")?;
            let pairs = &args[1..];
            let mut i = 0;
            while i + 1 < pairs.len() {
                if val == &pairs[i] {
                    return Ok(pairs[i + 1].clone());
                }
                i += 2;
            }
            if pairs.len() % 2 == 1 {
                Ok(pairs.last().cloned().unwrap_or(Value::None))
            } else {
                Ok(Value::None)
            }
        },
        "map" => |args| {
            let val = arg_f64(&args, 0, "map")?;
            let in_min = arg_f64(&args, 1, "map")?;
            let in_max = arg_f64(&args, 2, "map")?;
            let out_min = arg_f64(&args, 3, "map")?;
            let out_max = arg_f64(&args, 4, "map")?;
            let t = (val - in_min) / (in_max - in_min);
            Ok(Value::F64(out_min + t * (out_max - out_min)))
        },
        "duration" => |args| {
            let secs = arg_time_seconds(args.as_slice(), 0, "duration")?;
            Ok(Value::String(format_duration(secs)))
        },
        "elapsed" => |args| {
            let timestamp = arg_time_seconds(args.as_slice(), 0, "elapsed")?;

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| FunctionError::ConversionFailed {
                    from: "system time",
                    to: "unix timestamp",
                    function: "elapsed",
                })?
                .as_secs_f64();

            let elapsed = now - timestamp;

            let result = if elapsed >= 0.0 {
                format!("{} ago", format_elapsed(elapsed))
            } else {
                format!("in {}", format_elapsed(-elapsed))
            };

            Ok(Value::String(result))
        },
        "fancy_time" => |args| {
            let secs = arg_time_seconds(&args, 0, "fancy_time")?;
            let day_seconds = arg_f64(&args, 2, "fancy_time")?;
            let max_parts = arg_i64(&args, 3, "fancy_time")? as usize;
            let use_day_seconds = arg_bool(&args, 4, "fancy_time")?;
            Ok(Value::String(format_custom_time(
                secs,
                day_seconds,
                max_parts,
                use_day_seconds,
            )))
        },
        "datetime" => |args| {
            let timestamp = arg_time_seconds(&args, 0, "datetime")?;

            let dt = DateTime::<Utc>::from_timestamp(
                timestamp as i64,
                ((timestamp.fract().abs()) * 1_000_000_000.0) as u32,
            )
            .ok_or(FunctionError::ConversionFailed {
                from: "unix timestamp",
                to: "datetime",
                function: "datetime",
            })?;

            Ok(Value::String(
                dt.format("%A, %d %B %Y, %H:%M:%S UTC").to_string(),
            ))
        },
        "debug" => |args| {
            let val = get_arg(&args, 0, "debug")?;
            Ok(Value::String(format!("{:?}", val)))
        },
        "typeof" => |args| {
            let val = get_arg(&args, 0, "typeof")?;
            Ok(Value::String(val.type_name().to_string()))
        },
        "defined" => |args| {
            Ok(Value::Bool(
                !args.first().map(|v| v.is_none()).unwrap_or(true),
            ))
        },
        "empty" => |args| match args.first() {
            Some(Value::String(s)) => Ok(Value::Bool(s.is_empty())),
            Some(Value::Array(arr)) => Ok(Value::Bool(arr.is_empty())),
            Some(Value::None) => Ok(Value::Bool(true)),
            _ => Ok(Value::Bool(false)),
        },
        "is_some" => |args| {
            //println!("In is_some(xyz): {:?}", args.first());
            match args.first() {
                Some(v) => Ok(Value::Bool(value_exists(v))),
                _ => Err(FunctionError::MissingArgument {
                    index: 0,
                    function: "is_some",
                }),
            }
        },
        "hsv_to_rgb" => |args| match args.first() {
            Some(Value::Array(hsv)) => {
                if hsv.len() < 3 {
                    return Err(FunctionError::InvalidValue {
                        message: "expected array of 3 elements [h, s, v]".to_string(),
                        function: "hsv_to_rgb",
                    });
                }
                let hsv_obj = HSV {
                    h: hsv[0].as_f64().ok_or(FunctionError::ConversionFailed {
                        from: hsv[0].type_name(),
                        to: "f32",
                        function: "hsv_to_rgb",
                    })? as f32,
                    s: hsv[1].as_f64().ok_or(FunctionError::ConversionFailed {
                        from: hsv[1].type_name(),
                        to: "f32",
                        function: "hsv_to_rgb",
                    })? as f32,
                    v: hsv[2].as_f64().ok_or(FunctionError::ConversionFailed {
                        from: hsv[2].type_name(),
                        to: "f32",
                        function: "hsv_to_rgb",
                    })? as f32,
                };
                Ok(Value::from_vec(hsv_to_rgb(hsv_obj)))
            }
            Some(Value::F64(hue)) => {
                let h = *hue as f32;
                let s = arg_f64(&args, 1, "hsv_to_rgb")? as f32;
                let v = arg_f64(&args, 2, "hsv_to_rgb")? as f32;
                let hsv_obj = HSV { h, s, v };
                Ok(Value::from_vec(hsv_to_rgb(hsv_obj)))
            }
            Some(v) => Err(FunctionError::InvalidType {
                expected: "array or number",
                got: v.type_name(),
                function: "hsv_to_rgb",
            }),
            None => Err(FunctionError::MissingArgument {
                index: 0,
                function: "hsv_to_rgb",
            }),
        },
        "random" => |args| {
            let mut rng = ThreadRng::default();
            match args.first() {
                Some(Value::Array(array)) => {
                    if array.len() == 2 {
                        match (&array[0], &array[1]) {
                            (Value::F64(min), Value::F64(max)) => {
                                Ok(Value::F64(rng.random_range(*min..=*max)))
                            }
                            (Value::I64(min), Value::I64(max)) => {
                                Ok(Value::I64(rng.random_range(*min..=*max)))
                            }
                            _ => Err(FunctionError::InvalidType {
                                expected: "matching numeric pair",
                                got: "mixed types",
                                function: "random",
                            }),
                        }
                    } else if !array.is_empty() {
                        let idx = rng.random_range(0..array.len());
                        Ok(array[idx].clone())
                    } else {
                        Err(FunctionError::EmptyInput { function: "random" })
                    }
                }
                Some(Value::F64(first)) => match args.get(1) {
                    Some(Value::F64(second)) => Ok(Value::F64(rng.random_range(*first..=*second))),
                    Some(Value::I64(second)) => {
                        Ok(Value::F64(rng.random_range(*first..=*second as f64)))
                    }
                    _ => Ok(Value::F64(rng.random_range(0.0..=*first))),
                },
                Some(Value::I64(first)) => match args.get(1) {
                    Some(Value::F64(second)) => {
                        Ok(Value::F64(rng.random_range(*first as f64..=*second)))
                    }
                    Some(Value::I64(second)) => Ok(Value::I64(rng.random_range(*first..=*second))),
                    _ => Ok(Value::I64(rng.random_range(0..=*first))),
                },
                Some(Value::String(string)) => {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    string.hash(&mut hasher);
                    Ok(Value::F64((hasher.finish() % 1000000) as f64 / 1000000.0))
                }
                Some(Value::Bool(_)) => Ok(Value::Bool(rng.random())),
                Some(Value::None) => Ok(Value::F64(rng.random())),
                None => Ok(Value::F64(rng.random())),
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
    ErrorInFunction {
        name: String,
        args: Vec<Value>,
        function_error: FunctionError,
    },
}

#[derive(Debug, Clone)]
pub enum AnnoyingError {
    DivisionByZero { pos: usize },
    NoneValue { pos: usize },
    OverflowWarning { pos: usize },
}

use crate::renderer::ui_text_rendering::anchor_to;
use crate::ui::actions::ElementContext;
use crate::ui::ui_edit_manager::ColorComponent;
use crate::ui::ui_editor::Menus;
use crate::ui::ui_touch_manager::{ElementRef, Touchable};
use owo_colors::OwoColorize;

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
                    "{} at position {}: expected {}, but found {:?}",
                    "Parse error".bright_red().bold(),
                    pos.bright_yellow(),
                    expected.bright_blue(),
                    found
                )
            }
            Self::UnexpectedEnd { expected } => {
                write!(
                    f,
                    "{}: unexpected end of input, expected {}",
                    "Parse error".bright_red().bold(),
                    expected.bright_blue()
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
                    "{} at position {}: '{}' requires {}, but got {}",
                    "Type error".bright_red().bold(),
                    pos.bright_yellow(),
                    operation.bright_blue(),
                    expected.bright_blue(),
                    found.bright_magenta()
                )
            }
            Self::UndefinedVariable { name, pos } => {
                write!(
                    f,
                    "{} at position {}: undefined variable '{}'",
                    "Reference error".bright_red().bold(),
                    pos.bright_yellow(),
                    name.bright_blue()
                )
            }
            Self::UndefinedFunction { name, pos } => {
                write!(
                    f,
                    "{} at position {}: undefined function '{}'",
                    "Reference error".bright_red().bold(),
                    pos.bright_yellow(),
                    name.bright_blue()
                )
            }
            Self::InvalidPropertyAccess {
                property,
                on_type,
                pos,
            } => {
                write!(
                    f,
                    "{} at position {}: type '{}' has no property '{}'",
                    "Property error".bright_red().bold(),
                    pos.bright_yellow(),
                    on_type.bright_magenta(),
                    property.bright_blue()
                )
            }
            Self::InvalidIndexAccess {
                index_type,
                on_type,
                pos,
            } => {
                write!(
                    f,
                    "{} at position {}: cannot index '{}' with '{}'",
                    "Index error".bright_red().bold(),
                    pos.bright_yellow(),
                    on_type.bright_magenta(),
                    index_type.bright_magenta()
                )
            }
            Self::ErrorInFunction {
                name,
                args,
                function_error,
            } => {
                write!(
                    f,
                    "{} in function '{}' with args '{:?}': {:?}",
                    "Function error".bright_red().bold(),
                    name.bright_blue(),
                    args,
                    function_error
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
            Self::NoneValue { pos } => {
                write!(
                    f,
                    "Warning at position {}: none value used in operation",
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

// fn get_var(variables: &Variables, settings: &Settings, menus: &Menus, element_ctx: &ElementContext, name: &str) -> Value {
//     if let Some(key) = SettingKey::from_str(name) {
//         return settings.read_setting(key).to_value(); // TODO: setting.options!! Is missing, that's what I wanted to say, insecure mind, lizard brain...- HUH?!
//     }
//     Value::load_variable(variables, menus, element_ctx, name)
// }
fn get_var_opt(
    variables: &Variables,
    settings: &Settings,
    menus: &Menus,
    element_ctx: &ElementContext,
    name: &str,
) -> Option<Value> {
    if let Some(key) = SettingKey::from_str(name) {
        return Some(settings.read_setting(key).to_value());
    }

    Value::load_variable(variables, menus, element_ctx, name)
}

fn combine_array_values<F>(left: Vec<Value>, right: Vec<Value>, mut op: F) -> Option<Vec<Value>>
where
    F: FnMut(Value, Value) -> Option<Value>,
{
    let min_len = left.len().min(right.len());
    let mut out = Vec::with_capacity(left.len().max(right.len()));

    for i in 0..min_len {
        out.push(op(left[i].clone(), right[i].clone())?);
    }

    if left.len() > right.len() {
        out.extend(left[min_len..].iter().cloned());
    } else if right.len() > left.len() {
        out.extend(right[min_len..].iter().cloned());
    }

    Some(out)
}

fn map_array_values<F>(array: Vec<Value>, value: Value, mut op: F) -> Option<Vec<Value>>
where
    F: FnMut(Value, Value) -> Option<Value>,
{
    array
        .into_iter()
        .map(|element| op(element, value.clone()))
        .collect()
}

fn map_array_values_right<F>(value: Value, array: Vec<Value>, mut op: F) -> Option<Vec<Value>>
where
    F: FnMut(Value, Value) -> Option<Value>,
{
    array
        .into_iter()
        .map(|element| op(value.clone(), element))
        .collect()
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
        (Value::Array(x), Value::Array(y)) => {
            combine_array_values(x, y, add_values).map(Value::Array)
        }
        (Value::Array(x), v) => map_array_values(x, v, add_values).map(Value::Array),
        (v, Value::Array(y)) => map_array_values_right(v, y, add_values).map(Value::Array),
        (Value::String(s), Value::None) => Some(Value::String(s)),
        (Value::None, Value::String(s)) => Some(Value::String(s)),
        _ => None,
    }
}

fn sub_values(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => Some(Value::F64(x - y)),
        (Value::I64(x), Value::F64(y)) => Some(Value::F64(x as f64 - y)),
        (Value::F64(x), Value::I64(y)) => Some(Value::F64(x - y as f64)),
        (Value::I64(x), Value::I64(y)) => Some(Value::I64(x - y)),
        (Value::Array(x), Value::Array(y)) => {
            combine_array_values(x, y, sub_values).map(Value::Array)
        }
        (Value::Array(x), v) => map_array_values(x, v, sub_values).map(Value::Array),
        (v, Value::Array(y)) => map_array_values_right(v, y, sub_values).map(Value::Array),
        _ => None,
    }
}

fn multiply_values(l: Value, r: Value) -> Option<Value> {
    match (&l, &r) {
        (Value::Array(x), Value::Array(y)) => {
            combine_array_values(x.clone(), y.clone(), multiply_values).map(Value::Array)
        }
        (Value::Array(x), _) => map_array_values(x.clone(), r, multiply_values).map(Value::Array),
        (_, Value::Array(y)) => {
            map_array_values_right(l, y.clone(), multiply_values).map(Value::Array)
        }
        (Value::String(s), Value::I64(n)) => Some(Value::String(s.repeat(*n as usize))),
        (Value::I64(n), Value::String(s)) => Some(Value::String(s.repeat(*n as usize))),
        _ => Some(Value::F64(l.as_f64()? * r.as_f64()?)),
    }
}

fn get_property(value: &Value, prop: &str) -> Option<Value> {
    match prop {
        "is_some" => return Some(Value::Bool(value_exists(value))),
        _ => {}
    }
    match value {
        Value::String(s) => string_property(s, prop),
        Value::Array(arr) => array_property(arr, prop),
        Value::F64(n) => float_property(*n, prop),
        Value::I64(n) => int_property(*n, prop),
        _ => None,
    }
}
fn value_exists(value: &Value) -> bool {
    //println!("In xyz.is_some: {}", value);
    match value {
        Value::None => false,
        Value::F64(_) => true,
        Value::I64(_) => true,
        Value::Bool(_) => true,
        Value::String(_) => true,
        Value::Array(_) => true,
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

static PRINTED_PARSE_ERRORS: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new(); // The hash of the expression String!

fn printed_parse_errors() -> &'static Mutex<HashSet<u64>> {
    PRINTED_PARSE_ERRORS.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn resolve_template(
    template: &str,
    vars: &Variables,
    settings: &Settings,
    menus: &Menus,
    element_ctx: &ElementContext,
) -> String {
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
                    Value::from_str(settings, vars, menus, element_ctx, inside.trim()).into_string()
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

#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    Number(f64),
    Ident(String),
    StrLit(String),
    True,
    False,
    None,
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

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Array(Box<Type>),
    VarOrSetting(String),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64, Span),
    String(String, Span),
    Bool(bool, Span),
    None(Span),
    VarOrSetting(String, Span),
    Array(Vec<Expr>, Span),
    Binary {
        left: Box<Expr>,
        operator: TokenKind,
        right: Box<Expr>,
        span: Span,
    },
    Unary {
        operator: TokenKind,
        expr: Box<Expr>,
        span: Span,
    },
    FunctionCall {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    Index {
        value: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Property {
        value: Box<Expr>,
        property: String,
        span: Span,
    },
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        span: Span,
    },
    Format {
        value: Box<Expr>,
        precision: Box<Expr>,
        span: Span,
    },
}

fn get_span(expr: &Expr) -> Span {
    match expr {
        Expr::Number(_, s)
        | Expr::String(_, s)
        | Expr::Bool(_, s)
        | Expr::None(s)
        | Expr::VarOrSetting(_, s)
        | Expr::Array(_, s) => s.clone(),
        Expr::Binary { span, .. }
        | Expr::Unary { span, .. }
        | Expr::FunctionCall { span, .. }
        | Expr::Index { span, .. }
        | Expr::Property { span, .. }
        | Expr::Ternary { span, .. }
        | Expr::Format { span, .. } => span.clone(),
    }
}

pub fn tokenize_expr(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();

    while let Some(&(i, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c == '"' || c == '\'' || c == '`' {
            let start = i;
            let quote = c;
            chars.next();
            let mut s = String::new();
            while let Some(&(j, d)) = chars.peek() {
                chars.next();
                if d == quote {
                    let end = j + 1;
                    tokens.push(Token {
                        kind: TokenKind::StrLit(s),
                        span: Span { start, end },
                    });
                    break;
                } else if d == '\\' {
                    if let Some(&(_, e)) = chars.peek() {
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
            continue;
        }

        let prev_is_operand = matches!(
            tokens.last().map(|t| &t.kind),
            Some(TokenKind::Number(_))
                | Some(TokenKind::Ident(_))
                | Some(TokenKind::RParen)
                | Some(TokenKind::RBracket)
                | Some(TokenKind::StrLit(_))
        );

        if c.is_ascii_digit()
            || (c == '.'
                && !prev_is_operand
                && !matches!(tokens.last().map(|t| &t.kind), Some(TokenKind::Colon))
                && chars
                    .clone()
                    .nth(1)
                    .map_or(false, |(_, n)| n.is_ascii_digit()))
        {
            let start = i;
            let mut s = String::new();
            let mut has_dot = false;
            let mut has_exp = false;

            if c == '0' {
                s.push(c);
                chars.next();
                if let Some(&(_, next)) = chars.peek() {
                    match next {
                        'x' | 'X' => {
                            s.push(next);
                            chars.next();
                            while let Some(&(_, d)) = chars.peek() {
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
                                let end = i + s.len();
                                tokens.push(Token {
                                    kind: TokenKind::Number(v as f64),
                                    span: Span { start, end },
                                });
                            }
                            continue;
                        }
                        'b' | 'B' => {
                            s.push(next);
                            chars.next();
                            while let Some(&(_, d)) = chars.peek() {
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
                                let end = i + s.len();
                                tokens.push(Token {
                                    kind: TokenKind::Number(v as f64),
                                    span: Span { start, end },
                                });
                            }
                            continue;
                        }
                        'o' | 'O' => {
                            s.push(next);
                            chars.next();
                            while let Some(&(_, d)) = chars.peek() {
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
                                let end = i + s.len();
                                tokens.push(Token {
                                    kind: TokenKind::Number(v as f64),
                                    span: Span { start, end },
                                });
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

            while let Some(&(_, d)) = chars.peek() {
                if d.is_ascii_digit() || d == '_' {
                    if d != '_' {
                        s.push(d);
                    }
                    chars.next();
                } else if d == '.' && !has_dot && !has_exp {
                    let mut peek_chars = chars.clone();
                    peek_chars.next();
                    if peek_chars.peek().map(|&(_, c)| c) == Some('.') {
                        break;
                    }
                    has_dot = true;
                    s.push(d);
                    chars.next();
                } else if (d == 'e' || d == 'E') && !has_exp {
                    has_exp = true;
                    s.push(d);
                    chars.next();
                    if let Some(&(_, sign)) = chars.peek() {
                        if sign == '+' || sign == '-' {
                            s.push(sign);
                            chars.next();
                        }
                    }
                } else {
                    break;
                }
            }
            let end = i + s.len();
            if let Ok(v) = s.parse() {
                tokens.push(Token {
                    kind: TokenKind::Number(v),
                    span: Span { start, end },
                });
            }
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            let mut s = String::new();
            while let Some(&(_, d)) = chars.peek() {
                if d.is_alphanumeric() || d == '_' {
                    s.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            let end = i + s.len();
            let kind = match s.to_ascii_lowercase().as_str() {
                "true" => TokenKind::True,
                "false" => TokenKind::False,
                "none" | "null" => TokenKind::None,
                _ => TokenKind::Ident(s),
            };
            tokens.push(Token {
                kind,
                span: Span { start, end },
            });
            continue;
        }

        let start = i;
        chars.next();
        let mut end = i + 1;
        let kind = match c {
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => {
                if chars.peek().map(|&(_, c)| c) == Some('*') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Power
                } else {
                    TokenKind::Star
                }
            }
            '/' => TokenKind::Slash,
            '%' => TokenKind::Percent,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            ',' => TokenKind::Comma,
            '$' => TokenKind::Dollar,
            '~' => TokenKind::BitNot,
            '?' => {
                if chars.peek().map(|&(_, c)| c) == Some('?') {
                    chars.next();
                    end = i + 2;
                    TokenKind::NullCoalesce
                } else if chars.peek().map(|&(_, c)| c) == Some('.') {
                    chars.next();
                    end = i + 2;
                    TokenKind::OptChain
                } else {
                    TokenKind::Question
                }
            }
            ':' => TokenKind::Colon,
            '.' => {
                if chars.peek().map(|&(_, c)| c) == Some('.') {
                    chars.next();
                    if chars.peek().map(|&(_, c)| c) == Some('=') {
                        chars.next();
                        end = i + 3;
                        TokenKind::DotDotEq
                    } else {
                        end = i + 2;
                        TokenKind::DotDot
                    }
                } else {
                    TokenKind::Dot
                }
            }
            '!' => {
                if chars.peek().map(|&(_, c)| c) == Some('=') {
                    chars.next();
                    if chars.peek().map(|&(_, c)| c) == Some('=') {
                        chars.next();
                        end = i + 3;
                        TokenKind::StrictNeq
                    } else {
                        end = i + 2;
                        TokenKind::Neq
                    }
                } else {
                    TokenKind::Not
                }
            }
            '=' => {
                if chars.peek().map(|&(_, c)| c) == Some('=') {
                    chars.next();
                    if chars.peek().map(|&(_, c)| c) == Some('=') {
                        chars.next();
                        end = i + 3;
                        TokenKind::StrictEq
                    } else {
                        end = i + 2;
                        TokenKind::Eq
                    }
                } else {
                    return Err(ParseError::UnexpectedToken {
                        expected: "==".to_string(),
                        found: Token {
                            kind: TokenKind::Ident(c.to_string()),
                            span: Span { start, end },
                        },
                        pos: start,
                    });
                }
            }
            '<' => {
                if chars.peek().map(|&(_, c)| c) == Some('=') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Le
                } else if chars.peek().map(|&(_, c)| c) == Some('<') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Shl
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                if chars.peek().map(|&(_, c)| c) == Some('=') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Ge
                } else if chars.peek().map(|&(_, c)| c) == Some('>') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Shr
                } else {
                    TokenKind::Gt
                }
            }
            '&' => {
                if chars.peek().map(|&(_, c)| c) == Some('&') {
                    chars.next();
                    end = i + 2;
                    TokenKind::And
                } else {
                    TokenKind::BitAnd
                }
            }
            '|' => {
                if chars.peek().map(|&(_, c)| c) == Some('|') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Or
                } else if chars.peek().map(|&(_, c)| c) == Some('>') {
                    chars.next();
                    end = i + 2;
                    TokenKind::Pipe
                } else {
                    TokenKind::BitOr
                }
            }
            '^' => TokenKind::BitXor,
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "valid character".to_string(),
                    found: Token {
                        kind: TokenKind::Ident(c.to_string()),
                        span: Span { start, end },
                    },
                    pos: start,
                });
            }
        };
        tokens.push(Token {
            kind,
            span: Span { start, end },
        });
    }

    tokens.push(Token {
        kind: TokenKind::End,
        span: Span {
            start: input.len(),
            end: input.len(),
        },
    });
    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token {
            kind: TokenKind::End,
            span: Span { start: 0, end: 0 },
        })
    }
    fn peek_n(&self, n: usize) -> Token {
        self.tokens.get(self.pos + n).cloned().unwrap_or(Token {
            kind: TokenKind::End,
            span: Span { start: 0, end: 0 },
        })
    }
    pub fn parse(&mut self) -> ParseResult<Expr> {
        let value = self.parse_expr_bp(0)?;
        if self.peek().kind != TokenKind::End {
            return Err(ParseError::UnexpectedToken {
                expected: "end of expression".to_string(),
                found: self.peek(),
                pos: self.peek().span.start,
            });
        }
        Ok(value)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<Expr> {
        let mut left = self.parse_primary()?;
        let left_span = get_span(&left);

        loop {
            let op = self.peek();

            // If it's a Colon NOT followed by a Dot, it's a ternary colon, so break
            // and let the Question handler consume it without advancing the position.
            if op.kind == TokenKind::Colon && self.peek_n(1).kind != TokenKind::Dot {
                break;
            }

            let (l_bp, r_bp) = match op.kind {
                TokenKind::Pipe => (1, 2),
                TokenKind::Question => (2, 0),
                TokenKind::Colon => (16, 17),
                TokenKind::NullCoalesce => (3, 4),
                TokenKind::Or => (4, 5),
                TokenKind::And => (5, 6),
                TokenKind::BitOr => (6, 7),
                TokenKind::BitXor => (7, 8),
                TokenKind::BitAnd => (8, 9),
                TokenKind::Eq | TokenKind::Neq | TokenKind::StrictEq | TokenKind::StrictNeq => {
                    (9, 10)
                }
                TokenKind::Lt | TokenKind::Gt | TokenKind::Le | TokenKind::Ge => (10, 11),
                TokenKind::Shl | TokenKind::Shr => (11, 12),
                TokenKind::Plus | TokenKind::Minus => (12, 13),
                TokenKind::Star | TokenKind::Slash | TokenKind::Percent => (13, 14),
                TokenKind::Power => (15, 14),
                TokenKind::Dot => (16, 17),
                _ => break,
            };

            if l_bp < min_bp {
                break;
            }

            self.pos += 1;

            if op.kind == TokenKind::Question {
                let yes = self.parse_expr_bp(0)?;
                if self.peek().kind != TokenKind::Colon {
                    return Err(ParseError::UnexpectedToken {
                        expected: ":".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }
                self.pos += 1;

                let no = self.parse_expr_bp(2)?;
                let span = Span {
                    start: left_span.start,
                    end: get_span(&no).end,
                };
                left = Expr::Ternary {
                    cond: Box::new(left),
                    then_expr: Box::new(yes),
                    else_expr: Box::new(no),
                    span,
                };
                continue;
            }

            if op.kind == TokenKind::Pipe {
                let name = match self.peek().kind {
                    TokenKind::Ident(s) => {
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

                if self.peek().kind == TokenKind::LParen {
                    self.pos += 1;
                    if self.peek().kind != TokenKind::RParen {
                        loop {
                            args.push(self.parse_expr_bp(0)?);
                            match self.peek().kind {
                                TokenKind::Comma => {
                                    self.pos += 1;
                                }
                                TokenKind::RParen => break,
                                _ => {
                                    return Err(ParseError::UnexpectedToken {
                                        expected: "',' or ')'".to_string(),
                                        found: self.peek(),
                                        pos: self.pos,
                                    });
                                }
                            }
                        }
                    }
                    self.pos += 1;
                }
                let span = Span {
                    start: left_span.start,
                    end: self.peek().span.start,
                };
                left = Expr::FunctionCall { name, args, span };
                continue;
            }

            if op.kind == TokenKind::Colon {
                // We already checked above that it's followed by a Dot
                self.pos += 1; // consume the Dot
                let precision = self.parse_expr_bp(17)?;
                let span = Span {
                    start: left_span.start,
                    end: get_span(&precision).end,
                };
                left = Expr::Format {
                    value: Box::new(left),
                    precision: Box::new(precision),
                    span,
                };
                continue;
            }

            if op.kind == TokenKind::Dot {
                let prop_tok = self.peek();
                self.pos += 1;

                match prop_tok.kind {
                    TokenKind::Ident(name) => {
                        if self.peek().kind == TokenKind::LParen {
                            self.pos += 1;
                            let mut args = vec![left.clone()];
                            if self.peek().kind != TokenKind::RParen {
                                loop {
                                    args.push(self.parse_expr_bp(0)?);
                                    match self.peek().kind {
                                        TokenKind::Comma => {
                                            self.pos += 1;
                                        }
                                        TokenKind::RParen => break,
                                        _ => {
                                            return Err(ParseError::UnexpectedToken {
                                                expected: "',' or ')'".to_string(),
                                                found: self.peek(),
                                                pos: self.pos,
                                            });
                                        }
                                    }
                                }
                            }
                            self.pos += 1;
                            let span = Span {
                                start: left_span.start,
                                end: self.peek().span.start,
                            };
                            left = Expr::FunctionCall { name, args, span };
                        } else {
                            let span = Span {
                                start: left_span.start,
                                end: prop_tok.span.end,
                            };
                            left = Expr::Property {
                                value: Box::new(left),
                                property: name,
                                span,
                            };
                        }
                    }
                    TokenKind::Number(n) => {
                        let span = Span {
                            start: left_span.start,
                            end: prop_tok.span.end,
                        };
                        left = Expr::Index {
                            value: Box::new(left),
                            index: Box::new(Expr::Number(n, prop_tok.span)),
                            span,
                        };
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "identifier or numeric index".to_string(),
                            found: prop_tok.clone(),
                            pos: self.pos - 1,
                        });
                    }
                }
            } else {
                let right = self.parse_expr_bp(r_bp)?;
                let span = Span {
                    start: left_span.start,
                    end: get_span(&right).end,
                };
                left = Expr::Binary {
                    left: Box::new(left),
                    operator: op.kind,
                    right: Box::new(right),
                    span,
                };
            }
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let tok = self.peek();
        self.pos += 1;
        let pos = tok.span.start;

        match &tok.kind {
            TokenKind::Number(n) => Ok(Expr::Number(*n, tok.span)),
            TokenKind::StrLit(s) => Ok(Expr::String(s.clone(), tok.span)),
            TokenKind::True => Ok(Expr::Bool(true, tok.span)),
            TokenKind::False => Ok(Expr::Bool(false, tok.span)),
            TokenKind::None => Ok(Expr::None(tok.span)),

            TokenKind::Minus | TokenKind::Not | TokenKind::BitNot => {
                let expr = self.parse_expr_bp(15)?;
                let span = Span {
                    start: pos,
                    end: get_span(&expr).end,
                };
                Ok(Expr::Unary {
                    operator: tok.kind,
                    expr: Box::new(expr),
                    span,
                })
            }

            TokenKind::LBrace => {
                let next = self.peek();
                if let TokenKind::Ident(mut name) = next.kind {
                    self.pos += 1;

                    while self.peek().kind == TokenKind::Dot {
                        self.pos += 1;
                        match self.peek().kind {
                            TokenKind::Ident(prop) => {
                                name.push('.');
                                name.push_str(&prop);
                                self.pos += 1;
                            }
                            TokenKind::Number(n) => {
                                name.push_str(&format!(".{}", n as i64));
                                self.pos += 1;
                            }
                            _ => {
                                return Err(ParseError::UnexpectedToken {
                                    expected: "identifier or number after '.'".to_string(),
                                    found: self.peek(),
                                    pos: self.pos,
                                });
                            }
                        }
                    }
                    // Optional inline default: {name ?? default_expr}
                    let default = if self.peek().kind == TokenKind::NullCoalesce {
                        self.pos += 1;
                        Some(self.parse_expr_bp(4)?)
                    } else {
                        None
                    };

                    if self.peek().kind != TokenKind::RBrace {
                        return Err(ParseError::UnexpectedToken {
                            expected: "}".to_string(),
                            found: self.peek(),
                            pos: self.pos,
                        });
                    }
                    let end = self.peek().span.end;
                    self.pos += 1;

                    let var_expr = Expr::VarOrSetting(name, Span { start: pos, end });
                    Ok(match default {
                        Some(default_expr) => Expr::Binary {
                            left: Box::new(var_expr),
                            operator: TokenKind::NullCoalesce,
                            right: Box::new(default_expr),
                            span: Span { start: pos, end },
                        },
                        None => var_expr,
                    })
                } else {
                    Err(ParseError::UnexpectedToken {
                        expected: "identifier inside braces for variable/setting access"
                            .to_string(),
                        found: next.clone(),
                        pos: next.span.start,
                    })
                }
            }

            TokenKind::LParen => {
                let v = self.parse_expr_bp(0)?;
                if self.peek().kind != TokenKind::RParen {
                    return Err(ParseError::UnexpectedToken {
                        expected: ")".to_string(),
                        found: self.peek(),
                        pos: self.pos,
                    });
                }
                self.pos += 1;
                Ok(v)
            }

            TokenKind::LBracket => {
                let mut items = Vec::new();
                let start = pos;

                if self.peek().kind != TokenKind::RBracket {
                    loop {
                        let item_tok = self.peek();

                        // Accept bare identifiers inside arrays as string literals
                        let item = if let TokenKind::Ident(name) = &item_tok.kind {
                            self.pos += 1;
                            Expr::String(name.clone(), item_tok.span.clone())
                        } else {
                            self.parse_expr_bp(0)?
                        };

                        if self.peek().kind == TokenKind::DotDot
                            || self.peek().kind == TokenKind::DotDotEq
                        {
                            let inclusive = self.peek().kind == TokenKind::DotDotEq;
                            self.pos += 1;

                            let end_val = self.parse_expr_bp(0)?;
                            let span = Span {
                                start: get_span(&item).start,
                                end: get_span(&end_val).end,
                            };
                            items.push(Expr::Binary {
                                left: Box::new(item),
                                operator: if inclusive {
                                    TokenKind::DotDotEq
                                } else {
                                    TokenKind::DotDot
                                },
                                right: Box::new(end_val),
                                span,
                            });
                        } else {
                            items.push(item);
                        }

                        match self.peek().kind {
                            TokenKind::Comma => {
                                self.pos += 1;
                            }
                            TokenKind::RBracket => break,
                            _ => {
                                return Err(ParseError::UnexpectedToken {
                                    expected: "',' or ']'".to_string(),
                                    found: self.peek(),
                                    pos: self.pos,
                                });
                            }
                        }
                    }
                }

                self.pos += 1;
                let end = self
                    .tokens
                    .get(self.pos - 1)
                    .map(|t| t.span.end)
                    .unwrap_or(pos);
                Ok(Expr::Array(items, Span { start, end }))
            }

            TokenKind::Ident(name) => {
                if self.peek().kind == TokenKind::LParen {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if self.peek().kind != TokenKind::RParen {
                        loop {
                            args.push(self.parse_expr_bp(0)?);
                            match self.peek().kind {
                                TokenKind::Comma => {
                                    self.pos += 1;
                                }
                                TokenKind::RParen => break,
                                _ => {
                                    return Err(ParseError::UnexpectedToken {
                                        expected: "',' or ')'".to_string(),
                                        found: self.peek(),
                                        pos: self.pos,
                                    });
                                }
                            }
                        }
                    }
                    self.pos += 1;
                    let end = self
                        .tokens
                        .get(self.pos - 1)
                        .map(|t| t.span.end)
                        .unwrap_or(pos);
                    Ok(Expr::FunctionCall {
                        name: name.clone(),
                        args,
                        span: Span { start: pos, end },
                    })
                } else {
                    Err(ParseError::UnexpectedToken {
                        expected: "string literal, variable {name}, or function call".to_string(),
                        found: tok.clone(),
                        pos,
                    })
                }
            }

            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: tok.clone(),
                pos,
            }),
        }
    }
}

pub fn type_check_and_resolve(
    expr: &Expr,
    vars: &Variables,
    settings: &Settings,
    menus: &Menus,
    element_ctx: &ElementContext,
) -> Result<Expr, ParseError> {
    match expr {
        Expr::VarOrSetting(name, span) => {
            if SettingKey::from_str(name).is_some()
                || Value::load_variable(vars, menus, element_ctx, name).is_some()
            {
                Ok(expr.clone())
            } else {
                Err(ParseError::UndefinedVariable {
                    name: name.clone(),
                    pos: span.start,
                })
            }
        }
        Expr::FunctionCall { name, args, span } => {
            if get_builtin(name).is_none() {
                return Err(ParseError::UndefinedFunction {
                    name: name.clone(),
                    pos: span.start,
                });
            }
            for arg in args {
                type_check_and_resolve(arg, vars, settings, menus, element_ctx)?;
            }
            Ok(expr.clone())
        }
        Expr::Binary {
            left,
            operator,
            right,
            ..
        } => {
            if matches!(operator, TokenKind::NullCoalesce) {
                type_check_and_resolve(right, vars, settings, menus, element_ctx)?; // default must be valid
                match type_check_and_resolve(left, vars, settings, menus, element_ctx) {
                    Ok(_) | Err(ParseError::UndefinedVariable { .. }) => {}
                    Err(e) => return Err(e), // other errors still propagate
                }
                return Ok(expr.clone());
            }
            type_check_and_resolve(left, vars, settings, menus, element_ctx)?;
            type_check_and_resolve(right, vars, settings, menus, element_ctx)?;
            Ok(expr.clone())
        }
        Expr::Unary {
            operator,
            expr,
            span,
        } => {
            let resolved = type_check_and_resolve(expr, vars, settings, menus, element_ctx)?;
            Ok(Expr::Unary {
                operator: operator.clone(),
                expr: Box::new(resolved),
                span: span.clone(),
            })
        }
        Expr::Index { value, index, .. } => {
            type_check_and_resolve(value, vars, settings, menus, element_ctx)?;
            type_check_and_resolve(index, vars, settings, menus, element_ctx)?;
            Ok(expr.clone())
        }
        Expr::Property { value, .. } => {
            type_check_and_resolve(value, vars, settings, menus, element_ctx)
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            type_check_and_resolve(cond, vars, settings, menus, element_ctx)?;
            type_check_and_resolve(then_expr, vars, settings, menus, element_ctx)?;
            type_check_and_resolve(else_expr, vars, settings, menus, element_ctx)?;
            Ok(expr.clone())
        }
        Expr::Format {
            value, precision, ..
        } => {
            type_check_and_resolve(value, vars, settings, menus, element_ctx)?;
            type_check_and_resolve(precision, vars, settings, menus, element_ctx)?;
            Ok(expr.clone())
        }
        Expr::Array(items, _) => {
            for item in items {
                type_check_and_resolve(item, vars, settings, menus, element_ctx)?;
            }
            Ok(expr.clone())
        }
        _ => Ok(expr.clone()),
    }
}

pub fn eval_ast(
    expr: &Expr,
    vars: &Variables,
    settings: &Settings,
    menus: &Menus,
    element_ctx: &ElementContext,
) -> Result<Value, ParseError> {
    match expr {
        Expr::Number(n, _) => Ok(Value::F64(*n)),
        Expr::String(s, _) => Ok(Value::String(s.clone())),
        Expr::Bool(b, _) => Ok(Value::Bool(*b)),
        Expr::None(_) => Ok(Value::None),
        Expr::VarOrSetting(name, span) => get_var_opt(vars, settings, menus, element_ctx, name)
            .ok_or(ParseError::UndefinedVariable {
                name: name.clone(),
                pos: span.start,
            }),
        Expr::Array(items, _) => {
            let mut vals = Vec::new();
            for item in items {
                vals.push(eval_ast(item, vars, settings, menus, element_ctx)?);
            }
            Ok(Value::Array(vals))
        }
        Expr::FunctionCall { name, args, span } => {
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(eval_ast(arg, vars, settings, menus, element_ctx)?);
            }
            let function = get_builtin(name).ok_or(ParseError::UndefinedFunction {
                name: name.clone(),
                pos: span.start,
            })?;
            function(arg_vals.clone()).map_err(|e| ParseError::ErrorInFunction {
                name: name.clone(),
                args: arg_vals,
                function_error: e,
            })
        }
        Expr::Binary {
            left,
            operator,
            right,
            span,
        } => {
            if matches!(operator, TokenKind::NullCoalesce) {
                match eval_ast(left, vars, settings, menus, element_ctx) {
                    Ok(l) if l.is_truthy() => Ok(l),
                    Ok(_) => eval_ast(right, vars, settings, menus, element_ctx),
                    Err(ParseError::UndefinedVariable { .. }) => {
                        eval_ast(right, vars, settings, menus, element_ctx)
                    }
                    Err(e) => Err(e),
                }
            } else {
                let l = eval_ast(left, vars, settings, menus, element_ctx)?;
                let r = eval_ast(right, vars, settings, menus, element_ctx)?;
                eval_binary(l, r, operator, span.start)
            }
        }
        Expr::Unary {
            operator,
            expr,
            span,
        } => {
            let val = eval_ast(expr, vars, settings, menus, element_ctx)?;
            match operator {
                TokenKind::Minus => Ok(Value::F64(-val.as_f64().ok_or(
                    ParseError::TypeMismatch {
                        operation: "negation".to_string(),
                        expected: "number".to_string(),
                        found: val.type_name().to_string(),
                        pos: span.start,
                    },
                )?)),
                TokenKind::Not => Ok(Value::Bool(!val.is_truthy())),
                TokenKind::BitNot => Ok(Value::I64(!val.as_i64().ok_or(
                    ParseError::TypeMismatch {
                        operation: "bitnot".to_string(),
                        expected: "integer".to_string(),
                        found: val.type_name().to_string(),
                        pos: span.start,
                    },
                )?)),
                _ => unreachable!(),
            }
        }
        Expr::Index { value, index, span } => {
            let val = eval_ast(value, vars, settings, menus, element_ctx)?;
            let idx = eval_ast(index, vars, settings, menus, element_ctx)?;
            get_index(&val, &idx).ok_or(ParseError::InvalidIndexAccess {
                index_type: idx.type_name().to_string(),
                on_type: val.type_name().to_string(),
                pos: span.start,
            })
        }
        Expr::Property {
            value,
            property,
            span,
        } => {
            let val = eval_ast(value, vars, settings, menus, element_ctx)?;
            if let Some(idx) = Variables::component_index(property) {
                get_index(&val, &Value::I64(idx as i64)).ok_or(ParseError::InvalidIndexAccess {
                    index_type: property.clone(),
                    on_type: val.type_name().to_string(),
                    pos: span.start,
                })
            } else {
                get_property(&val, property).ok_or(ParseError::InvalidPropertyAccess {
                    property: property.clone(),
                    on_type: val.type_name().to_string(),
                    pos: span.start,
                })
            }
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            let cond_val = eval_ast(cond, vars, settings, menus, element_ctx)?;
            if cond_val.is_truthy() {
                eval_ast(then_expr, vars, settings, menus, element_ctx)
            } else {
                eval_ast(else_expr, vars, settings, menus, element_ctx)
            }
        }
        Expr::Format {
            value,
            precision,
            span,
        } => {
            let val = eval_ast(value, vars, settings, menus, element_ctx)?;
            let prec_val = eval_ast(precision, vars, settings, menus, element_ctx)?;
            let n = val.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "format".to_string(),
                expected: "number".to_string(),
                found: val.type_name().to_string(),
                pos: span.start,
            })?;
            let p = prec_val.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "precision".to_string(),
                expected: "number".to_string(),
                found: prec_val.type_name().to_string(),
                pos: span.start,
            })? as usize;
            Ok(Value::String(format!("{:.*}", p, n)))
        }
    }
}

fn eval_binary(left: Value, right: Value, op: &TokenKind, pos: usize) -> ParseResult<Value> {
    match op {
        TokenKind::Plus => {
            add_values(left.clone(), right.clone()).ok_or(ParseError::TypeMismatch {
                operation: "addition".to_string(),
                expected: "number, string, or array".to_string(),
                found: format!("{} + {}", left.type_name(), right.type_name()),
                pos,
            })
        }
        TokenKind::Minus => {
            sub_values(left.clone(), right.clone()).ok_or(ParseError::TypeMismatch {
                operation: "subtraction".to_string(),
                expected: "number or array".to_string(),
                found: format!("{} - {}", left.type_name(), right.type_name()),
                pos,
            })
        }
        TokenKind::Star => {
            multiply_values(left.clone(), right.clone()).ok_or(ParseError::TypeMismatch {
                operation: "multiplication".to_string(),
                expected: "number, string, or array".to_string(),
                found: format!("{} * {}", left.type_name(), right.type_name()),
                pos,
            })
        }
        TokenKind::Slash | TokenKind::Percent => {
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
            Ok(Value::F64(if matches!(op, TokenKind::Slash) {
                a / b
            } else {
                a % b
            }))
        }
        TokenKind::Power => {
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
        TokenKind::Eq | TokenKind::StrictEq => Ok(Value::Bool(left == right)),
        TokenKind::Neq | TokenKind::StrictNeq => Ok(Value::Bool(left != right)),
        TokenKind::Lt | TokenKind::Gt | TokenKind::Le | TokenKind::Ge => {
            let a = left.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "comparison".to_string(),
                expected: "number".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let b = right.as_f64().ok_or(ParseError::TypeMismatch {
                operation: "comparison".to_string(),
                expected: "number".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            Ok(Value::Bool(match op {
                TokenKind::Lt => a < b,
                TokenKind::Gt => a > b,
                TokenKind::Le => a <= b,
                TokenKind::Ge => a >= b,
                _ => unreachable!(),
            }))
        }
        TokenKind::Shl | TokenKind::Shr => {
            let a = left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "shift".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let b = right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "shift".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            Ok(Value::I64(if matches!(op, TokenKind::Shl) {
                a << (b as u32)
            } else {
                a >> (b as u32)
            }))
        }
        TokenKind::BitAnd | TokenKind::BitOr | TokenKind::BitXor => {
            let a = left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitwise".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let b = right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "bitwise".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            Ok(Value::I64(match op {
                TokenKind::BitAnd => a & b,
                TokenKind::BitOr => a | b,
                TokenKind::BitXor => a ^ b,
                _ => unreachable!(),
            }))
        }
        TokenKind::And => Ok(Value::Bool(left.is_truthy() && right.is_truthy())),
        TokenKind::Or => Ok(Value::Bool(left.is_truthy() || right.is_truthy())),
        TokenKind::NullCoalesce => Ok(if left.is_truthy() { left } else { right }),
        TokenKind::DotDot | TokenKind::DotDotEq => {
            let start = left.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "range".to_string(),
                expected: "integer".to_string(),
                found: left.type_name().to_string(),
                pos,
            })?;
            let end = right.as_i64().ok_or(ParseError::TypeMismatch {
                operation: "range".to_string(),
                expected: "integer".to_string(),
                found: right.type_name().to_string(),
                pos,
            })?;
            let range: Box<dyn Iterator<Item = i64>> = if matches!(op, TokenKind::DotDotEq) {
                Box::new(start..=end)
            } else {
                Box::new(start..end)
            };
            Ok(Value::Array(range.map(|i| Value::F64(i as f64)).collect()))
        }
        _ => unreachable!(),
    }
}

pub fn eval_expr(
    expr: &str,
    vars: &Variables,
    settings: &Settings,
    menus: &Menus,
    element_ctx: &ElementContext,
) -> Option<Value> {
    let hasher = &mut DefaultHasher::new();
    hasher.write(expr.as_bytes());
    let expr_hash = hasher.finish();
    fn print_error(err: String, expr_hash: u64) {
        let dedup_parse_errors = true;

        if dedup_parse_errors {
            let mut printed = printed_parse_errors().lock().unwrap();
            if printed.insert(expr_hash) {
                eprintln!("{err}");
            }
        } else {
            eprintln!("{err}");
        }
    }

    let tokens = match tokenize_expr(expr) {
        Ok(t) => t,
        Err(e) => {
            if settings.print_parse_errors {
                print_error(format!("LexError for '{}': {}", expr, e), expr_hash);
            }
            return None;
        }
    };
    //println!("TOKENS: {tokens:?}");
    let mut parser = Parser::new(&tokens);
    let ast = match parser.parse() {
        Ok(a) => a,
        Err(e) => {
            if settings.print_parse_errors {
                print_error(format!("ParseError for '{}': {}", expr, e), expr_hash);
            }
            return None;
        }
    };
    //println!("AST: {ast:?}");
    let resolved_ast = match type_check_and_resolve(&ast, vars, settings, menus, element_ctx) {
        Ok(a) => a,
        Err(e) => {
            if settings.print_parse_errors {
                print_error(format!("TypeError for '{}': {}", expr, e), expr_hash);
            }
            return None;
        }
    };
    //println!("RESOLVED_AST: {resolved_ast:?}");
    match eval_ast(&resolved_ast, vars, settings, menus, element_ctx) {
        Ok(v) => Some(v),
        Err(e) => {
            if settings.print_parse_errors {
                print_error(format!("EvalError for '{}': {}", expr, e), expr_hash);
            }
            None
        }
    }
}
