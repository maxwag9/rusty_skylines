use crate::data::{SettingKey, Settings};
use crate::helpers::hsv::{HSV, hsv_to_rgb};
use crate::ui::variables::Variables;
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
    pub fn from_str(settings: &Settings, variables: &Variables, s: &str) -> Self {
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
            "true" | "yes" | "on" => return Value::Bool(true),
            "false" | "no" | "off" => return Value::Bool(false),
            _ => {}
        }

        // =========================
        // Auto-detect integer
        // =========================
        if let Ok(i) = s.parse::<i64>() {
            return Value::I64(i);
        }

        // =========================
        // Auto-detect float
        // =========================
        if let Ok(f) = s.parse::<f64>() {
            return Value::F64(f);
        }

        // =========================
        // Default: String
        // =========================
        Value::String(s.to_string())
    }
    pub fn from_vec<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<f64>,
    {
        Value::Array(iter.into_iter().map(|e| Value::F64(e.into())).collect())
    }
    fn load_variable(variables: &Variables, name: &String) -> Option<Value> {
        variables.get(name).cloned()
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

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }

    pub fn as_array(&self) -> Option<Vec<Value>> {
        match self {
            Value::Array(arr) => Some(arr.clone()),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::I64(i) => *i != 0,
            Value::F64(f) => *f != 0.0,
            Value::String(s) => !s.is_empty() && s != "false" && s != "0",
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

fn lookup_var(vars: &Variables, name: &str) -> Option<Value> {
    vars.get(name).cloned()
}

// ------------------------------------------------------------
// Lexer / tokens
// ------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    StrLit(String),
    True,
    False,
    Null,

    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent, // %
    Power,   // **

    // Bitwise
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    BitNot, // ~
    Shl,    // <<
    Shr,    // >>

    // Comparison
    Eq,  // ==
    Neq, // !=
    Lt,  // <
    Gt,  // >
    Le,  // <=
    Ge,  // >=

    // Strict equality
    StrictEq,  // ===
    StrictNeq, // !==

    // Logical
    And, // &&
    Or,  // ||
    Not, // !

    // Null coalescing
    NullCoalesce, // ??

    // Optional chaining (for future)
    OptChain, // ?.

    // Ternary
    Question,
    Colon,

    // Grouping & access
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,

    // Range
    DotDot,   // ..
    DotDotEq, // ..=

    // String interpolation
    Dollar, // $

    // Pipeline
    Pipe, // |>

    End,
}

//noinspection GrazieInspection
fn tokenize_expr(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '"' {
            chars.next();
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                chars.next();
                if d == '"' {
                    break;
                } else if d == '\\' {
                    // Escape sequences
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
                            'x' => {
                                // Hex escape \xNN
                                let mut hex = String::new();
                                for _ in 0..2 {
                                    if let Some(&h) = chars.peek() {
                                        if h.is_ascii_hexdigit() {
                                            hex.push(h);
                                            chars.next();
                                        }
                                    }
                                }
                                if let Ok(code) = u8::from_str_radix(&hex, 16) {
                                    s.push(code as char);
                                }
                            }
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
        } else if c == '\'' {
            chars.next();
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                chars.next();
                if d == '\'' {
                    break;
                } else if d == '\\' {
                    if let Some(&e) = chars.peek() {
                        chars.next();
                        match e {
                            'n' => s.push('\n'),
                            't' => s.push('\t'),
                            'r' => s.push('\r'),
                            '\\' => s.push('\\'),
                            '\'' => s.push('\''),
                            '"' => s.push('"'),
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
        } else if c == '`' {
            // Template literal (backtick strings)
            chars.next();
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                chars.next();
                if d == '`' {
                    break;
                } else {
                    s.push(d);
                }
            }
            tokens.push(Token::StrLit(s));
        } else if c.is_ascii_digit()
            || (c == '.' && chars.clone().nth(1).map_or(false, |n| n.is_ascii_digit()))
        {
            let mut s = String::new();
            let mut has_dot = false;
            let mut has_exp = false;

            // Check for hex/binary/octal
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
                    // Check it's not .. range operator
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

            // Keep consuming .identifier parts (handles self.center.x, color_picker_hsv.h, etc.)
            while chars.peek() == Some(&'.') {
                let mut peek_chars = chars.clone();
                peek_chars.next(); // skip the '.'

                // Check if next char starts a valid identifier continuation
                if let Some(&next) = peek_chars.peek() {
                    if next.is_alphabetic() || next == '_' || next.is_ascii_digit() {
                        // Consume the '.'
                        chars.next();
                        s.push('.');

                        // Consume the identifier part
                        while let Some(&d) = chars.peek() {
                            if d.is_alphanumeric() || d == '_' {
                                s.push(d);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                    } else {
                        break;
                    }
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
                ':' => tokens.push(Token::Colon),
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

    tokens.push(Token::End);
    tokens
}

// ------------------------------------------------------------
// Built-in functions registry
// ------------------------------------------------------------
type BuiltinFn = fn(Vec<Value>) -> Option<Value>;

fn get_builtin(name: &str) -> Option<BuiltinFn> {
    Some(match name {
        "fix" => |args| {
            let val = args.first()?;
            let n = val.as_f64()?;

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
        "sin" => |args| args.first()?.as_f64().map(|n| Value::F64(n.sin())),
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
            println!("{}, {}, {}, {}", val, min, max, val.clamp(min, max));
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
        "str" => |args| Some(Value::String(args.first()?.to_string())),
        "bool" => |args| Some(Value::Bool(args.first()?.is_truthy())),
        "int" => |args| args.first()?.as_i64().map(|n| Value::I64(n)),
        "float" => |args| args.first()?.as_f64().map(|n| Value::F64(n)),

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
        _ => return None,
    })
}

// ------------------------------------------------------------
// Parser
// ------------------------------------------------------------
struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    vars: &'a Variables,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], vars: &'a Variables) -> Self {
        Self {
            tokens,
            pos: 0,
            vars,
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::End)
    }

    fn next(&mut self) -> Token {
        let t = self.peek().clone();
        if !matches!(t, Token::End) {
            self.pos += 1;
        }
        t
    }

    // expr = pipeline
    fn parse_expr(&mut self) -> Option<Value> {
        let mut value = self.parse_ternary()?;

        while let Token::Pipe = self.peek() {
            self.next();
            // Get the function name
            if let Token::Ident(name) = self.next() {
                // Check if it's a function call with parens
                if let Token::LParen = self.peek() {
                    self.next();
                    let mut args = vec![value];
                    args = self.parse_comma_separated_until_rparen(|s| s.parse_expr())?;

                    self.next(); // consume )

                    if let Some(func) = get_builtin(&name) {
                        value = func(args)?;
                    } else {
                        return None;
                    }
                } else {
                    // Just function name, pass current value as only arg
                    if let Some(func) = get_builtin(&name) {
                        value = func(vec![value])?;
                    } else {
                        return None;
                    }
                }
            } else {
                return None;
            }
        }

        Some(value)
    }

    // ternary = null_coalesce ('?' expr ':' expr)?
    fn parse_ternary(&mut self) -> Option<Value> {
        let condition = self.parse_null_coalesce()?;

        if let Token::Question = self.peek() {
            self.next();
            let yes = self.parse_expr()?;
            match self.next() {
                Token::Colon => (),
                _ => return None,
            }
            let no = self.parse_expr()?;

            let cond_bool = condition.is_truthy();
            return Some(if cond_bool { yes } else { no });
        }

        Some(condition)
    }

    // null_coalesce = logical_or (?? logical_or)*
    fn parse_null_coalesce(&mut self) -> Option<Value> {
        let mut value = self.parse_logical_or()?;

        while let Token::NullCoalesce = self.peek() {
            self.next();
            if value.is_null() {
                value = self.parse_logical_or()?;
            } else {
                // Still need to parse the RHS but discard it
                self.parse_logical_or()?;
            }
        }

        Some(value)
    }

    // logical_or = logical_and ('||' logical_and)*
    fn parse_logical_or(&mut self) -> Option<Value> {
        let mut value = self.parse_logical_and()?;
        loop {
            match self.peek() {
                Token::Or => {
                    self.next();
                    let rhs = self.parse_logical_and()?;
                    let a = value.is_truthy();
                    let b = rhs.is_truthy();
                    value = Value::Bool(a || b);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // logical_and = bitwise_or ('&&' bitwise_or)*
    fn parse_logical_and(&mut self) -> Option<Value> {
        let mut value = self.parse_bitwise_or()?;
        loop {
            match self.peek() {
                Token::And => {
                    self.next();
                    let rhs = self.parse_bitwise_or()?;
                    let a = value.is_truthy();
                    let b = rhs.is_truthy();
                    value = Value::Bool(a && b);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // bitwise_or = bitwise_xor ('|' bitwise_xor)*
    fn parse_bitwise_or(&mut self) -> Option<Value> {
        let mut value = self.parse_bitwise_xor()?;
        loop {
            match self.peek() {
                Token::BitOr => {
                    self.next();
                    let rhs = self.parse_bitwise_xor()?;
                    value = Value::I64(value.as_i64()? | rhs.as_i64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // bitwise_xor = bitwise_and ('^' bitwise_and)*
    fn parse_bitwise_xor(&mut self) -> Option<Value> {
        let mut value = self.parse_bitwise_and()?;
        loop {
            match self.peek() {
                Token::BitXor => {
                    self.next();
                    let rhs = self.parse_bitwise_and()?;
                    value = Value::I64(value.as_i64()? ^ rhs.as_i64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // bitwise_and = equality ('&' equality)*
    fn parse_bitwise_and(&mut self) -> Option<Value> {
        let mut value = self.parse_equality()?;
        loop {
            match self.peek() {
                Token::BitAnd => {
                    self.next();
                    let rhs = self.parse_equality()?;
                    value = Value::I64(value.as_i64()? & rhs.as_i64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // equality = comparison (== | != | === | !== comparison)*
    fn parse_equality(&mut self) -> Option<Value> {
        let mut value = self.parse_comparison()?;
        loop {
            match self.peek() {
                Token::Eq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(value == rhs);
                }
                Token::Neq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(!(value == rhs));
                }
                Token::StrictEq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(value == rhs);
                }
                Token::StrictNeq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(!(value == rhs));
                }
                _ => break,
            }
        }
        Some(value)
    }

    // comparison = shift (< | > | <= | >= shift)*
    fn parse_comparison(&mut self) -> Option<Value> {
        let mut value = self.parse_shift()?;
        loop {
            match self.peek() {
                Token::Lt => {
                    self.next();
                    let rhs = self.parse_shift()?;
                    value = Value::Bool(value.as_f64()? < rhs.as_f64()?);
                }
                Token::Gt => {
                    self.next();
                    let rhs = self.parse_shift()?;
                    value = Value::Bool(value.as_f64()? > rhs.as_f64()?);
                }
                Token::Le => {
                    self.next();
                    let rhs = self.parse_shift()?;
                    value = Value::Bool(value.as_f64()? <= rhs.as_f64()?);
                }
                Token::Ge => {
                    self.next();
                    let rhs = self.parse_shift()?;
                    value = Value::Bool(value.as_f64()? >= rhs.as_f64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // shift = addition (<< | >> addition)*
    fn parse_shift(&mut self) -> Option<Value> {
        let mut value = self.parse_addition()?;
        loop {
            match self.peek() {
                Token::Shl => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::I64((value.as_i64()?) << (rhs.as_i64()? as u32));
                }
                Token::Shr => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::I64((value.as_i64()?) >> (rhs.as_i64()? as u32));
                }
                _ => break,
            }
        }
        Some(value)
    }

    // addition = multiplication (+ | - multiplication)*
    fn parse_addition(&mut self) -> Option<Value> {
        let mut value = self.parse_multiplication()?;
        loop {
            match self.peek() {
                Token::Plus => {
                    self.next();
                    let rhs = self.parse_multiplication()?;
                    value = add_values(value, rhs)?;
                }
                Token::Minus => {
                    self.next();
                    let rhs = self.parse_multiplication()?;
                    value = Value::F64(value.as_f64()? - rhs.as_f64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // multiplication = power (* | / | % power)*
    fn parse_multiplication(&mut self) -> Option<Value> {
        let mut value = self.parse_power()?;
        loop {
            match self.peek() {
                Token::Star => {
                    self.next();
                    let rhs = self.parse_power()?;
                    // String repetition: "abc" * 3 = "abcabcabc"
                    match (&value, &rhs) {
                        (Value::String(s), Value::I64(n)) => {
                            value = Value::String(s.repeat(*n as usize));
                        }
                        (Value::I64(n), Value::String(s)) => {
                            value = Value::String(s.repeat(*n as usize));
                        }
                        _ => {
                            value = Value::F64(value.as_f64()? * rhs.as_f64()?);
                        }
                    }
                }
                Token::Slash => {
                    self.next();
                    let rhs = self.parse_power()?;
                    value = Value::F64(value.as_f64()? / rhs.as_f64()?);
                }
                Token::Percent => {
                    self.next();
                    let rhs = self.parse_power()?;
                    value = Value::F64(value.as_f64()? % rhs.as_f64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // power = unary (** unary)*  (right associative)
    fn parse_power(&mut self) -> Option<Value> {
        let base = self.parse_unary()?;

        if let Token::Power = self.peek() {
            self.next();
            let exp = self.parse_power()?; // Right associative
            return Some(Value::F64(base.as_f64()?.powf(exp.as_f64()?)));
        }

        Some(base)
    }

    // unary = !unary | -unary | ~unary | postfix
    fn parse_unary(&mut self) -> Option<Value> {
        match self.peek() {
            Token::Not => {
                self.next();
                let v = self.parse_unary()?;
                Some(Value::Bool(!v.is_truthy()))
            }
            Token::Minus => {
                self.next();
                let v = self.parse_unary()?;
                Some(Value::F64(-v.as_f64()?))
            }
            Token::BitNot => {
                self.next();
                let v = self.parse_unary()?;
                Some(Value::F64((!v.as_i64()?) as f64))
            }
            _ => self.parse_postfix(),
        }
    }

    // postfix = primary (. ident | [expr] | (args))*
    fn parse_postfix(&mut self) -> Option<Value> {
        let mut value = self.parse_primary()?;

        loop {
            match self.peek() {
                Token::Dot => {
                    self.next();
                    if let Token::Ident(prop) = self.next() {
                        value = self.get_property(&value, &prop)?;
                    } else {
                        return None;
                    }
                }
                Token::LBracket => {
                    self.next();
                    let index = self.parse_expr()?;
                    if !matches!(self.next(), Token::RBracket) {
                        return None;
                    }
                    value = self.get_index(&value, &index)?;
                }
                Token::LParen if matches!(value, Value::String(_)) => {
                    // This is a function call where we got the name as a string
                    break;
                }
                _ => break,
            }
        }

        Some(value)
    }

    fn get_property(&self, value: &Value, prop: &str) -> Option<Value> {
        let result = match value {
            Value::String(s) => match prop {
                "length" | "len" => Some(Value::I64(s.len() as i64)),
                "upper" => Some(Value::String(s.to_uppercase())),
                "lower" => Some(Value::String(s.to_lowercase())),
                "trim" => Some(Value::String(s.trim().to_string())),
                "reverse" => Some(Value::String(s.chars().rev().collect())),
                "first" => s.chars().next().map(|c| Value::String(c.to_string())),
                "last" => s.chars().last().map(|c| Value::String(c.to_string())),
                "empty" => Some(Value::Bool(s.is_empty())),
                "chars" => Some(Value::Array(
                    s.chars().map(|c| Value::String(c.to_string())).collect(),
                )),
                "lines" => Some(Value::Array(
                    s.lines().map(|l| Value::String(l.to_string())).collect(),
                )),
                "words" => Some(Value::Array(
                    s.split_whitespace()
                        .map(|w| Value::String(w.to_string()))
                        .collect(),
                )),
                _ => None,
            },

            Value::Array(arr) => match prop {
                "length" | "len" | "count" => Some(Value::I64(arr.len() as i64)),
                "first" => arr.first().cloned(),
                "last" => arr.last().cloned(),
                "empty" => Some(Value::Bool(arr.is_empty())),

                "sum" => {
                    let sum: f64 = arr.iter().filter_map(|v| v.as_f64()).sum();
                    Some(Value::F64(sum))
                }

                "avg" => {
                    let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    if nums.is_empty() {
                        None
                    } else {
                        Some(Value::F64(nums.iter().sum::<f64>() / nums.len() as f64))
                    }
                }

                "min" => arr
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .fold(None, |min, n| Some(min.map_or(n, |m: f64| m.min(n))))
                    .map(Value::F64),

                "max" => arr
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .fold(None, |max, n| Some(max.map_or(n, |m: f64| m.max(n))))
                    .map(Value::F64),

                "reverse" => Some(Value::Array(arr.iter().rev().cloned().collect())),

                _ => None,
            },

            Value::F64(n) => match prop {
                "abs" => Some(Value::F64(n.abs())),
                "floor" => Some(Value::F64(n.floor())),
                "ceil" => Some(Value::F64(n.ceil())),
                "round" => Some(Value::F64(n.round())),
                "trunc" => Some(Value::F64(n.trunc())),
                "sqrt" => Some(Value::F64(n.sqrt())),
                "sign" => Some(Value::F64(n.signum())),
                "fract" => Some(Value::F64(n.fract())),
                "int" => Some(Value::I64(n.trunc() as i64)),
                "neg" => Some(Value::F64(-n)),

                "isnan" => Some(Value::Bool(n.is_nan())),
                "isinf" => Some(Value::Bool(n.is_infinite())),
                "isfinite" => Some(Value::Bool(n.is_finite())),

                "hex" => Some(Value::String(format!("{:x}", *n as i64))),
                "bin" => Some(Value::String(format!("{:b}", *n as i64))),
                "oct" => Some(Value::String(format!("{:o}", *n as i64))),

                _ => None,
            },

            Value::I64(n) => match prop {
                "abs" => Some(Value::I64(n.abs())),
                "neg" => Some(Value::I64(-n)),
                "sign" => Some(Value::I64(n.signum())),
                "int" => Some(Value::I64(*n)),

                "tofloat" => Some(Value::F64(*n as f64)),

                "isnan" => Some(Value::Bool(false)),
                "isinf" => Some(Value::Bool(false)),
                "isfinite" => Some(Value::Bool(true)),

                "hex" => Some(Value::String(format!("{:x}", n))),
                "bin" => Some(Value::String(format!("{:b}", n))),
                "oct" => Some(Value::String(format!("{:o}", n))),

                _ => None,
            },

            _ => None,
        };
        match result {
            None => None,
            Some(v) => Some(v),
        }
    }

    fn get_index(&self, value: &Value, index: &Value) -> Option<Value> {
        match value {
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let mut idx = index.as_f64()? as i64;
                if idx < 0 {
                    idx += chars.len() as i64;
                }
                if idx >= 0 && (idx as usize) < chars.len() {
                    Some(Value::String(chars[idx as usize].to_string()))
                } else {
                    Some(Value::Null)
                }
            }
            Value::Array(arr) => {
                let mut idx = index.as_f64()? as i64;
                if idx < 0 {
                    idx += arr.len() as i64;
                }
                if idx >= 0 && (idx as usize) < arr.len() {
                    Some(arr[idx as usize].clone())
                } else {
                    Some(Value::Null)
                }
            }
            _ => None,
        }
    }

    // primary = number | string | true/false/null | ident | function call | array | '(' expr ')'
    fn parse_primary(&mut self) -> Option<Value> {
        match self.next() {
            Token::Number(v) => Some(Value::F64(v)),
            Token::StrLit(s) => Some(Value::String(s)),
            Token::True => Some(Value::Bool(true)),
            Token::False => Some(Value::Bool(false)),
            Token::Null => Some(Value::Null),
            Token::Ident(name) => {
                // Check for function call
                if let Token::LParen = self.peek() {
                    self.next();
                    let args: Vec<Value>;
                    args = self.parse_comma_separated_until_rparen(|s| s.parse_expr())?;
                    self.next(); // consume )

                    // Try built-in function
                    if let Some(func) = get_builtin(&name) {
                        return func(args);
                    }

                    // Unknown function
                    return None;
                }

                // Check for known math constants
                match name.as_str() {
                    "PI" | "pi" => return Some(Value::F64(std::f64::consts::PI)),
                    "TAU" | "tau" => return Some(Value::F64(std::f64::consts::TAU)),
                    "E" | "e" => return Some(Value::F64(std::f64::consts::E)),
                    "INF" | "inf" => return Some(Value::F64(f64::INFINITY)),
                    "NAN" | "NaN" => return Some(Value::F64(f64::NAN)),
                    _ => {}
                }

                // Variable lookup
                lookup_var(self.vars, &name).or(Some(Value::Null))
            }
            Token::LBracket => {
                // Array literal
                let mut items = Vec::new();
                if !matches!(self.peek(), Token::RBracket) {
                    loop {
                        // Check for range syntax: [1..5] or [1..=5]
                        let item = self.parse_expr()?;

                        if matches!(self.peek(), Token::DotDot | Token::DotDotEq) {
                            let inclusive = matches!(self.peek(), Token::DotDotEq);
                            self.next();
                            let end = self.parse_expr()?;

                            let start = item.as_i64()?;
                            let end_val = end.as_i64()?;

                            if inclusive {
                                for i in start..=end_val {
                                    items.push(Value::F64(i as f64));
                                }
                            } else {
                                for i in start..end_val {
                                    items.push(Value::F64(i as f64));
                                }
                            }
                        } else {
                            items.push(item);
                        }

                        match self.peek() {
                            Token::Comma => {
                                self.next();
                            }
                            Token::RBracket => break,
                            _ => return None,
                        }
                    }
                }
                self.next(); // consume ]
                Some(Value::Array(items))
            }
            Token::LParen => {
                let v = self.parse_expr()?;
                match self.next() {
                    Token::RParen => Some(v),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn parse_comma_separated_until_rparen<F>(&mut self, mut parse_item: F) -> Option<Vec<Value>>
    where
        F: FnMut(&mut Self) -> Option<Value>,
    {
        let mut items = Vec::new();

        if !matches!(self.peek(), Token::RParen) {
            loop {
                items.push(parse_item(self)?);
                match self.peek() {
                    Token::Comma => {
                        self.next();
                    }
                    Token::RParen => break,
                    _ => return None,
                }
            }
        }

        Some(items)
    }
}

fn add_values(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => Some(Value::F64(x + y)),
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

pub fn evaluate_placeholder(src: &str, vars: &Variables) -> String {
    format_slot(src.trim(), vars)
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
fn format_slot(src: &str, vars: &Variables) -> String {
    let (expr, modifiers) = split_format_chain(src);

    let mut value = match eval_expr(expr, vars) {
        Some(v) => v,
        None => return src.to_string(),
    };

    for modifier in modifiers {
        value = match apply_modifier(value, modifier) {
            Some(v) => v,
            None => return src.to_string(),
        };
    }

    value_to_text(&value)
}

fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    }
}
fn split_format_chain(src: &str) -> (&str, Vec<&str>) {
    let mut paren_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;
    let mut brace_depth: i32 = 0;
    let mut ternary_depth: i32 = 0;
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
            '?' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                ternary_depth += 1;
            }
            ':' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                if ternary_depth > 0 {
                    ternary_depth -= 1;
                } else {
                    let expr = src[..idx].trim();
                    let mods = src[idx + 1..]
                        .split(':')
                        .map(|m| m.trim())
                        .filter(|m| !m.is_empty())
                        .collect::<Vec<_>>();
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
pub fn eval_expr(expr: &str, vars: &Variables) -> Option<Value> {
    let tokens = tokenize_expr(expr);
    let mut parser = Parser::new(&tokens, vars);
    parser.parse_expr()
}

//noinspection GrazieInspection
pub fn resolve_template(template: &str, vars: &Variables, settings: &Settings) -> String {
    let mut out = String::new();
    let mut chars = template.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if c == '{' {
            // Check for escaped brace {{
            if let Some(&(_, '{')) = chars.peek() {
                chars.next();
                out.push('{');
                continue;
            }

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
                    evaluate_placeholder(inside.trim(), vars)
                };
                out.push_str(&val);
            } else {
                out.push('{');
            }
        } else if c == '}' {
            // Check for escaped brace }}
            if let Some(&(_, '}')) = chars.peek() {
                chars.next();
                out.push('}');
            } else {
                out.push(c);
            }
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
fn is_component_suffix(s: &str) -> bool {
    matches!(s, "x" | "y" | "z" | "w" | "r" | "g" | "b" | "h" | "s" | "v")
        || s.parse::<usize>().is_ok()
}
