use crate::ui::variables::{UiValue, UiVariableRegistry};
use std::fmt;

// ------------------------------------------------------------
// Value type
// ------------------------------------------------------------
#[derive(Clone, Debug)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    Array(Vec<Value>),
    Null,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::Str(s) => write!(f, "{}", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", items.join(", "))
            }
            Value::Null => write!(f, "null"),
        }
    }
}

impl Value {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            Value::Str(s) => s.parse().ok(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Array(arr) => Some(arr.len() as f64),
            Value::Null => Some(0.0),
        }
    }

    fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }

    fn to_string_value(self) -> String {
        match self {
            Value::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", n as i64)
                } else {
                    n.to_string()
                }
            }
            Value::Str(s) => s,
            Value::Bool(b) => b.to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.into_iter().map(|v| v.to_string_value()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Null => "null".to_string(),
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    fn type_name(&self) -> &'static str {
        match self {
            Value::Num(_) => "number",
            Value::Str(_) => "string",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::Null => "null",
        }
    }
}

fn lookup_var(vars: &UiVariableRegistry, name: &str) -> Option<Value> {
    match vars.get(name)? {
        UiValue::Bool(v) => Some(Value::Bool(*v)),
        UiValue::I32(v) => Some(Value::Num(*v as f64)),
        UiValue::F32(v) => Some(Value::Num(*v as f64)),
        UiValue::String(v) => Some(Value::Str(v.clone())),
    }
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
        // Math functions
        "abs" => |args| args.first()?.as_f64().map(|n| Value::Num(n.abs())),
        "floor" => |args| args.first()?.as_f64().map(|n| Value::Num(n.floor())),
        "ceil" => |args| args.first()?.as_f64().map(|n| Value::Num(n.ceil())),
        "round" => |args| {
            let n = args.first()?.as_f64()?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
            let factor = 10f64.powi(decimals);
            Some(Value::Num((n * factor).round() / factor))
        },
        "trunc" => |args| args.first()?.as_f64().map(|n| Value::Num(n.trunc())),
        "sqrt" => |args| args.first()?.as_f64().map(|n| Value::Num(n.sqrt())),
        "cbrt" => |args| args.first()?.as_f64().map(|n| Value::Num(n.cbrt())),
        "pow" => |args| {
            let base = args.first()?.as_f64()?;
            let exp = args.get(1)?.as_f64()?;
            Some(Value::Num(base.powf(exp)))
        },
        "exp" => |args| args.first()?.as_f64().map(|n| Value::Num(n.exp())),
        "ln" => |args| args.first()?.as_f64().map(|n| Value::Num(n.ln())),
        "log" => |args| {
            let n = args.first()?.as_f64()?;
            let base = args.get(1).and_then(|v| v.as_f64()).unwrap_or(10.0);
            Some(Value::Num(n.log(base)))
        },
        "log2" => |args| args.first()?.as_f64().map(|n| Value::Num(n.log2())),
        "log10" => |args| args.first()?.as_f64().map(|n| Value::Num(n.log10())),
        "sin" => |args| args.first()?.as_f64().map(|n| Value::Num(n.sin())),
        "cos" => |args| args.first()?.as_f64().map(|n| Value::Num(n.cos())),
        "tan" => |args| args.first()?.as_f64().map(|n| Value::Num(n.tan())),
        "asin" => |args| args.first()?.as_f64().map(|n| Value::Num(n.asin())),
        "acos" => |args| args.first()?.as_f64().map(|n| Value::Num(n.acos())),
        "atan" => |args| args.first()?.as_f64().map(|n| Value::Num(n.atan())),
        "atan2" => |args| {
            let y = args.first()?.as_f64()?;
            let x = args.get(1)?.as_f64()?;
            Some(Value::Num(y.atan2(x)))
        },
        "sinh" => |args| args.first()?.as_f64().map(|n| Value::Num(n.sinh())),
        "cosh" => |args| args.first()?.as_f64().map(|n| Value::Num(n.cosh())),
        "tanh" => |args| args.first()?.as_f64().map(|n| Value::Num(n.tanh())),
        "degrees" => |args| args.first()?.as_f64().map(|n| Value::Num(n.to_degrees())),
        "radians" => |args| args.first()?.as_f64().map(|n| Value::Num(n.to_radians())),
        "min" => |args| {
            let mut min_val: Option<f64> = None;
            for arg in &args {
                if let Some(n) = arg.as_f64() {
                    min_val = Some(min_val.map_or(n, |m| m.min(n)));
                }
            }
            min_val.map(Value::Num)
        },
        "max" => |args| {
            let mut max_val: Option<f64> = None;
            for arg in &args {
                if let Some(n) = arg.as_f64() {
                    max_val = Some(max_val.map_or(n, |m| m.max(n)));
                }
            }
            max_val.map(Value::Num)
        },
        "clamp" => |args| {
            let val = args.first()?.as_f64()?;
            let min = args.get(1)?.as_f64()?;
            let max = args.get(2)?.as_f64()?;
            Some(Value::Num(val.clamp(min, max)))
        },
        "lerp" => |args| {
            let a = args.first()?.as_f64()?;
            let b = args.get(1)?.as_f64()?;
            let t = args.get(2)?.as_f64()?;
            Some(Value::Num(a + (b - a) * t))
        },
        "sign" => |args| {
            let n = args.first()?.as_f64()?;
            Some(Value::Num(if n > 0.0 {
                1.0
            } else if n < 0.0 {
                -1.0
            } else {
                0.0
            }))
        },
        "fract" => |args| args.first()?.as_f64().map(|n| Value::Num(n.fract())),
        "mod" => |args| {
            let a = args.first()?.as_f64()?;
            let b = args.get(1)?.as_f64()?;
            Some(Value::Num(a % b))
        },
        "hypot" => |args| {
            let a = args.first()?.as_f64()?;
            let b = args.get(1)?.as_f64()?;
            Some(Value::Num(a.hypot(b)))
        },

        // Constants
        "pi" => |_| Some(Value::Num(std::f64::consts::PI)),
        "e" => |_| Some(Value::Num(std::f64::consts::E)),
        "tau" => |_| Some(Value::Num(std::f64::consts::TAU)),
        "inf" => |_| Some(Value::Num(f64::INFINITY)),
        "nan" => |_| Some(Value::Num(f64::NAN)),

        // Number checks
        "isnan" => |args| args.first()?.as_f64().map(|n| Value::Bool(n.is_nan())),
        "isinf" => |args| args.first()?.as_f64().map(|n| Value::Bool(n.is_infinite())),
        "isfinite" => |args| args.first()?.as_f64().map(|n| Value::Bool(n.is_finite())),

        // String functions
        "len" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Num(s.len() as f64)),
            Value::Array(arr) => Some(Value::Num(arr.len() as f64)),
            _ => None,
        },
        "upper" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Str(s.to_uppercase())),
            v => Some(Value::Str(v.to_string().to_uppercase())),
        },
        "lower" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Str(s.to_lowercase())),
            v => Some(Value::Str(v.to_string().to_lowercase())),
        },
        "capitalize" => |args| match args.first()? {
            Value::Str(s) => {
                let mut chars = s.chars();
                let result = match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    None => String::new(),
                };
                Some(Value::Str(result))
            }
            _ => None,
        },
        "title" => |args| match args.first()? {
            Value::Str(s) => {
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
                Some(Value::Str(result))
            }
            _ => None,
        },
        "trim" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Str(s.trim().to_string())),
            _ => None,
        },
        "ltrim" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Str(s.trim_start().to_string())),
            _ => None,
        },
        "rtrim" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Str(s.trim_end().to_string())),
            _ => None,
        },
        "reverse" => |args| match args.first()? {
            Value::Str(s) => Some(Value::Str(s.chars().rev().collect())),
            Value::Array(arr) => Some(Value::Array(arr.iter().rev().cloned().collect())),
            _ => None,
        },
        "repeat" => |args| {
            let s = match args.first()? {
                Value::Str(s) => s.clone(),
                v => v.to_string(),
            };
            let n = args.get(1)?.as_f64()? as usize;
            Some(Value::Str(s.repeat(n)))
        },
        "replace" => |args| match args.first()? {
            Value::Str(s) => {
                let from = match args.get(1)? {
                    Value::Str(s) => s.clone(),
                    v => v.to_string(),
                };
                let to = match args.get(2)? {
                    Value::Str(s) => s.clone(),
                    v => v.to_string(),
                };
                Some(Value::Str(s.replace(&from, &to)))
            }
            _ => None,
        },
        "split" => |args| match args.first()? {
            Value::Str(s) => {
                let delim = match args.get(1) {
                    Some(Value::Str(d)) => d.clone(),
                    _ => " ".to_string(),
                };
                let parts: Vec<Value> =
                    s.split(&delim).map(|p| Value::Str(p.to_string())).collect();
                Some(Value::Array(parts))
            }
            _ => None,
        },
        "join" => |args| match args.first()? {
            Value::Array(arr) => {
                let delim = match args.get(1) {
                    Some(Value::Str(d)) => d.clone(),
                    _ => "".to_string(),
                };
                let result: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                Some(Value::Str(result.join(&delim)))
            }
            _ => None,
        },
        "substr" => |args| match args.first()? {
            Value::Str(s) => {
                let start = args.get(1)?.as_f64()? as usize;
                let len = args.get(2).and_then(|v| v.as_f64()).map(|n| n as usize);
                let chars: Vec<char> = s.chars().collect();
                let end = len
                    .map(|l| (start + l).min(chars.len()))
                    .unwrap_or(chars.len());
                if start >= chars.len() {
                    Some(Value::Str(String::new()))
                } else {
                    Some(Value::Str(chars[start..end].iter().collect()))
                }
            }
            _ => None,
        },
        "contains" => |args| match args.first()? {
            Value::Str(s) => {
                let needle = match args.get(1)? {
                    Value::Str(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Bool(s.contains(&needle)))
            }
            Value::Array(arr) => {
                let needle = args.get(1)?;
                Some(Value::Bool(arr.iter().any(|v| values_equal(v, needle))))
            }
            _ => None,
        },
        "startswith" => |args| match args.first()? {
            Value::Str(s) => {
                let prefix = match args.get(1)? {
                    Value::Str(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Bool(s.starts_with(&prefix)))
            }
            _ => None,
        },
        "endswith" => |args| match args.first()? {
            Value::Str(s) => {
                let suffix = match args.get(1)? {
                    Value::Str(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Bool(s.ends_with(&suffix)))
            }
            _ => None,
        },
        "indexof" => |args| match args.first()? {
            Value::Str(s) => {
                let needle = match args.get(1)? {
                    Value::Str(n) => n.clone(),
                    v => v.to_string(),
                };
                Some(Value::Num(
                    s.find(&needle).map(|i| i as f64).unwrap_or(-1.0),
                ))
            }
            Value::Array(arr) => {
                let needle = args.get(1)?;
                for (i, v) in arr.iter().enumerate() {
                    if values_equal(v, needle) {
                        return Some(Value::Num(i as f64));
                    }
                }
                Some(Value::Num(-1.0))
            }
            _ => None,
        },
        "padleft" => |args| match args.first()? {
            Value::Str(s) => {
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::Str(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::Str(s.clone()))
                } else {
                    let padding: String =
                        std::iter::repeat(pad_char).take(width - s.len()).collect();
                    Some(Value::Str(padding + s))
                }
            }
            v => {
                let s = v.to_string();
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::Str(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::Str(s))
                } else {
                    let padding: String =
                        std::iter::repeat(pad_char).take(width - s.len()).collect();
                    Some(Value::Str(padding + &s))
                }
            }
        },
        "padright" => |args| match args.first()? {
            Value::Str(s) => {
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::Str(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::Str(s.clone()))
                } else {
                    let padding: String =
                        std::iter::repeat(pad_char).take(width - s.len()).collect();
                    Some(Value::Str(s.clone() + &padding))
                }
            }
            _ => None,
        },
        "center" => |args| match args.first()? {
            Value::Str(s) => {
                let width = args.get(1)?.as_f64()? as usize;
                let pad_char = match args.get(2) {
                    Some(Value::Str(p)) if !p.is_empty() => p.chars().next().unwrap(),
                    _ => ' ',
                };
                if s.len() >= width {
                    Some(Value::Str(s.clone()))
                } else {
                    let total_pad = width - s.len();
                    let left_pad = total_pad / 2;
                    let right_pad = total_pad - left_pad;
                    let left: String = std::iter::repeat(pad_char).take(left_pad).collect();
                    let right: String = std::iter::repeat(pad_char).take(right_pad).collect();
                    Some(Value::Str(left + s + &right))
                }
            }
            _ => None,
        },
        "char" => |args| {
            let code = args.first()?.as_f64()? as u32;
            char::from_u32(code).map(|c| Value::Str(c.to_string()))
        },
        "ord" => |args| match args.first()? {
            Value::Str(s) => s.chars().next().map(|c| Value::Num(c as u32 as f64)),
            _ => None,
        },
        "hex" => |args| {
            let n = args.first()?.as_f64()? as i64;
            Some(Value::Str(format!("{:x}", n)))
        },
        "bin" => |args| {
            let n = args.first()?.as_f64()? as i64;
            Some(Value::Str(format!("{:b}", n)))
        },
        "oct" => |args| {
            let n = args.first()?.as_f64()? as i64;
            Some(Value::Str(format!("{:o}", n)))
        },

        // Array functions
        "first" => |args| match args.first()? {
            Value::Array(arr) => arr.first().cloned(),
            Value::Str(s) => s.chars().next().map(|c| Value::Str(c.to_string())),
            _ => None,
        },
        "last" => |args| match args.first()? {
            Value::Array(arr) => arr.last().cloned(),
            Value::Str(s) => s.chars().last().map(|c| Value::Str(c.to_string())),
            _ => None,
        },
        "sum" => |args| match args.first()? {
            Value::Array(arr) => {
                let sum: f64 = arr.iter().filter_map(|v| v.as_f64()).sum();
                Some(Value::Num(sum))
            }
            _ => {
                let sum: f64 = args.iter().filter_map(|v| v.as_f64()).sum();
                Some(Value::Num(sum))
            }
        },
        "avg" => |args| match args.first()? {
            Value::Array(arr) => {
                let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                if nums.is_empty() {
                    None
                } else {
                    Some(Value::Num(nums.iter().sum::<f64>() / nums.len() as f64))
                }
            }
            _ => {
                let nums: Vec<f64> = args.iter().filter_map(|v| v.as_f64()).collect();
                if nums.is_empty() {
                    None
                } else {
                    Some(Value::Num(nums.iter().sum::<f64>() / nums.len() as f64))
                }
            }
        },
        "count" => |args| match args.first()? {
            Value::Array(arr) => Some(Value::Num(arr.len() as f64)),
            Value::Str(s) => Some(Value::Num(s.chars().count() as f64)),
            _ => Some(Value::Num(args.len() as f64)),
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
                    result.push(Value::Num(i as f64));
                    i += step;
                }
            } else {
                while i > end {
                    result.push(Value::Num(i as f64));
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
            Value::Str(s) => {
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
                    Some(Value::Str(String::new()))
                } else {
                    Some(Value::Str(chars[start..end].iter().collect()))
                }
            }
            _ => None,
        },

        // Type functions
        "type" => |args| Some(Value::Str(args.first()?.type_name().to_string())),
        "isnull" => |args| {
            Some(Value::Bool(
                args.first().map(|v| v.is_null()).unwrap_or(true),
            ))
        },
        "isnum" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::Num(_))))),
        "isstr" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::Str(_))))),
        "isbool" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::Bool(_))))),
        "isarray" => |args| Some(Value::Bool(matches!(args.first(), Some(Value::Array(_))))),

        // Conversion functions
        "num" => |args| match args.first()? {
            Value::Num(n) => Some(Value::Num(*n)),
            Value::Str(s) => s.parse().ok().map(Value::Num),
            Value::Bool(b) => Some(Value::Num(if *b { 1.0 } else { 0.0 })),
            _ => None,
        },
        "str" => |args| Some(Value::Str(args.first()?.to_string())),
        "bool" => |args| Some(Value::Bool(is_truthy(args.first()?))),
        "int" => |args| args.first()?.as_f64().map(|n| Value::Num(n.trunc())),
        "float" => |args| args.first()?.as_f64().map(Value::Num),

        // Formatting functions
        "format" => |args| {
            let val = args.first()?;
            let precision = args.get(1).and_then(|v| v.as_f64()).map(|n| n as usize);
            if let (Value::Num(n), Some(p)) = (val, precision) {
                Some(Value::Str(format!("{:.*}", p, n)))
            } else {
                Some(Value::Str(val.to_string()))
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

            let with_commas: String = digits
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
                .collect();

            let result = if negative {
                format!("-{}", with_commas)
            } else {
                with_commas
            };

            match frac_part {
                Some(frac) => Some(Value::Str(format!("{}.{}", result, frac))),
                None => Some(Value::Str(result)),
            }
        },
        "percent" => |args| {
            let n = args.first()?.as_f64()?;
            let decimals = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;
            Some(Value::Str(format!("{:.*}%", decimals, n * 100.0)))
        },
        "currency" => |args| {
            let n = args.first()?.as_f64()?;
            let symbol = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
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
            let with_commas: String = digits
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
                .collect();

            let num_str = match frac_part {
                Some(frac) => format!("{}.{}", with_commas, frac),
                None => with_commas,
            };

            if n < 0.0 {
                Some(Value::Str(format!("-{}{}", symbol, num_str)))
            } else {
                Some(Value::Str(format!("{}{}", symbol, num_str)))
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
            Some(Value::Str(format!("{}{}", n, suffix)))
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
            Some(Value::Str(format!(
                "{}{:.*} {}",
                sign, decimals, value, units[unit_idx]
            )))
        },

        // Conditional helpers
        "if" => |args| {
            let cond = is_truthy(args.first()?);
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
            if val.is_null() || matches!(val, Value::Str(s) if s.is_empty()) {
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
                if values_equal(val, &pairs[i]) {
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
            Some(Value::Num(out_min + t * (out_max - out_min)))
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

            Some(Value::Str(if negative {
                format!("-{}", result)
            } else {
                result
            }))
        },
        "elapsed" => |args| {
            // Format seconds as "X minutes ago" style
            let secs = args.first()?.as_f64()?.abs();
            let result = if secs < 60.0 {
                format!("{:.0} seconds", secs)
            } else if secs < 3600.0 {
                format!("{:.0} minutes", secs / 60.0)
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
            Some(Value::Str(result))
        },

        // Debug/utility
        "debug" => |args| Some(Value::Str(format!("{:?}", args.first()?))),
        "typeof" => |args| Some(Value::Str(args.first()?.type_name().to_string())),
        "defined" => |args| {
            Some(Value::Bool(
                !args.first().map(|v| v.is_null()).unwrap_or(true),
            ))
        },
        "empty" => |args| match args.first() {
            Some(Value::Str(s)) => Some(Value::Bool(s.is_empty())),
            Some(Value::Array(arr)) => Some(Value::Bool(arr.is_empty())),
            Some(Value::Null) => Some(Value::Bool(true)),
            _ => Some(Value::Bool(false)),
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
    vars: &'a UiVariableRegistry,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], vars: &'a UiVariableRegistry) -> Self {
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
        self.parse_pipeline()
    }

    // pipeline = ternary (|> ternary)*
    fn parse_pipeline(&mut self) -> Option<Value> {
        let mut value = self.parse_ternary()?;

        while let Token::Pipe = self.peek() {
            self.next();
            // Get the function name
            if let Token::Ident(name) = self.next() {
                // Check if it's a function call with parens
                if let Token::LParen = self.peek() {
                    self.next();
                    let mut args = vec![value];
                    if !matches!(self.peek(), Token::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            match self.peek() {
                                Token::Comma => {
                                    self.next();
                                }
                                Token::RParen => break,
                                _ => return None,
                            }
                        }
                    }
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

            let cond_bool = is_truthy(&condition);
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
                    let a = is_truthy(&value);
                    let b = is_truthy(&rhs);
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
                    let a = is_truthy(&value);
                    let b = is_truthy(&rhs);
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
                    value = Value::Num((value.as_i64()? | rhs.as_i64()?) as f64);
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
                    value = Value::Num((value.as_i64()? ^ rhs.as_i64()?) as f64);
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
                    value = Value::Num((value.as_i64()? & rhs.as_i64()?) as f64);
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
                    value = Value::Bool(values_equal(&value, &rhs));
                }
                Token::Neq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(!values_equal(&value, &rhs));
                }
                Token::StrictEq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(values_strict_equal(&value, &rhs));
                }
                Token::StrictNeq => {
                    self.next();
                    let rhs = self.parse_comparison()?;
                    value = Value::Bool(!values_strict_equal(&value, &rhs));
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
                    value = Value::Num(((value.as_i64()?) << (rhs.as_i64()? as u32)) as f64);
                }
                Token::Shr => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::Num(((value.as_i64()?) >> (rhs.as_i64()? as u32)) as f64);
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
                    value = Value::Num(value.as_f64()? - rhs.as_f64()?);
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
                        (Value::Str(s), Value::Num(n)) => {
                            value = Value::Str(s.repeat(*n as usize));
                        }
                        (Value::Num(n), Value::Str(s)) => {
                            value = Value::Str(s.repeat(*n as usize));
                        }
                        _ => {
                            value = Value::Num(value.as_f64()? * rhs.as_f64()?);
                        }
                    }
                }
                Token::Slash => {
                    self.next();
                    let rhs = self.parse_power()?;
                    value = Value::Num(value.as_f64()? / rhs.as_f64()?);
                }
                Token::Percent => {
                    self.next();
                    let rhs = self.parse_power()?;
                    value = Value::Num(value.as_f64()? % rhs.as_f64()?);
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
            return Some(Value::Num(base.as_f64()?.powf(exp.as_f64()?)));
        }

        Some(base)
    }

    // unary = !unary | -unary | ~unary | postfix
    fn parse_unary(&mut self) -> Option<Value> {
        match self.peek() {
            Token::Not => {
                self.next();
                let v = self.parse_unary()?;
                Some(Value::Bool(!is_truthy(&v)))
            }
            Token::Minus => {
                self.next();
                let v = self.parse_unary()?;
                Some(Value::Num(-v.as_f64()?))
            }
            Token::BitNot => {
                self.next();
                let v = self.parse_unary()?;
                Some(Value::Num((!v.as_i64()?) as f64))
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
                Token::LParen if matches!(value, Value::Str(_)) => {
                    // This is a function call where we got the name as a string
                    break;
                }
                _ => break,
            }
        }

        Some(value)
    }

    fn get_property(&self, value: &Value, prop: &str) -> Option<Value> {
        match value {
            Value::Str(s) => match prop {
                "length" | "len" => Some(Value::Num(s.len() as f64)),
                "upper" => Some(Value::Str(s.to_uppercase())),
                "lower" => Some(Value::Str(s.to_lowercase())),
                "trim" => Some(Value::Str(s.trim().to_string())),
                "reverse" => Some(Value::Str(s.chars().rev().collect())),
                "first" => s.chars().next().map(|c| Value::Str(c.to_string())),
                "last" => s.chars().last().map(|c| Value::Str(c.to_string())),
                "empty" => Some(Value::Bool(s.is_empty())),
                "chars" => Some(Value::Array(
                    s.chars().map(|c| Value::Str(c.to_string())).collect(),
                )),
                "lines" => Some(Value::Array(
                    s.lines().map(|l| Value::Str(l.to_string())).collect(),
                )),
                "words" => Some(Value::Array(
                    s.split_whitespace()
                        .map(|w| Value::Str(w.to_string()))
                        .collect(),
                )),
                _ => None,
            },
            Value::Array(arr) => match prop {
                "length" | "len" | "count" => Some(Value::Num(arr.len() as f64)),
                "first" => arr.first().cloned(),
                "last" => arr.last().cloned(),
                "empty" => Some(Value::Bool(arr.is_empty())),
                "sum" => {
                    let sum: f64 = arr.iter().filter_map(|v| v.as_f64()).sum();
                    Some(Value::Num(sum))
                }
                "avg" => {
                    let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    if nums.is_empty() {
                        None
                    } else {
                        Some(Value::Num(nums.iter().sum::<f64>() / nums.len() as f64))
                    }
                }
                "min" => arr
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .fold(None, |min, n| Some(min.map_or(n, |m: f64| m.min(n))))
                    .map(Value::Num),
                "max" => arr
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .fold(None, |max, n| Some(max.map_or(n, |m: f64| m.max(n))))
                    .map(Value::Num),
                "reverse" => Some(Value::Array(arr.iter().rev().cloned().collect())),
                _ => None,
            },
            Value::Num(n) => match prop {
                "abs" => Some(Value::Num(n.abs())),
                "floor" => Some(Value::Num(n.floor())),
                "ceil" => Some(Value::Num(n.ceil())),
                "round" => Some(Value::Num(n.round())),
                "trunc" => Some(Value::Num(n.trunc())),
                "sqrt" => Some(Value::Num(n.sqrt())),
                "sign" => Some(Value::Num(n.signum())),
                "fract" => Some(Value::Num(n.fract())),
                "int" => Some(Value::Num(n.trunc())),
                "neg" => Some(Value::Num(-n)),
                "isnan" => Some(Value::Bool(n.is_nan())),
                "isinf" => Some(Value::Bool(n.is_infinite())),
                "isfinite" => Some(Value::Bool(n.is_finite())),
                "hex" => Some(Value::Str(format!("{:x}", *n as i64))),
                "bin" => Some(Value::Str(format!("{:b}", *n as i64))),
                "oct" => Some(Value::Str(format!("{:o}", *n as i64))),
                _ => None,
            },
            _ => None,
        }
    }

    fn get_index(&self, value: &Value, index: &Value) -> Option<Value> {
        match value {
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let mut idx = index.as_f64()? as i64;
                if idx < 0 {
                    idx += chars.len() as i64;
                }
                if idx >= 0 && (idx as usize) < chars.len() {
                    Some(Value::Str(chars[idx as usize].to_string()))
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
            Token::Number(v) => Some(Value::Num(v)),
            Token::StrLit(s) => Some(Value::Str(s)),
            Token::True => Some(Value::Bool(true)),
            Token::False => Some(Value::Bool(false)),
            Token::Null => Some(Value::Null),
            Token::Ident(name) => {
                // Check for function call
                if let Token::LParen = self.peek() {
                    self.next();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Token::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            match self.peek() {
                                Token::Comma => {
                                    self.next();
                                }
                                Token::RParen => break,
                                _ => return None,
                            }
                        }
                    }
                    self.next(); // consume )

                    // Try built-in function
                    if let Some(func) = get_builtin(&name) {
                        return func(args);
                    }

                    // Unknown function
                    return None;
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
                                    items.push(Value::Num(i as f64));
                                }
                            } else {
                                for i in start..end_val {
                                    items.push(Value::Num(i as f64));
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
}

// ------------------------------------------------------------
// Helper functions
// ------------------------------------------------------------
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Num(n) => *n != 0.0 && !n.is_nan(),
        Value::Str(s) => !s.is_empty(),
        Value::Array(arr) => !arr.is_empty(),
        Value::Null => false,
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Null, Value::Null) => true,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal(a, b))
        }
        // Loose equality: compare numbers and numeric strings
        (Value::Num(n), Value::Str(s)) | (Value::Str(s), Value::Num(n)) => {
            s.parse::<f64>().map(|sn| sn == *n).unwrap_or(false)
        }
        _ => false,
    }
}

fn values_strict_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Null, Value::Null) => true,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(a, b)| values_strict_equal(a, b))
        }
        _ => false,
    }
}

fn add_values(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Some(Value::Num(x + y)),
        (Value::Str(s), Value::Num(n)) => Some(Value::Str(format!("{}{}", s, n))),
        (Value::Num(n), Value::Str(s)) => Some(Value::Str(format!("{}{}", n, s))),
        (Value::Str(x), Value::Str(y)) => Some(Value::Str(x + &y)),
        (Value::Bool(x), Value::Str(y)) => Some(Value::Str(format!("{}{}", x, y))),
        (Value::Str(x), Value::Bool(y)) => Some(Value::Str(format!("{}{}", x, y))),
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
        (Value::Str(s), Value::Null) => Some(Value::Str(s)),
        (Value::Null, Value::Str(s)) => Some(Value::Str(s)),
        _ => None,
    }
}

// ------------------------------------------------------------
// Template formatting
// -----------------------------------------------------------

fn apply_format(value: Value, precision: Option<usize>, opts: &str) -> String {
    let mut use_fix = false;
    let mut int_width = 4usize;
    let mut dec_width = 4usize;
    let mut use_sign = false;
    let mut use_space = false;
    let mut pad_char = ' ';
    let mut align_left = false;
    let mut total_width: Option<usize> = None;
    let mut use_upper = false;
    let mut use_lower = false;
    let mut use_hex = false;
    let mut use_bin = false;
    let mut use_oct = false;
    let mut use_exp = false;
    let mut use_percent = false;

    // parse flags: fix, int=, dec=, sign, space, pad=, left, width=, upper, lower, hex, bin, oct, exp, percent
    for part in opts.split('|') {
        let p = part.trim();
        if p.eq_ignore_ascii_case("fix") {
            use_fix = true;
        } else if p.eq_ignore_ascii_case("sign") || p == "+" {
            use_sign = true;
        } else if p.eq_ignore_ascii_case("space") {
            use_space = true;
        } else if p.eq_ignore_ascii_case("left") || p == "-" {
            align_left = true;
        } else if p.eq_ignore_ascii_case("upper") {
            use_upper = true;
        } else if p.eq_ignore_ascii_case("lower") {
            use_lower = true;
        } else if p.eq_ignore_ascii_case("hex") || p == "x" {
            use_hex = true;
        } else if p.eq_ignore_ascii_case("HEX") || p == "X" {
            use_hex = true;
            use_upper = true;
        } else if p.eq_ignore_ascii_case("bin") || p == "b" {
            use_bin = true;
        } else if p.eq_ignore_ascii_case("oct") || p == "o" {
            use_oct = true;
        } else if p.eq_ignore_ascii_case("exp") || p == "e" {
            use_exp = true;
        } else if p.eq_ignore_ascii_case("percent") || p == "%" {
            use_percent = true;
        } else if let Some(v) = p.strip_prefix("int=") {
            if let Ok(n) = v.parse() {
                int_width = n;
            }
        } else if let Some(v) = p.strip_prefix("dec=") {
            if let Ok(n) = v.parse() {
                dec_width = n;
            }
        } else if let Some(v) = p.strip_prefix("pad=") {
            pad_char = v.chars().next().unwrap_or(' ');
        } else if let Some(v) = p.strip_prefix("width=") {
            total_width = v.parse().ok();
        } else if let Some(v) = p.strip_prefix("w") {
            total_width = v.parse().ok();
        }
    }

    // Handle special formats first
    if use_hex {
        if let Some(n) = value.as_f64() {
            let hex = format!("{:x}", n as i64);
            let result = if use_upper { hex.to_uppercase() } else { hex };
            return apply_width_padding(&result, total_width, pad_char, align_left);
        }
    }

    if use_bin {
        if let Some(n) = value.as_f64() {
            let result = format!("{:b}", n as i64);
            return apply_width_padding(&result, total_width, pad_char, align_left);
        }
    }

    if use_oct {
        if let Some(n) = value.as_f64() {
            let result = format!("{:o}", n as i64);
            return apply_width_padding(&result, total_width, pad_char, align_left);
        }
    }

    if use_exp {
        if let Some(n) = value.as_f64() {
            let prec = precision.unwrap_or(6);
            let result = format!("{:.*e}", prec, n);
            let result = if use_upper {
                result.replace('e', "E")
            } else {
                result
            };
            return apply_width_padding(&result, total_width, pad_char, align_left);
        }
    }

    if use_percent {
        if let Some(n) = value.as_f64() {
            let prec = precision.unwrap_or(0);
            let result = format!("{:.*}%", prec, n * 100.0);
            return apply_width_padding(&result, total_width, pad_char, align_left);
        }
    }

    // Handle string formatting
    if let Value::Str(s) = &value {
        let mut result = s.clone();
        if use_upper {
            result = result.to_uppercase();
        } else if use_lower {
            result = result.to_lowercase();
        }
        return apply_width_padding(&result, total_width, pad_char, align_left);
    }

    // normal formatting path, int/dec ignored
    if !use_fix {
        let base = if let (Value::Num(n), Some(p)) = (&value, precision) {
            let mut s = format!("{:.*}", p, n);
            if use_sign && *n >= 0.0 {
                s = format!("+{}", s);
            } else if use_space && *n >= 0.0 {
                s = format!(" {}", s);
            }
            s
        } else {
            let mut s = value.clone().to_string_value();
            if let Value::Num(n) = &value {
                if use_sign && *n >= 0.0 {
                    s = format!("+{}", s);
                } else if use_space && *n >= 0.0 {
                    s = format!(" {}", s);
                }
            }
            s
        };

        if use_upper {
            return apply_width_padding(&base.to_uppercase(), total_width, pad_char, align_left);
        } else if use_lower {
            return apply_width_padding(&base.to_lowercase(), total_width, pad_char, align_left);
        }
        return apply_width_padding(&base, total_width, pad_char, align_left);
    }

    // fix-mode: precision overrides dec_width
    if let Some(p) = precision {
        dec_width = p;
    }

    let n = match value.as_f64() {
        Some(v) => v,
        None => return value.to_string_value(),
    };

    let neg = n.is_sign_negative();
    let abs = n.abs();

    // integer magnitude and fractional part from |n|
    let mut int_mag = abs.trunc() as i64;
    let mut frac_scaled = if dec_width > 0 {
        (abs.fract() * 10f64.powi(dec_width as i32)).round() as i64
    } else {
        0
    };

    // rounding overflow: 9.999 → 10.000
    if dec_width > 0 {
        let base = 10i64.pow(dec_width as u32);
        if frac_scaled >= base {
            frac_scaled -= base;
            int_mag += 1;
        }
    }

    // now reapply sign to the integer and let formatting handle the '-'
    let signed_int = if neg { -int_mag } else { int_mag };

    let int_str = if use_sign && !neg {
        format!("{:>+iw$}", signed_int, iw = int_width)
    } else if use_space && !neg {
        format!(
            " {:>iw$}",
            signed_int.abs(),
            iw = int_width.saturating_sub(1)
        )
    } else {
        format!("{:>iw$}", signed_int, iw = int_width)
    };

    let result = if dec_width > 0 {
        let frac_str = format!("{:0dw$}", frac_scaled, dw = dec_width);
        format!("{int_str}.{frac_str}")
    } else {
        int_str
    };

    apply_width_padding(&result, total_width, pad_char, align_left)
}

fn apply_width_padding(s: &str, width: Option<usize>, pad_char: char, align_left: bool) -> String {
    match width {
        Some(w) if s.len() < w => {
            let padding: String = std::iter::repeat(pad_char).take(w - s.len()).collect();
            if align_left {
                format!("{}{}", s, padding)
            } else {
                format!("{}{}", padding, s)
            }
        }
        _ => s.to_string(),
    }
}

// ------------------------------------------------------------
// evaluate a placeholder
// ------------------------------------------------------------
fn is_plain_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
}

pub fn evaluate_placeholder(src: &str, vars: &UiVariableRegistry) -> String {
    format_slot(src.trim(), vars)
}

fn format_slot(src: &str, vars: &UiVariableRegistry) -> String {
    // extract |options (but not || which is logical or)
    let mut expr = src;
    let mut opts = "";

    // Find the last | that's not part of ||
    let bytes = src.as_bytes();
    let mut split_pos = None;
    let mut i = src.len();
    while i > 0 {
        i -= 1;
        if bytes[i] == b'|' {
            if i > 0 && bytes[i - 1] == b'|' {
                i -= 1; // skip ||
                continue;
            }
            if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                continue; // skip ||
            }
            if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                continue; // skip |>
            }
            split_pos = Some(i);
            break;
        }
    }

    if let Some(i) = split_pos {
        // Verify it's actually format options (contains = or known keywords)
        let potential_opts = &src[i + 1..];
        let is_format_opts = potential_opts.split('|').any(|p| {
            let p = p.trim();
            p.contains('=')
                || matches!(
                    p.to_lowercase().as_str(),
                    "fix"
                        | "sign"
                        | "space"
                        | "left"
                        | "upper"
                        | "lower"
                        | "hex"
                        | "bin"
                        | "oct"
                        | "exp"
                        | "percent"
                        | "+"
                        | "-"
                        | "%"
                        | "x"
                        | "b"
                        | "o"
                        | "e"
                )
        });

        if is_format_opts {
            expr = &src[..i];
            opts = potential_opts;
        }
    }

    // extract :precision
    let mut precision = None;

    // Find : that's not inside parentheses (for ternary operator)
    let mut paren_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;
    let mut found_question = false;
    let mut colon_pos = None;

    for (idx, ch) in expr.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '?' if paren_depth == 0 && bracket_depth == 0 => found_question = true,
            ':' if paren_depth == 0 && bracket_depth == 0 => {
                if found_question {
                    // This is a ternary colon, skip it
                    found_question = false;
                } else {
                    // This might be a format specifier
                    colon_pos = Some(idx);
                }
            }
            _ => {}
        }
    }

    if let Some(idx) = colon_pos {
        let after_colon = expr[idx + 1..].trim();
        if let Some(prec_str) = after_colon.strip_prefix('.') {
            if let Ok(p) = prec_str.parse::<usize>() {
                precision = Some(p);
                expr = &expr[..idx];
            }
        } else if let Ok(p) = after_colon.parse::<usize>() {
            precision = Some(p);
            expr = &expr[..idx];
        }
    }

    expr = expr.trim();

    let val = match eval_expr(expr, vars) {
        Some(v) => v,
        None => return src.to_string(),
    };

    if is_plain_ident(expr) && precision.is_none() && opts.is_empty() {
        return val.to_string();
    }

    apply_format(val, precision, opts)
}

pub fn eval_expr(expr: &str, vars: &UiVariableRegistry) -> Option<Value> {
    let tokens = tokenize_expr(expr);
    let mut parser = Parser::new(&tokens, vars);
    parser.parse_expr()
}

//noinspection GrazieInspection
// ------------------------------------------------------------
// Final public function
// ------------------------------------------------------------
pub fn resolve_template(template: &str, vars: &UiVariableRegistry) -> String {
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
                let val = evaluate_placeholder(inside.trim(), vars);
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

pub fn set_input_box(template: &str, current_text: &str, _vars: &mut UiVariableRegistry) -> String {
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
