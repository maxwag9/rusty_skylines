use crate::ui::ui_editor::UiVariableRegistry;
use crate::ui::variables::UiValue;
use std::fmt;

// ------------------------------------------------------------
// Value type
// ------------------------------------------------------------
#[derive(Clone, Debug)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Num(n) => write!(f, "{}", n),
            Value::Str(s) => write!(f, "{}", s),
            Value::Bool(b) => write!(f, "{}", b),
        }
    }
}

impl Value {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            Value::Str(s) => s.parse().ok(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        }
    }

    fn to_string_value(self) -> String {
        match self {
            Value::Num(n) => n.to_string(),
            Value::Str(s) => s,
            Value::Bool(b) => b.to_string(),
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

    Plus,
    Minus,
    Star,
    Slash,

    Eq,  // ==
    Neq, // !=
    Lt,  // <
    Gt,  // >
    Le,  // <=
    Ge,  // >=

    And, // &&
    Or,  // ||
    Not, // !

    Question,
    Colon,

    LParen,
    RParen,

    End,
}

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
                } else {
                    s.push(d);
                }
            }
            tokens.push(Token::StrLit(s));
        } else if c.is_ascii_digit() || c == '.' {
            let mut s = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() || d == '.' {
                    s.push(d);
                    chars.next();
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
                if d.is_alphanumeric() || d == '_' || d == '.' {
                    s.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            match s.as_str() {
                "true" => tokens.push(Token::True),
                "false" => tokens.push(Token::False),
                _ => tokens.push(Token::Ident(s)),
            }
        } else {
            chars.next();
            match c {
                '+' => tokens.push(Token::Plus),
                '-' => tokens.push(Token::Minus),
                '*' => tokens.push(Token::Star),
                '/' => tokens.push(Token::Slash),
                '(' => tokens.push(Token::LParen),
                ')' => tokens.push(Token::RParen),
                '?' => tokens.push(Token::Question),
                ':' => tokens.push(Token::Colon),
                '!' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Neq);
                    } else {
                        tokens.push(Token::Not);
                    }
                }
                '=' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Eq);
                    }
                }
                '<' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Le);
                    } else {
                        tokens.push(Token::Lt);
                    }
                }
                '>' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Ge);
                    } else {
                        tokens.push(Token::Gt);
                    }
                }
                '&' => {
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        tokens.push(Token::And);
                    }
                }
                '|' => {
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        tokens.push(Token::Or);
                    }
                }
                _ => {}
            }
        }
    }

    tokens.push(Token::End);
    tokens
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

    // expr = ternary
    fn parse_expr(&mut self) -> Option<Value> {
        self.parse_ternary()
    }

    // ternary = logical_or ('?' expr ':' expr)?
    fn parse_ternary(&mut self) -> Option<Value> {
        let condition = self.parse_logical_or()?;

        if let Token::Question = self.peek() {
            self.next();
            let yes = self.parse_expr()?;
            match self.next() {
                Token::Colon => (),
                _ => return None,
            }
            let no = self.parse_expr()?;

            let cond_bool = match condition {
                Value::Bool(b) => b,
                Value::Num(n) => n != 0.0,
                Value::Str(ref s) => !s.is_empty(),
            };

            return Some(if cond_bool { yes } else { no });
        }

        Some(condition)
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

    // logical_and = equality ('&&' equality)*
    fn parse_logical_and(&mut self) -> Option<Value> {
        let mut value = self.parse_equality()?;
        loop {
            match self.peek() {
                Token::And => {
                    self.next();
                    let rhs = self.parse_equality()?;
                    let a = is_truthy(&value);
                    let b = is_truthy(&rhs);
                    value = Value::Bool(a && b);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // equality = comparison (== | != comparison)*
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
                _ => break,
            }
        }
        Some(value)
    }

    // comparison = addition (< | > | <= | >= addition)*
    fn parse_comparison(&mut self) -> Option<Value> {
        let mut value = self.parse_addition()?;
        loop {
            match self.peek() {
                Token::Lt => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::Bool(value.as_f64()? < rhs.as_f64()?);
                }
                Token::Gt => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::Bool(value.as_f64()? > rhs.as_f64()?);
                }
                Token::Le => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::Bool(value.as_f64()? <= rhs.as_f64()?);
                }
                Token::Ge => {
                    self.next();
                    let rhs = self.parse_addition()?;
                    value = Value::Bool(value.as_f64()? >= rhs.as_f64()?);
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

    // multiplication = unary (* | / unary)*
    fn parse_multiplication(&mut self) -> Option<Value> {
        let mut value = self.parse_unary()?;
        loop {
            match self.peek() {
                Token::Star => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    value = Value::Num(value.as_f64()? * rhs.as_f64()?);
                }
                Token::Slash => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    value = Value::Num(value.as_f64()? / rhs.as_f64()?);
                }
                _ => break,
            }
        }
        Some(value)
    }

    // unary = !unary | -unary | primary
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
            _ => self.parse_primary(),
        }
    }

    // primary = number | string | true/false | ident | '(' expr ')'
    fn parse_primary(&mut self) -> Option<Value> {
        match self.next() {
            Token::Number(v) => Some(Value::Num(v)),
            Token::StrLit(s) => Some(Value::Str(s)),
            Token::True => Some(Value::Bool(true)),
            Token::False => Some(Value::Bool(false)),
            Token::Ident(name) => lookup_var(self.vars, &name),
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
        Value::Num(n) => *n != 0.0,
        Value::Str(s) => !s.is_empty(),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
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

    // parse flags: fix, int=, dec=
    for part in opts.split('|') {
        let p = part.trim();
        if p.eq_ignore_ascii_case("fix") {
            use_fix = true;
        } else if let Some(v) = p.strip_prefix("int=") {
            if let Ok(n) = v.parse() {
                int_width = n;
            }
        } else if let Some(v) = p.strip_prefix("dec=") {
            if let Ok(n) = v.parse() {
                dec_width = n;
            }
        }
    }

    // normal formatting path, int/dec ignored
    if !use_fix {
        if let (Value::Num(n), Some(p)) = (&value, precision) {
            return format!("{:.*}", p, n);
        }
        return value.to_string_value();
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

    let int_str = format!("{:>iw$}", signed_int, iw = int_width);

    if dec_width > 0 {
        let frac_str = format!("{:0dw$}", frac_scaled, dw = dec_width);
        format!("{int_str}.{frac_str}")
    } else {
        int_str
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
    // extract |options
    let mut expr = src;
    let mut opts = "";

    if let Some(i) = src.find('|') {
        expr = &src[..i];
        opts = &src[i + 1..];
    }

    // extract :precision
    let mut precision = None;
    if let Some(i) = expr.find(':') {
        precision = expr[i + 1..]
            .trim()
            .strip_prefix('.')
            .and_then(|p| p.parse().ok());
        expr = &expr[..i];
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

// ------------------------------------------------------------
// Final public function
// ------------------------------------------------------------
pub fn resolve_template(template: &str, vars: &UiVariableRegistry) -> String {
    let mut out = String::new();
    let mut chars = template.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if c == '{' {
            let start = i + 1;
            let mut end_opt = None;

            while let Some(&(j, cj)) = chars.peek() {
                if cj == '}' {
                    end_opt = Some(j);
                    chars.next();
                    break;
                } else {
                    chars.next();
                }
            }

            if let Some(end) = end_opt {
                let inside = &template[start..end];
                let val = evaluate_placeholder(inside.trim(), vars);
                out.push_str(&val);
            } else {
                out.push('{');
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
