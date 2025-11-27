use crate::renderer::ui_editor::UiVariableRegistry;
use std::fmt;

// ------------------------------------------------------------
// Value type
// ------------------------------------------------------------
#[derive(Clone, Debug)]
enum Value {
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

// ------------------------------------------------------------
// Registry lookup
// ------------------------------------------------------------
fn lookup_var(vars: &UiVariableRegistry, name: &str) -> Option<Value> {
    let raw = vars.get(name)?;

    if raw == "true" {
        return Some(Value::Bool(true));
    }
    if raw == "false" {
        return Some(Value::Bool(false));
    }
    if let Ok(n) = raw.parse::<f64>() {
        return Some(Value::Num(n));
    }
    Some(Value::Str(raw.to_string()))
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
fn apply_full_format(value: Value, precision: Option<usize>, fmt_opts: Option<&str>) -> String {
    let mut s = match (value.clone(), precision) {
        (Value::Num(n), Some(p)) => format!("{:.*}", p, n),
        _ => value.to_string_value(),
    };

    if let Some(opts) = fmt_opts {
        let mut width = None;
        let mut align = "left";

        for part in opts.split(',') {
            let p = part.trim();
            if let Some(v) = p.strip_prefix("width=") {
                width = v.parse::<usize>().ok();
            } else if let Some(v) = p.strip_prefix("align=") {
                align = v;
            }
        }

        if let Some(w) = width {
            if s.len() < w {
                s = match align {
                    "right" => format!("{:>width$}", s, width = w),
                    "center" => format!("{:^width$}", s, width = w),
                    _ => format!("{:<width$}", s, width = w),
                };
            }
        }
    }

    s
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
    let inner = src.trim();

    // no comma → single expression, simple path
    if !inner.contains(',') {
        return format_slot(inner, vars);
    }

    // ---------- multi-expression: {a, b, c} ----------
    #[derive(Debug)]
    struct FixSlot {
        raw: String,  // full "part" including |fix etc
        expr: String, // pure expression, no : or |
        precision: Option<usize>,
        is_fix: bool,
        numeric: Option<f64>,
    }

    let mut slots: Vec<FixSlot> = Vec::new();
    let mut any_fix = false;
    let mut any_fix_numeric = false;

    for part in inner.split(',') {
        let raw = part.trim();
        if raw.is_empty() {
            continue;
        }

        // split |options
        let mut expr = raw;
        let mut fmt_opts_str: Option<String> = None;
        if let Some(idx) = expr.find('|') {
            fmt_opts_str = Some(expr[idx + 1..].trim().to_string());
            expr = &expr[..idx];
        }

        // split :precision
        let mut prec = None;
        if let Some(idx) = expr.find(':') {
            prec = expr[idx + 1..]
                .trim()
                .strip_prefix('.')
                .and_then(|p| p.parse::<usize>().ok());
            expr = &expr[..idx];
        }

        let expr = expr.trim().to_string();

        // detect |fix inside options
        let is_fix = fmt_opts_str
            .as_ref()
            .map(|s| s.split(',').any(|p| p.trim().eq_ignore_ascii_case("fix")))
            .unwrap_or(false);

        let mut numeric = None;
        if is_fix {
            if let Some(val) = eval_expr(&expr, vars) {
                numeric = val.as_f64();
                if numeric.is_some() {
                    any_fix_numeric = true;
                }
            }
        }

        any_fix |= is_fix;

        slots.push(FixSlot {
            raw: raw.to_string(),
            expr,
            precision: prec,
            is_fix,
            numeric,
        });
    }

    // no |fix at all → just format each slot individually
    if !any_fix || !any_fix_numeric {
        let mut out = String::new();
        let mut first = true;
        for slot in slots {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(&format_slot(&slot.raw, vars));
        }
        return out;
    }

    // ---------- compute common integer/fraction widths for all |fix numeric slots ----------
    let mut max_int = 0usize;
    let mut max_frac = 0usize;
    let mut max_specified_prec = 0usize;

    for slot in &slots {
        if !slot.is_fix {
            continue;
        }
        let Some(n) = slot.numeric else { continue };

        if let Some(p) = slot.precision {
            if p > max_specified_prec {
                max_specified_prec = p;
            }
        }

        let abs = n.abs();
        let s = if let Some(p) = slot.precision {
            format!("{:.*}", p, abs)
        } else {
            abs.to_string()
        };

        let parts: Vec<&str> = s.split('.').collect();
        let int_digits = parts
            .get(0)
            .map(|t| t.chars().filter(|c| c.is_ascii_digit()).count())
            .unwrap_or(0);

        let frac_digits = if parts.len() > 1 { parts[1].len() } else { 0 };

        if int_digits > max_int {
            max_int = int_digits;
        }
        if frac_digits > max_frac {
            max_frac = frac_digits;
        }
    }

    // if any precision explicitly given via :.N, use the max of those for all |fix entries
    if max_specified_prec > 0 {
        max_frac = max_specified_prec;
    }

    // ---------- format all slots ----------
    let mut out = String::new();
    let mut first = true;

    for slot in slots {
        if !first {
            out.push_str(", ");
        }
        first = false;

        // non-fix or non-numeric → fallback to normal formatting
        if !slot.is_fix || slot.numeric.is_none() {
            out.push_str(&format_slot(&slot.raw, vars));
            continue;
        }

        let n = slot.numeric.unwrap();
        let sign = if n < 0.0 { "-" } else { "" };
        let abs = n.abs();

        let mut int_part = abs.trunc() as i64;
        let mut frac_scaled = if max_frac > 0 {
            (abs.fract() * 10f64.powi(max_frac as i32)).round() as i64
        } else {
            0
        };

        // handle rounding overflow: 9.999 → 10.000
        if max_frac > 0 {
            let base = 10i64.pow(max_frac as u32);
            if frac_scaled >= base {
                frac_scaled -= base;
                int_part += 1;
            }
        }

        let int_str = format!("{:>width$}", int_part, width = max_int);

        if max_frac > 0 {
            let frac_str = format!("{:0width$}", frac_scaled, width = max_frac);
            out.push_str(&format!("{sign}{int_str}.{frac_str}"));
        } else {
            out.push_str(&format!("{sign}{int_str}"));
        }
    }

    out
}

fn format_slot(src: &str, vars: &UiVariableRegistry) -> String {
    let mut expr = src;
    let mut fmt_opts = None;

    // split |options
    if let Some(idx) = expr.find('|') {
        fmt_opts = Some(expr[idx + 1..].trim());
        expr = &expr[..idx];
    }

    // split :precision  like  expr:.3
    let mut prec = None;
    if let Some(idx) = expr.find(':') {
        prec = expr[idx + 1..]
            .trim()
            .strip_prefix('.')
            .and_then(|p| p.parse::<usize>().ok());
        expr = &expr[..idx];
    }

    expr = expr.trim();

    // fast path for plain variable with no formatting
    if prec.is_none() && fmt_opts.is_none() && is_plain_ident(expr) {
        if let Some(raw) = vars.get(expr) {
            return raw.to_string();
        }
    }

    match eval_expr(expr, vars) {
        Some(val) => apply_full_format(val, prec, fmt_opts),
        None => src.to_string(),
    }
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
