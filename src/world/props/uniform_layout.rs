use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_yaml::Value;
use std::collections::HashSet;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniformYaml {
    pub fields: Vec<UniformFieldYaml>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniformFieldYaml {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct UniformMember {
    pub name: String,
    pub wgsl_type: String,
    pub offset: u64,
    pub size: u64,
    pub alignment: u64,
}

#[derive(Debug, Clone)]
pub struct UniformLayout {
    pub size: u64,
    pub members: Vec<UniformMember>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
enum ScalarType {
    F32,
    I32,
    U32,
}

#[derive(Debug, Clone)]
enum UniformType {
    Scalar(ScalarType),
    Vector {
        lanes: usize,
        scalar: ScalarType,
    },
    Matrix {
        columns: usize,
        rows: usize,
    },
    Array {
        element: Box<UniformType>,
        count: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct TypeLayout {
    alignment: u64,
    size: u64,
    stride: Option<u64>,
}

impl UniformLayout {
    pub fn from_yaml(source: &str) -> Result<Self> {
        Self::from_yaml_with_limit(source, 65_536)
    }

    pub fn from_yaml_with_limit(source: &str, max_size: u64) -> Result<Self> {
        let declaration: UniformYaml =
            serde_yaml::from_str(source).context("Failed to parse uniform YAML")?;

        Self::from_declaration(declaration, max_size)
    }

    pub fn from_value(value: &Value, max_size: u64) -> Result<Self> {
        let declaration: UniformYaml =
            serde_yaml::from_value(value.clone()).context("Invalid uniform declaration")?;

        Self::from_declaration(declaration, max_size)
    }

    pub fn from_declaration(declaration: UniformYaml, max_size: u64) -> Result<Self> {
        ensure!(
            !declaration.fields.is_empty(),
            "A uniform must contain at least one field"
        );

        let mut names = HashSet::new();
        let mut members = Vec::with_capacity(declaration.fields.len());
        let mut prepared = Vec::with_capacity(declaration.fields.len());

        let mut cursor = 0u64;
        let mut struct_alignment = 16u64;

        for field in declaration.fields {
            validate_identifier(&field.name)?;

            ensure!(
                names.insert(field.name.clone()),
                "Duplicate uniform field '{}'",
                field.name
            );

            let ty = UniformType::parse(&field.ty)
                .with_context(|| format!("Invalid type for uniform field '{}'", field.name))?;

            let layout = ty.layout()?;

            cursor = align_up(cursor, layout.alignment)?;
            let offset = cursor;

            cursor = cursor
                .checked_add(layout.size)
                .context("Uniform size overflow")?;

            struct_alignment = struct_alignment.max(layout.alignment);

            members.push(UniformMember {
                name: field.name.clone(),
                wgsl_type: ty.to_wgsl(),
                offset,
                size: layout.size,
                alignment: layout.alignment,
            });

            prepared.push((field, ty, offset, layout.size));
        }

        let size = align_up(cursor, struct_alignment)?;

        ensure!(
            size <= max_size,
            "Uniform requires {} bytes, exceeding the configured limit of {} bytes",
            size,
            max_size
        );

        let byte_count = usize::try_from(size).context("Uniform is too large for this platform")?;
        let mut bytes = vec![0u8; byte_count];

        for (field, ty, offset, field_size) in prepared {
            let start = usize::try_from(offset)?;
            let end = start
                .checked_add(usize::try_from(field_size)?)
                .context("Uniform field range overflow")?;

            ty.write_value(
                &field.value,
                &mut bytes[start..end],
            )
                .with_context(|| format!("Invalid value for uniform field '{}'", field.name))?;
        }

        Ok(Self {
            size,
            members,
            bytes,
        })
    }

    pub fn create_buffer(
        &self,
        device: &wgpu::Device,
        label: &str,
    ) -> Result<wgpu::Buffer> {
        let device_limit = device.limits().max_uniform_buffer_binding_size as u64;

        ensure!(
            self.size <= device_limit,
            "Uniform requires {} bytes, but this device allows at most {} bytes per uniform binding",
            self.size,
            device_limit
        );

        Ok(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &self.bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }))
    }

    pub fn member(&self, name: &str) -> Option<&UniformMember> {
        self.members.iter().find(|member| member.name == name)
    }
}

impl UniformType {
    fn parse(input: &str) -> Result<Self> {
        let compact: String = input.chars().filter(|c| !c.is_whitespace()).collect();

        if let Some(inner) = generic_inner(&compact, "array<") {
            let (element_type, count) = split_generic_args(inner)?;

            let element = Box::new(Self::parse(element_type)?);
            let count: usize = count
                .parse()
                .context("Array length must be a positive integer")?;

            ensure!(count > 0, "Uniform arrays cannot have zero elements");

            return Ok(Self::Array { element, count });
        }

        for lanes in 2..=4 {
            let prefix = format!("vec{}<", lanes);

            if let Some(inner) = generic_inner(&compact, &prefix) {
                return Ok(Self::Vector {
                    lanes,
                    scalar: parse_scalar(inner)?,
                });
            }
        }

        if let Some(rest) = compact.strip_prefix("mat") {
            let (dimensions, scalar) = rest
                .split_once('<')
                .context("Invalid matrix type")?;

            let scalar = scalar
                .strip_suffix('>')
                .context("Invalid matrix type")?;

            ensure!(scalar == "f32", "Only f32 matrices are supported");

            let (columns, rows) = dimensions
                .split_once('x')
                .context("Matrix type must use matCxR syntax")?;

            let columns: usize = columns.parse().context("Invalid matrix column count")?;
            let rows: usize = rows.parse().context("Invalid matrix row count")?;

            ensure!(
                (2..=4).contains(&columns) && (2..=4).contains(&rows),
                "Matrix dimensions must be between 2 and 4"
            );

            return Ok(Self::Matrix { columns, rows });
        }

        Ok(Self::Scalar(parse_scalar(&compact)?))
    }

    fn layout(&self) -> Result<TypeLayout> {
        match self {
            Self::Scalar(_) => Ok(TypeLayout {
                alignment: 4,
                size: 4,
                stride: None,
            }),

            Self::Vector { lanes, .. } => {
                let alignment = match lanes {
                    2 => 8,
                    3 | 4 => 16,
                    _ => bail!("Invalid vector width"),
                };

                Ok(TypeLayout {
                    alignment,
                    size: (*lanes as u64) * 4,
                    stride: None,
                })
            }

            Self::Matrix { columns, rows } => {
                let vector_alignment = match rows {
                    2 => 8,
                    3 | 4 => 16,
                    _ => bail!("Invalid matrix row count"),
                };

                let vector_size = (*rows as u64) * 4;
                let column_stride = align_up(vector_size, vector_alignment)?;
                let size = column_stride
                    .checked_mul(*columns as u64)
                    .context("Matrix size overflow")?;

                Ok(TypeLayout {
                    alignment: vector_alignment,
                    size,
                    stride: Some(column_stride),
                })
            }

            Self::Array { element, count } => {
                let element_layout = element.layout()?;
                let alignment = align_up(element_layout.alignment, 16)?;
                let stride = align_up(element_layout.size, alignment)?;
                let size = stride
                    .checked_mul(*count as u64)
                    .context("Array size overflow")?;

                Ok(TypeLayout {
                    alignment,
                    size,
                    stride: Some(stride),
                })
            }
        }
    }

    fn to_wgsl(&self) -> String {
        match self {
            Self::Scalar(scalar) => scalar.to_wgsl().to_string(),

            Self::Vector { lanes, scalar } => {
                format!("vec{}<{}>", lanes, scalar.to_wgsl())
            }

            Self::Matrix { columns, rows } => {
                format!("mat{}x{}<f32>", columns, rows)
            }

            Self::Array { element, count } => {
                format!("array<{}, {}>", element.to_wgsl(), count)
            }
        }
    }

    fn write_value(&self, value: &Value, output: &mut [u8]) -> Result<()> {
        match self {
            Self::Scalar(scalar) => {
                write_scalar(*scalar, value, output)
            }

            Self::Vector { lanes, scalar } => {
                let values = sequence(value, *lanes)?;

                for (index, value) in values.iter().enumerate() {
                    let start = index * 4;
                    write_scalar(
                        *scalar,
                        value,
                        &mut output[start..start + 4],
                    )?;
                }

                Ok(())
            }

            Self::Matrix { columns, rows } => {
                let column_values = sequence(value, *columns)?;
                let layout = self.layout()?;
                let column_stride = layout.stride.unwrap();

                for (column_index, column_value) in column_values.iter().enumerate() {
                    let row_values = sequence(column_value, *rows)?;
                    let column_start = column_index * column_stride as usize;

                    for (row_index, row_value) in row_values.iter().enumerate() {
                        let start = column_start + row_index * 4;

                        write_scalar(
                            ScalarType::F32,
                            row_value,
                            &mut output[start..start + 4],
                        )?;
                    }
                }

                Ok(())
            }

            Self::Array { element, count } => {
                let values = sequence(value, *count)?;
                let element_layout = element.layout()?;
                let array_layout = self.layout()?;
                let stride = array_layout.stride.unwrap() as usize;
                let element_size = element_layout.size as usize;

                for (index, value) in values.iter().enumerate() {
                    let start = index * stride;
                    element.write_value(
                        value,
                        &mut output[start..start + element_size],
                    )?;
                }

                Ok(())
            }
        }
    }
}

impl ScalarType {
    fn to_wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
        }
    }
}

fn parse_scalar(input: &str) -> Result<ScalarType> {
    match input {
        "f32" => Ok(ScalarType::F32),
        "i32" => Ok(ScalarType::I32),
        "u32" => Ok(ScalarType::U32),
        _ => bail!("Unsupported uniform scalar type '{}'", input),
    }
}

fn generic_inner<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    input
        .strip_prefix(prefix)?
        .strip_suffix('>')
}

fn split_generic_args(input: &str) -> Result<(&str, &str)> {
    let mut depth = 0usize;

    for (index, character) in input.char_indices() {
        match character {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                return Ok((input[..index].trim(), input[index + 1..].trim()));
            }
            _ => {}
        }
    }

    bail!("Invalid generic type arguments")
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    ensure!(alignment.is_power_of_two(), "Invalid alignment");

    let mask = alignment - 1;

    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .context("Uniform alignment overflow")
}

fn sequence(value: &Value, expected: usize) -> Result<&[Value]> {
    let values = value
        .as_sequence()
        .context("Expected a YAML sequence")?;

    ensure!(
        values.len() == expected,
        "Expected {} elements, got {}",
        expected,
        values.len()
    );

    Ok(values)
}

fn write_scalar(scalar: ScalarType, value: &Value, output: &mut [u8]) -> Result<()> {
    match scalar {
        ScalarType::F32 => {
            let number = value
                .as_f64()
                .or_else(|| value.as_i64().map(|number| number as f64))
                .or_else(|| value.as_u64().map(|number| number as f64))
                .context("Expected a numeric f32 value")?;

            ensure!(number.is_finite(), "f32 value must be finite");

            let number = number as f32;
            ensure!(number.is_finite(), "f32 value is out of range");

            output[..4].copy_from_slice(&number.to_le_bytes());
        }

        ScalarType::I32 => {
            let number = value
                .as_i64()
                .context("Expected an integer i32 value")?;

            let number = i32::try_from(number)
                .context("Integer is outside the i32 range")?;

            output[..4].copy_from_slice(&number.to_le_bytes());
        }

        ScalarType::U32 => {
            let number = value
                .as_u64()
                .context("Expected a non-negative integer u32 value")?;

            let number = u32::try_from(number)
                .context("Integer is outside the u32 range")?;

            output[..4].copy_from_slice(&number.to_le_bytes());
        }
    }

    Ok(())
}

fn validate_identifier(name: &str) -> Result<()> {
    let mut characters = name.chars();

    let first = characters
        .next()
        .context("Uniform field name cannot be empty")?;

    ensure!(
        first.is_ascii_alphabetic() || first == '_',
        "Invalid WGSL identifier '{}'",
        name
    );

    ensure!(
        characters.all(|character| character.is_ascii_alphanumeric() || character == '_'),
        "Invalid WGSL identifier '{}'",
        name
    );

    Ok(())
}
