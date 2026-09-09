use glam::Vec3;
use revision::{
    DeserializeRevisioned, Revisioned, SerializeRevisioned, SkipRevisioned, revisioned,
};
use smallvec::SmallVec;
use std::io::{Read, Write};

#[revisioned(revision = 1)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SerializableVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl SerializableVec3 {
    pub fn as_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    pub fn from_vec3(vec3: Vec3) -> SerializableVec3 {
        SerializableVec3 {
            x: vec3.x,
            y: vec3.y,
            z: vec3.z,
        }
    }
}
impl From<Vec3> for SerializableVec3 {
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<SerializableVec3> for Vec3 {
    fn from(v: SerializableVec3) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}
impl PartialEq for SerializableVec3 {
    fn eq(&self, other: &Self) -> bool {
        self.x.to_bits() == other.x.to_bits()
            && self.y.to_bits() == other.y.to_bits()
            && self.z.to_bits() == other.z.to_bits()
    }
}

impl Eq for SerializableVec3 {}

impl std::hash::Hash for SerializableVec3 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
        self.z.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RevisionedSmallVec<T, const N: usize>(pub SmallVec<T, N>);

impl<T, const N: usize> Revisioned for RevisionedSmallVec<T, N>
where
    T: Revisioned,
{
    fn revision() -> u16 {
        1
    }
}

impl<T, const N: usize> SerializeRevisioned for RevisionedSmallVec<T, N>
where
    T: SerializeRevisioned + 'static,
{
    #[inline]
    fn serialize_revisioned<W: Write>(&self, writer: &mut W) -> Result<(), revision::Error> {
        let len = self.0.len();

        len.serialize_revisioned(writer)?;

        for value in &self.0 {
            value.serialize_revisioned(writer)?;
        }

        Ok(())
    }
}

impl<T, const N: usize> DeserializeRevisioned for RevisionedSmallVec<T, N>
where
    T: DeserializeRevisioned + 'static,
{
    #[inline]
    fn deserialize_revisioned<R: Read>(reader: &mut R) -> Result<Self, revision::Error> {
        let len = usize::deserialize_revisioned(reader)?;

        let mut values = SmallVec::<T, N>::with_capacity(len);

        for _ in 0..len {
            values.push(T::deserialize_revisioned(reader)?);
        }

        Ok(Self(values))
    }
}
impl<T, const N: usize> SkipRevisioned for RevisionedSmallVec<T, N>
where
    T: SkipRevisioned,
{
    #[inline]
    fn skip_revisioned<R: Read>(reader: &mut R) -> Result<(), revision::Error> {
        let len = usize::deserialize_revisioned(reader)?;

        for _ in 0..len {
            T::skip_revisioned(reader)?;
        }

        Ok(())
    }
}
