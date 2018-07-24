use std::path::Path;

mod encryption;
pub mod field;
pub mod groth16;

pub struct Relation {}
pub enum RelationErr {}

pub struct ReferenceString {}
pub enum SetupErr {}

pub struct Proof {}
pub enum ProveErr {}

pub struct Assignment {}

pub struct Input {}

pub fn relation<P>(_path: P) -> Result<Relation, RelationErr>
where
    P: AsRef<Path>,
{
    unimplemented!()
}

impl Relation {
    pub fn setup(&self) -> Result<ReferenceString, SetupErr> {
        unimplemented!()
    }

    pub fn prove(
        &self,
        _crs: &ReferenceString,
        _assignment: &Assignment,
    ) -> Result<Proof, ProveErr> {
        unimplemented!()
    }

    pub fn verify(&self, _crs: &ReferenceString, _input: &Input) -> bool {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {}
