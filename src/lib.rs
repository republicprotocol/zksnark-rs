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

pub struct CircuitInput {}

pub struct Assignment {}

pub fn relation<P>(_path: P) -> Result<Relation, RelationErr>
where
    P: AsRef<Path>,
{
    unimplemented!()
}

pub fn setup(_relation: Relation) -> Result<ReferenceString, SetupErr> {
    unimplemented!()
}

pub fn prove(_relation: Relation, _crs: &ReferenceString, _input: &CircuitInput) -> Result<Proof, ProveErr> {
    unimplemented!()
}

pub fn verify(_relation: Relation, _crs: &ReferenceString, _input: &Assignment) -> bool {
    unimplemented!()
}

#[cfg(test)]
mod tests {}
