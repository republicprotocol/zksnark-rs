use super::super::field::*;

mod ast;
pub mod dummy_rep;

pub trait RootRepresentation<T>
where
    T: Field,
{
    type Row: Iterator<Item = Self::Column>;
    type Column: Iterator<Item = (T, T)>;
    type Roots: Iterator<Item = T>;

    fn u(&self) -> Self::Row;
    fn v(&self) -> Self::Row;
    fn w(&self) -> Self::Row;
    fn roots(&self) -> Self::Roots;
    fn input(&self) -> usize;
}

pub trait TryParse<T, F>
where
    T: RootRepresentation<F>,
    F: Field,
{
    fn try_parse(&str) -> T;
}
