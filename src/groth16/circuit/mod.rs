use super::super::field::*;

pub mod dummy_rep;
mod ast;

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