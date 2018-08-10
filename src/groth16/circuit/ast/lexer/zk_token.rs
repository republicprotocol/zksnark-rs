use super::{Interval, Position, Span, Token};
use failure::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::str::FromStr;

use super::super::{Host, Visitor};

#[derive(Debug, PartialEq)]
pub enum ZKToken<T> {
    // Keywords
    In(Interval),
    Out(Interval),
    Verify(Interval),
    Program(Interval),

    // Operators
    Assign(Interval),
    Mul(Interval),
    Add(Interval),

    // Reserved symbols
    ParenL(Interval),
    ParenR(Interval),

    // Values
    Var(Interval, String),
    Literal(Interval, T),
}

impl<T> Span for ZKToken<T> {
    fn interval(&self) -> &Interval {
        use self::ZKToken::*;

        match self {
            In(i) => i,
            Out(i) => i,
            Verify(i) => i,
            Program(i) => i,
            Assign(i) => i,
            Mul(i) => i,
            Add(i) => i,
            ParenL(i) => i,
            ParenR(i) => i,
            Var(i, _) => i,
            Literal(i, _) => i,
        }
    }
}

impl<T> Token for ZKToken<T>
where
    T: FromStr,
{
    fn is_reserved(c: &char) -> bool {
        let reserved = "=*+()";
        reserved.contains(*c)
    }

    fn try_from_str(s: &str, interval: &Interval) -> Option<Self> {
        use self::ZKToken::*;

        match s {
            "in" => Some(In(interval.clone())),
            "out" => Some(Out(interval.clone())),
            "verify" => Some(Verify(interval.clone())),
            "program" => Some(Program(interval.clone())),
            "=" => Some(Assign(interval.clone())),
            "*" => Some(Mul(interval.clone())),
            "+" => Some(Add(interval.clone())),
            "(" => Some(ParenL(interval.clone())),
            ")" => Some(ParenR(interval.clone())),
            s if first_is_numeric(s) => T::from_str(s).ok().map(|t| Literal(interval.clone(), t)),
            s => Some(Var(interval.clone(), s.to_owned())),
        }
    }
}

fn first_is_numeric(s: &str) -> bool {
    s.chars().next().map_or(false, |c| c.is_numeric())
}

struct TokenList<T>(Vec<ZKToken<T>>);

impl<T> Host for TokenList<T> {}

struct ParenMatchVisitor<T> {
    phantom: PhantomData<T>,
}

#[derive(Debug, Fail)]
enum ParenMatchError {
    MissingR(Position),
    MissingL(Position),
}

impl Display for ParenMatchError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::ParenMatchError::*;

        match self {
            MissingR(pos) => write!(f, "'(' at {} has no matching ')'", pos),
            MissingL(pos) => write!(f, "')' at {} has no matching '('", pos),
        }
    }
}

impl<T> Visitor for ParenMatchVisitor<T> {
    type Target = TokenList<T>;
    type Ret = Result<(), Error>;

    fn visit(&mut self, host: &Self::Target) -> Self::Ret {
        use self::ParenMatchError::*;
        use self::ZKToken::*;

        for (i, token) in host.0.iter().enumerate() {
            match token {
                ParenL(Interval(pos, _)) => {
                    let mut depth = 1;

                    for token in host.0.iter().skip(i) {
                        depth += match token {
                            ParenL(_) => 1,
                            ParenR(_) => -1,
                            _ => 0,
                        };

                        if depth == 0 {
                            break;
                        }
                    }

                    if depth != 0 {
                        Err(MissingR(*pos))?;
                    }
                }
                ParenR(Interval(pos, _)) => {
                    let mut depth = 1;

                    for token in host.0.iter().rev().skip(host.0.len() - i) {
                        depth += match token {
                            ParenR(_) => 1,
                            ParenL(_) => -1,
                            _ => 0,
                        };

                        if depth == 0 {
                            break;
                        }
                    }

                    if depth != 0 {
                        Err(MissingL(*pos))?;
                    }
                }
            }
        }

        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Position;
    use super::*;

    #[test]
    fn is_reserved_test() {
        let reserved = "=*+()";
        let not_reserved = "`1234567890-qwertyuiop[]asdfghjkl;'zxcvbnm,./~!@#$%^&_QWERTYUIOP|ASDFGHJKL:ZXCVBNNM<>?";

        for c in reserved.chars() {
            assert!(ZKToken::<usize>::is_reserved(&c));
        }
        for c in not_reserved.chars() {
            assert!(!ZKToken::<usize>::is_reserved(&c));
        }
    }

    #[test]
    fn try_from_str_test() {
        use self::ZKToken::*;

        let start = Position {
            filename: "foo.zk".to_string(),
            line: 15,
            column: 23589,
        };
        let end = Position {
            filename: "foo.zk".to_string(),
            line: 198,
            column: 4,
        };
        let interval = Interval(start, end);

        // Ok
        assert_eq!(
            ZKToken::<usize>::try_from_str("in", &interval),
            Some(In(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("out", &interval),
            Some(Out(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("verify", &interval),
            Some(Verify(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("program", &interval),
            Some(Program(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("=", &interval),
            Some(Assign(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("*", &interval),
            Some(Mul(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("+", &interval),
            Some(Add(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("(", &interval),
            Some(ParenL(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str(")", &interval),
            Some(ParenR(interval.clone()))
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("variable", &interval),
            Some(Var(interval.clone(), "variable".to_owned()))
        );

        // Err
        assert_eq!(ZKToken::<usize>::try_from_str("6variable", &interval), None);
    }
}
