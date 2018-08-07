use super::{Interval, Span, Token};
use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug)]
enum ZKTokenError {
    ParseLiteral(String, Interval),
}

impl Display for ZKTokenError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::ZKTokenError::*;

        match self {
            ParseLiteral(token, Interval(start, _)) => {
                write!(f, "{}: could not parse literal '{}'", start, token)
            }
        }
    }
}

impl Error for ZKTokenError {}

#[derive(Debug, PartialEq)]
enum ZKToken<T> {
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

impl<T, E> Token for ZKToken<T>
where
    T: FromStr<Err = E>,
    E: Error + 'static,
{
    fn is_reserved(c: &char) -> bool {
        let reserved = "=*+()";
        reserved.contains(*c)
    }

    fn try_from_str(s: &str, interval: &Interval) -> Result<Self, Box<Error>> {
        use self::ZKToken::*;

        match s {
            "in" => Ok(In(interval.clone())),
            "out" => Ok(Out(interval.clone())),
            "verify" => Ok(Verify(interval.clone())),
            "program" => Ok(Program(interval.clone())),
            "=" => Ok(Assign(interval.clone())),
            "*" => Ok(Mul(interval.clone())),
            "+" => Ok(Add(interval.clone())),
            "(" => Ok(ParenL(interval.clone())),
            ")" => Ok(ParenR(interval.clone())),
            s if first_is_numeric(s) => Ok(Literal(
                interval.clone(),
                T::from_str(s)
                    .map_err(|_| ZKTokenError::ParseLiteral(s.to_owned(), interval.clone()))?,
            )),
            s => Ok(Var(interval.clone(), s.to_owned())),
        }
    }
}

fn first_is_numeric(s: &str) -> bool {
    s.chars().next().map_or(false, |c| c.is_numeric())
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
            ZKToken::<usize>::try_from_str("in", &interval).unwrap(),
            In(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("out", &interval).unwrap(),
            Out(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("verify", &interval).unwrap(),
            Verify(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("program", &interval).unwrap(),
            Program(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("=", &interval).unwrap(),
            Assign(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("*", &interval).unwrap(),
            Mul(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("+", &interval).unwrap(),
            Add(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("(", &interval).unwrap(),
            ParenL(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str(")", &interval).unwrap(),
            ParenR(interval.clone())
        );
        assert_eq!(
            ZKToken::<usize>::try_from_str("variable", &interval).unwrap(),
            Var(interval.clone(), "variable".to_owned())
        );

        // Err
        assert!(ZKToken::<usize>::try_from_str("6variable", &interval).is_err());
    }
}
