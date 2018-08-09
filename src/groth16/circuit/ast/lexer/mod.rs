use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::path::Path;

mod zk_token;

#[derive(Clone, Debug, PartialEq)]
struct Position {
    filename: String,
    line: usize,
    column: usize,
}

impl Position {
    fn col_offset(&self, offset: isize) -> Self {
        let Position {
            filename,
            line,
            column,
        } = self.clone();

        let column = column as isize + offset;
        let column = if column < 0 {
            panic!("Attempted to offset a Position too far to the left");
        } else {
            column as usize
        };

        Position {
            filename,
            line,
            column,
        }
    }
}

impl Display for Position {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:{}:{}", self.filename, self.line, self.column)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Interval(Position, Position);

pub trait Span {
    fn interval(&self) -> &Interval;
}

pub trait Token: Sized + Span {
    fn is_reserved(&char) -> bool;
    fn try_from_str(&str, &Interval) -> Option<Self>;
}

#[derive(Debug, PartialEq)]
pub enum LexerError {
    ParseLiteral(String, Interval),
    Filename,
}

impl Display for LexerError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::LexerError::*;

        match self {
            ParseLiteral(token, Interval(start, _)) => {
                write!(f, "{}: could not parse literal '{}'", start, token)
            }
            Filename => write!(f, "Could not convert filename to String"),
        }
    }
}

impl Error for LexerError {}

impl From<::std::io::Error> for LexerError {
    fn from(_: ::std::io::Error) -> Self {
        LexerError::Filename
    }
}

pub fn lex_file<T, P>(file: P) -> Result<Vec<T>, LexerError>
where
    T: Token,
    P: AsRef<Path>,
{
    use self::LexerError::*;

    let mut tokens = Vec::new();
    let filename = file.as_ref().to_str().ok_or(Filename)?.to_owned();

    let mut file = File::open(file)?;
    let mut code = String::new();
    file.read_to_string(&mut code)?;

    let mut column: usize;
    let mut line = 1;
    let mut context = "".to_string();

    let mut start_position: Position;
    let mut current_position = Position {
        filename: filename.clone(),
        line: 1,
        column: 1,
    };

    for ln in code.lines() {
        column = 2;
        start_position = Position {
            filename: filename.clone(),
            line,
            column: 1,
        };

        for c in ln.chars() {
            current_position = Position {
                filename: filename.clone(),
                line,
                column,
            };
            let prev_position = current_position.col_offset(-1);

            if T::is_reserved(&c) || c.is_whitespace() {
                if context.len() > 0 {
                    let interval = Interval(start_position, prev_position.clone());

                    tokens.push(T::try_from_str(&context, &interval)
                        .ok_or(ParseLiteral(context.clone(), interval))?);
                    context = "".to_string();
                }

                if T::is_reserved(&c) {
                    let interval = Interval(prev_position, current_position.clone());

                    tokens.push(T::try_from_str(&c.to_string(), &interval)
                        .ok_or(ParseLiteral(context.clone(), interval))?);
                }

                start_position = current_position.clone();
            } else {
                context.push(c);
            }

            column += 1;
        }

        if context.len() > 0 {
            let interval = Interval(start_position, current_position.clone());

            tokens.push(T::try_from_str(&context, &interval)
                .ok_or(ParseLiteral(context.clone(), interval))?);
            context = "".to_string();
        }

        line += 1;
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::zk_token::ZKToken::*;
    use super::*;

    #[test]
    fn lex_file_test() {
        let file = "test_programs/simple.zk";
        let actual = lex_file(file);

        let positions = [
            Position {
                // (
                filename: file.to_owned(),
                line: 1,
                column: 1,
            },
            Position {
                filename: file.to_owned(),
                line: 1,
                column: 2,
            },
            Position {
                // in
                filename: file.to_owned(),
                line: 1,
                column: 2,
            },
            Position {
                filename: file.to_owned(),
                line: 1,
                column: 4,
            },
            Position {
                // a
                filename: file.to_owned(),
                line: 1,
                column: 5,
            },
            Position {
                filename: file.to_owned(),
                line: 1,
                column: 6,
            },
            Position {
                // b
                filename: file.to_owned(),
                line: 1,
                column: 7,
            },
            Position {
                filename: file.to_owned(),
                line: 1,
                column: 8,
            },
            Position {
                // c
                filename: file.to_owned(),
                line: 1,
                column: 9,
            },
            Position {
                filename: file.to_owned(),
                line: 1,
                column: 10,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 1,
                column: 10,
            },
            Position {
                filename: file.to_owned(),
                line: 1,
                column: 11,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 2,
                column: 1,
            },
            Position {
                filename: file.to_owned(),
                line: 2,
                column: 2,
            },
            Position {
                // out
                filename: file.to_owned(),
                line: 2,
                column: 2,
            },
            Position {
                filename: file.to_owned(),
                line: 2,
                column: 5,
            },
            Position {
                // x
                filename: file.to_owned(),
                line: 2,
                column: 6,
            },
            Position {
                filename: file.to_owned(),
                line: 2,
                column: 7,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 2,
                column: 7,
            },
            Position {
                filename: file.to_owned(),
                line: 2,
                column: 8,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 3,
                column: 1,
            },
            Position {
                filename: file.to_owned(),
                line: 3,
                column: 2,
            },
            Position {
                // verify
                filename: file.to_owned(),
                line: 3,
                column: 2,
            },
            Position {
                filename: file.to_owned(),
                line: 3,
                column: 8,
            },
            Position {
                // b
                filename: file.to_owned(),
                line: 3,
                column: 9,
            },
            Position {
                filename: file.to_owned(),
                line: 3,
                column: 10,
            },
            Position {
                // x
                filename: file.to_owned(),
                line: 3,
                column: 11,
            },
            Position {
                filename: file.to_owned(),
                line: 3,
                column: 12,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 3,
                column: 12,
            },
            Position {
                filename: file.to_owned(),
                line: 3,
                column: 13,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 5,
                column: 1,
            },
            Position {
                filename: file.to_owned(),
                line: 5,
                column: 2,
            },
            Position {
                // program
                filename: file.to_owned(),
                line: 5,
                column: 2,
            },
            Position {
                filename: file.to_owned(),
                line: 5,
                column: 9,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 6,
                column: 5,
            },
            Position {
                filename: file.to_owned(),
                line: 6,
                column: 6,
            },
            Position {
                // =
                filename: file.to_owned(),
                line: 6,
                column: 6,
            },
            Position {
                filename: file.to_owned(),
                line: 6,
                column: 7,
            },
            Position {
                // temp
                filename: file.to_owned(),
                line: 6,
                column: 8,
            },
            Position {
                filename: file.to_owned(),
                line: 6,
                column: 12,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 7,
                column: 9,
            },
            Position {
                filename: file.to_owned(),
                line: 7,
                column: 10,
            },
            Position {
                // *
                filename: file.to_owned(),
                line: 7,
                column: 10,
            },
            Position {
                filename: file.to_owned(),
                line: 7,
                column: 11,
            },
            Position {
                // a
                filename: file.to_owned(),
                line: 7,
                column: 12,
            },
            Position {
                filename: file.to_owned(),
                line: 7,
                column: 13,
            },
            Position {
                // b
                filename: file.to_owned(),
                line: 7,
                column: 14,
            },
            Position {
                filename: file.to_owned(),
                line: 7,
                column: 15,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 7,
                column: 15,
            },
            Position {
                filename: file.to_owned(),
                line: 7,
                column: 16,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 7,
                column: 16,
            },
            Position {
                filename: file.to_owned(),
                line: 7,
                column: 17,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 8,
                column: 5,
            },
            Position {
                filename: file.to_owned(),
                line: 8,
                column: 6,
            },
            Position {
                // =
                filename: file.to_owned(),
                line: 8,
                column: 6,
            },
            Position {
                filename: file.to_owned(),
                line: 8,
                column: 7,
            },
            Position {
                // x
                filename: file.to_owned(),
                line: 8,
                column: 8,
            },
            Position {
                filename: file.to_owned(),
                line: 8,
                column: 9,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 9,
                column: 9,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 10,
            },
            Position {
                // *
                filename: file.to_owned(),
                line: 9,
                column: 10,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 11,
            },
            Position {
                // 1
                filename: file.to_owned(),
                line: 9,
                column: 12,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 13,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 9,
                column: 14,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 15,
            },
            Position {
                // +
                filename: file.to_owned(),
                line: 9,
                column: 15,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 16,
            },
            Position {
                // (
                filename: file.to_owned(),
                line: 9,
                column: 17,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 18,
            },
            Position {
                // *
                filename: file.to_owned(),
                line: 9,
                column: 18,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 19,
            },
            Position {
                // 4
                filename: file.to_owned(),
                line: 9,
                column: 20,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 21,
            },
            Position {
                // temp
                filename: file.to_owned(),
                line: 9,
                column: 22,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 26,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 9,
                column: 26,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 27,
            },
            Position {
                // c
                filename: file.to_owned(),
                line: 9,
                column: 28,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 29,
            },
            Position {
                // 6
                filename: file.to_owned(),
                line: 9,
                column: 30,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 31,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 9,
                column: 31,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 32,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 9,
                column: 32,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 33,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 9,
                column: 33,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 34,
            },
            Position {
                // )
                filename: file.to_owned(),
                line: 9,
                column: 34,
            },
            Position {
                filename: file.to_owned(),
                line: 9,
                column: 35,
            },
        ];

        let mut intervals = Vec::new();
        for i in 0..45 {
            intervals.push(Interval(
                positions[2 * i].clone(),
                positions[2 * i + 1].clone(),
            ));
        }

        let expected = vec![
            ParenL::<usize>(intervals[0].clone()),
            In::<usize>(intervals[1].clone()),
            Var::<usize>(intervals[2].clone(), "a".to_owned()),
            Var::<usize>(intervals[3].clone(), "b".to_owned()),
            Var::<usize>(intervals[4].clone(), "c".to_owned()),
            ParenR::<usize>(intervals[5].clone()),
            ParenL::<usize>(intervals[6].clone()),
            Out::<usize>(intervals[7].clone()),
            Var::<usize>(intervals[8].clone(), "x".to_owned()),
            ParenR::<usize>(intervals[9].clone()),
            ParenL::<usize>(intervals[10].clone()),
            Verify::<usize>(intervals[11].clone()),
            Var::<usize>(intervals[12].clone(), "b".to_owned()),
            Var::<usize>(intervals[13].clone(), "x".to_owned()),
            ParenR::<usize>(intervals[14].clone()),
            ParenL::<usize>(intervals[15].clone()),
            Program::<usize>(intervals[16].clone()),
            ParenL::<usize>(intervals[17].clone()),
            Assign::<usize>(intervals[18].clone()),
            Var::<usize>(intervals[19].clone(), "temp".to_owned()),
            ParenL::<usize>(intervals[20].clone()),
            Mul::<usize>(intervals[21].clone()),
            Var::<usize>(intervals[22].clone(), "a".to_owned()),
            Var::<usize>(intervals[23].clone(), "b".to_owned()),
            ParenR::<usize>(intervals[24].clone()),
            ParenR::<usize>(intervals[25].clone()),
            ParenL::<usize>(intervals[26].clone()),
            Assign::<usize>(intervals[27].clone()),
            Var::<usize>(intervals[28].clone(), "x".to_owned()),
            ParenL::<usize>(intervals[29].clone()),
            Mul::<usize>(intervals[30].clone()),
            Literal::<usize>(intervals[31].clone(), 1),
            ParenL::<usize>(intervals[32].clone()),
            Add::<usize>(intervals[33].clone()),
            ParenL::<usize>(intervals[34].clone()),
            Mul::<usize>(intervals[35].clone()),
            Literal::<usize>(intervals[36].clone(), 4),
            Var::<usize>(intervals[37].clone(), "temp".to_owned()),
            ParenR::<usize>(intervals[38].clone()),
            Var::<usize>(intervals[39].clone(), "c".to_owned()),
            Literal::<usize>(intervals[40].clone(), 6),
            ParenR::<usize>(intervals[41].clone()),
            ParenR::<usize>(intervals[42].clone()),
            ParenR::<usize>(intervals[43].clone()),
            ParenR::<usize>(intervals[44].clone()),
        ];

        assert_eq!(Ok(expected), actual);
    }
}
