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

impl Display for Position {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:{}:{}", self.filename, self.line, self.column)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Interval(Position, Position);

impl Interval {
    fn single(pos: Position) -> Self {
        Interval(pos.clone(), pos)
    }
}

trait Span {
    fn interval(&self) -> &Interval;
}

trait Token: Sized + Span {
    fn is_reserved(&char) -> bool;
    fn try_from_str(&str, &Interval) -> Result<Self, Box<Error>>;
}

#[derive(Debug)]
enum LexerError {
    UnknownToken(String, Interval),
    Filename,
}

impl Display for LexerError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use self::LexerError::*;

        match self {
            UnknownToken(token, Interval(start, _)) => {
                write!(f, "{}: Unknown token '{}'", start, token)
            }
            Filename => write!(f, "Could not convert filename to String"),
        }
    }
}

impl Error for LexerError {}

fn lex_file<T, P>(file: P) -> Result<Vec<T>, Box<Error>>
where
    T: Token,
    P: AsRef<Path>,
{
    use self::LexerError::*;

    let mut line = 1;
    let mut context = "".to_string();
    let mut tokens = Vec::new();
    let mut start_position: Position;

    let filename = file.as_ref().to_str().ok_or(Filename)?.to_owned();

    let mut file = File::open(file)?;
    let mut code = String::new();
    file.read_to_string(&mut code)?;

    for ln in code.lines() {
        let mut column = 1;
        start_position = Position {
            filename: filename.clone(),
            line,
            column,
        };

        for c in ln.chars() {
            if T::is_reserved(&c) || c.is_whitespace() {
                if context.len() > 0 {
                    let current_position = {
                        let filename = filename.clone();
                        Position {
                            filename,
                            line,
                            column: column - 1,
                        }
                    };
                    let interval = Interval(start_position, current_position);

                    tokens.push(T::try_from_str(&context, &interval)?);
                    context = "".to_string();
                }

                start_position = if T::is_reserved(&c) {
                    let current_position = Position {
                        filename: filename.clone(),
                        line,
                        column,
                    };
                    let interval = Interval::single(current_position);

                    tokens.push(T::try_from_str(&c.to_string(), &interval)?);
                    
                    Position {
                        filename: filename.clone(),
                        line,
                        column: column + 1,
                    }
                } else {
                    Position {
                        filename: filename.clone(),
                        line,
                        column,
                    }
                }
            } else {
                context.push(c);
            }

            column += 1;
        }

        line += 1;
    }

    Ok(tokens)
}
