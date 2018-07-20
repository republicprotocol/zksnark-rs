use super::super::super::field::Z251;

struct TokenList {
    tokens: Vec<Token<Z251>>,
}

enum Key {
    In,
    Witness,
    Program,
    Equal,
    Mul,
    Add,
}

enum Token<T> {
    Keyword(Key),
    Var(String),
    Parenthesis(ParenCase),
    Literal(T),
}

enum ParenCase {
    Open,
    Close,
}

impl From<String> for TokenList {
    fn from(code: String) -> Self {
        use self::Key::*;
        use self::ParenCase::*;
        use self::Token::*;

        let mut tokens: Vec<Token<Z251>> = Vec::new();
        let substrs = code.split_whitespace();

        for mut substr in substrs {
            if substr.starts_with("(") {
                tokens.push(Parenthesis(Open));
                let (_, s) = substr.split_at(1);
                substr = s;
            }

            match substr.len() {
                0 => panic!("Open parenthesis must be followed by keyword"),
                1 => match substr {
                    "=" => tokens.push(Keyword(Equal)),
                    "*" => tokens.push(Keyword(Mul)),
                    "+" => tokens.push(Keyword(Add)),
                    s @ _ => tokens.push(Var(s.to_owned())),
                },
                _ => {
                    if substr.contains("(") {
                        panic!("Unexpected open parenthesis in token");
                    } else if substr.contains("*") || substr.contains("+") {
                        panic!("Unexpected operator in token");
                    } else if substr.contains("=") {
                        panic!("Unexpected '=' in token");
                    }

                    let paren_at_end = substr.ends_with(")");
                    if paren_at_end {
                        let (s, _) = substr.split_at(substr.len() - 1);
                        substr = s;
                    }

                    // It is safe to unwrap because in this match arm substr.len() >= 2
                    let first = substr.chars().nth(0).unwrap();

                    if first.is_numeric() {
                        match substr.parse::<usize>() {
                            Ok(n) => tokens.push(Literal(n.into())),
                            _ => panic!("Could not parse literal"),
                        }
                    } else {
                        tokens.push(Var(substr.to_owned()));
                    }

                    if paren_at_end {
                        tokens.push(Parenthesis(Close));
                    }
                }
            }
        }

        TokenList { tokens }
    }
}

enum TokenParseErr {}

fn parse_token<T>(substr: &str) -> Result<Vec<Token<T>>, TokenParseErr> {
    unimplemented!()
}
