use std::str::FromStr;
use super::super::super::field::Z251;

struct TokenList {
    tokens: Vec<Token<Z251>>,
}

#[derive(Debug, PartialEq)]
enum Key {
    In,
    Witness,
    Program,
    Equal,
    Mul,
    Add,
}

#[derive(Debug, PartialEq)]
enum Token<T> {
    Keyword(Key),
    Var(String),
    Parenthesis(ParenCase),
    Literal(T),
}

#[derive(Debug, PartialEq)]
enum ParenCase {
    Open,
    Close,
}

impl From<String> for TokenList {
    fn from(code: String) -> Self {
        use self::TokenParseErr::*;
        
        let mut current_line = 1;
        let mut tokens: Vec<Token<Z251>> = Vec::new();
        
        for line in code.lines() {
            for substr in line.split_whitespace() {
                match parse_token::<Z251>(substr) {
                    Err(MissingKey) => panic!("Error on line {}: no key found after '('", current_line),
                    Err(UnexpectedParen) => panic!("Error on line {}: unexpected parenthesis", current_line),
                    Err(UnexpectedKey) => panic!("Error on line {}: unexpected keyword", current_line),
                    Err(ParseLiteral) => panic!("Error on line {}: could not parse literal", current_line),
                    Ok(ref mut t) => tokens.append(t),
                }
            }

            current_line += 1;
        }

        TokenList { tokens }
    }
}

#[derive(Debug, PartialEq)]
enum TokenParseErr {
    MissingKey,
    UnexpectedParen,
    UnexpectedKey,
    ParseLiteral,
}

fn parse_token<T>(mut substr: &str) -> Result<Vec<Token<T>>, TokenParseErr>
where
    T: FromStr,
{
    use self::Key::*;
    use self::ParenCase::*;
    use self::Token::*;
    use self::TokenParseErr::*;

    // Possible valid substrings:
    // ({Keyword}
    // {Var}
    // {Var})
    // {Literal}

    let mut tokens: Vec<Token<T>> = Vec::new();

    if substr.starts_with("(") {
        tokens.push(Parenthesis(Open));
        let (_, s) = substr.split_at(1);
        substr = s;
    }

    if substr.len() == 0 {
        return Err(MissingKey);
    }

    match substr {
        "in" => tokens.push(Keyword(In)),
        "witness" => tokens.push(Keyword(Witness)),
        "program" => tokens.push(Keyword(Program)),
        "=" => tokens.push(Keyword(Equal)),
        "*" => tokens.push(Keyword(Mul)),
        "+" => tokens.push(Keyword(Add)),
        _ => {
            if substr.contains("(") {
                return Err(UnexpectedParen);
            } else if substr.contains("*") || substr.contains("+") || substr.contains("=") {
                return Err(UnexpectedKey);
            }

            let paren_at_end = substr.ends_with(")");
            if paren_at_end {
                if tokens.len() != 0 {
                    return Err(UnexpectedParen);
                }

                let (s, _) = substr.split_at(substr.len() - 1);
                substr = s;
            }

            // It is safe to unwrap because substr.len() >= 1
            let first = substr.chars().nth(0).unwrap();

            if first.is_numeric() {
                match substr.parse::<T>() {
                    Ok(n) => tokens.push(Literal(n)),
                    _ => return Err(ParseLiteral),
                }
            } else {
                tokens.push(Var(substr.to_owned()));
            }

            if paren_at_end {
                tokens.push(Parenthesis(Close));
            }
        }
    }

    Ok(tokens)
}

#[test]
fn parse_token_test() {
    use self::Key::*;
    use self::ParenCase::*;
    use self::Token::*;
    use self::TokenParseErr::*;
    
    // Valid substring examples
    let substr = "(in";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Parenthesis(Open), Keyword(In)]));
    let substr = "(witness";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Parenthesis(Open), Keyword(Witness)]));
    let substr = "(program";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Parenthesis(Open), Keyword(Program)]));
    let substr = "(=";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Parenthesis(Open), Keyword(Equal)]));
    let substr = "(*";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Parenthesis(Open), Keyword(Mul)]));
    let substr = "(+";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Parenthesis(Open), Keyword(Add)]));
    let substr = "x";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Var("x".to_string())]));
    let substr = "y)";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Var("y".to_string()), Parenthesis(Close)]));
    let substr = "9";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Literal(9.into())]));
    let substr = "9)";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Literal(9.into()), Parenthesis(Close)]));

    // Invalid substring examples
    let substr = "(";
    assert_eq!(parse_token::<Z251>(substr), Err(MissingKey));
    let substr = "(vari(able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedParen));
    let substr = "vari(able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedParen));
    let substr = "(variable)";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedParen));
    let substr = "vari=able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedKey));
    let substr = "vari*able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedKey));
    let substr = "vari+able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedKey));
    let substr = "(vari=able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedKey));
    let substr = "(vari*able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedKey));
    let substr = "(vari+able";
    assert_eq!(parse_token::<Z251>(substr), Err(UnexpectedKey));
    let substr = "9variable";
    assert_eq!(parse_token::<Z251>(substr), Err(ParseLiteral));
}