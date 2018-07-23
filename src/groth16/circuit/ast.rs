use super::super::super::field::Z251;
use super::TryParse;
use std::str::FromStr;

#[derive(Debug, PartialEq)]
struct TokenList<T> {
    tokens: Vec<Token<T>>,
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

// impl From<String> for TokenList<Z251> {
//     fn from(code: String) -> Self {
//         use self::TokenParseErr::*;

//         let mut current_line = 1;
//         let mut tokens: Vec<Token<Z251>> = Vec::new();

//         for line in code.lines() {
//             for substr in line.split_whitespace() {
//                 match parse_token::<Z251>(substr) {
//                     Err(MissingKey) => {
//                         panic!("Error on line {}: no key found after '('", current_line)
//                     }
//                     Err(UnexpectedParen) => {
//                         panic!("Error on line {}: unexpected parenthesis", current_line)
//                     }
//                     Err(UnexpectedKey) => {
//                         panic!("Error on line {}: unexpected keyword", current_line)
//                     }
//                     Err(ParseLiteral) => {
//                         panic!("Error on line {}: could not parse literal", current_line)
//                     }
//                     Ok(ref mut t) => tokens.append(t),
//                 }
//             }

//             current_line += 1;
//         }

//         TokenList { tokens }
//     }
// }

fn try_to_list<T>(code: String) -> Result<TokenList<T>, ParseErr>
where
    T: FromStr,
{
    use self::ParseErr::*;
    use self::TokenParseErr::*;

    let mut current_line = 1;
    let mut tokens: Vec<Token<T>> = Vec::new();

    for line in code.lines() {
        for substr in line.split_whitespace() {
            match parse_token::<T>(substr) {
                Err(TokenErr(e)) => {
                    return Err(LineErr(current_line, e));
                }
                Ok(ref mut t) => tokens.append(t),
            }
        }

        current_line += 1;
    }

    Ok(TokenList { tokens })
}

#[derive(Debug, PartialEq)]
enum ParseErr {
    LineErr(usize, String),
}

#[derive(Debug, PartialEq)]
enum TokenParseErr {
    TokenErr(String),
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
        return Err(TokenErr("found whitespace after '('".to_string()));
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
                return Err(TokenErr("unexpected '('".to_string()));
            } else if substr.contains("*") || substr.contains("+") || substr.contains("=") {
                return Err(TokenErr("unexpected operator".to_string()));
            }

            let (start, end) = split_at_char(substr, ')');
            if tokens.len() != 0 && end.len() != 0 {
                return Err(TokenErr("unexpected ')'".to_string()));
            }

            // It is safe to unwrap because substr.len() >= 1
            let first = start.chars().nth(0).unwrap();

            if first.is_numeric() {
                match start.parse::<T>() {
                    Ok(n) => tokens.push(Literal(n)),
                    _ => return Err(TokenErr("could not parse literal".to_string())),
                }
            } else {
                tokens.push(Var(start.to_owned()));
            }

            for c in end.chars() {
                if c != ')' {
                    return Err(TokenErr("expected ')'".to_string()));
                } else {
                    tokens.push(Parenthesis(Close));
                }
            }
        }
    }

    Ok(tokens)
}

fn split_at_char(s: &str, c: char) -> (&str, &str) {
    let first = &s.chars().take_while(|&x| x != c).collect::<String>();
    s.split_at(first.len())
}

#[test]
fn split_at_char_test() {
    let s = "variable";
    assert_eq!(split_at_char(s, ')'), ("variable", ""));
    let s = "variable)";
    assert_eq!(split_at_char(s, ')'), ("variable", ")"));
    let s = "variable))";
    assert_eq!(split_at_char(s, ')'), ("variable", "))"));
    let s = "variable)))";
    assert_eq!(split_at_char(s, ')'), ("variable", ")))"));
}

#[test]
fn parse_token_test() {
    use self::Key::*;
    use self::ParenCase::*;
    use self::Token::*;
    use self::TokenParseErr::*;

    // Valid substring examples
    let substr = "(in";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Parenthesis(Open), Keyword(In)])
    );
    let substr = "(witness";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Parenthesis(Open), Keyword(Witness)])
    );
    let substr = "(program";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Parenthesis(Open), Keyword(Program)])
    );
    let substr = "(=";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Parenthesis(Open), Keyword(Equal)])
    );
    let substr = "(*";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Parenthesis(Open), Keyword(Mul)])
    );
    let substr = "(+";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Parenthesis(Open), Keyword(Add)])
    );
    let substr = "x";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Var("x".to_string())]));
    let substr = "y)";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Var("y".to_string()), Parenthesis(Close)])
    );
    let substr = "y))";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![
            Var("y".to_string()),
            Parenthesis(Close),
            Parenthesis(Close),
        ])
    );
    let substr = "9";
    assert_eq!(parse_token::<Z251>(substr), Ok(vec![Literal(9.into())]));
    let substr = "9)";
    assert_eq!(
        parse_token::<Z251>(substr),
        Ok(vec![Literal(9.into()), Parenthesis(Close)])
    );

    // Invalid substring examples
    let substr = "(";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("found whitespace after '('".to_string()))
    );
    let substr = "(vari(able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected '('".to_string()))
    );
    let substr = "vari(able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected '('".to_string()))
    );
    let substr = "(variable)";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected ')'".to_string()))
    );
    let substr = "vari=able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected operator".to_string()))
    );
    let substr = "vari*able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected operator".to_string()))
    );
    let substr = "vari+able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected operator".to_string()))
    );
    let substr = "(vari=able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected operator".to_string()))
    );
    let substr = "(vari*able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected operator".to_string()))
    );
    let substr = "(vari+able";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("unexpected operator".to_string()))
    );
    let substr = "9variable";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("could not parse literal".to_string()))
    );
    let substr = "variabl)e))";
    assert_eq!(
        parse_token::<Z251>(substr),
        Err(TokenErr("expected ')'".to_string()))
    );
}

#[test]
fn tokenlist_from_string() {
    use self::Key::*;
    use self::ParenCase::*;
    use self::Token::*;

    let code = "(in x y)
                (witness a b c)

                (program
                    (= t1
                        (* x a))
                    (= t2
                        (* x (+ t1 b)))
                    (= y
                        (* 1 (+ t2 c))))";

    let expected = TokenList::<Z251> {
        tokens: vec![
            Parenthesis(Open),
            Keyword(In),
            Var("x".to_string()),
            Var("y".to_string()),
            Parenthesis(Close),
            Parenthesis(Open),
            Keyword(Witness),
            Var("a".to_string()),
            Var("b".to_string()),
            Var("c".to_string()),
            Parenthesis(Close),
            Parenthesis(Open),
            Keyword(Program),
            Parenthesis(Open),
            Keyword(Equal),
            Var("t1".to_string()),
            Parenthesis(Open),
            Keyword(Mul),
            Var("x".to_string()),
            Var("a".to_string()),
            Parenthesis(Close),
            Parenthesis(Close),
            Parenthesis(Open),
            Keyword(Equal),
            Var("t2".to_string()),
            Parenthesis(Open),
            Keyword(Mul),
            Var("x".to_string()),
            Parenthesis(Open),
            Keyword(Add),
            Var("t1".to_string()),
            Var("b".to_string()),
            Parenthesis(Close),
            Parenthesis(Close),
            Parenthesis(Close),
            Parenthesis(Open),
            Keyword(Equal),
            Var("y".to_string()),
            Parenthesis(Open),
            Keyword(Mul),
            Literal(1.into()),
            Parenthesis(Open),
            Keyword(Add),
            Var("t2".to_string()),
            Var("c".to_string()),
            Parenthesis(Close),
            Parenthesis(Close),
            Parenthesis(Close),
            Parenthesis(Close),
        ],
    };

    let actual = try_to_list::<Z251>(code.to_string());

    assert_eq!(Ok(expected), actual);
}
