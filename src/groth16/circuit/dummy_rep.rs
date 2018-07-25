use super::super::super::field::z251::Z251;
use super::RootRepresentation;
use std::vec::IntoIter;

#[derive(Debug, PartialEq)]
pub struct DummyRep {
    pub u: Vec<Vec<(Z251, Z251)>>,
    pub v: Vec<Vec<(Z251, Z251)>>,
    pub w: Vec<Vec<(Z251, Z251)>>,
    pub roots: Vec<Z251>,
    pub input: usize,
}

impl RootRepresentation<Z251> for DummyRep {
    type Row = IntoIter<Self::Column>;
    type Column = IntoIter<(Z251, Z251)>;
    type Roots = IntoIter<Z251>;

    fn u(&self) -> Self::Row {
        self.u
            .clone()
            .into_iter()
            .map(|x| x.into_iter())
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn v(&self) -> Self::Row {
        self.v
            .clone()
            .into_iter()
            .map(|x| x.into_iter())
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn w(&self) -> Self::Row {
        self.w
            .clone()
            .into_iter()
            .map(|x| x.into_iter())
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn roots(&self) -> Self::Roots {
        self.roots.clone().into_iter()
    }
    fn input(&self) -> usize {
        self.input
    }
}

impl<'a> From<&'a str> for DummyRep {
    fn from(code: &'a str) -> Self {
        let mut line_count = 0;
        let mut lines = code.lines();
        let inputs = lines.next().unwrap().split(' ').collect::<Vec<_>>();
        let witness = lines.next().unwrap().split(' ').collect::<Vec<_>>();
        let temp_vars = lines.next().unwrap().split(' ').collect::<Vec<_>>();
        lines.next();

        let num_vars = inputs.len() + witness.len() + temp_vars.len() + 1;
        let mut u: Vec<Vec<(Z251, Z251)>> = vec![Vec::new(); num_vars];
        let mut v: Vec<Vec<(Z251, Z251)>> = vec![Vec::new(); num_vars];
        let mut w: Vec<Vec<(Z251, Z251)>> = vec![Vec::new(); num_vars];

        for (n, line) in lines.enumerate() {
            line_count += 1;

            let mut symbols = line.split(' ');
            let first = symbols.next().unwrap();
            let pos = inputs
                .clone()
                .into_iter()
                .chain(
                    witness
                        .clone()
                        .into_iter()
                        .chain(temp_vars.clone().into_iter()),
                )
                .position(|s| s == first)
                .unwrap() + 1;

            w[pos].push(((n + 1).into(), 1.into()));
            symbols.next();

            let left = symbols
                .by_ref()
                .take_while(|&c| c != ")")
                .collect::<Vec<_>>();

            for l in left {
                if l == "1" {
                    u[0].push(((n + 1).into(), 1.into()));
                } else {
                    let pos = inputs
                        .clone()
                        .into_iter()
                        .chain(
                            witness
                                .clone()
                                .into_iter()
                                .chain(temp_vars.clone().into_iter()),
                        )
                        .position(|s| s == l)
                        .unwrap() + 1;

                    u[pos].push(((n + 1).into(), 1.into()));
                }
            }
            symbols.next();

            let right = symbols.take_while(|&c| c != ")").collect::<Vec<_>>();

            for r in right {
                let pos = inputs
                    .clone()
                    .into_iter()
                    .chain(
                        witness
                            .clone()
                            .into_iter()
                            .chain(temp_vars.clone().into_iter()),
                    )
                    .position(|s| s == r)
                    .unwrap() + 1;

                v[pos].push(((n + 1).into(), 1.into()));
            }
        }

        DummyRep {
            u,
            v,
            w,
            roots: (1..line_count + 1).map(|n| n.into()).collect::<Vec<_>>(),
            input: inputs.len(),
        }
    }
}
