# Conformance

What every implementation of JPEG-Unround has to reproduce, and the exact values its
tests compare with.

- `references/`: exact references, written by `references.py` from decimals of 60 to
  70 digits: the DCT's basis and blocks transformed, the Laplace scale, the shrinkage
  and the centres of docs/math.md, 1.1 and 2. Each value is a pair of doubles
  `hi:lo` whose sum is the value to about 2^-106 of it. `python conformance/references.py`
  writes them, and `--check` compares them with their definitions.
- `cases/`: the conformance cases (docs/math.md, 9), with their tolerances, which
  `bounds.py` computes.

## The cases

A case is the quantized coefficients of a picture, one component or three, and the
options of its reconstruction, as the command line takes them (docs/cli.md). Its file
holds what the reference of the cases made of it, and what bounds the rounding of that
run: the implementation in Python that came first, which ran its iterations again,
operation for operation, to keep the norms of the magnitudes of their rounding. It was
removed, with the program that wrote the cases (`generate.py`), once the implementation
in Rust reproduced them; both are in the repository's history. The files are fixed: a
new case needs a program that traces a run of the reference implementation as that one
did. Their options give every value of the model and of the steps that their runs took,
the ratio of the steps and the relaxation where those were the defaults of the time,
since the defaults change. An implementation reproduces the case when

- the intervals (`lower`, `upper`), the weights of the data term (`weights`) and the
  steps (`steps`) are the same to the last bit;
- every centre lies within (2|q| + 20) u Q of its exact value (`exact-centres`), where
  q is the level and Q the step of an AC coefficient whose centre is the MMSE one, and
  is that value itself elsewhere (u = 2^-53);
- the canvas after the last iteration lies within `tolerance canvas` of the case's, in
  the Euclidean norm over all its samples;
- the records are taken at the same iterations, and the primal value of each lies
  within the first tolerance of its `tolerance record` line of the case's; where the
  second is not -1, each implementation's dual value lies at most that far above the
  other's primal value; and where the third is not -1, the two gaps lie within it of
  each other;
- the method stops after the same iterations, and for the same reason.

A case of one iteration from a state given (its `first-*` lines) starts the method
from those coefficients and dual fields, and from the field w of TGV, as the library
takes a start (docs/math.md, 5); the command line does not take one.

## The files

Each line is a keyword and its values, numbers in decimal; a double is written as the
shortest decimal that reads back as it, and `nan` where there is none. Lines starting
with `#` are comments. Arrays of coefficients are in the order of the component's
blocks, row by row, each block's 64 coefficients row by row; fields of the canvas are
indexed by entry, channel and row.

| Keyword | Values |
| --- | --- |
| `name` | the case's name |
| `picture` | rows, columns and the colour space (`grayscale`, `ycbcr` or `rgb`) |
| `component` | index, sampling factors h and v, rows and columns of blocks |
| `table` | index, the 64 steps of its quantization table, row by row |
| `levels` | index, block row, the levels of that row of blocks |
| `options` | the command line's options of the reconstruction |
| `steps` | tau and sigma of the primal-dual method |
| `weights` | index, the 64 weights of the data term, one for each frequency |
| `lower`, `upper`, `centres` | index, block row, the values of that row of blocks |
| `exact-centres` | index, block row, the exact centres as `hi:lo` pairs |
| `first-coefficients` | index, block row, the coefficients of the state given |
| `first-p`, `first-w`, `first-r` | entry, channel, row: the fields of the state given |
| `record` | iteration, primal value, dual value, TGV's scaling and partial gap |
| `canvas` | channel, row, the samples of the last canvas |
| `constants`, `start`, `majorant`, `recordbound` | what the bounds depend on (`bounds.py`) |
| `stop` | the iterations, and `converged` or `iterations` |
| `tolerance canvas` | the canvas's tolerance |
| `tolerance record` | iteration, the primal value's tolerance, the dual value's above the other primal value, and the gaps' (-1 where there is none) |

## Commands

In the environment of `environment.yml`:

- `python conformance/bounds.py --check` computes the tolerances again from what the
  files hold and compares them; without `--check` it writes them.
- `python conformance/references.py --check` computes the exact references again and
  compares them; without `--check` it writes them.

The Rust implementation's tests read the cases (`rust/unround/tests/conformance.rs`);
`cargo test --release -p jpeg-unround --test conformance -- --ignored --nocapture`
prints how far it lies from each, against the tolerances.
