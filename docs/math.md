<!-- SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com> -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# The mathematics of JPEG-Unround

This is the one statement of the model and of the algorithms, which the
implementations follow. Each section says in which module it is implemented: a
module of the Rust reference implementation (`rust/unround/src/`, the crate
`jpeg-unround`), and one of the same name of the Python implementation
(`python/src/unround/`), which the Rust one replaces.

Every value this document gives as a default, of the model, of the steps of
the solvers and of when they stop, is an option of the implementations: the
defaults are what is used when nothing else is given.

## Arithmetic

What is rational is computed exactly, and is the same to the last bit in every
implementation: the levels and the steps; the counts of 2.1 and the integers
made of them ($D$ needs up to about 80 bits: 128-bit integers in C++ and Rust);
the intervals without slack, $(q - \tfrac12) Q$, which binary64 holds exactly for
levels and steps of 16 bits; the coefficients of the series of 2.2, computed in
rationals; and, for pictures of integers, the sums and means of the metrics. A
rational enters floating point once, rounded to the nearest double. With a slack
$s$, the ends are $((q - \tfrac12) - s) Q$ and $((q + \tfrac12) + s) Q$, with
1024 added to those of DC last (1.1); they are computed in floating point.

What needs floating point (the basis of the DCT, square roots, logarithms and
exponentials, and the iterations) is computed in binary64 at least; binary32 is
not used. A constant is the double nearest to its value, as the cosines of 1.1
are. The tests take their tolerances from bounds of rounding error, and check
what holds exactly in exact arithmetic.

In floating point, speed comes first. Where the processor has a fused
multiply–add, an implementation may fuse a product with the sum or difference
that takes it, and round once where the two operations would round twice. The
bounds of rounding error hold all the same: they are derived in the standard
model, each operation exact to a factor $1 + \delta$ with $|\delta| \le 2^{-53}$
in binary64, and a fused product is the case $\delta = 0$. What is computed in
floating point may therefore differ in its last bits between implementations,
builds and processors; what is rational is computed exactly first, and is the
same everywhere. A computation that relies on what holds only without rounding,
such as $x^2 - y^2 \ge 0$ where $|x| \ge |y|$, is written so that no fusing
breaks it.

A sum of many terms, such as the values of a record (6), is taken in an order
that depends only on the order of its terms. The terms of a row go to eight
lanes, term $8k + l$ to lane $l$, each lane adding its terms one after another,
the last terms of the row as one more eight padded with 0; then the lanes are
added one after another. The sums of the rows are added in blocks of 128, one
after another within a block, and the blocks in a tree. A term of a row of $n$
goes through at most $\lceil n/8 \rceil + 7$ additions, so that the rounding of
the row's sum is at most $\gamma_{\lceil n/8 \rceil + 7}$ times the sum of the
magnitudes of its terms, $\gamma_m = m u / (1 - m u)$ with $u = 2^{-53}$ (Higham,
3.1). The lanes are what the processor's vectors add at once.

The result can be written in binary64, and is then the last iterate's canvas
itself, cut to the picture, to the last bit: no operation stands between the
computation and the file. The canvas holds the samples in their own units (0 to
255 for 8-bit files), not level-shifted (1.1) and not normalized, and the file
is a TIFF with 64-bit IEEE floating-point samples. Output of 8 or 16 bits is
rounded from it.

## 1. What a JPEG file says about a component

### 1.1 Samples, blocks and the DCT

*Implemented in `dct`; the intervals and the projection in `model`.*

A component is decoded on its **canvas**: the whole blocks that its
coefficients cover, $H = 8\,B_y$ rows of $W = 8\,B_x$ samples. The picture is
the top-left $h \times w$ of it. The encoder filled the samples beyond the
picture itself (libjpeg repeats the last row and column); here they are free
variables, which the regularizer alone decides, and they are cut off at the
end. The unknown is the canvas $x \in \mathbb{R}^{H \times W}$ of the samples
themselves.

JPEG transforms the samples level-shifted, $D(x - 128)$. The shift is constant
over a block, and so moves only its DC coefficient, by exactly
$8 \times 128 = 1024$ ($C_{0,n} = \sqrt{1/8}$). It is added to the DC
interval and the DC centre below, which are exact rationals; the samples are
never shifted, and the canvas is the result as it is.

$D$ is the orthonormal two-dimensional DCT-II of every 8×8 block. Within a
block, with $C_{k,n} = \gamma_k \cos\bigl(\pi (2n+1) k / 16\bigr)$,
$\gamma_0 = \sqrt{1/8}$ and $\gamma_k = \sqrt{2/8}$ for $k > 0$,

$$ (D x)_{v,u} = \sum_{m,n} C_{v,m}\, C_{u,n}\, x_{m,n}, $$

where $v$ is the vertical frequency and $u$ the horizontal one (natural
order). This is the DCT of the JPEG standard, $F(u,v) = \tfrac14 C(u) C(v)
\sum\sum f \cos\cos$, and `scipy.fft.dctn(block, norm="ortho")`. $D$ is
orthogonal: $D^\top D = D D^\top = I$, and $D^\top$ is the inverse DCT.

Every entry of $C$ is the double nearest to its exact value. The cosines are
taken from the nine constants $\cos(\pi j / 16)$, $j = 0, \dots, 8$, written to
25 digits, with $(2n+1)k$ reduced modulo 32 in integers: over a period,
$[0, 8]$ as it is, $(8, 16]$ and $(16, 24]$ negated, $(24, 32)$ mirrored.
Halving is exact and $\sqrt{1/8}$ is correctly rounded. A library cosine of
$\pi (2n+1) k / 16$ would not give the nearest double: the angle is rounded
before the cosine is taken, which near $\pi/2$ magnifies that rounding several
times. All three implementations use the same constants, and so the same basis
to the last bit.

The 8-point transforms go by their even and odd halves. Forward, the sums
$s_j = x_j + x_{7-j}$ and the differences $d_j = x_j - x_{7-j}$, $j < 4$, give
the even frequencies $y_{2i} = \sum_{j<4} C_{2i,j}\, s_j$ and the odd ones
$y_{2i+1} = \sum_{j<4} C_{2i+1,j}\, d_j$; inversely, the even part
$e_n = \sum_{i<4} C_{2i,n}\, c_{2i}$ and the odd part
$o_n = \sum_{i<4} C_{2i+1,n}\, c_{2i+1}$, $n < 4$, give $x_n = e_n + o_n$ and
$x_{7-n} = e_n - o_n$. Each sum of four products is taken from its first term
on, each product fused with the sum that takes it where the processor can
(Arithmetic). A block is transformed down its columns and then along its rows,
and back along its rows and then down its columns. Each output of an 8-point
transform is then within $e_1 = (1 + u)^2 (1 + \gamma_4) - 1 \approx 6u$ of its
exact value, times $\sum_n |C_{k,n}|\, |x_n|$ (the sum or difference rounded
once, the four products and sums, and the entries of $C$ each within $u$), and
each coefficient of a block within $e_1 (2 + e_1) \approx 12u$ times the entry
of $|C|\, |X|\, |C|^\top$; the inverse likewise, with $|C|^\top |Z|\, |C|$.

The file holds, for every coefficient $k$ (a block and a frequency), an integer
$q_k$, and for every frequency the step $Q_k$ of the component's quantization
table. The encoder chose $q_k$ as the nearest integer to $(D y)_k / Q_k$, where
$y$ is the canvas that it was given.

### 1.2 The quantization constraint set

The coefficient of the level-shifted canvas lies within half a step of
$q_k Q_k$. With a slack $s \ge 0$, in steps, the interval of coefficient $k$ of
the canvas is

$$ a_k = (q_k - \tfrac12 - s)\,Q_k + 1024\,[k \text{ is DC}], \qquad
   b_k = (q_k + \tfrac12 + s)\,Q_k + 1024\,[k \text{ is DC}], $$

the level shift of 1.1 in the DC interval, and the **quantization constraint set** (QCS) is
$\mathcal{C} = \{x : a \le D x \le b\}$. Because $D$ is orthogonal and the
box is a product of intervals, the projection onto $\mathcal{C}$ is

$$ P_{\mathcal{C}}(x) = D^\top \mathrm{clip}(D x, a, b). $$

Every output of JPEG-Unround lies in $\mathcal{C}$: the coefficients of the
floating-point result are within their intervals, to $10^{-4} Q_k$, checked
in binary64. The slack is 0 unless it is asked for.

A slack can have a cost (4.1): the widened interval stays hard, and a coefficient
that leaves the file's own interval, $[(q_k - \tfrac12) Q_k, (q_k + \tfrac12) Q_k]$
with 1024 on DC, costs a price per step of the way. The result then leaves that
interval only where the regularizer gains more than the price; it lies within
the widened one, as with any slack. By default there is no cost.

Where the original lies (measured in `experiments/results/phase0.md`): a file
of libjpeg's encoder has the original within its intervals, apart from the
encoder's own rounding. The encoder rounds its YCbCr samples to 8 bits, and in
a flat block that rounding moves every sample alike, so the DC coefficient can
move by up to $8 \times \tfrac12 = 4$. mozjpeg's trellis quantization chooses
levels up to about two steps from the nearest, and its overshoot deringing
encodes white samples pushed beyond 255.

### 1.3 Several components on one canvas

A file has one component (greyscale) or three: Y, Cb and Cr, or R, G and B where
the file says so. Component $c$ has the sampling factors $h_c$ across and $v_c$
down, and $h_{\max}$ and $v_{\max}$ are the largest of them. Each of its samples
stands for a cell of

$$ r_c^{\mathrm v} \times r_c^{\mathrm h} = \frac{v_{\max}}{v_c} \times \frac{h_{\max}}{h_c} $$

samples of the picture, and only files where these are integers are taken:
4:4:4, 4:2:2, 4:2:0, 4:4:0 and 4:1:1 are; the others are refused. A cell has
$n_c = r_c^{\mathrm v} r_c^{\mathrm h}$ samples.

The unknowns of all the components lie on one canvas, at the resolution of the
picture. With one component, the canvas is its blocks, as in 1.1. With several,
it is the picture of $h$ rows and $w$ columns rounded up to whole MCUs,

$$ H = 8 v_{\max} \bigl\lceil h / (8 v_{\max}) \bigr\rceil, \qquad
   W = 8 h_{\max} \bigl\lceil w / (8 h_{\max}) \bigr\rceil, $$

and each component $c$ is a canvas $x_c$ of $H \times W$ samples. The means of
its cells are the component's samples, and the first
$b_c^{\mathrm v} = \lceil h / (8 r_c^{\mathrm v}) \rceil$ rows and
$b_c^{\mathrm h} = \lceil w / (8 r_c^{\mathrm h}) \rceil$ columns of blocks of
them are its blocks, as libjpeg counts them. The file's other blocks of the
component, which complete its MCUs, carry nothing that a decoder shows, and are
left out. With

$$ (S_c x)_{i,j} = \frac{1}{n_c} \sum_{a < r_c^{\mathrm v}} \sum_{b < r_c^{\mathrm h}}
   x_{r_c^{\mathrm v} i + a,\ r_c^{\mathrm h} j + b}, $$

$E_c$ the restriction to the component's blocks and $D$ the DCT of 1.1, the
coefficients of the component are $A_c x_c$, where $A_c = D E_c S_c$. The blocks
lie within the canvas: with $m = h / (8 v_{\max})$,
$r_c^{\mathrm v} \lceil h / (8 r_c^{\mathrm v}) \rceil = r_c^{\mathrm v} \lceil v_c m \rceil
\le v_{\max} \lceil m \rceil$, so that $8 r_c^{\mathrm v} b_c^{\mathrm v} \le H$; and
across alike.

The rows of $S_c$ have $n_c$ entries $1/n_c$ each, and no two share a column, so
$S_c S_c^\top = I / n_c$; with $E_c E_c^\top = I$ and $D D^\top = I$,

$$ A_c A_c^\top = \nu_c I, \qquad \nu_c = 1 / n_c. $$

$\nu_c^{-1} A_c^\top$ takes coefficients to the canvas: their inverse DCT, each
sample repeated over its cell, and 0 beyond the blocks.
$\Pi_c = \nu_c^{-1} A_c^\top A_c$ is the orthogonal projection onto the canvases
that are constant on every cell of the blocks and 0 beyond them: it replaces each
cell of the blocks by its mean, and the samples beyond the blocks by 0. What it
leaves, $x_c - \Pi_c x_c$ (the deviations of the samples from the means of their
cells, and the samples beyond the blocks), no interval constrains: those samples
are free. The samples beyond the blocks all lie beyond the picture. With one
component, and with several whose canvas is their blocks (4:4:4), $n_c = 1$,
nothing lies beyond the blocks, $A_c = D$ and $\Pi_c = I$: nothing is free.

The encoder averaged its samples over the cells and rounded the means to
integers, having filled the picture out to whole MCUs by repeating its last row
and column (some of the rows after averaging). A canvas that holds the encoder's
samples within the picture, and elsewhere values that give the cells the means
the encoder coded, is in the set up to that rounding, as a greyscale picture is
(1.2).

## 2. A Laplace model of the AC coefficients

*Implemented in `laplace`.*

### 2.1 The scale, from the quantized coefficients

For a component and an AC frequency, the coefficients of all its blocks are
taken as draws from the Laplace distribution
$p(c) = \frac{1}{2\beta} e^{-|c|/\beta}$. The file shows only the bin of each:
with step $Q$, the bin of $q = 0$ has probability
$1 - e^{-Q/(2\beta)}$, and the bin of $q \ne 0$
$\tfrac12 e^{-(|q| - 1/2) Q/\beta}\,(1 - e^{-Q/\beta})$.

Write $t = e^{-Q/(2\beta)} \in (0, 1)$. With $n_0$ the number of zeros,
$n_1$ the number of non-zeros and $S = \sum_{q \ne 0} (2|q| - 1)$, the
log-likelihood is, up to a constant,

$$ L(t) = n_0 \log(1 - t) + S \log t + n_1 \log(1 - t^2). $$

Setting $L'(t) = 0$ and multiplying by $t(1 - t)(1 + t) > 0$ gives the
quadratic $A t^2 + n_0 t - S = 0$ with $A = n_0 + S + 2 n_1$. Its root in
$[0, 1)$, written so that nothing cancels, is the maximum-likelihood estimate:

$$ t^\ast = \frac{2S}{n_0 + \sqrt{n_0^2 + 4 A S}}, \qquad
   \beta^\ast = \frac{Q}{2 \log(1/t^\ast)}. $$

It is the maximum: the numerator $S - n_0 t - A t^2$ of $L'$ falls on
$(0, 1)$ and is positive at 0 and negative at 1, and $t^\ast < 1$ whenever
there is a coefficient. When every coefficient is 0, $S = 0$, $t^\ast = 0$ and
$\beta^\ast = 0$: the distribution has collapsed onto 0.

When $\beta \gg Q$, $t^\ast$ is close to 1 and $\log(1/t^\ast)$ would lose
digits to cancellation. From $t^\ast \ge \tfrac12$ on, it is taken as
$-\mathrm{log1p}(-(1 - t^\ast))$ with

$$ 1 - t^\ast = \frac{8 S (n_0 + n_1)}{\bigl(\sqrt{D} + 2S - n_0\bigr)\bigl(n_0 + \sqrt{D}\bigr)}, \qquad
   D = n_0^2 + 4 A S, $$

which follows from multiplying $\sqrt{D} - (2S - n_0)$ by its conjugate, since
$D - (2S - n_0)^2 = 8 S (n_0 + n_1)$. Every term is positive there
($t^\ast \ge \tfrac12$ gives $S \ge n_0 / 2$), so nothing cancels, and the
scale is within a few units in the last place of $\beta^\ast$.

### 2.2 The mean of a bin

Given its bin $[l, l + Q]$ with $l \ge 0$, a coefficient's conditional mean
under the model is the mean of an exponential distribution cut to the bin:

$$ \mathbb{E}[c \mid l \le c \le l + Q]
   = l + \beta - \frac{Q}{e^{Q/\beta} - 1}. $$

Bins below 0 mirror it, and the bin of 0, being symmetric, has mean 0. So the
**MMSE centre** of a coefficient is its interval's centre moved towards 0 by a
fraction of the step that depends on $\rho = Q/\beta$ alone:

$$ \hat c_k = \mathrm{sign}(q_k)\,\bigl(|q_k| Q_k - \delta_k\bigr), \qquad
   \frac{\delta}{Q} = \frac12 - \frac1\rho + \frac{1}{e^\rho - 1}
   \in \bigl[0, \tfrac12\bigr), $$

and $\hat c_k = 0$ for $q_k = 0$. As $\beta \to \infty$ (a flat distribution)
$\delta \to 0$, the centre of the interval; as $\beta \to 0$,
$\delta \to Q/2$, its end nearer 0. A scale of 0 gives $\delta = Q/2$.

For small $\rho$ the three terms nearly cancel. Below $\rho = 1$, $\delta / Q$
is its Taylor series,

$$ \frac{\delta}{Q} = \sum_{k \ge 1} \frac{B_{2k}}{(2k)!}\, \rho^{2k-1}
   = \frac{\rho}{12} - \frac{\rho^3}{720} + \frac{\rho^5}{30240} - \dots, $$

with $B_{2k}$ the Bernoulli numbers, which converges for $\rho < 2\pi$. Eleven
terms leave less than $2 \times 10^{-19}$ below 1; the coefficients are
computed exactly in rational numbers and rounded once. From 1 on, the closed
form is used, with $1/(e^\rho - 1) = e^{-\rho}/(1 - e^{-\rho})$; its terms are
then at most 1. Either way $\delta / Q$ is within a few units of $2^{-53}$.

In floating point, too, every centre lies within its interval:
$|q| - \delta/Q$ rounds to within $[|q| - \tfrac12, |q|]$, whose ends are
exact, and so does its product with $Q$ to within the interval.

The MAP estimate under the same model would be the end of the interval nearest
0, which is worse in mean square: the centres are the conditional means.

The DC coefficient does not follow a Laplace distribution. Its centre is the
centre of its interval, $q_0 Q_0 + 1024$, and the data term below leaves it
out by default.

The data term can take the middles of the intervals, $q_k Q_k$ (with 1024
added on DC), as its centres instead. They are exact, and their decoder is the
plain one, without rounding.

### 2.3 The MMSE decoder

$x = D^\top \hat c$, with the DC centres as above, is a decoder of its own: the
coefficients' conditional means. It costs one inverse DCT, lies in
$\mathcal{C}$, and is the default starting point of the solvers below.

## 3. Finite differences

*Implemented in `operators`.*

### 3.1 Gradient and divergence

On an $H \times W$ grid, with $i$ the row and $j$ the column, the gradient uses
forward differences with Neumann boundaries (the difference out of the grid is
0):

$$ (\partial_x x)_{i,j} = \begin{cases} x_{i,j+1} - x_{i,j} & j < W - 1 \\ 0 & j = W - 1 \end{cases}
   \qquad
   \nabla x = (\partial_x x, \partial_y x), $$

and $\partial_y$ likewise down the columns. The divergence is its negative
adjoint, $\mathrm{div} = -\nabla^\top$, which is the backward difference

$$ (\delta_x f)_{i,j} = f_{i,j}\,[j < W - 1] - f_{i,j-1}\,[j > 0], \qquad
   \mathrm{div}(p_1, p_2) = \delta_x p_1 + \delta_y p_2, $$

where $[\cdot]$ is 1 when its condition holds and 0 otherwise. So
$\langle \nabla x, p \rangle = -\langle x, \mathrm{div}\, p \rangle$.

### 3.2 The symmetrized gradient

Symmetric 2×2 tensor fields $r = (r_{11}, r_{22}, r_{12})$ have the inner
product that counts the off-diagonal entry twice, as the full matrix does:

$$ \langle r, s \rangle = \sum_{i,j} r_{11} s_{11} + r_{22} s_{22} + 2\, r_{12} s_{12},
   \qquad |r|_F = \sqrt{r_{11}^2 + r_{22}^2 + 2 r_{12}^2}. $$

The symmetrized gradient of a vector field $w = (w_1, w_2)$ uses the backward
differences of 3.1, and its negative adjoint, the divergence of a tensor field,
the forward differences:

$$ \mathcal{E} w = \bigl(\delta_x w_1,\ \delta_y w_2,\ \tfrac12 (\delta_y w_1 + \delta_x w_2)\bigr), \qquad
   \mathrm{div}_2 r = (\partial_x r_{11} + \partial_y r_{12},\ \partial_x r_{12} + \partial_y r_{22}). $$

Then $\langle \mathcal{E} w, r \rangle = -\langle w, \mathrm{div}_2 r \rangle$,
because $\delta_x = -\partial_x^\top$. This is the discretization of Bredies,
Kunisch and Pock. The norm of a tensor, and the projection of a tensor field
onto a ball, are $|\cdot|_F$ of that inner product, off-diagonal counted twice.

### 3.3 Norms

$\|\partial_x\|^2 < 4$, so $\|\nabla\|^2 < 8$. Also
$|\mathcal{E} w|_F^2 \le |\delta w_1|^2 + |\delta w_2|^2$ pixel by pixel,
so $\|\mathcal{E}\|^2 < 8$. The operator of TGV,
$K(x, w) = (\nabla x - w, \mathcal{E} w)$, is bounded by the matrix
$\begin{pmatrix} \sqrt8 & 1 \\ 0 & \sqrt8 \end{pmatrix}$, so

$$ \|K\|^2 \le \tfrac12 \bigl(17 + \sqrt{33}\bigr) \approx 11.37 < 12. $$

## 4. The models

*Implemented in `model` for one component, and in `frames` for several (4.4).*

### 4.1 The data term and the constraint

With $c = D x$, $\mu \ge 0$, the weights $\omega_k = 1 / Q_k^p$ on AC
coefficients and $\omega_k = \omega_{\mathrm{DC}} / Q_k^p$ on DC with
$\omega_{\mathrm{DC}} \ge 0$ and $p \ge 0$, and the centres $\hat c_k$ (those of
2.2, or the middles of the intervals),

$$ G(x) = \frac{\mu}{2} \sum_k \omega_k (c_k - \hat c_k)^2 + \iota_{[a, b]}(c), $$

where $\iota$ is 0 on its set and $+\infty$ off it. $G$ is separable in the
coefficients, and so are its proximal map and its conjugate. By default
$\mu = 10^{-3}$, $p = 2$, $\omega_{\mathrm{DC}} = 0$ (DC follows no Laplace model,
and its centre is only the middle of its interval), the centres are the MMSE
ones, and there is no slack.

$\mu$ can instead follow the quantization of the component:
$\mu = \mu_s \bar Q^{\,r}$, where $\bar Q$ is the mean of its 64 steps. $\bar Q$ is
exact (an integer sum over 64), and with $r = 1$ nothing more rounds but the
product. The scale $\mu_s$ and the power $r$ are options; the rule is not the
default yet.

**Proximal map.** For $\tau > 0$ and $e = D v$,
$\mathrm{prox}_{\tau G}(v) = D^\top \zeta$ with

$$ \zeta_k = \mathrm{clip}\!\left(\frac{e_k + \tau \mu \omega_k \hat c_k}{1 + \tau \mu \omega_k},\ a_k,\ b_k\right), $$

since a strictly convex quadratic of one variable, restricted to an interval,
is least at the clipped unconstrained minimizer.

**Conjugate.** For $\xi$ and $s = D \xi$,
$G^\ast(\xi) = \sum_k g_k^\ast(s_k)$ with, where $m_k = \mu \omega_k > 0$,

$$ g_k^\ast(s) = s\, c^\ast - \frac{m_k}{2} (c^\ast - \hat c_k)^2, \qquad
   c^\ast = \mathrm{clip}(\hat c_k + s / m_k,\ a_k,\ b_k), $$

and where $m_k = 0$ (DC with $\omega_{\mathrm{DC}} = 0$, or $\mu = 0$),
$g_k^\ast(s) = \max(s\, a_k,\ s\, b_k)$.
Both are finite everywhere, because the intervals are bounded.

**A slack with a cost.** With a slack $s > 0$ and a price $\beta > 0$ per step
(1.2), $G$ adds $\sum_k \lambda_k\, \mathrm{dist}(c_k, [a^\circ_k, b^\circ_k])$,
$\lambda_k = \beta / Q_k$, where $[a^\circ_k, b^\circ_k]$ is the file's own interval
and $[a_k, b_k]$ the widened one. Beyond $b^\circ_k$ the objective of the proximal
map is the quadratic above plus the linear $\tau \lambda_k (c - b^\circ_k)$, so with
$z_k$ the quadratic's minimizer and $d_k = 1 + \tau m_k$,

$$ \zeta_k = \mathrm{clip}\bigl(\max(b^\circ_k,\ z_k - \tau \lambda_k / d_k),\ a_k,\ b_k\bigr)
   \text{ where } z_k > b^\circ_k, $$

alike below $a^\circ_k$ with $\min(a^\circ_k,\ z_k + \tau \lambda_k / d_k)$, and
$\zeta_k = z_k$ within $[a^\circ_k, b^\circ_k]$. The conjugate's maximizer is
$\hat c_k + s / m_k$ where that lies within $[a^\circ_k, b^\circ_k]$; above,
$\hat c_k + (s - \lambda_k) / m_k$ but not below $b^\circ_k$ nor above $b_k$; alike
below; and where $m_k = 0$, $b_k$ for $s > \lambda_k$, $b^\circ_k$ for
$0 < s \le \lambda_k$, and alike for $s < 0$. $g_k^\ast(s)$ is the objective there,
$s c^\ast - \frac{m_k}{2} (c^\ast - \hat c_k)^2 - \lambda_k \mathrm{dist}(c^\ast, [a^\circ_k, b^\circ_k])$.

### 4.2 Total variation

$$ \min_x\ P(x) = \alpha \|\nabla x\|_{2,1} + G(x), \qquad
   \|p\|_{2,1} = \sum_{i,j} |p_{i,j}|_2. $$

Without the data term ($\mu = 0$) $\alpha$ only scales the objective, and the
minimizer is the least total variation within $\mathcal{C}$.

### 4.3 Total generalized variation of second order

$$ \min_{x, w}\ P(x, w) = \alpha_1 \|\nabla x - w\|_{2,1} + \alpha_0 \|\mathcal{E} w\|_{F,1} + G(x), \qquad
   \|r\|_{F,1} = \sum_{i,j} |r_{i,j}|_F. $$

This is TGV² itself, an infimal convolution over $w$: where the picture is a
ramp, $w$ takes up its gradient, and only changes of the ramp cost.

### 4.4 Several components

$G(x) = \sum_c G_c(x_c)$ with $G_c = g_c \circ A_c$ (1.3), where $g_c$ is the $G$
of 4.1 in the coefficients of component $c$: its intervals, its weights, and its
centres, whose Laplace scales (2.1) are those of its own coefficients.

**Proximal map.** For a linear map $A$ with $A A^\top = \nu I$, the minimizer of
$g(Ax) + \frac{1}{2\tau} \|x - v\|^2$ differs from $v$ only within the range of
$A^\top$: with $x = v + A^\top u$, $Ax = Av + \nu u$ and
$\|x - v\|^2 = \|Ax - Av\|^2 / \nu$, so that $Ax$ minimizes
$g(z) + \frac{1}{2 \nu \tau} \|z - Av\|^2$. Hence

$$ \mathrm{prox}_{\tau G_c}(v) = v + \nu_c^{-1} A_c^\top \bigl(\zeta - A_c v\bigr), \qquad
   \zeta = \mathrm{prox}_{\nu_c \tau\, g_c}(A_c v): $$

the component's coefficients take the proximal map of 4.1 with the step
$\nu_c \tau$, and the canvas moves by the change, repeated over the cells; the
free samples stay as they are. $A_c v$ is the DCT of the means $\bar v$ of $v$'s
cells, and the result is computed as $v$ plus $D^\top \zeta - \bar v$ repeated
over the cells (0 beyond the blocks); where $n_c = 1$, as $D^\top \zeta$ on the
blocks, as in 4.1, and $v$ beyond them. The projection onto $\mathcal{C}$ is the
same with $\zeta = \mathrm{clip}(A_c v, a_c, b_c)$.

**Conjugate.** $G_c^\ast(\xi)$ is finite only where $\xi = \Pi_c \xi$: along a
free direction $f$ ($A_c f = 0$) with $\langle \xi, f \rangle > 0$,
$\langle \xi, x + t f \rangle - G_c(x + t f)$ grows without bound. Where
$\xi = \Pi_c \xi$, $\langle \xi, x \rangle = \nu_c^{-1} \langle A_c \xi, A_c x \rangle$
and $A_c$ is onto, so

$$ G_c^\ast(\xi) = g_c^\ast\bigl(\nu_c^{-1} A_c \xi\bigr), $$

where $\nu_c^{-1} A_c \xi$ is the DCT of the sums of $\xi$ over the cells of the
blocks. 6.6 bounds what lies elsewhere.

**Channels.** The regularizers take the channels together, pixel by pixel, or
each on its own, and weight the differences of channel $c$ by $\gamma_c > 0$.
Coupled,

$$ \mathrm{TV}(x) = \alpha \sum_{i,j} \Bigl( \sum_c \gamma_c^2\, |(\nabla x_c)_{i,j}|^2 \Bigr)^{1/2}, $$

$$ \mathrm{TGV}(x) = \min_w\ \alpha_1 \sum_{i,j} \Bigl( \sum_c \gamma_c^2\, |(\nabla x_c - w_c)_{i,j}|^2 \Bigr)^{1/2}
   + \alpha_0 \sum_{i,j} \Bigl( \sum_c \gamma_c^2\, |(\mathcal{E} w_c)_{i,j}|_F^2 \Bigr)^{1/2}; $$

each on its own, the sum over the channels of $\gamma_c$ times the TV or TGV of
4.2 and 4.3 of $x_c$ (for TGV, each channel with its own $w_c$). Both are
$F(Kz)$: $K$ is the operator of 4.2 or 4.3 on every channel, times $\gamma_c$,
and $F$ sums over the pixels the Euclidean norm of all the channels' entries
(those of a tensor with the off-diagonal twice), coupled, or of each channel's
alone. The dual balls are those of the same norms, pixel by pixel, and
$\|K\|^2$ is at most $\max_c \gamma_c^2$ times the bound of 3.3. Coupled, an edge
that the channels share costs less than the same edges apart
($\sqrt{a^2 + b^2} \le a + b$), which keeps the chroma's edges where the luma's
are (Bresson and Chan). With one channel and $\gamma = 1$, both are those of 4.2
and 4.3. By default the channels are coupled, and every $\gamma_c$ is 1. On the
colour tuning files, coupled, the 8-bit PSNR of TV's results was 0.6 dB higher
in the median than apart (by 0.02 to 2.6 dB, file by file), and that of TGV's
0.3 dB (on 12 files, one of them 0.1 dB lower; `experiments/results/phase2-colour.md`).

## 5. The primal–dual hybrid gradient method

*Implemented in `pdhg`.*

Both models are $\min_z F(K z) + G(z)$ with $F$ a norm, and so the saddle-point
problem $\min_z \max_y \langle K z, y \rangle + G(z) - F^\ast(y)$, where
$F^\ast$ is the indicator of the dual ball of the norm. The method of
Chambolle and Pock, in the form of Condat with a relaxation $\rho$, is

$$ \tilde z = \mathrm{prox}_{\tau G}\bigl(z^n - \tau K^\top y^n\bigr), \quad
   \tilde y = \mathrm{prox}_{\sigma F^\ast}\bigl(y^n + \sigma K (2 \tilde z - z^n)\bigr), \quad
   (z^{n+1}, y^{n+1}) = \rho\, (\tilde z, \tilde y) + (1 - \rho)\, (z^n, y^n). $$

With $\sigma \tau \|K\|^2 < 1$ and $0 < \rho < 2$ the iterates converge to a
saddle point (Condat, Theorem 3.1, with no smooth term). With $\rho = 1$ this is
Chambolle and Pock's Algorithm 1, with the dual one step ahead: from $y^0 = 0$
and an $x^0$ that $\mathrm{prox}_{\tau G}$ leaves as it is (the default start,
whose coefficients are the centres), its $\tilde z$ at step $n + 1$ is their $z$
at step $n$.

The proximal map of the indicator of a ball is the projection onto it, pixel
by pixel: $y \mapsto y / \max(1, |y| / \alpha)$.

**Total variation.** $K = \nabla$, $y = p$, $K^\top p = -\mathrm{div}\, p$:

$$ \tilde x = \mathrm{prox}_{\tau G}\bigl(x + \tau \mathrm{div}\, p\bigr), \qquad
   \tilde p = \mathrm{proj}_{\alpha}\bigl(p + \sigma \nabla (2 \tilde x - x)\bigr), $$

then $x \leftarrow \rho \tilde x + (1 - \rho) x$ and
$p \leftarrow \rho \tilde p + (1 - \rho) p$.

**TGV.** $z = (x, w)$, $y = (p, r)$,
$K^\top (p, r) = (-\mathrm{div}\, p,\ -p - \mathrm{div}_2 r)$, and
$G$ does not depend on $w$, whose proximal map is the identity:

$$ \begin{aligned}
\tilde x &= \mathrm{prox}_{\tau G}\bigl(x + \tau \mathrm{div}\, p\bigr), &
\tilde w &= w + \tau (p + \mathrm{div}_2 r), \\
\tilde p &= \mathrm{proj}_{\alpha_1}\bigl(p + \sigma (\nabla \bar x - \bar w)\bigr), &
\tilde r &= \mathrm{proj}_{\alpha_0, F}\bigl(r + \sigma \mathcal{E} \bar w\bigr),
\end{aligned} $$

with $\bar x = 2 \tilde x - x$ and $\bar w = 2 \tilde w - w$, then each of
$x, w, p, r$ moves to $\rho$ times its tilde plus $1 - \rho$ times itself.

**Several channels.** $x = (x_c)$, and so are $w$, $p$ and $r$. Every difference
of channel $c$ is multiplied by $\gamma_c$ (4.4): $\mathrm{div}\, p_c$ in the step
of $x_c$ becomes $\gamma_c\, \mathrm{div}\, p_c$, $\nabla \bar x_c$ in that of $p_c$
becomes $\gamma_c \nabla \bar x_c$, and for TGV so do $p_c + \mathrm{div}_2 r_c$,
$\nabla \bar x_c - \bar w_c$ and $\mathcal{E} \bar w_c$. The proximal map of $G$ is
that of 4.4, channel by channel, and the projections are onto the balls of 4.4:
over all the channels at a pixel where they are coupled.

**Steps.** With $L^2$ a bound of $\|K\|^2$, a ratio $\kappa = \tau / \sigma$ and
a product $\vartheta$ in $(0, 1)$, the steps are
$\tau = \sqrt{\vartheta \kappa} / L$ and $\sigma = \sqrt{\vartheta / \kappa} / L$,
so that $\sigma \tau L^2 = \vartheta$. By default $L^2$ is the bound of 3.3 times
$\max_c \gamma_c^2$ (8 for TV, 11.37 for TGV, where every $\gamma_c$ is 1) and
$\vartheta = 0.99$; with an $L^2$ below $\|K\|^2$ the condition above need not
hold, nor the convergence.

**The ratio.** For the averages $Z^N, Y^N$ of the first $N$ iterates, Theorem 1
of Chambolle and Pock bounds the gap at a saddle point $(z^\ast, y^\ast)$:

$$ \mathcal{L}(Z^N, y^\ast) - \mathcal{L}(z^\ast, Y^N) \le \frac{1}{N}
   \Bigl(\frac{\|z^\ast - z^0\|^2}{2\tau} + \frac{\|y^\ast - y^0\|^2}{2\sigma}\Bigr)
   = \frac{L}{2N\sqrt{\vartheta}} \Bigl(\frac{\|z^\ast - z^0\|^2}{\sqrt{\kappa}}
   + \|y^\ast - y^0\|^2 \sqrt{\kappa}\Bigr), $$

with $\mathcal{L}(z, y) = \langle K z, y \rangle + G(z) - F^\ast(y)$. The bound is
least at

$$ \kappa^\ast = \frac{\|z^\ast - z^0\|^2}{\|y^\ast - y^0\|^2}: $$

the ratio is that of squared distances, of the primal start from the solution
(in grey levels, and grey levels per sample for $w$) over the dual one's (in
units of $\alpha$). Neither is known before solving, and the bound is far from
tight: on the tuning images, $\kappa^\ast$ taken from the last points of long
runs is 2 to 540 for TV and 150 to 2900 for TGV, while the ratios that bring the
gap down fastest are 10 to 30 for TV and 3 to 10 for TGV. The default ratio
(6.5) is the one measured to stop soonest.

**Start.** By default $x$ is the MMSE decoder's output (2.3; with several
components, $\nu_c^{-1} A_c^\top \hat c_c$ for each, the inverse DCT of its
centres repeated over its cells, and 128 beyond its blocks), $w = \nabla x$, and
the dual variables are 0. Any start can be given instead: its coefficients are
clipped to their intervals, and its $p$ and $r$ projected onto their balls.
(Starting $w$ at 0 puts it 3 to 10 times nearer
the solution's $w$, which is small across the jumps of the picture, and divides
the bound's $\kappa^\ast$ by 10 to 100. On two tuning files the gap then fell
no faster with the ratios 300 to 3000, and with the default ratio 3 at most 15
per cent sooner on one file and no sooner on the other.)

Every $\tilde x$ is an output of $\mathrm{prox}_{\tau G}$, and so lies in
$\mathcal{C}$, and every $\tilde p$ (and $\tilde r$) in its ball. With
$\rho > 1$ the relaxed $x$ and $p$ need not: what is recorded, what the gap is
taken at, and the result are therefore the tilde points. The coefficients
$\zeta$ of the last $\tilde x$ are kept: the result is $D^\top \zeta$, whose
coefficients are $\zeta$ to rounding error.

**What else was tried.** Diagonal preconditioning (Pock and Chambolle) takes a
step for each variable from the sums of the absolute values of $K$'s rows and
columns. For TGV's $K$ (with the off-diagonal of $r$ scaled by $\sqrt 2$, where
its norm counts it twice) those are about 4 for $x$, $3 + \sqrt 2$ for $w$, 3 for
$p$ and $2 \sqrt 2$ for $r$: nearly the same step for every block, and so
nearly the method above with a ratio of about 0.7, which is less than the one
chosen. Adaptive steps (Goldstein, Li and Yuan), which change the ratio to
keep the primal and dual residuals of an iteration in proportion to one
another, drove it to 0.08 to 2 with the proportions 1 and 3, and kept it near
where it started with 0.3; on three tuning files none of them brought the gap
down faster than the fixed ratio. Starting TGV from TV's solution, with $w = 0$
and TV's $p$, was no faster either: TV's solution is another picture than
TGV's. (`experiments/results/phase2-solver.md`.)

### 5.1 The layout of the solvers

*Implemented in `planar`, `sweep`, `kernels` and `records`.*

The iterations compute the formulas above, operation for operation and each
rounded as it is written; what the layout changes is the order in which the
samples are visited and where they lie in memory.

**The planes.** The canvas is cut into MCUs of $R \times P$ samples, $P = 8a$
and $R = 8b$ for the largest ratios $a$ across and $b$ down of the components'
cells (1.3), so that every component's blocks tile an MCU, and the canvas is
$M$ MCUs across. A row of the canvas holds its $W = P M$ samples as $P$ planes
of $M$: column $c$ at $(c \bmod P)\, M + \lfloor c / P \rfloor$. The same column
of every MCU is then one stream, and so is the same coefficient of every block
of a component at the same place in its MCU: for a component of $P_c = P / a_c$
columns in an MCU, coefficient $(v, u)$ of block $h m + t$ of a block row
($h = P_c / 8$ blocks across an MCU, $t < h$) is at
$(v P_c + 8 t + u)\, M + m$ in the block row. An 8-point transform of eight such
streams, a proximal map, a difference and a projection are each one loop over
the MCUs, the same operation on neighbouring values, which the compiler makes
vector code of the processor's width. The next sample across is in the next
plane at the same MCU, and after the last plane in the first plane at the next
MCU.

**The sweep.** An iteration is one pass over the canvas, band of $R$ rows by
band. For each band, the primal step: $x + \tau \gamma\, \mathrm{div}\, p$ row by
row, and the proximal map of $G$ block row by block row, the DCT down the
columns of every plane at once, then along the rows of each set of blocks, the
map, and back; and for TGV the step of $w$, which needs $p$ and the differences
of $r$ down to the row below the band. Then the dual steps and the relaxation of
the band before, whose last row's forward differences down need the first row
of the new band's $\tilde x$ (and $\tilde w$), and whose backward differences of
$\bar w$ need the row above. Two bands of $\tilde x$ (and of $\tilde w$) are kept;
the point is read and written once in an iteration, and the outputs of the
proximal steps are written out where a record takes them.

**The records.** The values of 6 are computed in the same layout from those
outputs, row by row, their sums in the order of Arithmetic, which differs from
that of the natural layout: the two agree within the bounds of their rounding.

## 6. When to stop: duality gaps

*Implemented in `frames` and `pdhg`.*

The dual of $\min F(Kz) + G(z)$ is $\max_y -F^\ast(y) - G^\ast(-K^\top y)$.
For any feasible pair, the gap $P(z) - D(y) \ge P(z) - P^\ast \ge 0$ bounds
how far $z$ is from the least value, and it is 0 at a saddle point.

### 6.1 Total variation: in closed form

$-K^\top p = \mathrm{div}\, p$, and every iterate has $|p| \le \alpha$, so

$$ \mathrm{gap}(x, p) = \alpha \|\nabla x\|_{2,1} + G(x) + G^\ast(\mathrm{div}\, p), $$

with $G^\ast$ of 4.1: finite, and in closed form.

### 6.2 TGV: a feasible dual

Here $-K^\top(p, r) = (\mathrm{div}\, p,\ p + \mathrm{div}_2 r)$, and
$G$ does not depend on $w$, so the dual is finite only where
$p = -\mathrm{div}_2 r$: at the iterates of the method it is
$-\infty$, and the gap $+\infty$. The pair is made feasible by scaling:

$$ \theta = \min\Bigl(1,\ \frac{\alpha_1}{\max_{i,j} |(\mathrm{div}_2 r)_{i,j}|}\Bigr), \qquad
   \tilde r = \theta r, \qquad \tilde p = -\mathrm{div}_2 \tilde r. $$

Then $|\tilde p| \le \alpha_1$, $|\tilde r|_F \le \alpha_0$ and
$\tilde p + \mathrm{div}_2 \tilde r = 0$, so

$$ \mathrm{gap}(x, w, r) = P(x, w) + G^\ast(\mathrm{div}\, \tilde p)
   = P(x, w) + G^\ast(-\mathrm{div}\, \mathrm{div}_2 \tilde r) $$

is an upper bound of $P(x, w) - P^\ast$. At a saddle point
$p = -\mathrm{div}_2 r$ already, $\theta = 1$, and the gap is 0.

### 6.3 TGV: a partial gap

The gap of Chambolle and Pock restricted to a bounded set,
$\mathcal{G}_B = \sup_{y} L(z, y) - \inf_{z' \in B} L(z', y)$ with
$L(z, y) = \langle K z, y \rangle + G(z) - F^\ast(y)$, bounds $P(z) - P^\ast$
when $B$ contains the primal part of a saddle point. With
$B = \{(x, w) : |w_{i,j}| \le \varrho\}$,

$$ \mathcal{G}_B(x, w, p, r) = P(x, w) + G^\ast(\mathrm{div}\, p)
   + \varrho\, \|p + \mathrm{div}_2 r\|_{2,1}. $$

It needs no change of the iterates, but only holds for a $\varrho$ at least as
large as the solution's $w$, which is not known beforehand.

### 6.4 When the solvers stop

Every few iterations (ten by default) a solver takes the gap, and it stops at
the first of these where one of the following holds, each tolerance of 0
turning its test off; or else after the most iterations allowed (6.5).

- The gap per sample is at most a tolerance $\varepsilon$ (by default that of 6.5).
- The gap is at most $\varepsilon_{\mathrm{rel}} P(z)$, which bounds
  $(P(z) - P^\ast) / P(z)$ by $\varepsilon_{\mathrm{rel}}$ ($P \ge 0$). Off by
  default.
- For TGV, the partial gap of 6.3 per sample is at most its own tolerance. It
  bounds how far $P$ is from $P^\ast$ only if its radius $\varrho$ is at least the
  solution's largest $|w|$. Off by default.

The gaps are sums over the canvas. They are divided by its number of samples,
that of all the channels ($C H W$ for $C$ channels), so that the tolerance is in
the units of the objective per sample and does not depend on the size of the
picture.

A gap bounds how far the objective is above its least value, not how far the
iterate is from the point where it is least. $P$ is strongly convex only in the
AC coefficients (and in DC where $\omega_{\mathrm{DC}} > 0$, which the bound below
leaves out), with the modulus $\mu \omega_k = \mu / Q_k^2$ (4.1): with $c^\ast$ the
least point, $0 \in \partial P(c^\ast)$ gives
$P(c) - P^\ast \ge \frac{\mu}{2} \sum_{k\ \mathrm{AC}} \omega_k (c_k - c^\ast_k)^2$,
and so a gap of $\varepsilon$ per sample bounds

$$ \frac{1}{N} \sum_{k\ \mathrm{AC}} \Bigl(\frac{c_k - c^\ast_k}{Q_k}\Bigr)^2 \le \frac{2 \varepsilon}{\mu}. $$

At $\varepsilon = 10^{-2}$ and $\mu = 10^{-3}$ that allows the coefficients
$\sqrt{20} \approx 4.5$ steps from the least point on average: more than their
intervals are wide. The least point is unique in its AC coefficients, but the
objective is almost flat around it: on the tuning images, the iterates of TV
still move by hundredths to tenths of a grey level (RMS) thousands of
iterations after the objective, and the PSNR of the result, have settled. The
tolerance is therefore chosen by how much stopping changes the result.

For the same reason what stopping at a gap leaves depends on the path the
iterates took, and not on the gap alone. Stopped at a gap per sample of
$2 \cdot 10^{-4}$, a few tuning files still differ from a much longer run by
0.010 to 0.014 dB of PSNR, whatever the ratio of the steps and the relaxation, and
smaller tolerances bring that largest difference down only slowly: a bound on
every file, at that precision, is not what a gap can give. The criterion is
therefore on the distribution of the differences over the files, in its median
and its 90th percentile, and the largest difference is reported.

### 6.5 The defaults

*In `pdhg` (`TV_RATIO`, `TV_RELAXATION`, `TV_TOLERANCE`,
`TV_ITERATIONS`, and those of TGV), chosen on the tuning images by
`experiments/phase2_solver.py` (`experiments/results/phase2-solver.md`).*

Stopping where the gap per sample first falls within a tolerance changes the
result, against the last point of a much longer run (TV: $\tau / \sigma = 30$,
$\rho = 1.5$, 8000 iterations; TGV: 3, 1.9, 12000). A tolerance is accepted when,
over the tuning files, the change of the PSNR of the binary64 result is at most
0.001 dB in the median and 0.005 dB in the 90th percentile, and that of the SSIM
of its 8-bit samples at most $10^{-5}$ and $5 \cdot 10^{-5}$; when every file
reaches it; and when the longer runs can tell it, their median last gap per
sample being at most a tenth of it. For each ratio of the steps and relaxation,
the tolerance is the largest accepted one whose smaller ones are accepted too;
the pair is the one that then stops the files in the fewest iterations in all;
and the most iterations allowed is twice the most a tuning file took, rounded
up to 1, 2 or 5 times a power of ten.

| | $\tau / \sigma$ | $\rho$ | gap per sample | most iterations |
|---|---|---|---|---|
| TV | 30 | 1.9 | $2 \cdot 10^{-4}$ | 20000 |
| TGV | 10 | 1.9 | $10^{-2}$ | 10000 |

TV at these defaults stopped the tuning files after 1050 to 5070 iterations
(median 1800). Its PSNR changed by 0.0002 dB in the median, 0.0019 dB in the
90th percentile and 0.0100 dB at most, and its SSIM by $2.1 \cdot 10^{-6}$,
$3.7 \cdot 10^{-5}$ and $1.1 \cdot 10^{-4}$. For TGV no tolerance that the longer
runs tell (from $6.7 \cdot 10^{-3}$) was accepted; the default is the least of
them that every file reached, with the pair that got there soonest: 1170 to
4570 iterations (median 2735), and a change of the PSNR of 0.0087 dB in the
median, 0.093 dB in the 90th percentile and 0.105 dB at most. That is TGV's
default until its convergence is improved.

Phase 1 chose a tolerance of $5 \cdot 10^{-4}$ for TV, unrelaxed, by the
largest change over the files, and found it within 0.0083 dB. That was measured
against long runs that went on from the stops along the same paths; against
the longer runs above, the same stops differ by 0.0008 dB in the median, 0.0062
dB in the 90th percentile and 0.014 dB at most.

**Colour.** The defaults were chosen on greyscale files, and do not meet the
criterion on colour ones (`experiments/results/phase2-colour.md`). On the colour
tuning files in 4:2:0 and 4:4:4 at the qualities 20 and 50 (48 files), TV,
coupled, stopped at the gap per sample of $2 \cdot 10^{-4}$ (the partial gap of
6.6 where samples are free) after a median of 2245 iterations, and its RGB's
PSNR changed against longer runs on another path by 0.0024 dB in the median,
0.025 dB in the 90th percentile and 0.071 dB at most; at $10^{-4}$, the least
tolerance those runs tell, by 0.0014, 0.016 and 0.057 dB. There are two causes.
Y settles within about 2000 iterations, to 0.001 dB, but Cb and Cr keep moving
for thousands more, and move the RGB by some hundredths of a dB while the gap,
which they add little to, is already small; two paths come within 0.002 dB of
each other by 8000 iterations, and within 0.0004 dB by 16000. And on smooth gradients TV keeps changing the picture at
a gap of $10^{-6}$ per sample, its PSNR falling as it nears its least point:
that is the staircasing of TV, not an error of the stop. The defaults stand for
colour files as well, until they are chosen again with the weights of the
channels.

These are for $\alpha = 1$ (for TGV $\alpha_1 = 1$, with $\alpha_0 = 2 \alpha_1$).
The dual variables are in units of $\alpha$ and the objective scales with it,
so with another weight the ratio is divided by $\alpha^2$ and the tolerance
multiplied by $\alpha$, unless that scaling is turned off. Neither applies to a
ratio or a tolerance that is given.

### 6.6 Several components: free samples

With free samples (1.3), $G^\ast$ is $+\infty$ off the range of $A^\top$ (4.4), and
so is the gap at the iterates, where $\xi = -K^\top y$ has deviations within the
cells, or samples beyond the blocks, that are not 0. The partial gap of 6.3
bounds $P - P^\ast$ instead, over the set

$$ B_R = \bigl\{x : |(x_c - \Pi_c x_c - m_c)_{i,j}| \le R \text{ for every } c, i, j\bigr\}, $$

where $m_c$ is 128 beyond the blocks of component $c$ and 0 within them: every
deviation from the mean of a cell is at most $R$, and every sample beyond the
blocks within $R$ of 128. With $\zeta_c = \xi_c - \Pi_c \xi_c$,

$$ \langle \xi_c, x_c \rangle = \langle \Pi_c \xi_c, \Pi_c x_c \rangle
   + \langle \zeta_c, x_c - \Pi_c x_c \rangle. $$

$G_c$ depends on $\Pi_c x_c$ alone, and the first part gives
$g_c^\ast(\nu_c^{-1} A_c \xi_c)$, as in 4.4. The second is at most
$\langle \zeta_c, m_c \rangle + R \|\zeta_c\|_1$ over the box, which holds the
deviations of $B_R$ (and more: those of a cell have the sum 0). So

$$ \mathrm{gap}_R = P(x) + \sum_c \Bigl( g_c^\ast\bigl(\nu_c^{-1} A_c \xi_c\bigr)
   + \langle \zeta_c, m_c \rangle + R\, \|\zeta_c\|_1 \Bigr) $$

is at least $P(x) - P^\ast$ when a solution lies in $B_R$. For TV,
$\xi_c = \gamma_c\, \mathrm{div}\, p_c$. For TGV, $\xi$ is that of the feasible
dual of 6.2, $\xi_c = -\gamma_c\, \mathrm{div}\, \mathrm{div}_2 \tilde r_c$, with
$\theta$ taken over the norm of 4.4; the partial gap of 6.3 takes
$\xi_c = \gamma_c\, \mathrm{div}\, p_c$ instead, and adds
$\varrho \sum_c \gamma_c \|p_c + \mathrm{div}_2 r_c\|_{2,1}$ for $|(w_c)_{i,j}| \le \varrho$.
At a saddle point, $-K^\top y^\ast$ lies in $\partial G(x^\ast)$, which is in the range
of $A^\top$: $\zeta = 0$, and the gap is 0 whatever $R$ is. Where nothing is free,
$\zeta = 0$, and this is the gap of 6.1 and 6.2.

$R$ is 255 by default. A solution whose samples lie within $[0, 255]$ is in
$B_{255}$: its deviations within cells are at most $255 (1 - 1/n_c)$, and its
samples beyond the blocks within 128 of 128. Nothing keeps the solution within
$[0, 255]$, and the gap bounds the distance from the least value only as far as
it does.

## 7. A subgradient method of jpeg2png's kind

*Implemented in `subgradient`, to compare with.*

jpeg2png minimizes a non-smooth objective within $\mathcal{C}$ by normalized
subgradient steps, a step length that falls as $1/\sqrt{n}$, and the
extrapolation of FISTA. The same scheme, applied to the objective $P$ of 4.2,
with a step $\eta$ and a decay $\beta$:

$$ \begin{aligned}
g &= -\alpha \mathrm{div}\bigl(\nabla y / |\nabla y|\bigr) + D^\top \bigl(\mu\, \omega\, (D y - \hat c)\bigr)
   && (\text{0 where } \nabla y = 0), \\
x^{n+1} &= P_{\mathcal{C}}\bigl(y^n - h_n\, g / \|g\|\bigr), \qquad h_n = \frac{\eta \sqrt{N}}{(1 + n)^\beta}, \\
t_{n+1} &= \tfrac12 \bigl(1 + \sqrt{1 + 4 t_n^2}\bigr), \qquad
y^{n+1} = x^{n+1} + \frac{t_n - 1}{t_{n+1}} \bigl(x^{n+1} - x^n\bigr),
\end{aligned} $$

from $y^0 = x^0$ and $t_0 = 1$, where $N$ is the number of samples, so that
the first step moves a sample by $\eta$ grey levels on average (root mean
square). By default $\eta = \beta = \tfrac12$ and the extrapolation is on;
without it, $y^{n+1} = x^{n+1}$. $g$ is a subgradient of
$P - \iota_{\mathcal{C}}$. The scheme has no convergence guarantee and no
measure of how far it is from the least value.

## 8. Colour

*Implemented in `colour`.*

A file in YCbCr is solved in YCbCr (4.4), and its result is the RGB that the
conversion of JFIF (ITU-T T.871) gives of the solution. With $K_R = 0.299$,
$K_B = 0.114$ and $K_G = 1 - K_R - K_B = 0.587$, the YCbCr of JFIF is

$$ Y = K_R R + K_G G + K_B B, \qquad C_B = 128 + \frac{B - Y}{2 (1 - K_B)}, \qquad
   C_R = 128 + \frac{R - Y}{2 (1 - K_R)}, $$

and its exact inverse is

$$ R = Y + \tfrac{701}{500} (C_R - 128), \qquad B = Y + \tfrac{443}{250} (C_B - 128), \qquad
   G = Y - \tfrac{25251}{73375} (C_B - 128) - \tfrac{209599}{293500} (C_R - 128), $$

the rationals being $2 (1 - K_R) = 1.402$, $2 (1 - K_B) = 1.772$,
$2 K_B (1 - K_B) / K_G$ and $2 K_R (1 - K_R) / K_G$. T.871 gives the last two to
six digits, 0.344136 and 0.714136. Those are off by $2.9 \cdot 10^{-7}$, which
moves $G$ by up to $7 \cdot 10^{-5}$, and the DC coefficient of a block converted
back by up to $3 \cdot 10^{-4}$: more than $10^{-4} Q$ where $Q \le 3$. The
rationals, rounded to the nearest doubles, leave only rounding.

**The result.** It is computed in binary64, in the order

$$ R = Y + c_R (C_R - 128), \qquad B = Y + c_B (C_B - 128), \qquad
   G = \bigl(Y - c_{GB} (C_B - 128)\bigr) - c_{GR} (C_R - 128), $$

where $c_R$, $c_B$, $c_{GB}$ and $c_{GR}$ are the doubles nearest to the four
rationals, from the solution's canvas cut to the picture. Each product may be
fused with the sum or difference that takes it (Arithmetic). The result is these
values themselves: it is neither rounded nor clamped, and a file of binary64
samples holds them to the last bit. Output of 8 or 16 bits rounds them to the
nearest, halves away from 0, and clamps them. The YCbCr of the solution can be
written instead, in binary64: the canvas itself, cut to the picture.

libjpeg's decoder clamps Y, Cb and Cr to 0–255 before it converts them; here
nothing is clamped before the conversion, and 8-bit output clamps the RGB alone.
Where a sample of Y overshoots 255 and its chroma is not neutral, the two differ
by more than rounding.

A file in RGB (Adobe's, without a transform) is solved in R, G and B, and its
result is the canvas itself.

**Going back.** In the same way, the YCbCr of an RGB picture is
$Y = (k_R R + k_G G) + k_B B$, $C_B = 128 + k_{CB} (B - Y)$ and
$C_R = 128 + k_{CR} (R - Y)$, with the doubles nearest to $K_R$, $K_G$, $K_B$,
$1 / 1.772 = 250 / 443$ and $1 / 1.402 = 500 / 701$. The RGB result converted
back is the solution to within rounding, and its coefficients are within their
intervals to within rounding (the tests bound both).

## References

- K. Bredies, K. Kunisch, T. Pock. Total generalized variation. SIAM Journal on
  Imaging Sciences 3(3), 2010.
- K. Bredies, M. Holler. Artifact-free decompression and zooming of JPEG
  compressed images with total generalized variation. 2013.
- X. Bresson, T. F. Chan. Fast dual minimization of the vectorial total
  variation norm and applications to color image processing. Inverse Problems
  and Imaging 2(4), 2008.
- A. Chambolle, T. Pock. A first-order primal-dual algorithm for convex problems
  with applications to imaging. Journal of Mathematical Imaging and Vision 40,
  2011.
- L. Condat. A primal-dual splitting method for convex optimization involving
  Lipschitzian, proximable and linear composite terms. Journal of Optimization
  Theory and Applications 158, 2013.
- T. Pock, A. Chambolle. Diagonal preconditioning for first order primal-dual
  algorithms in convex optimization. ICCV, 2011.
- T. Goldstein, M. Li, X. Yuan. Adaptive primal-dual splitting methods for
  statistical learning and image processing. NeurIPS, 2015.
- F. Alter, S. Durand, J. Froment. Adapted total variation for artifact free
  decompression of JPEG images. Journal of Mathematical Imaging and Vision 23,
  2005.
