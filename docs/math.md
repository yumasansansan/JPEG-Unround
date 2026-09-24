<!-- SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com> -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# The mathematics of JPEG-Unround

This is the one statement of the model and of the algorithms, which the three
implementations follow. Each section says where it is implemented. So far that
is the Python reference, for one component without chroma subsampling
(`python/src/unround/`). Chroma subsampling and the coupling of colour
components are still to be written here, before they are implemented.

## Arithmetic

What is rational is computed exactly, and is the same to the last bit in every
implementation: the levels and the steps; the counts of 2.1 and the integers
made of them ($D$ needs up to about 80 bits: 128-bit integers in C++ and Rust);
the intervals without slack, $(q - \tfrac12) Q$, which binary64 holds exactly for
levels and steps of 16 bits; the coefficients of the series of 2.2, computed in
rationals; and, for pictures of integers, the sums and means of the metrics. A
rational enters floating point once, rounded to the nearest double. With a slack
$s$, the ends are $((q - \tfrac12) - s) Q$ and $((q + \tfrac12) + s) Q$, in that
order, so that every implementation rounds them alike.

What needs floating point (the basis of the DCT, square roots, logarithms and
exponentials, and the iterations) is computed in binary64 at least; binary32 is
not used. A constant is the double nearest to its value, as the cosines of 1.1
are. The tests take their tolerances from bounds of rounding error, and check
what holds exactly in exact arithmetic.

## 1. What a JPEG file says about a component

### 1.1 Samples, blocks and the DCT

*Implemented in `unround/dct.py`; the intervals and the projection in
`unround/model.py`.*

A component is decoded on its **canvas**: the whole blocks that its
coefficients cover, $H = 8\,B_y$ rows of $W = 8\,B_x$ samples. The picture is
the top-left $h \times w$ of it. The encoder filled the samples beyond the
picture itself (libjpeg repeats the last row and column); here they are free
variables, which the regularizer alone decides, and they are cut off at the
end. The unknown is the canvas $x \in \mathbb{R}^{H \times W}$, level-shifted:
the sample value less 128.

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

The file holds, for every coefficient $k$ (a block and a frequency), an integer
$q_k$, and for every frequency the step $Q_k$ of the component's quantization
table. The encoder chose $q_k$ as the nearest integer to $(D y)_k / Q_k$, where
$y$ is the canvas that it was given.

### 1.2 The quantization constraint set

The coefficient lies within half a step of $q_k Q_k$. With a slack
$s \ge 0$, in steps, the interval of coefficient $k$ is

$$ a_k = (q_k - \tfrac12 - s)\,Q_k, \qquad b_k = (q_k + \tfrac12 + s)\,Q_k, $$

and the **quantization constraint set** (QCS) is
$\mathcal{C} = \{x : a \le D x \le b\}$. Because $D$ is orthogonal and the
box is a product of intervals, the projection onto $\mathcal{C}$ is

$$ P_{\mathcal{C}}(x) = D^\top \operatorname{clip}(D x, a, b). $$

Every output of JPEG-Unround lies in $\mathcal{C}$: the coefficients of the
floating-point result are within their intervals, to $10^{-4} Q_k$, checked
in binary64. The slack is 0 unless it is asked for.

Where the original lies (measured in `experiments/results/phase0.md`): a file
of libjpeg's encoder has the original within its intervals, apart from the
encoder's own rounding. The encoder rounds its YCbCr samples to 8 bits, and in
a flat block that rounding moves every sample alike, so the DC coefficient can
move by up to $8 \times \tfrac12 = 4$. mozjpeg's trellis quantization chooses
levels up to about two steps from the nearest, and its overshoot deringing
encodes white samples pushed beyond 255.

## 2. A Laplace model of the AC coefficients

*Implemented in `unround/laplace.py`.*

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
$-\operatorname{log1p}(-(1 - t^\ast))$ with

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

$$ \hat c_k = \operatorname{sign}(q_k)\,\bigl(|q_k| Q_k - \delta_k\bigr), \qquad
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
centre of its interval, $q_0 Q_0$, and the data term below leaves it out.

### 2.3 The MMSE decoder

$x = D^\top \hat c$, with the DC centres as above, is a decoder of its own: the
coefficients' conditional means. It costs one inverse DCT, lies in
$\mathcal{C}$, and is the starting point of the solvers below.

## 3. Finite differences

*Implemented in `unround/operators.py`.*

### 3.1 Gradient and divergence

On an $H \times W$ grid, with $i$ the row and $j$ the column, the gradient uses
forward differences with Neumann boundaries (the difference out of the grid is
0):

$$ (\partial_x x)_{i,j} = \begin{cases} x_{i,j+1} - x_{i,j} & j < W - 1 \\ 0 & j = W - 1 \end{cases}
   \qquad
   \nabla x = (\partial_x x, \partial_y x), $$

and $\partial_y$ likewise down the columns. The divergence is its negative
adjoint, $\operatorname{div} = -\nabla^\top$, which is the backward difference

$$ (\delta_x f)_{i,j} = f_{i,j}\,[j < W - 1] - f_{i,j-1}\,[j > 0], \qquad
   \operatorname{div}(p_1, p_2) = \delta_x p_1 + \delta_y p_2, $$

where $[\cdot]$ is 1 when its condition holds and 0 otherwise. So
$\langle \nabla x, p \rangle = -\langle x, \operatorname{div} p \rangle$.

### 3.2 The symmetrized gradient

Symmetric 2×2 tensor fields $r = (r_{11}, r_{22}, r_{12})$ have the inner
product that counts the off-diagonal entry twice, as the full matrix does:

$$ \langle r, s \rangle = \sum_{i,j} r_{11} s_{11} + r_{22} s_{22} + 2\, r_{12} s_{12},
   \qquad |r|_F = \sqrt{r_{11}^2 + r_{22}^2 + 2 r_{12}^2}. $$

The symmetrized gradient of a vector field $w = (w_1, w_2)$ uses the backward
differences of 3.1, and its negative adjoint, the divergence of a tensor field,
the forward differences:

$$ \mathcal{E} w = \bigl(\delta_x w_1,\ \delta_y w_2,\ \tfrac12 (\delta_y w_1 + \delta_x w_2)\bigr), \qquad
   \operatorname{div}_2 r = (\partial_x r_{11} + \partial_y r_{12},\ \partial_x r_{12} + \partial_y r_{22}). $$

Then $\langle \mathcal{E} w, r \rangle = -\langle w, \operatorname{div}_2 r \rangle$,
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

*Implemented in `unround/model.py`, for one component without subsampling.*

### 4.1 The data term and the constraint

With $c = D x$, the weights $\omega_k = 1 / Q_k^2$ on AC coefficients and
$\omega_k = 0$ on DC, and $\mu \ge 0$,

$$ G(x) = \frac{\mu}{2} \sum_k \omega_k (c_k - \hat c_k)^2 + \iota_{[a, b]}(c), $$

where $\iota$ is 0 on its set and $+\infty$ off it. $G$ is separable in the
coefficients, and so are its proximal map and its conjugate.

**Proximal map.** For $\tau > 0$ and $e = D v$,
$\operatorname{prox}_{\tau G}(v) = D^\top \zeta$ with

$$ \zeta_k = \operatorname{clip}\!\left(\frac{e_k + \tau \mu \omega_k \hat c_k}{1 + \tau \mu \omega_k},\ a_k,\ b_k\right), $$

since a strictly convex quadratic of one variable, restricted to an interval,
is least at the clipped unconstrained minimizer.

**Conjugate.** For $\xi$ and $s = D \xi$,
$G^\ast(\xi) = \sum_k g_k^\ast(s_k)$ with, where $m_k = \mu \omega_k > 0$,

$$ g_k^\ast(s) = s\, c^\ast - \frac{m_k}{2} (c^\ast - \hat c_k)^2, \qquad
   c^\ast = \operatorname{clip}(\hat c_k + s / m_k,\ a_k,\ b_k), $$

and where $m_k = 0$ (DC, or $\mu = 0$), $g_k^\ast(s) = \max(s\, a_k,\ s\, b_k)$.
Both are finite everywhere, because the intervals are bounded.

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

## 5. The primal–dual hybrid gradient method

*Implemented in `unround/pdhg.py`.*

Both models are $\min_z F(K z) + G(z)$ with $F$ a norm, and so the saddle-point
problem $\min_z \max_y \langle K z, y \rangle + G(z) - F^\ast(y)$, where
$F^\ast$ is the indicator of the dual ball of the norm. The method of
Chambolle and Pock, with $\sigma \tau \|K\|^2 < 1$, is

$$ y^{n+1} = \operatorname{prox}_{\sigma F^\ast}\bigl(y^n + \sigma K \bar z^n\bigr), \quad
   z^{n+1} = \operatorname{prox}_{\tau G}\bigl(z^n - \tau K^\top y^{n+1}\bigr), \quad
   \bar z^{n+1} = 2 z^{n+1} - z^n. $$

The proximal map of the indicator of a ball is the projection onto it, pixel
by pixel: $y \mapsto y / \max(1, |y| / \alpha)$.

**Total variation.** $K = \nabla$, $y = p$, $K^\top p = -\operatorname{div} p$:

$$ p \leftarrow \operatorname{proj}_{\alpha}\bigl(p + \sigma \nabla \bar x\bigr), \qquad
   x^+ \leftarrow \operatorname{prox}_{\tau G}\bigl(x + \tau \operatorname{div} p\bigr), \qquad
   \bar x \leftarrow 2 x^+ - x. $$

**TGV.** $z = (x, w)$, $y = (p, r)$,
$K^\top (p, r) = (-\operatorname{div} p,\ -p - \operatorname{div}_2 r)$, and
$G$ does not depend on $w$, whose proximal map is the identity:

$$ \begin{aligned}
p &\leftarrow \operatorname{proj}_{\alpha_1}\bigl(p + \sigma (\nabla \bar x - \bar w)\bigr), &
r &\leftarrow \operatorname{proj}_{\alpha_0, F}\bigl(r + \sigma \mathcal{E} \bar w\bigr), \\
x^+ &\leftarrow \operatorname{prox}_{\tau G}\bigl(x + \tau \operatorname{div} p\bigr), &
w^+ &\leftarrow w + \tau (p + \operatorname{div}_2 r),
\end{aligned} $$

then $\bar x = 2 x^+ - x$ and $\bar w = 2 w^+ - w$.

**Steps.** With $L^2$ the bound of 3.3 (8 for TV, 11.37 for TGV) and a ratio
$\kappa = \tau / \sigma$, the steps are $\tau = \sqrt{0.99\, \kappa} / L$ and
$\sigma = \sqrt{0.99 / \kappa} / L$, so that $\sigma \tau L^2 = 0.99$. The
samples are in grey levels and the dual variables within $\alpha$, so a ratio
above 1 lets the picture move faster.

**Start.** $x$ is the MMSE decoder's output (2.3), $w = \nabla x$, and the
dual variables are 0.

Every iterate $x$ is an output of $\operatorname{prox}_{\tau G}$, and so lies in
$\mathcal{C}$. The coefficients $\zeta$ of the last one are kept: the result is
$D^\top \zeta$, whose coefficients are $\zeta$ to rounding error.

## 6. When to stop: duality gaps

*Implemented in `unround/model.py` and `unround/pdhg.py`.*

The dual of $\min F(Kz) + G(z)$ is $\max_y -F^\ast(y) - G^\ast(-K^\top y)$.
For any feasible pair, the gap $P(z) - D(y) \ge P(z) - P^\ast \ge 0$ bounds
how far $z$ is from the least value, and it is 0 at a saddle point.

### 6.1 Total variation: in closed form

$-K^\top p = \operatorname{div} p$, and every iterate has $|p| \le \alpha$, so

$$ \operatorname{gap}(x, p) = \alpha \|\nabla x\|_{2,1} + G(x) + G^\ast(\operatorname{div} p), $$

with $G^\ast$ of 4.1: finite, and in closed form.

### 6.2 TGV: a feasible dual

Here $-K^\top(p, r) = (\operatorname{div} p,\ p + \operatorname{div}_2 r)$, and
$G$ does not depend on $w$, so the dual is finite only where
$p = -\operatorname{div}_2 r$: at the iterates of the method it is
$-\infty$, and the gap $+\infty$. The pair is made feasible by scaling:

$$ \theta = \min\Bigl(1,\ \frac{\alpha_1}{\max_{i,j} |(\operatorname{div}_2 r)_{i,j}|}\Bigr), \qquad
   \tilde r = \theta r, \qquad \tilde p = -\operatorname{div}_2 \tilde r. $$

Then $|\tilde p| \le \alpha_1$, $|\tilde r|_F \le \alpha_0$ and
$\tilde p + \operatorname{div}_2 \tilde r = 0$, so

$$ \operatorname{gap}(x, w, r) = P(x, w) + G^\ast(\operatorname{div} \tilde p)
   = P(x, w) + G^\ast(-\operatorname{div} \operatorname{div}_2 \tilde r) $$

is an upper bound of $P(x, w) - P^\ast$. At a saddle point
$p = -\operatorname{div}_2 r$ already, $\theta = 1$, and the gap is 0.

### 6.3 TGV: a partial gap

The gap of Chambolle and Pock restricted to a bounded set,
$\mathcal{G}_B = \sup_{y} L(z, y) - \inf_{z' \in B} L(z', y)$ with
$L(z, y) = \langle K z, y \rangle + G(z) - F^\ast(y)$, bounds $P(z) - P^\ast$
when $B$ contains the primal part of a saddle point. With
$B = \{(x, w) : |w_{i,j}| \le \varrho\}$,

$$ \mathcal{G}_B(x, w, p, r) = P(x, w) + G^\ast(\operatorname{div} p)
   + \varrho\, \|p + \operatorname{div}_2 r\|_{2,1}. $$

It needs no change of the iterates, but only holds for a $\varrho$ at least as
large as the solution's $w$, which is not known beforehand.

### 6.4 What is compared with the tolerance

The gaps are sums over the canvas. They are divided by its number of samples,
so that the tolerance is in the units of the objective per sample and does not
depend on the size of the picture.

## 7. A subgradient method of jpeg2png's kind

*Implemented in `unround/subgradient.py`, to compare with.*

jpeg2png minimizes a non-smooth objective within $\mathcal{C}$ by normalized
subgradient steps, a step length that falls as $1/\sqrt{n}$, and the
extrapolation of FISTA. The same scheme, applied to the objective $P$ of 4.2:

$$ \begin{aligned}
g &= -\alpha \operatorname{div}\bigl(\nabla y / |\nabla y|\bigr) + D^\top \bigl(\mu\, \omega\, (D y - \hat c)\bigr)
   && (\text{0 where } \nabla y = 0), \\
x^{n+1} &= P_{\mathcal{C}}\bigl(y^n - h_n\, g / \|g\|\bigr), \qquad h_n = \frac{\sqrt{N}/2}{\sqrt{1 + n}}, \\
t_{n+1} &= \tfrac12 \bigl(1 + \sqrt{1 + 4 t_n^2}\bigr), \qquad
y^{n+1} = x^{n+1} + \frac{t_n - 1}{t_{n+1}} \bigl(x^{n+1} - x^n\bigr),
\end{aligned} $$

from $y^0 = x^0$ and $t_0 = 1$, where $N$ is the number of samples, so that
the first step moves a sample by half a grey level on average. $g$ is a
subgradient of $P - \iota_{\mathcal{C}}$. The scheme has no convergence
guarantee and no measure of how far it is from the least value.

## References

- K. Bredies, K. Kunisch, T. Pock. Total generalized variation. SIAM Journal on
  Imaging Sciences 3(3), 2010.
- K. Bredies, M. Holler. Artifact-free decompression and zooming of JPEG
  compressed images with total generalized variation. 2013.
- A. Chambolle, T. Pock. A first-order primal-dual algorithm for convex problems
  with applications to imaging. Journal of Mathematical Imaging and Vision 40,
  2011.
- F. Alter, S. Durand, J. Froment. Adapted total variation for artifact free
  decompression of JPEG images. Journal of Mathematical Imaging and Vision 23,
  2005.
