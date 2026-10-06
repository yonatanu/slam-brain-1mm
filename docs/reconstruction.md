# Reconstruction conventions

For each independent 2D slice, let $x$ be a complex image, $S_c$ its coil
sensitivity maps, and $M$ the real binary acquisition mask. The operator is

$$
(Ax)_c=M\mathcal F(S_c x),\qquad
A^H y=\sum_c\overline{S_c}\,\mathcal F^H(My_c).
$$

$\mathcal F$ is a centered, orthonormal 2D discrete Fourier transform on the
readout and phase axes.

CG solves

$$
\widehat x=\arg\min_x\|Ax-y\|_2^2+\lambda\|x\|_2^2,
\qquad (A^HA+\lambda I)\widehat x=A^Hy.
$$

The default is $\lambda=0$, a zero initial image, at most 40 iterations, and
relative normal-equation residual below $10^{-5}$. Each slice has its own step
sizes and convergence decision. Converged slices are frozen. The returned
residual is recomputed from the final image; check it when using a fixed iteration
budget. A zero right-hand side returns a zero image with zero iterations.

The returned image is in the same stored units as the released reference.
Normalize an image and its measurements by the same positive per-slice factor
when changing units. Maps and masks retain their original values.
