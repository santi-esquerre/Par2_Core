Displacement Matrix :math:`\mathbf{B}`
=======================================

The particle displacement in the RWPT scheme requires a matrix
:math:`\mathbf{B}` such that :math:`\mathbf{B}\mathbf{B}^T = 2\mathbf{D}\,\Delta t`,
where :math:`\mathbf{D}` is the local dispersion tensor
(see :doc:`dispersion_tensor`).

Par2\_Core constructs :math:`\mathbf{B}` **analytically** via explicit
eigendecomposition of :math:`\mathbf{D}`.  This is a deliberate HPC design
choice: iterative eigensolvers (Jacobi, QR) would require loops with
unpredictable iteration counts inside each CUDA thread, destroying warp
convergence and throughput.  The analytical path is branch-free in the
common case ([Rizzo2019]_, Eqs. 9–14).

.. contents::
   :local:
   :depth: 2

----

Algebraic setup
---------------

The dispersion tensor can be decomposed as
([Rizzo2019]_, Eq. 9):

.. math::

   \mathbf{D} = a\,\mathbf{I} + b\,\mathbf{W}

where:

.. math::

   a = \alpha_T\,|\mathbf{v}| + D_m, \qquad
   b = \frac{\alpha_L - \alpha_T}{|\mathbf{v}|}, \qquad
   \mathbf{W} = \mathbf{v}\,\mathbf{v}^T

This is a rank-1 perturbation of a scalar matrix.  The eigenvalues
and eigenvectors are known in closed form.

Eigenvalues
-----------

([Rizzo2019]_, Eq. 10):

.. math::

   \lambda_L &= a + b\,|\mathbf{v}|^2 = \alpha_L\,|\mathbf{v}| + D_m
     \qquad \text{(longitudinal)} \\
   \lambda_T &= a = \alpha_T\,|\mathbf{v}| + D_m
     \qquad \text{(transverse, 2× degenerate)}

Eigenvectors
------------

Par2\_Core uses three **unnormalised** orthogonal eigenvectors
([Rizzo2019]_, Eqs. 12–14):

.. math::

   \mathbf{e}_0 &= (v_x,\; v_y,\; v_z)
     & \text{along flow, eigenvalue } \lambda_L \\
   \mathbf{e}_1 &= (-v_y,\; v_x,\; 0)
     & \text{perpendicular in XY, eigenvalue } \lambda_T \\
   \mathbf{e}_2 &= (-v_z v_x,\; -v_z v_y,\; v_x^2 {+} v_y^2)
     & \text{perpendicular to both, eigenvalue } \lambda_T

Their squared norms are:

.. math::

   |\mathbf{e}_0|^2 &= |\mathbf{v}|^2 \\
   |\mathbf{e}_1|^2 &= v_x^2 + v_y^2 \\
   |\mathbf{e}_2|^2 &= (v_x^2 + v_y^2)(v_x^2 + v_y^2 + v_z^2)
                      = |\mathbf{e}_1|^2 \cdot |\mathbf{v}|^2

Gamma coefficients and assembly
--------------------------------

Define the gamma coefficients:

.. math::

   \gamma_i = \frac{\sqrt{\lambda_i}}{|\mathbf{e}_i|^2}

The :math:`\mathbf{B}` matrix is then
([Rizzo2019]_, Eq. 11):

.. math::

   B_{jk} = \sqrt{2\,\Delta t}\;
     \sum_{i=0}^{2} \gamma_i \; e_i[j] \; e_i[k]

The result is a **symmetric** 3×3 matrix.  Only 6 unique components
are computed and stored: ``B00``, ``B11``, ``B22``, ``B01``, ``B02``,
``B12``.

.. note::

   The factor :math:`\sqrt{2\,\Delta t}` is baked into :math:`\mathbf{B}`
   at construction time.  The move kernel applies
   :math:`\Delta\mathbf{x}_{\text{stochastic}} = \mathbf{B}\,\boldsymbol{\xi}`
   directly, without a separate :math:`\sqrt{\Delta t}` multiplier.

Special cases
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Condition
     - Behaviour
   * - :math:`D_m = \alpha_L = \alpha_T = 0`
     - :math:`\mathbf{B} = \mathbf{0}` — pure advection, no RNG calls
   * - :math:`\alpha_L = \alpha_T = 0,\; D_m > 0`
     - :math:`\mathbf{B} = \sqrt{2\,D_m\,\Delta t}\;\mathbf{I}` — isotropic
       diffusion, three multiply-adds
   * - General
     - Full eigendecomposition (formulae above)

These short-circuits avoid redundant computation and division by zero
when :math:`|\mathbf{v}| = 0`.

Zero-velocity handling
----------------------

When :math:`|\mathbf{v}| \to 0`, both :math:`|\mathbf{e}_1|^2` and
:math:`|\mathbf{e}_2|^2` approach zero, producing division-by-zero in
the gamma coefficients.  The legacy PAR² solution
([Rizzo2019]_, implemented in ``displacementMatrix``):

.. math::

   \text{toll} = 0.01 \cdot D_m / \alpha_L, \qquad
   v_x \leftarrow \max(v_x,\; \text{toll})

Only :math:`v_x` is clamped.  This ensures :math:`|\mathbf{e}_1|^2 \geq
\text{toll}^2 > 0`.

**Legacy edge case:** when :math:`\alpha_L = 0`, the division
``Dm / αL`` produces ``NaN``.  Par2\_Core returns ``toll = 1e-15``
instead, which is small enough to have no physical effect but prevents
NaN propagation.

**Extra guards** (``EngineConfig::nan_prevention = true``, not in
legacy):

* Clamps :math:`|\mathbf{e}_1|^2` and :math:`|\mathbf{e}_2|^2` to
  :math:`\text{toll}^2` when they approach zero (e.g., flow purely
  in the Z direction where :math:`v_x = v_y = 0`).
* In precomputed drift mode: sets :math:`\mathbf{D} = D_m\,\mathbf{I}`
  directly when :math:`|\mathbf{v}| < \text{toll}`.

HPC motivation
--------------

The analytical construction avoids:

* **Iterative eigensolvers** — Jacobi or QR iterations have
  data-dependent loop counts; in a warp of 32 threads, the slowest
  thread determines runtime.  Analytical formulae are constant-cost.
* **Cholesky decomposition** — requires :math:`\mathbf{D}` to be
  strictly positive-definite.  At :math:`|\mathbf{v}| = 0` with
  :math:`D_m > 0`, :math:`\mathbf{D} = D_m\,\mathbf{I}` is fine,
  but the general Cholesky branch adds complexity with no benefit.
* **Shared memory** — each thread computes B from its own velocity;
  no inter-thread communication is needed.

The entire B computation is ~40 FLOPs per particle per step (excluding
RNG), fitting comfortably in registers.

Source
------

Implementation: ``src/internal/math/dispersion.cuh``
(``compute_displacement_matrix_legacy``, ``compute_B_matrix``).
