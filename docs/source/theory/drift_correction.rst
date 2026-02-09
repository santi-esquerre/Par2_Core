Drift Correction
================

The drift correction term :math:`\nabla \cdot \mathbf{D}` arises from
the Fokker–Planck equivalence of the ADE.  It compensates for spatial
variation of the dispersion tensor — without it, particles accumulate
in low-dispersion regions ([Rizzo2019]_,
Eq. 6; [LaBolle1996]_; [LaBolle2000]_).

.. math::

   v_{\text{drift},i} = \sum_j \frac{\partial D_{ij}}{\partial x_j}

.. note::

   This is the second term of the full drift vector
   :math:`\mathbf{A} = \mathbf{u} + \nabla \cdot \mathbf{D}
   + \frac{1}{\Theta}\mathbf{D}\nabla\Theta\,`.
   The porosity gradient term (:math:`\frac{1}{\Theta}\mathbf{D}\nabla\Theta`)
   is **not implemented** — see :doc:`rwpt` for details.

Par2_Core Modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 25

   * - Mode
     - Description
     - Legacy YAML equivalent
   * - ``None``
     - No correction (D constant)
     - —
   * - ``Precomputed``
     - Finite-difference div(D)
     - ``"finite difference"``
   * - ``TrilinearOnFly``
     - Analytical trilinear derivatives
     - ``"trilinear"``

TrilinearOnFly Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes :math:`\nabla \cdot \mathbf{D}` at the **particle position**
using trilinear derivative interpolation of D-tensor evaluated at the
8 cell corners.  For example:

.. math::

   \frac{\partial D_{xx}}{\partial x}\bigg|_{\mathbf{p}}
     = \text{trilinearDevX}\!\left(t_x, t_y, t_z,\; D_{xx}^{\text{corners}}\right)

where ``trilinearDevX`` computes :math:`(v_1 - v_0)/\Delta x` on the
differentiation axis and interpolates in the remaining two dimensions.

**Zero-velocity tolerance** is applied at each corner before evaluating D:
``toll = 0.01 * Dm / αL``; if all velocity components are below toll,
:math:`v_x` is set to toll.

**Computational cost:**  24 velocity loads (8 corners × 3 components),
48 D-tensor evaluations, 9 trilinear derivative calls.

Precomputed Finite-Difference Stencil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-step approach (separate kernels):

1. **Step 1:** Compute full D tensor at each cell centre from face-velocity
   interpolation at the cell centre.

2. **Step 2:** Finite differences:

   .. math::

      \frac{\partial D_{ij}}{\partial x}\bigg|_k =
      \begin{cases}
        (D_{ij}[k+1] - D_{ij}[k]) / \Delta x
          & k = 0 \text{ (forward)} \\
        (D_{ij}[k+1] - D_{ij}[k-1]) / (2\Delta x)
          & 0 < k < n_x{-}1 \text{ (central)} \\
        (D_{ij}[k] - D_{ij}[k-1]) / \Delta x
          & k = n_x{-}1 \text{ (backward)}
      \end{cases}

   Requires 9 temporary arrays (6 D-tensor + 3 drift) of ``num_cells``
   elements each.

In the move kernel, precomputed drift is a **piecewise-constant** cell
lookup (no within-cell interpolation).

Source
------

* TrilinearOnFly: ``src/internal/fields/cornerfield_accessor.cuh``
  (``compute_drift_trilinear``)
* Precomputed: ``src/kernels/drift_correction.cu``
  (``compute_D_tensor_kernel``, ``compute_drift_from_D_kernel``)
