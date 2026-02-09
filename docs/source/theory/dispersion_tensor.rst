Dispersion Tensor
=================

The mechanical dispersion tensor in a saturated porous medium is
([Rizzo2019]_, Eq. 4):

.. math::

   D_{ij} = \left(\alpha_T\, |\mathbf{v}| + D_m\right) \delta_{ij}
            + \left(\alpha_L - \alpha_T\right) \frac{v_i\, v_j}{|\mathbf{v}|}

where:

* :math:`\alpha_L` — longitudinal dispersivity [L]
* :math:`\alpha_T` — transverse dispersivity [L]
* :math:`D_m` — effective molecular diffusion coefficient [L²/T]
* :math:`|\mathbf{v}|` — magnitude of the seepage velocity vector

Source: ``src/internal/math/dispersion.cuh`` (``compute_D_tensor``).

Eigenvalues
-----------

The tensor :math:`\mathbf{D}` can be written as
:math:`\mathbf{D} = a\,\mathbf{I} + b\,\mathbf{v}\mathbf{v}^T`
(Eq. 9), a rank-1 perturbation of a scalar matrix.
Its eigenvalues are ([Rizzo2019]_, Eq. 10):

.. math::

   \lambda_L &= \alpha_L\,|\mathbf{v}| + D_m
     \qquad \text{(longitudinal, along } \mathbf{v}\text{)} \\
   \lambda_T &= \alpha_T\,|\mathbf{v}| + D_m
     \qquad \text{(transverse, 2× degenerate)}


Displacement Matrix :math:`\mathbf{B}`
---------------------------------------

The particle displacement requires :math:`\mathbf{B}` such that
:math:`\mathbf{B}\mathbf{B}^T = 2\mathbf{D}\,\Delta t`.

Par2\_Core constructs :math:`\mathbf{B}` via **analytical
eigendecomposition** using the eigenvectors from
[Rizzo2019]_, Eqs. 12–14.  The full derivation, assembly
formula, special-case dispatching, and zero-velocity handling are
documented in :doc:`displacement_matrix`.

Special-Case Dispatching
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Condition
     - Behaviour
   * - :math:`D_m = \alpha_L = \alpha_T = 0`
     - :math:`\mathbf{B} = \mathbf{0}` — pure advection
   * - :math:`\alpha_L = \alpha_T = 0,\; D_m > 0`
     - :math:`\mathbf{B} = \sqrt{2 D_m \Delta t}\;\mathbf{I}` — isotropic
   * - General
     - Full eigendecomposition (see :doc:`displacement_matrix`)


Zero-Velocity Handling
----------------------

When :math:`|\mathbf{v}| \to 0`, the :math:`v_i v_j / |\mathbf{v}|`
term in :math:`D_{ij}` is ill-defined, and the eigenvector norms used
in B assembly approach zero.  Par2\_Core applies a velocity tolerance
inherited from legacy PAR²:

.. math::

   \text{toll} = 0.01 \cdot D_m / \alpha_L, \qquad
   v_x \leftarrow \max(v_x,\; \text{toll})

Full details of the zero-velocity handling — including the
``nan_prevention`` guards added by Par2\_Core — are in
:doc:`displacement_matrix`.
