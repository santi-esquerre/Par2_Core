RWPT: From ADE to Stochastic Particles
=======================================

This chapter derives the Random Walk Particle Tracking (RWPT) method
from the advection–dispersion equation (ADE) and maps each term to the
Par2\_Core implementation.  Equation numbers in parentheses refer to
[Rizzo2019]_.

.. contents::
   :local:
   :depth: 2

----

The Advection–Dispersion Equation
---------------------------------

Transport of a dissolved species in a saturated porous medium is governed
by ([Rizzo2019]_, Eq. 1):

.. math::

   \Theta \frac{\partial C}{\partial t}
   = -\nabla \cdot (\Theta\,\mathbf{u}\, C)
   + \nabla \cdot (\Theta\,\mathbf{D}\, \nabla C)

where :math:`C` is the resident concentration, :math:`\mathbf{u}` the
seepage (pore) velocity, :math:`\Theta` the porosity, and
:math:`\mathbf{D}` the local dispersion tensor
(see :doc:`dispersion_tensor`).

For **uniform porosity** the ADE simplifies to:

.. math::

   \frac{\partial C}{\partial t}
   = -\nabla \cdot (\mathbf{u}\, C)
   + \nabla \cdot (\mathbf{D}\, \nabla C)

Par2\_Core solves this simplified form.

Equivalent Itô SDE
-------------------

The equivalence between the ADE and the Fokker–Planck equation yields a
stochastic differential equation (SDE) for each particle
([Rizzo2019]_, Eq. 3):

.. math::

   d\mathbf{X} = \mathbf{A}(\mathbf{X})\,dt
                 + \mathbf{B}(\mathbf{X})\,d\mathbf{W}

where :math:`d\mathbf{W}` is a Wiener increment and:

* :math:`\mathbf{A}` is the **drift vector** (Eq. 6),
* :math:`\mathbf{B}` is the **displacement matrix** satisfying
  :math:`\mathbf{B}\mathbf{B}^T = 2\mathbf{D}` (Eqs. 9–14).

The drift vector
~~~~~~~~~~~~~~~~

The complete drift vector from the paper is
([Rizzo2019]_, Eq. 6):

.. math::

   \mathbf{A}(\mathbf{x})
   = \mathbf{u}(\mathbf{x})
   + \nabla \cdot \mathbf{D}(\mathbf{x})
   + \frac{1}{\Theta(\mathbf{x})}\,\mathbf{D}(\mathbf{x})\,
     \nabla\Theta(\mathbf{x})

.. important::

   **Porosity gradient term — not implemented.**
   The paper states explicitly that the third term
   (:math:`\frac{1}{\Theta}\mathbf{D}\nabla\Theta`) was not yet
   implemented in the original PAR² code.

   **Par2\_Core status:** this term is **also not implemented**.
   The engine assumes the porosity gradient is zero (uniform porosity).
   The velocity field passed to ``bind_velocity`` is assumed to be
   the seepage velocity :math:`\mathbf{u}`, not the Darcy flux —
   any porosity correction must be applied by the caller before binding.

   Supporting heterogeneous porosity would require an additional field
   binding (``bind_porosity``) and kernel modifications.

Itô–Taylor Stepping Scheme
---------------------------

Par2\_Core discretises the SDE using the explicit Euler–Maruyama
(Itô–Taylor order 1.0) scheme ([Rizzo2019]_, Eq. 5):

.. math::

   \mathbf{X}(t + \Delta t)
   = \mathbf{X}(t)
   + \mathbf{A}\bigl(\mathbf{X}(t)\bigr)\,\Delta t
   + \mathbf{B}\bigl(\mathbf{X}(t)\bigr)\,\boldsymbol{\xi}\,\sqrt{\Delta t}

where :math:`\boldsymbol{\xi} = (\xi_0, \xi_1, \xi_2)` are three
independent :math:`\mathcal{N}(0, 1)` variates drawn via cuRAND
(see :doc:`/hpc/rng_reproducibility`).

.. note::

   In the implementation, the :math:`\sqrt{\Delta t}` factor is absorbed
   into :math:`\mathbf{B}` at construction time
   (:math:`\mathbf{B}` includes :math:`\sqrt{2\,\Delta t}`).
   The kernel computes:

   .. math::

      \Delta\mathbf{x} = (\mathbf{v} + \mathbf{v}_{\text{drift}})\,\Delta t
        + \mathbf{B}\,\boldsymbol{\xi}

   See :doc:`displacement_matrix` for the :math:`\mathbf{B}` construction
   and :doc:`drift_correction` for the :math:`\nabla \cdot \mathbf{D}` modes.

Per-particle kernel algorithm
-----------------------------

Each active particle executes the following 8-step sequence inside
``move_particles_kernel_full`` (grid-stride loop, one thread per particle):

.. code-block:: text

   for each active particle i:
     1.  (idx, idy, idz) = cell_from_position(p[i])
     2.  v = sample_velocity(interp_mode, p[i])
     3.  v_drift = compute_drift(drift_mode, p[i])
     4.  v_B = sample_corner_velocity(p[i])     # for B matrix
     5.  B = displacement_matrix(v_B, Dm, αL, αT, dt)
     6.  ξ = (N(0,1), N(0,1), N(0,1))           # curand_normal_double
     7.  Δp = (v + v_drift)·dt + B·ξ
     8.  p[i] = apply_boundary(p[i], Δp)

.. note::

   **Corner velocity for B.**
   The displacement matrix :math:`\mathbf{B}` always uses
   **corner-centred velocity** (trilinear interpolation at the
   particle position), even when advection uses face-centred (Linear)
   interpolation.  This ensures the B matrix sees a smooth velocity
   field, avoiding discontinuities at cell faces that would create
   unphysical dispersion jumps.

   If corner velocity arrays are available (``update_derived_fields()``
   has been called), they are used directly.  Otherwise, face velocity
   serves as fallback (less smooth but functional).

   This hybrid approach — **linear at faces for advection, trilinear
   at corners for dispersion** — is a defining characteristic of the
   PAR² algorithm [Rizzo2019]_.

Source
------

* Kernel: ``src/kernels/move_particles.cu`` (``move_particles_kernel_full``)
* Displacement: step (7) maps to lines computing ``dpx``, ``dpy``, ``dpz``

References
----------

* [Rizzo2019]_ — primary reference for
  PAR² algorithm and analytical B matrix.
* [Salamon2006]_ — review of RWPT methods.
* [LaBolle1996]_ — foundational RWPT paper.
