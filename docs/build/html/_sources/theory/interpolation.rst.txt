Velocity Interpolation
======================

Par2\_Core supports two velocity interpolation modes at particle
positions.  The **hybrid scheme** — linear at faces for advection,
trilinear at corners for dispersion — is a defining characteristic of
the PAR² algorithm ([Rizzo2019]_).

Linear (Face-Centred)
---------------------

Each velocity component is interpolated independently along its staggered
axis from the two adjacent face values:

.. math::

   t_x &= \frac{p_x - c_x}{\Delta x} + 0.5 \quad \in [0, 1] \\
   v_x &= (1 - t_x)\, U[\text{left}] + t_x\, U[\text{right}]

where :math:`(c_x, c_y, c_z)` is the cell centre.  Similarly for
:math:`v_y` (Y-faces) and :math:`v_z` (Z-faces).

**Face-field indexing** (legacy ``mergeId``)::

   index = iz * (ny+1) * (nx+1) + iy * (nx+1) + ix

Array size per component: :math:`(n_x+1)(n_y+1)(n_z+1)`.

Trilinear (Corner-Based)
-------------------------

Trilinear interpolation using all 8 corner velocity values of the
containing cell.  Normalised coordinates:

.. math::

   t = \frac{\mathbf{p} - \mathbf{c}}{\Delta \mathbf{h}} + 0.5
     \quad \in [0, 1]^3

The interpolation proceeds along X (4 lerps), then Y (2 lerps), then Z
(1 lerp) — standard trilinear.

**Corner velocity computation:** Each corner value is the **average**
of the face velocities at up to 4 adjacent cells (2×2 stencil).  For
boundary corners, fewer faces contribute (1 or 2).  This is computed
once by ``update_derived_fields()`` and cached in workspace.

Choosing a Mode
---------------

+-------------+-----------------------------------------+-----------------------------------------------------------+
| Mode        | Pros                                    | Cons                                                      |
+=============+=========================================+===========================================================+
| Linear      | No precomputation; faster per step      | Velocity discontinuous at cell boundaries; no smooth      |
|             |                                         | gradients for drift correction                            |
+-------------+-----------------------------------------+-----------------------------------------------------------+
| Trilinear   | Smooth velocity within cells; needed    | Requires corner velocity precomputation;                  |
|             | for on-the-fly drift correction         | corner buffers use 3 × field_size memory                  |
+-------------+-----------------------------------------+-----------------------------------------------------------+

Use **Trilinear** when drift correction (``TrilinearOnFly``) is needed
or when smooth velocity gradients are required.  Use **Linear** for
fast exploratory runs without drift correction.

Source
------

* Linear: ``src/internal/fields/facefield_accessor.cuh``
* Trilinear: ``src/internal/fields/cornerfield_accessor.cuh``
* Corner computation: ``src/kernels/cornerfield.cu``
