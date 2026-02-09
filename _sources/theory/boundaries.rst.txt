Boundary Conditions
===================

Par2_Core supports per-axis, per-face boundary conditions:

.. list-table::
   :header-rows: 1

   * - Type
     - Behaviour
     - Use case
   * - ``Closed``
     - Reflective — displacement zeroed at boundary
     - Confined aquifer
   * - ``Periodic``
     - Position wraps to opposite face; wrap counter tracks crossings
     - Infinite domain approximation
   * - ``Open``
     - Particle exits domain and is marked inactive
     - Outflow boundaries

Wrap Counters (Periodic)
------------------------

For periodic axes, the engine tracks net domain crossings so that
the continuous (unwrapped) position can be recovered:

.. math::

   x_{\text{unwrapped}} = x + n_{\text{wrap}} \cdot L_x

**Status flags:**  Particles have a ``ParticleStatus`` (``uint8_t``):

* ``Active`` (0) — normal tracking.
* ``Exited`` (1) — left the domain via Open BC; skipped in subsequent steps.
* ``Inactive`` (2) — manually deactivated by user.

The ``StatsComputer`` can filter by status (``filter_active_only``).
Snapshots include a ``status`` column in extended format.

**Closed BC implementation detail:**  Par2_Core uses strict inequality
:math:`lo < x < hi` with an epsilon buffer :math:`\varepsilon = L \times
10^{-14}`.  This is a deliberate deviation from legacy (which uses exact
strict inequality) for floating-point robustness.

**Periodic BC implementation:**

* **Fast path:** assumes :math:`|\Delta p| < L`; wraps by adding or
  subtracting :math:`L` once.
* **Robust fallback:** floor-based algorithm handles multi-domain crossings.
* Wrap domain is :math:`[lo, hi)` — position landing exactly on :math:`hi`
  is snapped to :math:`lo`.
* Periodic boundary must be **symmetric** (both lo and hi set to Periodic);
  mixed periodic is treated as non-periodic for that axis.
