Semantic Map — Par2_Core Contracts
====================================

.. note::

   This document records the **real contracts** extracted from the source code
   of Par2_Core v0.1.0.  It is the single source of truth for API
   documentation.  Items marked **TODO/UNKNOWN** require further investigation
   or future design decisions.

.. contents:: Contents
   :local:
   :depth: 2


1  Public API Tree
------------------

Each header in ``include/par2_core/`` and its responsibilities:

+----------------------------+-------------------------------------------------------------+
| Header                     | Responsibilities                                            |
+============================+=============================================================+
| ``par2_core.hpp``          | Umbrella header — includes all public API.                  |
+----------------------------+-------------------------------------------------------------+
| ``transport_engine.hpp``   | ``TransportEngine<T>`` class: lifecycle, binding, stepping, |
|                            | stream/event management, injection, unwrapping.             |
+----------------------------+-------------------------------------------------------------+
| ``types.hpp``              | Value types: ``TransportParams``, ``EngineConfig``,         |
|                            | ``BoundaryType``, ``InterpolationMode``,                    |
|                            | ``DriftCorrectionMode``, ``ParticleStatus``, ``StepStats``. |
+----------------------------+-------------------------------------------------------------+
| ``grid.hpp``               | ``GridDesc<T>``: lightweight POD for uniform Cartesian      |
|                            | grid geometry.  Factory helpers ``make_grid``,              |
|                            | ``make_uniform_grid``.                                      |
+----------------------------+-------------------------------------------------------------+
| ``views.hpp``              | Non-owning GPU memory views: ``ParticlesView``,             |
|                            | ``ConstParticlesView``, ``UnwrappedPositionsView``,         |
|                            | ``DriftCorrectionView``, ``DeviceSpan``.                    |
+----------------------------+-------------------------------------------------------------+
| ``velocity_layout.hpp``    | Velocity field layout contracts: ``FaceFieldView`` (MAC     |
|                            | staggered), ``CornerFieldView``.  Aliases                   |
|                            | ``VelocityView = FaceFieldView``,                           |
|                            | ``CornerVelocityView = CornerFieldView``.                   |
|                            | Utility: ``merge_id()``, ``field_size()``.                  |
+----------------------------+-------------------------------------------------------------+
| ``boundary.hpp``           | ``AxisBoundary<T>``, ``BoundaryConfig<T>``: per-axis,       |
|                            | per-face boundary specification with factory helpers.       |
+----------------------------+-------------------------------------------------------------+
| ``injectors.hpp``          | Standalone GPU injection functions: ``inject_box``,         |
|                            | ``inject_grid``.                                            |
+----------------------------+-------------------------------------------------------------+
| ``stats.hpp``              | ``StatsComputer<T>`` (async, no hidden alloc),              |
|                            | ``Moments3``, ``ParticleCounts``, ``StatsResult``.          |
|                            | Legacy helpers: ``concentration_box``,                      |
|                            | ``concentration_past_plane``, ``count_by_status``.          |
+----------------------------+-------------------------------------------------------------+
| ``io.hpp``                 | Host-side I/O: ``download_positions_async``,                |
|                            | ``write_particles_csv``, ``CsvSnapshotWriter``,             |
|                            | ``PinnedParticlesBuffer``, ``write_legacy_csv_snapshot``.   |
+----------------------------+-------------------------------------------------------------+
| ``debug_policy.hpp``       | Compile-time debug macros (``PAR2_DEBUG_LEVEL`` 0/1/2),     |
|                            | derived feature flags, device-side assertion helpers.       |
+----------------------------+-------------------------------------------------------------+
| ``version.hpp``            | Version constants (``0.1.0``).                              |
+----------------------------+-------------------------------------------------------------+


2  Engine Lifecycle (State Machine)
-----------------------------------

.. code-block:: text

   ┌──────────────┐
   │  Constructed  │  (grid, params, bc, config stored; stream = default)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐     bind_velocity(vel)
   │ Velocity     │◄────────────────────────
   │ Bound        │     sets corner_dirty=true, drift_dirty=true
   └──────┬───────┘
          │  bind_particles(part)
          ▼  sets prepared=false
   ┌──────────────┐
   │ Particles    │
   │ Bound        │
   └──────┬───────┘
          │  prepare(stream)
          │   ├─ init_rng_states()
          │   ├─ ensure_tracking_arrays()
          │   └─ update_derived_fields() if needed
          ▼
   ┌──────────────┐
   │  Prepared     │  prepared=true
   └──────┬───────┘
          │
          │  step(dt) / advance(dt, n)
          │   └─ Asserts: has_velocity, has_particles, prepared
          │      Asserts: NOT corner_dirty, NOT drift_dirty
          │      Launches move_particles kernel
          │      NO alloc, NO sync
          ▼
   ┌──────────────┐
   │  Stepping     │  (can loop N times without leaving this state)
   └──────┬───────┘
          │
          │  synchronize()  ←  the ONLY sync point exposed
          ▼
   ┌──────────────┐
   │  Synced      │  particle data readable on CPU after D2H copy
   └──────────────┘

**Re-binding rules:**

* ``bind_velocity()`` → marks ``corner_dirty = true``, ``drift_dirty = true``.
  Must call ``update_derived_fields()`` before next ``step()`` if using
  trilinear interpolation or on-the-fly drift.
* ``bind_particles()`` → resets ``prepared = false``.
  Must call ``prepare()`` before next ``step()``.
* ``bind_corner_velocity()`` → sets ``corner_external = true``,
  ``corner_dirty = false``.  Overrides internal computation.
* ``bind_drift_correction()`` → sets ``drift_external = true``,
  ``drift_dirty = false``.  Overrides internal computation.
* ``set_stream()`` → can be called at any time; does NOT require re-prepare.

**What happens if you violate the lifecycle:**

* ``step()`` without ``prepare()`` → **assertion failure** (debug) or
  undefined behavior (release).
* ``step()`` with ``corner_dirty`` or ``drift_dirty`` still ``true`` →
  **assertion failure** (kernel will use stale/uninitialized data).


3  Memory Contracts (GPU-Native)
---------------------------------

3.1  Ownership Rules
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Resource
     - Owner
     - Lifetime requirement
   * - Velocity arrays (U, V, W)
     - **User**
     - Valid from ``bind_velocity`` until next ``bind_velocity``
       or engine destruction.
   * - Particle arrays (x, y, z)
     - **User**
     - Valid from ``bind_particles`` until next ``bind_particles``
       or engine destruction.
   * - Status / wrap arrays
     - **Engine** if auto-allocated; **User** if provided in view.
     - Created in ``prepare()`` via ``ensure_tracking_arrays()``.
       Engine-owned arrays freed on destruction.
   * - Corner velocity (Uc, Vc, Wc)
     - **Engine** (internal) or **User** via ``bind_corner_velocity``.
     - Engine-owned: lives in workspace. User-owned: same lifetime
       rules as velocity.
   * - RNG states
     - **Engine** (workspace).
     - Allocated in ``prepare()``, freed on destruction.
   * - Drift correction
     - **Engine** (workspace) or **User** via ``bind_drift_correction``.
     - Same as corner velocity.
   * - CUDA stream
     - **User**.
     - Engine does NOT own or create/destroy the stream.

3.2  Device Pointer Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Velocity fields** (both Face and Corner):

* Size: ``(nx+1) × (ny+1) × (nz+1)`` elements per component
* Indexing: ``index = iz × (ny+1) × (nx+1) + iy × (nx+1) + ix``
  (legacy ``mergeId`` convention)
* Three separate arrays: U (x-faces), V (y-faces), W (z-faces)

**Particle arrays** (SoA):

* Three separate arrays: ``x[n]``, ``y[n]``, ``z[n]``
* Optional: ``status[n]`` (``uint8_t``), ``wrapX[n]``, ``wrapY[n]``,
  ``wrapZ[n]`` (``int32_t``)
* Coalesced access in kernels (SoA optimal for GPU)

3.3  Workspace Growth Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workspace buffers use a **25% growth factor + 64 minimum**:
``new_cap = (n * 5 / 4) + 64``.  Buffers are only freed and re-allocated
when the requested count exceeds current capacity.  This ensures
``step()`` is allocation-free once ``prepare()`` has been called.

When ``cudaMallocAsync`` is available (CUDA ≥ 11.2), allocations are
stream-ordered.  Otherwise, they fall back to ``cudaMalloc``.


4  Stream and Event Semantics
------------------------------

4.1  Stream Usage
~~~~~~~~~~~~~~~~~

* The engine operates on **a single user-provided CUDA stream** (set via
  ``set_stream()``; defaults to the default stream / ``nullptr``).
* All kernel launches (``step``, ``inject_box``, ``update_derived_fields``,
  ``compute_unwrapped_positions``) are **enqueued** on this stream.
* The engine **never** calls ``cudaDeviceSynchronize()``.  The only
  synchronization it performs is through ``synchronize()``
  (= ``cudaStreamSynchronize(stream_)``).

4.2  Event Integration
~~~~~~~~~~~~~~~~~~~~~~

For multi-solver pipelines:

* ``record_event(event)`` — records a CUDA event on the engine stream.
  Other streams can ``cudaStreamWaitEvent`` on it.
* ``wait_event(event)`` — makes the engine stream wait for an external
  event.  **GPU-side wait**, does not block the CPU host.

4.3  Execution Order Guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within the engine stream, all operations execute in enqueue order.  A
typical pipeline cycle:

.. code-block:: text

   wait_event(vel_ready)     ← GPU waits for flow solver
   step(dt)                  ← kernel enqueued
   record_event(step_done)   ← signal for downstream

No CPU blocking occurs in this sequence.


5  Boundary Conditions
-----------------------

5.1  Per-Axis, Per-Face Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BoundaryConfig<T>`` defines six faces via three ``AxisBoundary<T>``
structs (``x.lo``, ``x.hi``, ``y.lo``, ``y.hi``, ``z.lo``, ``z.hi``).

5.2  Closed (Reflective)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Semantics:** if the proposed new position falls outside the domain
(using **strict inequalities**), the displacement is **zeroed** and the
particle stays at its current position.

* Domain validity: ``lo < x < hi`` (strict, matching legacy
  ``grid::validX/Y/Z``).
* An epsilon buffer ``ε = L × 1e-14`` is applied internally for
  floating-point robustness (**deviation from legacy**, which uses exact
  strict inequality without epsilon).  This prevents edge-case particle
  freezing at positions infinitesimally close to the boundary.
* This is a *rejection* scheme, not a bounce/reflection scheme:
  the particle simply does not move on that step.

5.3  Periodic (Wrap-Around)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Semantics:** the position wraps modulo the domain length ``L = hi - lo``.

* Wrap counter (``int32_t``) tracks **net domain crossings** (can be
  negative).
* Fast path assumes ``|dp| < L``; falls back to robust floor-based
  algorithm for multi-domain crossings.
* Wrapped domain is ``[lo, hi)`` — position is snapped to ``lo`` if it
  lands exactly on ``hi``.
* Periodic BC must be symmetric (both ``lo`` and ``hi`` set to
  ``Periodic``); mixed periodic is treated as non-periodic for that axis.

5.4  Open (Exit)
~~~~~~~~~~~~~~~~

**Semantics:** if the proposed new position falls outside the domain, the
particle is flagged as ``ParticleStatus::Exited`` and **retains its current
position** (it does not advance to the out-of-bounds location).

* Requires ``status`` array to be bound (auto-allocated by
  ``ensure_tracking_arrays()`` if boundary ``has_open()``).
* Exited particles are skipped in subsequent ``step()`` calls (the
  kernel checks ``status[i] != Active`` before moving).

5.5  2D Mode
~~~~~~~~~~~~~

When ``nz == 1``, the Z displacement is forced to zero and the Z boundary
is skipped.  This is handled inside the kernel, not by the boundary
configuration.


6  RNG and Determinism
-----------------------

6.1  Initialization
~~~~~~~~~~~~~~~~~~~~

* Each particle gets its own ``curandState_t`` (XORWOW generator).
* Initialized by ``curand_init(seed, tid, 0, &state)`` where ``seed``
  is ``EngineConfig::rng_seed`` (default ``12345ULL``) and ``tid`` is
  the particle index.
* Initialization runs as a kernel on the engine stream during
  ``prepare()``.

6.2  Per-Step Usage
~~~~~~~~~~~~~~~~~~~~

* Three independent :math:`N(0,1)` variates are drawn per particle per
  step via ``curand_normal_double()``.
* The state is loaded from global memory, used, then written back.
* State persistence: the curand state array persists in the workspace
  across steps — this is essential for correct statistical properties.

6.3  Reproducibility Guarantees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Same GPU + same seed + same particle count + same step sequence →
  identical results** (bitwise).
* Changing the number of particles changes the ``tid`` mapping → different
  streams of random numbers even for the same seed.
* Cross-GPU reproducibility is **NOT guaranteed** (different warp
  scheduling can affect curand internal state sequence).
* ``float`` vs ``double`` instantiations produce different results (expected).

.. note::

   **TODO:** PhiloxPolicy (counter-based RNG) is defined in internal headers
   but not yet wired into the public API.  This would enable
   cross-GPU reproducibility and skip-ahead capabilities.


7  Outputs, Stats, and Snapshots
---------------------------------

7.1  Position Data
~~~~~~~~~~~~~~~~~~~

* ``particles()`` returns a ``ConstParticlesView<T>`` with **device pointers**
  to wrapped positions (x, y, z), plus status and wrap counters if tracking
  is enabled.
* To read on host: ``synchronize()`` → ``cudaMemcpy`` (or use
  ``io::download_positions_async``).

7.2  Unwrapped Positions
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``compute_unwrapped_positions(out)`` computes on-demand:
  ``x_u = x + wrapX × Lx`` (for periodic axes).
* Non-periodic axes: ``x_u = x`` (identity copy).
* This is a **kernel launch** (async, no sync, no alloc).
* Output arrays must be user-allocated on device.

7.3  Statistics
~~~~~~~~~~~~~~~~

* ``StatsComputer<T>`` performs GPU reduction for moments (mean, variance,
  std) and status counts.
* ``compute_async()`` → async; ``fetch_result()`` → after sync.
* Variance uses unbiased estimator (N-1 denominator).
* ``use_unwrapped = true`` computes stats from unwrapped positions (essential
  for correct dispersion measurements under periodic BC).
* ``filter_active_only = true`` excludes Exited/Inactive particles from
  moment computation.

7.4  CSV Snapshots
~~~~~~~~~~~~~~~~~~~

**Legacy format** (``CsvSnapshotConfig::legacy_format = true``):

.. code-block:: text

   id,x coord,y coord,z coord
   0,1.234567890123456,2.345678901234567,3.456789012345678

* Precision: 15 decimal digits (full ``double``).
* No time, no status, no wrap columns.

**Extended format** — adds optional columns: ``t``, ``xu``, ``yu``, ``zu``,
``status``, ``wrapX``, ``wrapY``, ``wrapZ``.

7.5  Snapshot Timing
~~~~~~~~~~~~~~~~~~~~~

Snapshots reflect the state **after** the most recent ``step()`` that has
been synchronized.  Step 0 (initial state before any stepping) can be
exported after ``inject_*()`` + ``synchronize()`` if the user wants it.

.. note::

   **TODO:** There is no built-in snapshot-scheduling facility.  The user
   loop controls when to export (e.g., ``if step % interval == 0``).


8  Thread Safety
-----------------

* ``TransportEngine`` is **NOT thread-safe** for concurrent host calls.
  All ``bind_*``, ``step()``, ``synchronize()`` etc. must be serialized
  from the host side.
* Multiple engines on **different CUDA streams** can run concurrently
  (one host thread per engine or explicit serialization).
* ``StatsComputer`` and ``CsvSnapshotWriter`` are **NOT thread-safe**
  individually; use separate instances per thread.
* GPU kernels launched by different engines on different streams execute
  concurrently if GPU resources allow.


9  RWPT Algorithm (Mathematical Formulation)
---------------------------------------------

The Random Walk Particle Tracking (RWPT) method solves the
advection-dispersion equation (ADE) in Lagrangian form.  Each particle
represents a passive tracer whose position evolves per timestep as:

.. math::

   \Delta\mathbf{x} = (\mathbf{v}_{\text{interp}} + \mathbf{v}_{\text{drift}}) \, dt
                       + \mathbf{B} \cdot \boldsymbol{\xi}

where:

* :math:`\mathbf{v}_{\text{interp}}` — velocity at the particle position,
  sampled via linear face-field interpolation or trilinear corner-field
  interpolation (see Section 10).
* :math:`\mathbf{v}_{\text{drift}} = \nabla \cdot \mathbf{D}` — drift
  correction ensuring correct Fokker–Planck behaviour
  (see Section 12).
* :math:`\mathbf{B}` — displacement matrix satisfying
  :math:`\mathbf{B}\mathbf{B}^T = 2\,\mathbf{D}\,dt`
  (see Section 11).
* :math:`\boldsymbol{\xi} = (\xi_0, \xi_1, \xi_2)` — three independent
  :math:`N(0,1)` random variates from ``curand_normal_double()``.
* :math:`dt` — time step.

**Per-particle kernel pseudocode:**

.. code-block:: text

   for each active particle i:
     1. (idx, idy, idz) = cell_from_position(p[i])
     2. v = sample_velocity(interp_mode, p[i])
     3. v_drift = compute_drift(drift_mode, p[i])
     4. v_B = sample_corner_velocity(p[i])     ← for B computation
     5. B = displacement_matrix(v_B, Dm, αL, αT, dt)
     6. ξ = (N(0,1), N(0,1), N(0,1))
     7. Δp = (v + v_drift)·dt + B·ξ
     8. p[i] = apply_boundary(p[i], Δp)

**Important detail:** Step 4 always prefers **corner velocity** for
computing B, even under ``InterpolationMode::Linear``.  If corner velocity
is available, it is re-sampled at the particle position for B; if not, the
face-centered velocity serves as fallback.  This matches legacy PAR²
behaviour where dispersion was always based on corner-field values.


10  Velocity Layout and Interpolation
---------------------------------------

10.1  Staggered MAC Grid (FaceField)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Par2_Core inherits the legacy PAR² staggered MAC grid layout.  Each velocity
component (U, V, W) is stored as a flat array with size:

.. math::

   N_{\text{field}} = (n_x + 1)(n_y + 1)(n_z + 1)

Indexing:

.. math::

   \mathrm{mergeId}(i_x, i_y, i_z)
     = i_z \cdot (n_y+1)(n_x+1) + i_y \cdot (n_x+1) + i_x

**Semantic meaning:** despite using corner-point indexing, these are
**face-centred** values:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Component
     - Physical Location
     - Cell access
   * - ``U``
     - X-faces (perpendicular to X, in YZ plane)
     - left: ``U[id(ix,iy,iz)]``, right: ``U[id(ix+1,...)]``
   * - ``V``
     - Y-faces (perpendicular to Y, in XZ plane)
     - front: ``V[id(ix,iy,iz)]``, back: ``V[id(...,iy+1,...)]``
   * - ``W``
     - Z-faces (perpendicular to Z, in XY plane)
     - bottom: ``W[id(ix,...)]``, top: ``W[id(...,iz+1)]``

10.2  Linear Interpolation (InterpolationMode::Linear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source: ``internal/fields/facefield_accessor.cuh``

For a particle inside cell ``(idx, idy, idz)`` with cell centre
``(cx, cy, cz)``:

.. math::

   t_x &= \frac{p_x - c_x}{\Delta x} + 0.5 \quad \in [0, 1] \\
   v_x &= (1 - t_x) \cdot U[\text{left}] + t_x \cdot U[\text{right}]

Similarly for :math:`v_y` and :math:`v_z`.  Each component is interpolated
independently on its own staggered axis.

10.3  Trilinear Interpolation (InterpolationMode::Trilinear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source: ``internal/fields/cornerfield_accessor.cuh``

Uses the 8 **corner-centred** velocity values of the containing cell.
Normalized coordinates:

.. math::

   t = \frac{\mathbf{p} - \mathbf{c}}{\Delta \mathbf{h}} + 0.5
     \quad \in [0, 1]^3

Standard trilinear formula:  interpolate along X for all 4 YZ pairs →
interpolate along Y for the 2 Z pairs → interpolate along Z.

10.4  Corner Velocity Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source: ``kernels/cornerfield.cu``

Corner velocity is computed from face velocity by averaging adjacent faces:

.. math::

   U_c(i_x, i_y, i_z) = \frac{1}{|\mathcal{S}|}
     \sum_{(j_y, j_z) \in \mathcal{S}}
       U_{\text{face}}(i_x^{\prime}, j_y, j_z)

where :math:`\mathcal{S}` is the 2×2 stencil of valid adjacent cells in
the YZ plane, and :math:`i_x^{\prime}` selects the left or right face
(``XM`` if ``ix < nx``, ``XP`` if ``ix == nx``).  Boundary corners have
fewer contributing faces (1 or 2 instead of 4).


11  Dispersion Tensor and Displacement Matrix
-----------------------------------------------

11.1  Dispersion Tensor D
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   D_{ij} = (\alpha_T |\mathbf{v}| + D_m)\,\delta_{ij}
            + \frac{\alpha_L - \alpha_T}{|\mathbf{v}|}\,v_i\,v_j

Eigenvalues:

.. math::

   \lambda_L &= \alpha_L |\mathbf{v}| + D_m
     \qquad \text{(longitudinal, along } \mathbf{v}\text{)} \\
   \lambda_T &= \alpha_T |\mathbf{v}| + D_m
     \qquad \text{(transverse, 2× degenerate, } \perp \mathbf{v}\text{)}

11.2  Displacement Matrix B
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\mathbf{B}` is symmetric and satisfies
:math:`\mathbf{B}\mathbf{B}^T = 2\,\mathbf{D}\,dt`.
It is constructed via eigendecomposition:

.. math::

   \mathbf{B} = \sqrt{2\,dt}
     \sum_{i=0}^{2} \gamma_i \;(\mathbf{e}_i \otimes \mathbf{e}_i)

where the unnormalized eigenvectors are:

.. math::

   \mathbf{e}_0 &= (v_x, \; v_y, \; v_z)
     \qquad \text{(along flow)} \\
   \mathbf{e}_1 &= (-v_y, \; v_x, \; 0)
     \qquad \text{(⊥ in XY plane)} \\
   \mathbf{e}_2 &= (-v_z v_x, \; -v_z v_y, \; v_x^2 + v_y^2)
     \qquad \text{(⊥ to both)}

and the gamma coefficients incorporate the eigenvalue and normalization:

.. math::

   \gamma_0 &= \frac{\sqrt{\lambda_L}}{|\mathbf{e}_0|^2}
     = \frac{\sqrt{\alpha_L |\mathbf{v}| + D_m}}{|\mathbf{v}|^2} \\
   \gamma_1 &= \frac{\sqrt{\lambda_T}}{|\mathbf{e}_1|^2}
     = \frac{\sqrt{\alpha_T |\mathbf{v}| + D_m}}{v_x^2 + v_y^2} \\
   \gamma_2 &= \frac{\sqrt{\lambda_T}}{|\mathbf{e}_2|^2}
     = \frac{\sqrt{\alpha_T |\mathbf{v}| + D_m}}{(v_x^2 + v_y^2)(v_x^2 + v_y^2 + v_z^2)}

The 6 unique components of symmetric B:

.. math::

   B_{jk} = \sqrt{2\,dt} \sum_{i=0}^{2} \gamma_i \; e_i[j] \; e_i[k]

11.3  Zero‑Velocity Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two mechanisms prevent NaN from :math:`|\mathbf{v}| \to 0`:

**Mechanism 1 — Displacement matrix** (legacy line 432):

.. math::

   \text{toll} = 0.01 \cdot D_m / \alpha_L, \qquad
   v_x \leftarrow \max(v_x, \text{toll})

Only :math:`v_x` is clamped.  This ensures :math:`|\mathbf{v}| > 0` and
:math:`|\mathbf{e}_1|^2, |\mathbf{e}_2|^2 > 0`.

**Mechanism 2 — Drift correction** (legacy line 278):

.. math::

   \text{if } v_x < \text{toll} \;\wedge\; v_y < \text{toll}
              \;\wedge\; v_z < \text{toll}:
   \quad v_x \leftarrow \text{toll}

Applied at each of the 8 corners independently.

**Legacy bug fix:**  When :math:`\alpha_L = 0`, legacy computes
``toll = NaN``.  Par2_Core returns ``toll = 1e-15`` instead, which produces
correct zero drift without NaN propagation.

11.4  Special‑Case Dispatching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source: ``internal/math/dispersion.cuh``, ``compute_B_matrix()``

+----------------------------+------------------------------------------+
| Condition                  | Behaviour                                |
+============================+==========================================+
| Dm = αL = αT = 0           | B = 0 (zero matrix, no diffusion)        |
+----------------------------+------------------------------------------+
| αL = αT = 0, Dm > 0        | B = √(2·Dm·dt) · I (isotropic)           |
+----------------------------+------------------------------------------+
| otherwise                  | Full eigendecomposition (legacy path)    |
+----------------------------+------------------------------------------+

11.5  ``nan_prevention`` Flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``EngineConfig::nan_prevention`` adds guards **not present in legacy**:

* In the B-matrix: clamps :math:`|\mathbf{e}_1|^2` and
  :math:`|\mathbf{e}_2|^2` to :math:`\text{toll}^2` if they approach zero
  (e.g., flow purely in Z).
* In precomputed drift: when :math:`|\mathbf{v}| < \text{toll}` at a cell
  centre, sets :math:`\mathbf{D} = D_m \mathbf{I}` directly.


12  Drift Correction Algorithms
---------------------------------

12.1  TrilinearOnFly Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source: ``internal/fields/cornerfield_accessor.cuh``,
``compute_drift_trilinear()``

Computes :math:`\nabla \cdot \mathbf{D}` at the **particle position**
using trilinear derivative interpolation of D-tensor components evaluated
at the 8 cell corners:

.. math::

   v_{\text{drift},x} &= \frac{\partial D_{xx}}{\partial x}
                          + \frac{\partial D_{xy}}{\partial y}
                          + \frac{\partial D_{xz}}{\partial z} \\
   v_{\text{drift},y} &= \frac{\partial D_{xy}}{\partial x}
                          + \frac{\partial D_{yy}}{\partial y}
                          + \frac{\partial D_{yz}}{\partial z} \\
   v_{\text{drift},z} &= \frac{\partial D_{xz}}{\partial x}
                          + \frac{\partial D_{yz}}{\partial y}
                          + \frac{\partial D_{zz}}{\partial z}

The partial derivatives use trilinear derivative interpolation functions
(``trilinear_dx/dy/dz``) which compute :math:`(v_1 - v_0)/h` on the
differentiation axis, then interpolate in the remaining two dimensions.

**Computational cost:**  8 corner lookups × 3 components = 24 velocity
loads, 6 D-tensor evaluations × 8 corners = 48 D computations, plus
9 trilinear derivative evaluations.

12.2  Precomputed Mode (Finite Differences)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source: ``kernels/drift_correction.cu``

Two-step process:

1. **Step 1 — D tensor at cell centres:**  For each cell, sample
   face-field velocity at the cell centre, compute :math:`|\mathbf{v}|`,
   and evaluate the 6 unique D-tensor components.

2. **Step 2 — Finite differences:**

   .. math::

      \frac{\partial D_{ij}}{\partial x}\bigg|_k =
      \begin{cases}
        (D_{ij}[k+1] - D_{ij}[k]) / \Delta x
          & k = 0 \text{ (one-sided)} \\
        (D_{ij}[k+1] - D_{ij}[k-1]) / (2\Delta x)
          & 0 < k < n_x-1 \text{ (central)} \\
        (D_{ij}[k] - D_{ij}[k-1]) / \Delta x
          & k = n_x-1 \text{ (one-sided)}
      \end{cases}

   The result is stored per cell and looked up in the kernel as a
   piecewise-constant field (no interpolation within the cell).

**Trade-off:**  Precomputed mode is cheaper per particle per step
(one cell-index lookup vs. 48 D computations) but requires an extra
compute pass and 9 temporary arrays (6 D + 3 drift) of ``num_cells``
elements each.  Best for large particle counts with slowly-changing
velocity.


13  Performance Contracts
---------------------------

13.1  Allocation‑Free Stepping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``prepare()`` has been called, ``step()`` performs **zero**
device-memory allocations.  All GPU buffers (RNG states, corner velocity,
drift, status, wrap counters) are pre-allocated with a 25% growth margin
and only re-allocated if the particle count exceeds the workspace capacity.

This is critical for real-time coupling and deterministic performance.

13.2  Kernel Launch Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All kernels use ``EngineConfig::kernel_block_size`` (default 256).
Grid size: ``(n + block_size - 1) / block_size``.  Grid-stride loops
ensure correctness for arbitrary ``n``.

13.3  Memory Access Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Particle data:** coalesced SoA reads/writes (separate x, y, z arrays).
* **RNG state:** per-particle, 48 bytes per ``curandState_t`` —
  coalesced by thread index.
* **Velocity fields:** indirect lookups via cell index — **not** coalesced
  for randomly distributed particles.
* **Zero shared memory** used in the move kernel (each thread is
  independent).

13.4  Known Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``float`` instantiations still generate double-precision random
  numbers (``curand_normal_double``), wasting FP64 throughput.
* PhiloxPolicy is defined but not wired — XORWOW is the only option.
* The legacy kernel (``move_particles_kernel``, without ``_full``) is
  dead code still compiled into the library.
* Moments reduction (``moments.cuh``) uses global atomics — may become a
  bottleneck above ~1024 thread blocks.
