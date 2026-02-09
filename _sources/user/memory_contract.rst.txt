Memory Contract
===============

This is the **most important page** in the user guide. Every device
pointer, every lifetime rule, every "don't touch that" is listed here.
Read this before writing a single ``cudaMalloc``.

.. contents::
   :local:
   :depth: 2

----

Core principle: views are non-owning
--------------------------------------

All ``bind_*`` methods accept **view structs** (``VelocityView``,
``ParticlesView``, etc.). These are plain-old-data structs holding
device pointers and a size — they **do not own memory**.

.. code-block:: cpp

   // VelocityView (alias for FaceFieldView)
   struct VelocityView<T> {
       const T* U;     // device ptr to X-velocities
       const T* V;     // device ptr to Y-velocities
       const T* W;     // device ptr to Z-velocities
       size_t   size;  // (nx+1)*(ny+1)*(nz+1)
   };

   // ParticlesView
   struct ParticlesView<T> {
       T*        x, *y, *z;    // device ptrs, writable
       int       n;            // particle count
       uint8_t*  status;       // optional (Open BC)
       int32_t*  wrapX, *wrapY, *wrapZ;  // optional (Periodic BC)
   };

**You** allocate the memory, **you** free it, **you** keep it alive.
The engine just reads/writes through these pointers.

----

Velocity fields
---------------

Layout
~~~~~~

Par2\_Core inherits a **staggered MAC grid** layout from legacy PAR².
Each velocity component (``U``, ``V``, ``W``) is a flat array of
``(nx+1) * (ny+1) * (nz+1)`` elements.

Despite this corner-sized allocation, the values represent
**face-centred** velocities:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Component
     - Physical location
     - Cell access for cell (cx, cy, cz)
   * - ``U``
     - X-face centres (in YZ planes)
     - left: ``U[id(cx,cy,cz)]``, right: ``U[id(cx+1,cy,cz)]``
   * - ``V``
     - Y-face centres (in XZ planes)
     - front: ``V[id(cx,cy,cz)]``, back: ``V[id(cx,cy+1,cz)]``
   * - ``W``
     - Z-face centres (in XY planes)
     - bottom: ``W[id(cx,cy,cz)]``, top: ``W[id(cx,cy,cz+1)]``

The linear index function is:

.. code-block:: cpp

   size_t id(int ix, int iy, int iz) {
       return iz * (ny+1) * (nx+1) + iy * (nx+1) + ix;
   }

Use ``par2::merge_id(ix, iy, iz, nx, ny)`` for the canonical version
or ``par2::field_size(grid)`` for the total array length.

Allocation
~~~~~~~~~~

.. code-block:: cpp

   size_t sz = grid.num_corners();   // = (nx+1)*(ny+1)*(nz+1)

   double *d_U, *d_V, *d_W;
   cudaMalloc(&d_U, sz * sizeof(double));
   cudaMalloc(&d_V, sz * sizeof(double));
   cudaMalloc(&d_W, sz * sizeof(double));

Lifetime
~~~~~~~~

Velocity buffers must remain valid **from the call to** ``bind_velocity``
**until** the next ``bind_velocity`` or engine destruction.

The engine reads U/V/W during ``step()`` and ``update_derived_fields()``.
You may write new values into the same buffers between steps (e.g., from
a flow solver kernel on the same stream) — just call
``update_derived_fields()`` if the engine uses Trilinear interpolation
or drift correction.

----

Particles
---------

Layout: Structure of Arrays (SoA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Particle positions are stored as three separate arrays for GPU
coalescing:

.. code-block:: text

   x[0] x[1] x[2] ... x[N-1]    ← contiguous in memory
   y[0] y[1] y[2] ... y[N-1]
   z[0] z[1] z[2] ... z[N-1]

The engine writes positions in-place during ``step()``.

Allocation
~~~~~~~~~~

.. code-block:: cpp

   int N = 100000;
   double *d_x, *d_y, *d_z;
   cudaMalloc(&d_x, N * sizeof(double));
   cudaMalloc(&d_y, N * sizeof(double));
   cudaMalloc(&d_z, N * sizeof(double));

   engine.bind_particles({d_x, d_y, d_z, N});

Lifetime
~~~~~~~~

Particle buffers must remain valid **from** ``bind_particles`` **until**
the next ``bind_particles`` or engine destruction.

After ``bind_particles``, you **must** call ``prepare()`` before
``step()``.

Status array (Open BC)
~~~~~~~~~~~~~~~~~~~~~~

When any axis uses ``BoundaryType::Open``, the engine needs a
``uint8_t`` status array to flag exited particles:

.. code-block:: cpp

   // Option A: let the engine auto-allocate (in prepare())
   engine.bind_particles({d_x, d_y, d_z, N});
   engine.prepare();  // allocates status internally

   // Option B: provide your own
   uint8_t* d_status;
   cudaMalloc(&d_status, N * sizeof(uint8_t));
   cudaMemset(d_status, 0, N);  // 0 = Active

   par2::ParticlesView<double> pv{d_x, d_y, d_z, N};
   pv.status = d_status;
   engine.bind_particles(pv);

Wrap counters (Periodic BC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When any axis uses ``BoundaryType::Periodic``, the engine tracks
net domain crossings in ``int32_t`` wrap counters:

.. code-block:: cpp

   // Auto-allocated in prepare() if nullptr
   // Or provide your own:
   int32_t *d_wrapX, *d_wrapY, *d_wrapZ;
   cudaMalloc(&d_wrapX, N * sizeof(int32_t));
   cudaMalloc(&d_wrapY, N * sizeof(int32_t));
   cudaMalloc(&d_wrapZ, N * sizeof(int32_t));
   cudaMemset(d_wrapX, 0, N * sizeof(int32_t));
   cudaMemset(d_wrapY, 0, N * sizeof(int32_t));
   cudaMemset(d_wrapZ, 0, N * sizeof(int32_t));

   par2::ParticlesView<double> pv{d_x, d_y, d_z, N};
   pv.wrapX = d_wrapX;
   pv.wrapY = d_wrapY;
   pv.wrapZ = d_wrapZ;
   engine.bind_particles(pv);

----

Unwrapped positions
-------------------

What they are
~~~~~~~~~~~~~

For periodic BC, particles wrap modulo the domain length. The
**unwrapped** position recovers the continuous trajectory:

.. math::

   x_u = x + \text{wrapX} \times L_x

where :math:`L_x = n_x \cdot \Delta x` is the domain length.

How to compute
~~~~~~~~~~~~~~

.. code-block:: cpp

   // User-allocated output buffers
   double *d_xu, *d_yu, *d_zu;
   cudaMalloc(&d_xu, N * sizeof(double));
   cudaMalloc(&d_yu, N * sizeof(double));
   cudaMalloc(&d_zu, N * sizeof(double));

   par2::UnwrappedPositionsView<double> out{d_xu, d_yu, d_zu, N};
   engine.compute_unwrapped_positions(out);
   engine.synchronize();

Cost
~~~~

* **One kernel launch** (N threads, trivial arithmetic).
* **Three extra arrays** of ``N * sizeof(T)`` on device.
* For axes without periodic BC, the kernel copies ``x_u = x`` directly.

This is **not free** — do not call it every step unless you need
continuous trajectories.  Compute on-demand (e.g., for statistics or
snapshots).

----

Engine-owned temporaries
------------------------

The engine allocates internal workspace in ``prepare()``:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Resource
     - Owner
     - When allocated
   * - RNG states (``curandState_t``)
     - Engine (workspace)
     - ``prepare()``
   * - Corner velocity (Uc, Vc, Wc)
     - Engine (workspace), unless user binds via ``bind_corner_velocity``
     - ``prepare()`` / ``update_derived_fields()``
   * - Drift correction field
     - Engine (workspace), unless user binds via ``bind_drift_correction``
     - ``update_derived_fields()`` (Precomputed mode only)
   * - Status / wrap arrays
     - Engine, if user did not provide them
     - ``prepare()`` via ``ensure_tracking_arrays()``
   * - CUDA stream
     - **User** (the engine never creates or destroys streams)
     - —

All engine-owned memory is freed when the engine is destroyed.

----

Prohibitions
------------

These rules apply **while the engine has pending GPU work** (between
``step()`` and the next ``synchronize()``):

.. warning::

   **Do NOT** access particle buffers from the CPU (``cudaMemcpy``,
   ``thrust::copy``, pointer dereference on pageable memory) without
   synchronizing the transport stream first.

.. warning::

   **Do NOT** call ``cudaFree`` on any bound buffer. The engine holds
   raw pointers — freeing underneath it is undefined behavior.

.. warning::

   **Do NOT** call ``cudaDeviceSynchronize()`` in the hot loop. Use
   ``engine.synchronize()`` or stream-specific synchronization instead.
   Global sync stalls **all** streams.

.. warning::

   **Do NOT** call ``cudaMemcpy`` (synchronous) in the hot loop. Use
   ``cudaMemcpyAsync`` with pinned host memory on the appropriate stream.

Summary: the hot loop should contain **only** kernel launches,
event records, and stream waits. Anything that blocks the CPU or
triggers implicit synchronization (pageable ``cudaMemcpy``, ``printf``
from device, ``cudaDeviceSynchronize``) must be moved outside or
guarded by explicit stream sync at known safe points.
