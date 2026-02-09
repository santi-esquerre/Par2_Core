Performance Tips
================

A short checklist of things that matter for throughput and latency
when running Par2\_Core in an HPC pipeline.

.. contents::
   :local:
   :depth: 2

----

1. ``prepare()`` once, ``step()`` many
--------------------------------------

``prepare()`` allocates RNG states, tracking arrays, and internal
workspace. ``step()`` performs **zero** device-memory allocations.

.. code-block:: cpp

   engine.prepare();              // one-time cost

   for (int t = 0; t < 10000; ++t)
       engine.step(dt);           // allocation-free, every time

If particle count grows (new ``bind_particles`` with larger N),
call ``prepare()`` again — the workspace grows but never shrinks.

----

2. Reuse engine and workspace
------------------------------

Creating a new ``TransportEngine`` re-allocates everything. If you
need to reset particles, call ``bind_particles`` + ``prepare()``
on the existing engine rather than constructing a new one.

----

3. No allocs, no syncs in ``step()``
-------------------------------------

This is by design:

* ``step()`` launches **one** kernel (``move_particles_kernel_full``)
  and returns.
* No ``cudaMalloc``, no ``cudaMemcpy``, no ``cudaDeviceSynchronize``.
* Precondition: ``is_prepared() == true`` and
  ``needs_corner_update() == false``.

If you violate preconditions in a debug build, the engine asserts.
In release builds, behavior is undefined.

----

4. Streams and events beat global sync
----------------------------------------

.. code-block:: cpp

   // BAD — stalls all GPU work on all streams
   cudaDeviceSynchronize();

   // GOOD — stalls only the transport stream
   engine.synchronize();

   // BEST — GPU-side ordering, CPU never blocks
   engine.record_event(done_evt);
   cudaStreamWaitEvent(other_stream, done_evt, 0);

Use ``cudaEventDisableTiming`` when creating ordering-only events
to avoid the timing hardware overhead.

----

5. Know the hotspots
--------------------

The RWPT kernel is **memory-bound** (velocity reads dominate compute).
The three most expensive operations per particle per step:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Hotspot
     - Notes
   * - Velocity interpolation
     - 2 reads (Linear) or 8 reads (Trilinear) per component.
       Linear is cheaper in bandwidth.
   * - Drift correction
     - ``TrilinearOnFly``: 24 velocity loads + 48 D-tensor evaluations
       per particle. ``Precomputed``: one cell lookup (cheapest, but
       requires pre-step kernel + 9 temporary arrays).
       ``None``: free (but only valid when dispersion is spatially
       constant).
   * - RNG
     - Two ``curand_normal_double`` calls per particle per step
       (3 in 3-D). XORWOW state is 48 bytes — fits in registers.

If your velocity field is steady, ``update_derived_fields()`` needs
to run only **once** (after the initial ``bind_velocity``).

----

6. CUDA tips
------------

``cudaMallocAsync`` for workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Par2\_Core uses ``cudaMallocAsync`` (CUDA 11.2+) internally for
workspace growth in ``prepare()``. This avoids the implicit
synchronization of ``cudaMalloc`` and works well with
CUDA memory pools.

Avoid unnecessary atomics
~~~~~~~~~~~~~~~~~~~~~~~~~

The move kernel is fully independent per particle — no atomics in the
hot path. The only atomics are in ``StatsComputer`` reductions
(global atomicAdd). If you need stats every step, consider reducing
only every N steps.

Prefer SoA
~~~~~~~~~~~

Particle data is already SoA (``x[]``, ``y[]``, ``z[]``). If your
downstream kernels consume particle positions, read them in SoA layout
for coalesced access. Do not transpose to AoS on the GPU.

Pinned host memory for snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``cudaMallocHost`` for host buffers that receive
``cudaMemcpyAsync`` downloads:

.. code-block:: cpp

   double* h_x;
   cudaMallocHost(&h_x, N * sizeof(double));

   // Async download — does not block the GPU
   cudaMemcpyAsync(h_x, d_x, N * sizeof(double),
                   cudaMemcpyDeviceToHost, transport_stream);
   cudaStreamSynchronize(transport_stream);
   // Now h_x is safe to read on CPU

Pageable memory forces a synchronous path, killing overlap.

Block size
~~~~~~~~~~

Default: 256 threads/block (``EngineConfig::kernel_block_size``).
This is generally optimal for occupancy on CC 7.5+. Tune only if
profiling shows low occupancy on your specific GPU.

----

7. Debug vs release
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Control
     - Debug
     - Release / HPC
   * - ``EngineConfig::debug_checks``
     - ``true``
     - ``false``
   * - ``EngineConfig::nan_prevention``
     - ``true``
     - ``false`` (if velocity field is known-safe)
   * - ``PAR2_DEBUG_LEVEL`` (compile-time)
     - 1
     - 0
   * - NVCC flags
     - ``-G -lineinfo``
     - ``-O3 --use_fast_math``

For maximum throughput, compile with ``CMAKE_BUILD_TYPE=Release`` and
set both runtime flags to ``false``:

.. code-block:: cpp

   par2::EngineConfig config;
   config.debug_checks   = false;
   config.nan_prevention = false;

``nan_prevention`` adds a per-particle branch in the kernel. It is
safe to disable when your velocity field has no zero-velocity cells
in dispersive regions.
