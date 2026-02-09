Performance Notes
=================

Kernel Block Size
-----------------

Default: 256 threads/block (``EngineConfig::kernel_block_size``).
This is generally optimal for occupancy on compute capability ≥ 7.5.

Arithmetic Intensity
--------------------

The RWPT kernel is **memory-bound** (velocity reads dominate).
Key optimisations:

* SoA particle layout → coalesced reads/writes
* Corner velocity reuse across particles in the same cell
* ``--use_fast_math`` for transcendentals (sqrt, sin, cos)

Debug vs Release
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 30

   * - Flag
     - Debug
     - Release / HPC
   * - ``EngineConfig::debug_checks``
     - ``true``
     - ``false``
   * - ``EngineConfig::nan_prevention``
     - ``true``
     - ``false`` (safe fields)
   * - ``PAR2_DEBUG_LEVEL`` (compile)
     - 1
     - 0
   * - NVCC flags
     - ``-G -lineinfo``
     - ``-O3 --use_fast_math``

Allocation-Free Stepping
------------------------

Once ``prepare()`` has been called, ``step()`` performs **zero**
device-memory allocations.  This is critical for deterministic
performance in real-time coupling scenarios.

All kernels use grid-stride loops::

   for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
        tid < n;
        tid += blockDim.x * gridDim.x)

This handles arbitrary particle counts with a single kernel launch.

Known Limitations
-----------------

* ``float`` instantiations still use ``curand_normal_double()``
  (FP64 throughput waste).
* The legacy kernel (``move_particles_kernel``, without ``_full``) is
  dead code still compiled into the library.
* Moments reduction (``moments.cuh``) uses global atomics — may become
  a bottleneck above ~1024 thread blocks.
* Zero shared memory used in the move kernel — each thread is fully
  independent, which limits optimisation opportunities for cell-local
  velocity caching.
