RNG & Reproducibility
=====================

Par2\_Core uses cuRAND [NVIDIA-cuRAND]_ for parallel random number generation, following
the approach described in [Rizzo2019]_.

Random Number Generation
------------------------

The engine uses cuRAND's **XORWOW** generator (``curandState_t``).
Each particle has its own persistent RNG state, initialised during
``prepare()`` via:

.. code-block:: cpp

   curand_init(seed, tid, 0, &states[tid]);

where ``seed`` is ``EngineConfig::rng_seed`` (default ``12345ULL``) and
``tid`` is the particle index.

Per step, three independent :math:`N(0,1)` variates are drawn per particle
using ``curand_normal_double()``.  The RNG state is loaded from global
memory, used, and written back — persistence is essential for correct
statistical properties.

.. note::

   A ``PhiloxPolicy`` (``curandStatePhilox4_32_10_t``) is defined in
   ``internal/rng/rng_policy.cuh`` but is **not yet wired** into the
   public API or kernels.  The current implementation hardcodes XORWOW.

Seeding
-------

The seed is set via ``EngineConfig::rng_seed``.  Given the same seed,
particle count, and grid, results are **deterministic**
(same binary, same GPU architecture).

.. warning::

   Results are NOT bit-reproducible across different GPU architectures or
   CUDA toolkit versions due to floating-point non-associativity in
   reductions and math functions.

Reproducibility Guarantees
--------------------------

+---------------------------------------------+----------+
| Scenario                                    | Bitwise? |
+=============================================+==========+
| Same GPU + same seed + same N + same steps  | **Yes**  |
+---------------------------------------------+----------+
| Same GPU + different N (same seed)          | No       |
+---------------------------------------------+----------+
| Different GPU architecture (same seed)      | No       |
+---------------------------------------------+----------+
| ``float`` vs ``double`` (same seed)         | No       |
+---------------------------------------------+----------+

Changing the particle count changes the ``tid`` ↔ sequence mapping,
producing different random streams even with the same seed.

**Known limitation:** ``TransportEngine<float>`` calls
``curand_normal_double()`` (FP64), then implicitly truncates to float.
This wastes FP64 throughput but is correct.

Correlation risks
-----------------

Each particle draws from an **independent** XORWOW sub-sequence
(different ``tid`` in ``curand_init``).  The XORWOW period is
:math:`2^{192} - 1`, far exceeding any practical particle count.

Potential pitfalls:

* **Re-seeding mid-run** (e.g., after ``bind_particles`` + ``prepare()``)
  resets all RNG states, possibly replaying the same sequence prefix.
  Avoid re-preparing unless the particle set genuinely changes.
* **Inter-particle correlations** are not a concern for XORWOW with
  distinct ``tid`` values, but could matter if a counter-based generator
  (e.g., Philox) is adopted in the future with shared counters.

Source: ``src/internal/rng/rng_policy.cuh``, ``src/kernels/move_particles.cu``.

See also [NVIDIA-BestPractices]_ for general GPU performance guidance.
