Workspace Allocations
=====================

``prepare()`` vs ``step()``
---------------------------

Par2_Core follows a strict two-phase pattern:

1. **``prepare()``** — called once after binding velocity and particles.
   Allocates all internal workspace (RNG states, tracking arrays, corner
   fields).

2. **``step(dt)``** — called per timestep in the hot loop.  **Never
   allocates.**  Pure kernel launches on the configured stream.

This separation makes ``step()`` safe for real-time and latency-sensitive
pipelines.

What ``prepare()`` Allocates
----------------------------

* RNG states (cuRAND, one per particle)
* Status array (if Open BC, not user-provided)
* Wrap counters (if Periodic BC, not user-provided)
* Corner velocity field (if Trilinear mode, not externally bound)

Growth Policy
-------------

Workspace buffers use a **25% growth factor + 64 minimum**::

   new_capacity = (n * 5 / 4) + 64

Buffers are only freed and re-allocated when the requested count exceeds
current capacity.  This ensures ``step()`` is allocation-free once
``prepare()`` has completed.

``cudaMallocAsync`` (CUDA ≥ 11.2)
---------------------------------

When ``cudaMallocAsync`` is available (CUDA ≥ 11.2), all workspace
allocations are **stream-ordered**:

* Allocations enqueued on the stream passed to ``prepare()``.
* Frees are stream-ordered on destruction.
* This avoids global ``cudaMalloc`` locks and enables better memory reuse.

On older CUDA toolkits, workspace falls back to blocking ``cudaMalloc``.
