Pipeline Recipes
================

Par2\_Core is designed to slot into multi-solver GPU pipelines.
The engine **never** calls ``cudaDeviceSynchronize()``.
All kernel work is enqueued on the engine's CUDA stream and returns
immediately — the CPU is free to post work to other streams or do I/O.

This page provides two copy-pasteable integration patterns.

.. contents::
   :local:
   :depth: 2

----

Recipe A — Single stream (simple)
----------------------------------

Use this when there is **one** GPU solver feeding velocity to the engine
and no other solvers need to run concurrently.

.. code-block:: text

   Flow solver writes U/V/W  ──►  engine.step(dt)  ──►  read positions
                        ↕ all on the same stream ↕

Setup
~~~~~

.. code-block:: cpp

   // Engine on default stream (or your solver's stream)
   par2::TransportEngine<double> engine(grid, params, bc, config);
   engine.set_stream(my_stream);       // optional

   engine.bind_velocity({d_U, d_V, d_W, num_corners});
   engine.bind_particles({d_x, d_y, d_z, N});
   engine.inject_box(x0, y0, z0, x1, y1, z1);
   engine.prepare();                   // allocations happen HERE

Hot loop
~~~~~~~~

.. code-block:: cpp

   for (int t = 0; t < num_steps; ++t) {
       // Your solver writes velocity into d_U, d_V, d_W
       // (same stream — ordering is automatic)
       flow_solver_kernel<<<..., my_stream>>>(d_U, d_V, d_W, ...);

       // If using Trilinear or Precomputed drift: recompute derived fields
       if (engine.needs_corner_update() || engine.needs_drift_update())
           engine.update_derived_fields();

       // Step — async, zero-alloc, returns immediately
       engine.step(dt);

       // Another kernel can consume d_x, d_y, d_z on the same stream
       // (positions are updated in-place after step completes on GPU)
       reactive_kernel<<<..., my_stream>>>(d_x, d_y, d_z, N, ...);
   }

   // Sync ONLY when the host needs results
   engine.synchronize();

Key points
~~~~~~~~~~

* ``step()`` **does not synchronize**. It enqueues a kernel and returns.
* Everything on the same stream executes in FIFO order on the GPU — no
  events needed.
* Call ``engine.synchronize()`` (or ``cudaStreamSynchronize``)
  **only** when the CPU must read results (stats, snapshot, end of run).

----

Recipe B — Multi-stream with events
-------------------------------------

Use this when the flow solver and transport run on **separate** streams
so their GPU work can overlap.

.. code-block:: text

   stream_flow:       ┌─ update_velocity ──► record(vel_ready) ─────────────────┐
                      │                                                          │
   stream_transport:  │── wait(vel_ready) ──► update_derived ──► step(dt) ──►    │
                      │                          record(transport_done) ────┐    │
                      │                                                     │    │
   stream_reactive:   └────────── wait(transport_done) ──► react_kernel ────┘    │
                                                                                 │
                      └──────────────────── next iteration ──────────────────────┘

Setup
~~~~~

.. code-block:: cpp

   // Create streams
   cudaStream_t stream_flow, stream_transport;
   cudaStreamCreate(&stream_flow);
   cudaStreamCreate(&stream_transport);

   // Create events (disable timing for lower overhead)
   cudaEvent_t vel_ready, transport_done;
   cudaEventCreateWithFlags(&vel_ready,       cudaEventDisableTiming);
   cudaEventCreateWithFlags(&transport_done,  cudaEventDisableTiming);

   // Assign transport stream to engine
   engine.set_stream(stream_transport);
   engine.prepare();

Hot loop
~~~~~~~~

.. code-block:: cpp

   for (int t = 0; t < num_steps; ++t) {
       double time = t * dt;

       // (a) Flow solver updates velocity on stream_flow
       flow_kernel<<<..., stream_flow>>>(d_U, d_V, d_W, time, ...);
       cudaEventRecord(vel_ready, stream_flow);

       // (b) Transport waits for velocity, then steps
       engine.wait_event(vel_ready);

       if (engine.needs_corner_update() || engine.needs_drift_update())
           engine.update_derived_fields();

       engine.step(dt);
       engine.record_event(transport_done);

       // (c) Optional: another solver waits for transport
       // cudaStreamWaitEvent(stream_reactive, transport_done, 0);
   }

   engine.synchronize();

Key points
~~~~~~~~~~

* ``engine.wait_event(evt)`` is a GPU-side dependency — the **CPU never
  blocks**.
* ``engine.record_event(evt)`` marks completion on the transport stream
  so downstream streams can wait on it.
* The CPU posts all work and moves on immediately. Synchronize only for
  host-side reads.
* ``cudaEventCreateWithFlags(..., cudaEventDisableTiming)`` skips the
  timing circuitry → lower overhead for pure ordering events.

----

Velocity updates between steps
-------------------------------

When velocity changes between steps (time-dependent flow), the engine
must update internal derived fields before the next ``step()``:

.. code-block:: cpp

   // After velocity changes on GPU:
   engine.bind_velocity({d_U, d_V, d_W, num_corners});  // re-bind
   engine.update_derived_fields();                        // recompute corners / drift
   engine.step(dt);

When the velocity pointers **do not change** (same buffers, new values
written by a kernel), ``bind_velocity`` is not needed — but
``update_derived_fields()`` is still required if
``needs_corner_update()`` or ``needs_drift_update()`` returns ``true``.

.. note::

   ``update_derived_fields()`` is async and zero-alloc. It is safe to
   call on every iteration — it is a no-op when the engine uses
   ``InterpolationMode::Linear`` with ``DriftCorrectionMode::None``.

----

Extracting results
------------------

Positions
~~~~~~~~~

Particle positions live **in your device buffers** (``d_x``, ``d_y``,
``d_z``). After synchronizing the transport stream:

.. code-block:: cpp

   engine.synchronize();   // or cudaStreamSynchronize(transport_stream)

   // Option A: direct cudaMemcpy
   cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

   // Option B: async with pinned memory (preferred for HPC)
   cudaMemcpyAsync(h_x, d_x, N * sizeof(double),
                   cudaMemcpyDeviceToHost, transport_stream);
   cudaStreamSynchronize(transport_stream);

Statistics
~~~~~~~~~~

``StatsComputer`` performs GPU reductions — only 6 doubles (mean + var,
per axis) are copied to the host:

.. code-block:: cpp

   par2::StatsComputer<double> stats(N);   // pre-allocates reduction buffers
   par2::StatsConfig cfg;
   cfg.filter_active_only = true;          // skip exited particles

   stats.compute_async(engine.particles(), grid, cfg, transport_stream);
   cudaStreamSynchronize(transport_stream);
   auto r = stats.fetch_result();
   // r.moments.mean[0], r.moments.var[0], r.moments.std[0], ...

Unwrapped positions (periodic BC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With periodic boundaries, particles wrap around the domain. To recover
continuous trajectories:

.. code-block:: cpp

   // Allocate once
   double *d_xu, *d_yu, *d_zu;
   cudaMalloc(&d_xu, N * sizeof(double));
   cudaMalloc(&d_yu, N * sizeof(double));
   cudaMalloc(&d_zu, N * sizeof(double));
   par2::UnwrappedPositionsView<double> unwrap{d_xu, d_yu, d_zu, N};

   // After each step (or when needed):
   engine.compute_unwrapped_positions(unwrap);
   engine.synchronize();
   // d_xu now contains: x + wrapX * Lx
