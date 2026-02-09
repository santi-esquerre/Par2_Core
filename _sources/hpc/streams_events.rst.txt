CUDA Streams & Events
=====================


Par2_Core is designed as a "good citizen" in multi-solver GPU pipelines.
It never calls ``cudaDeviceSynchronize()`` internally.

Stream Ownership
----------------

The engine does **not** own its CUDA stream.  The caller creates and manages
the stream lifetime:

.. code-block:: cpp

   cudaStream_t transport_stream;
   cudaStreamCreate(&transport_stream);
   engine.set_stream(transport_stream);

Event-Based Synchronisation
---------------------------

For coupling with other solvers (flow, reactions, etc.):

.. code-block:: text

   stream_flow:       update_velocity → record(vel_ready)
                              ↓
   stream_transport:  wait(vel_ready) → update_derived_fields() → step(dt)
                              ↓
                      record(transport_done)

The CPU never blocks — waiting happens on the GPU via
``cudaStreamWaitEvent``.

Double-Buffered Velocity Pattern
---------------------------------

For time-varying velocity fields, a double-buffer pattern avoids
synchronisation stalls:

.. code-block:: text

   step N:  engine uses velocity buffer A
            flow solver writes next field to buffer B (different stream)
   step N+1: swap A ↔ B via bind_velocity
             update_derived_fields() re-computes corner/drift
             step(dt)
