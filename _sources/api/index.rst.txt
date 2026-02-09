API Reference
=============

This section is auto-generated from the C++ headers in ``include/par2_core/``
using Doxygen + Breathe.

How to integrate (TL;DR)
-------------------------

.. code-block:: cpp

   #include <par2_core/par2_core.hpp>

   // 1. Grid + physics + boundary
   auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
   par2::TransportParams<double> params{Dm, alphaL, alphaT};
   auto bc = par2::BoundaryConfig<double>::all_closed();

   // 2. Engine (configure once)
   par2::EngineConfig cfg;
   cfg.interpolation_mode = par2::InterpolationMode::Linear;
   cfg.drift_mode         = par2::DriftCorrectionMode::TrilinearOnFly;
   par2::TransportEngine<double> engine(grid, params, bc, cfg);

   // 3. Bind device-resident data (zero-copy)
   engine.set_stream(my_stream);
   engine.bind_velocity({d_U, d_V, d_W, grid.num_corners()});
   engine.bind_particles({d_x, d_y, d_z, N});
   engine.inject_box(x0, y0, z0, x1, y1, z1);

   // 4. Prepare (one-time allocs: RNG, workspace)
   engine.prepare();

   // 5. Hot loop — async, zero-alloc, zero-sync
   for (int t = 0; t < num_steps; ++t) {
       engine.update_derived_fields();  // after velocity changes
       engine.step(dt);
   }
   engine.synchronize();  // sync when host needs results

See :doc:`/user/pipeline_recipes` for multi-stream patterns and
:doc:`/user/memory_contract` for ownership rules.

Core Engine
-----------

.. doxygenfile:: transport_engine.hpp
   :project: par2_core

Types & Configuration
---------------------

.. doxygenfile:: types.hpp
   :project: par2_core

Grid
----

.. doxygenfile:: grid.hpp
   :project: par2_core

Views (Zero-Copy Binding)
-------------------------

.. doxygenfile:: views.hpp
   :project: par2_core

Velocity Layout
---------------

.. doxygenfile:: velocity_layout.hpp
   :project: par2_core

Boundary Conditions
-------------------

.. doxygenfile:: boundary.hpp
   :project: par2_core

Particle Injection
------------------

.. doxygenfile:: injectors.hpp
   :project: par2_core

Statistics
----------

.. doxygenfile:: stats.hpp
   :project: par2_core

I/O Helpers
-----------

.. doxygenfile:: io.hpp
   :project: par2_core

Version
-------

.. doxygenfile:: version.hpp
   :project: par2_core

Debug Policy
------------

.. doxygenfile:: debug_policy.hpp
   :project: par2_core
