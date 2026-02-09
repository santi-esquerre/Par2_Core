Quick Start
===========

Requirements
------------

.. list-table::
   :widths: 25 75

   * - CUDA Toolkit
     - 11.2+ (``cudaMallocAsync`` support); 12.x recommended
   * - C++ compiler
     - C++17 capable (GCC 9+, Clang 10+)
   * - CMake
     - 3.18+
   * - GPU
     - Compute capability 7.5+ (Turing / Ampere / Ada / Hopper)

Build the library
-----------------

.. code-block:: bash

   # Clone the repository
   git clone <repo-url> Par2_Core && cd Par2_Core

   # Release build (recommended)
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)

The output is ``build/libpar2_core.a`` (static library).

Build with examples
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cmake -S . -B build \
       -DCMAKE_BUILD_TYPE=Release \
       -DPAR2_BUILD_EXAMPLES=ON
   cmake --build build -j$(nproc)

Two example binaries are produced:

.. list-table::
   :widths: 40 60

   * - ``build/examples/par2_pipeline_example``
     - Two-stream pipeline with time-dependent velocity
   * - ``build/examples/par2_steady_flow_example``
     - Simplest hot loop with constant velocity

Run the pipeline example:

.. code-block:: bash

   ./build/examples/par2_pipeline_example

Output lands in ``output/stats.csv`` and ``output/snapshot_step*.csv``.

Link from your own CMake project
---------------------------------

If Par2\_Core is a subdirectory:

.. code-block:: cmake

   add_subdirectory(Par2_Core)

   add_executable(my_solver my_solver.cu)
   target_link_libraries(my_solver PRIVATE par2_core)

If installed system-wide:

.. code-block:: cmake

   find_package(par2_core REQUIRED)

   add_executable(my_solver my_solver.cu)
   target_link_libraries(my_solver PRIVATE par2::par2_core)

Minimal code
------------

One header includes the entire API:

.. code-block:: cpp

   #include <par2_core/par2_core.hpp>

The smallest working program:

.. code-block:: cpp

   #include <par2_core/par2_core.hpp>
   #include <cuda_runtime.h>

   int main() {
       // 1. Grid: 100x100 cells, 1 m spacing, 2-D (nz=1)
       auto grid = par2::make_grid<double>(100, 100, 1, 1.0, 1.0, 1.0);

       // 2. Physics
       par2::TransportParams<double> params;
       params.molecular_diffusion = 1e-9;
       params.alpha_l             = 0.1;
       params.alpha_t             = 0.01;

       // 3. Boundary conditions
       auto bc = par2::BoundaryConfig<double>::all_closed();

       // 4. Engine (default config)
       par2::TransportEngine<double> engine(grid, params, bc);

       // 5. Allocate GPU velocity field — (nx+1)*(ny+1)*(nz+1) per component
       size_t sz = grid.num_corners();
       double *d_U, *d_V, *d_W;
       cudaMalloc(&d_U, sz * sizeof(double));
       cudaMalloc(&d_V, sz * sizeof(double));
       cudaMalloc(&d_W, sz * sizeof(double));
       cudaMemset(d_U, 0, sz * sizeof(double));  // zero velocity
       cudaMemset(d_V, 0, sz * sizeof(double));
       cudaMemset(d_W, 0, sz * sizeof(double));

       // 6. Allocate GPU particle arrays (SoA)
       int N = 5000;
       double *d_x, *d_y, *d_z;
       cudaMalloc(&d_x, N * sizeof(double));
       cudaMalloc(&d_y, N * sizeof(double));
       cudaMalloc(&d_z, N * sizeof(double));

       // 7. Bind + inject + prepare
       engine.bind_velocity({d_U, d_V, d_W, sz});
       engine.bind_particles({d_x, d_y, d_z, N});
       engine.inject_box(10.0, 10.0, 0.0, 30.0, 30.0, 1.0);
       engine.prepare();

       // 8. Run 100 steps
       double dt = 0.1;
       for (int i = 0; i < 100; ++i)
           engine.step(dt);

       engine.synchronize();

       // 9. Cleanup
       cudaFree(d_U); cudaFree(d_V); cudaFree(d_W);
       cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
   }

CMake options reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Option
     - Default
     - Description
   * - ``PAR2_BUILD_EXAMPLES``
     - ``ON``
     - Build the two example binaries
   * - ``PAR2_BUILD_DOCS``
     - ``OFF``
     - Build Sphinx + Doxygen documentation
   * - ``CMAKE_CUDA_ARCHITECTURES``
     - ``75;80;86;89;90``
     - Target GPU architectures (semicolon-separated)
   * - ``CMAKE_BUILD_TYPE``
     - (none)
     - ``Release`` adds ``-O3 --use_fast_math`` to CUDA
