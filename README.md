# Par2\_Core

**GPU-native RWPT transport engine for HPC simulation pipelines.**

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)
![CMake](https://img.shields.io/badge/CMake-≥3.18-064F8C?logo=cmake)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=readthedocs)](https://santi-esquerre.github.io/Par2_Core/)
<!-- TODO: update docs URL once GitHub Pages is live -->

---

## What is Par2\_Core?

Par2\_Core is a **static CUDA library** that solves solute transport via
Random Walk Particle Tracking (RWPT) entirely on the GPU.
It is designed to plug into larger HPC simulation pipelines
(flow solvers, coupled multi-physics codes) with minimal friction.

- **RWPT transport on GPU** — Itô–Taylor stepping, analytical displacement
  matrix, per-axis boundary conditions.
- **Device-resident data** — velocity fields and particle arrays stay on
  GPU; no host↔device copies in the hot loop.
- **Stream / event aware** — the engine never calls
  `cudaDeviceSynchronize()`; synchronization is explicit via
  `record_event()` / `wait_event()`.
- **Pipeline-first API** — `prepare()` allocates once; `step(dt)` is
  a pure kernel launch with zero allocations.
- **Modular workspace** — grid, transport parameters, and boundary
  conditions are decoupled from the engine itself.
- **Scientific documentation** — full equation derivations
  (mapped to Rizzo et al., 2019) published on
  [GitHub Pages](https://GerryR.github.io/par2/).

---

## Key Features

| Feature | Details |
|---|---|
| **Hot path** | `step(dt)` / `advance(dt, n)` — async kernel launch, no alloc, no sync |
| **Zero-copy binding** | `bind_velocity()`, `bind_particles()` operate on user-owned device pointers |
| **Boundary conditions** | `Closed` (reflective), `Periodic` (wrap + counter), `Open` (exit + flag) — per axis, per face |
| **Particle injection** | `inject_box()` for uniform seeding inside an arbitrary box |
| **Snapshots / stats** | Optional CSV export (`io.hpp`) and moment statistics (`stats.hpp`) |
| **Precision** | `float` or `double` via template parameter `TransportEngine<T>` |
| **CUDA archs** | Default: `sm_75 sm_80 sm_86 sm_89 sm_90` (override with `CMAKE_CUDA_ARCHITECTURES`) |

---

## Quick Start

### Requirements

- CUDA Toolkit ≥ 12.x (tested with 12.x; 11.2+ may work)
- CMake ≥ 3.18
- C++17-capable host compiler (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 19.29)

### Build the library

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Build with examples

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPAR2_BUILD_EXAMPLES=ON
cmake --build build -j
```

Two example binaries are produced under `build/examples/`:

| Binary | Description |
|---|---|
| `par2_pipeline_example` | Time-dependent velocity, multi-stream + events |
| `par2_steady_flow_example` | Constant velocity, simplest hot loop |

### Build documentation (optional)

Requires Doxygen and a Python venv with Sphinx:

```bash
pip install -r docs/requirements.txt   # sphinx, breathe, etc.
cmake -S . -B build -DPAR2_BUILD_DOCS=ON
cmake --build build --target docs
# Output: docs/build/html/index.html
```

---

## Example — Pipeline Integration

```cpp
#include <par2_core/par2_core.hpp>

// 1. Describe the domain
auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
par2::TransportParams<double> params{Dm, alpha_L, alpha_T};
auto bc = par2::BoundaryConfig<double>::all_closed();

// 2. Create engine (owns CUDA workspace, not your data)
par2::TransportEngine<double> engine(grid, params, bc);
engine.set_stream(my_stream);               // optional: dedicated stream

// 3. Bind device pointers (zero-copy)
engine.bind_velocity({d_U, d_V, d_W, grid.num_corners()});
engine.bind_particles({d_x, d_y, d_z, num_particles});

// 4. One-time allocation + RNG init
engine.prepare();

// 5. Hot loop — async, no alloc, no sync
for (int t = 0; t < num_steps; ++t) {
    engine.wait_event(velocity_ready);       // GPU-side wait
    engine.step(dt);
    engine.record_event(transport_done);     // signal other solvers
}

// 6. Sync only when the host needs data
engine.synchronize();
```

> **Note:** For the full multi-stream pipeline pattern with events, see
> `examples/pipeline_example.cu`.

---

## Documentation

📖 **[Full documentation on GitHub Pages](https://santi-esquerre.github.io/Par2_Core/)**
<!-- TODO: verify URL once GitHub Pages deployment is set up -->

The docs are built with **Sphinx + MathJax + Breathe (Doxygen → Sphinx)**
and cover:

| Section | Content |
|---|---|
| **User Guide** | Quick start, pipeline recipes, memory contract, performance tips |
| **Scientific Background** | RWPT derivation, dispersion tensor, displacement matrix *B*, drift correction, interpolation — all mapped to Rizzo et al. (2019) |
| **HPC / GPU Architecture** | Memory layout, streams & events, workspace allocations, RNG reproducibility |
| **API Reference** | Auto-generated from Doxygen (every public header) |
| **Appendix** | References & paper equation cross-reference table |

---

## Project Structure

```
Par2_Core/
├── include/par2_core/     # Public API headers
│   ├── par2_core.hpp       #   Umbrella header
│   ├── transport_engine.hpp#   Main engine class
│   ├── types.hpp           #   TransportParams, EngineConfig, BoundaryType
│   ├── grid.hpp            #   GridDesc, make_grid()
│   ├── boundary.hpp        #   BoundaryConfig, AxisBoundary
│   ├── views.hpp           #   ParticlesView, VelocityView, etc.
│   ├── velocity_layout.hpp #   FaceFieldView, CornerFieldView
│   ├── injectors.hpp       #   Injection helpers
│   ├── io.hpp              #   CSV snapshot export
│   └── stats.hpp           #   Moment statistics
├── src/                    # Implementation (CUDA kernels, internal math)
│   ├── kernels/            #   move_particles, drift_correction, cornerfield
│   └── internal/           #   dispersion, RNG, fields, workspace
├── examples/               # Two pipeline examples
├── docs/                   # Sphinx + Doxygen documentation
│   ├── source/theory/      #   Scientific derivations
│   ├── source/hpc/         #   GPU architecture docs
│   └── source/api/         #   Auto-generated API ref
└── CMakeLists.txt
```

---

## Contributing

Contributions are welcome — bug reports, feature requests, and pull requests.

1. **Build** the library and examples (see [Quick Start](#quick-start)).
2. **Run an example** to verify your setup:
   ```bash
   ./build/examples/par2_pipeline_example
   ```
3. **Build the docs** to check rendering:
   ```bash
   cmake -S . -B build -DPAR2_BUILD_DOCS=ON && cmake --build build --target docs
   ```
4. Open a PR against `master`.

> **Style:** There is no `.clang-format` in the repo yet. Follow the
> existing code style (4-space indent, `snake_case` for functions/variables,
> `PascalCase` for types). <!-- TODO: add .clang-format -->

---

## Cite This Work

If you use Par2\_Core in your research, please cite the original paper that
this codebase is based on:

> Calogero B. Rizzo, Aiichiro Nakano, Felipe P.J. de Barros,
> "PAR²: Parallel Random Walk Particle Tracking Method for solute transport
> in porous media,"
> *Computer Physics Communications*, Volume 239, 2019, Pages 265–271.

```bibtex
@article{Rizzo2019,
  author  = {Rizzo, Calogero B. and Nakano, Aiichiro and de Barros, Felipe P.J.},
  title   = {{PAR\textsuperscript{2}}: Parallel Random Walk Particle Tracking
             Method for solute transport in porous media},
  journal = {Computer Physics Communications},
  volume  = {239},
  pages   = {265--271},
  year    = {2019},
  doi     = {10.1016/j.cpc.2019.01.013}
}
```

<!-- TODO: add CITATION.cff to the repo root for automatic GitHub citation support -->

---

## License

Par2\_Core is released under the **MIT License** — see [LICENSE](LICENSE).

The original PAR² codebase (`legacy/`) is released under the
**GNU General Public License v3.0** (see `legacy/LICENSE`).
