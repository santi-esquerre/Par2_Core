References
==========

Primary reference
-----------------

.. [Rizzo2019] Calogero B. Rizzo, Aiichiro Nakano, Felipe P.J. de Barros,
   "PAR²: Parallel Random Walk Particle Tracking Method for solute transport
   in porous media,"
   *Computer Physics Communications*, Volume 239, 2019, Pages 265–271.
   `doi:10.1016/j.cpc.2019.01.013 <https://doi.org/10.1016/j.cpc.2019.01.013>`_

Par2\_Core is a fork of the original PAR² codebase, restructured as a
GPU-native static library for integration into HPC pipelines.

Equation cross-reference
~~~~~~~~~~~~~~~~~~~~~~~~~

The table below maps paper equations to Par2\_Core documentation and source.

.. list-table::
   :header-rows: 1
   :widths: 12 30 30 28

   * - Paper Eq.
     - Description
     - Doc section
     - Source file
   * - Eq. 4
     - Dispersion tensor :math:`D_{ij}`
     - :doc:`theory/dispersion_tensor`
     - ``internal/math/dispersion.cuh``
   * - Eq. 5
     - Itô–Taylor stepping scheme
     - :doc:`theory/rwpt`
     - ``kernels/move_particles.cu``
   * - Eq. 6
     - Drift vector :math:`\mathbf{A}`
     - :doc:`theory/drift_correction`
     - ``internal/fields/cornerfield_accessor.cuh``, ``kernels/drift_correction.cu``
   * - Eqs. 9–11
     - :math:`\mathbf{D} = a\mathbf{I} + b\mathbf{W}`, B assembly
     - :doc:`theory/displacement_matrix`
     - ``internal/math/dispersion.cuh``
   * - Eqs. 12–14
     - Eigenvectors :math:`\mathbf{e}_0, \mathbf{e}_1, \mathbf{e}_2`
     - :doc:`theory/displacement_matrix`
     - ``internal/math/dispersion.cuh``

RWPT theory
-----------

.. [Salamon2006] Salamon, P., Fernàndez-Garcia, D., and Gómez-Hernández, J.J.
   "A review and numerical assessment of the random walk particle tracking
   method."
   *Journal of Contaminant Hydrology*, 87(3–4), 277–305, 2006.
   `doi:10.1016/j.jconhyd.2006.04.005 <https://doi.org/10.1016/j.jconhyd.2006.04.005>`_

.. [LaBolle1996] LaBolle, E.M., Fogg, G.E., and Tompson, A.F.B.
   "Random-walk simulation of transport in heterogeneous porous media:
   Local mass-conservation problem and implementation methods."
   *Water Resources Research*, 32(3), 583–593, 1996.
   `doi:10.1029/95WR03528 <https://doi.org/10.1029/95WR03528>`_

.. [LaBolle2000] LaBolle, E.M., Quastel, J., Fogg, G.E., and Gravner, J.
   "Diffusion processes in composite porous media and their numerical
   integration by random walks: Generalized stochastic differential
   equations with discontinuous coefficients."
   *Water Resources Research*, 36(3), 651–662, 2000.
   `doi:10.1029/1999WR900224 <https://doi.org/10.1029/1999WR900224>`_

GPU / CUDA
----------

.. [NVIDIA-cuRAND] NVIDIA Corporation. *cuRAND Library Documentation*.
   `docs.nvidia.com/cuda/curand <https://docs.nvidia.com/cuda/curand/index.html>`_

.. [NVIDIA-BestPractices] NVIDIA Corporation. *CUDA C++ Best Practices Guide*.
   `docs.nvidia.com/cuda/cuda-c-best-practices-guide <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>`_
