.. _quickmulti:

Using multi-GPU parallelization [NEW!]
+++++++++++++++++++++++++++++++

GX now has the capability to parallelize a calculation over multiple GPUs. 

.. contents::

Constraints
-----------

Parallelization is implemented over the species and Hermite indices, with an experimental decomposition over theta available through the expert input ``nproc_theta`` (or ``ptheta``). The computational grid must divide evenly into the number of GPUs requested, and decomposition of species index is prioritized within each theta slab. If :math:`N_\mathrm{sp}` is the number of species, :math:`N_m` is the number of Hermite modes, :math:`N_\theta` is the number of theta grid points, :math:`P_\theta` is ``nproc_theta``, and :math:`N_\mathrm{GPU}` is the number of GPUs to be used for the calculation, this means:

- :math:`N_\mathrm{GPU}` must be an integer multiple of :math:`P_\theta`
- :math:`N_\theta` must be an integer multiple of :math:`P_\theta`
- if :math:`N_\mathrm{GPU}/P_\theta \leq N_\mathrm{sp}`, :math:`N_\mathrm{sp}` must be an integer multiple of :math:`N_\mathrm{GPU}/P_\theta`
- if :math:`N_\mathrm{GPU}/P_\theta > N_\mathrm{sp}`, :math:`N_\mathrm{GPU}/P_\theta` must be an integer multiple of :math:`N_\mathrm{sp}` AND :math:`N_m` must be an integer multiple of :math:`(N_\mathrm{GPU}/P_\theta)/N_\mathrm{sp}`.

For example, if :math:`N_\mathrm{sp} = 2`, :math:`N_m = 16`, and ``nproc_theta = 1``, the number of GPUs can be any of :math:`N_\mathrm{GPU} = \{1,2,4,8,16,32\}`. With ``nproc_theta = 2``, each theta slab uses the same species/Hermite decomposition rules over :math:`N_\mathrm{GPU}/2` GPUs.

Theta decomposition currently has additional prototype restrictions:

- only periodic parallel boundary conditions are supported
- restart reads are not supported, and ``save_for_restart`` is disabled automatically
- ``use_NCCL`` must be true
- ``hyperz``, ``hypercollisions_kz``, and ``dealias_kz`` are not supported
- ``beer4+2`` and ``smith_par`` closures are not supported
- forcing with nonzero ``forcing_kz`` is not supported
- the theta derivative uses a second-order finite-difference halo exchange rather than the local spectral derivative.

Requesting a multi-GPU job (SLURM)
----------------------------------

On systems with SLURM job management, use the ``--nodes=[N]`` flag to specify the number of nodes, and the ``--gpus-per-node=[P]`` flag to specify the number of GPUs per node, so that the total number of GPUs requested is ``N*P``. The maximum number of GPUs per node is system-specific, so check the documentation for your system, but typical configurations are 2 or 4 GPUs per node.

For example, an interactive job that requests 4 GPUs on a system with 4 GPUs/node can be requested with something like

.. code-block:: bash

  salloc --nodes=1 -—gpus-per-node=4 ...

For a batch job, the submission script should include something like

.. code-block:: bash

  ...
  #SBATCH --nodes=1
  #SBATCH --gpus-per-node=4
  ...

Running the calculation
-----------------------

The number of GPUs used by the calculation is controlled at runtime by the number of MPI processes. Suppose we have a job with 4 GPUs allocated via one of the commands above. Now we can choose to use 1, 2, or 4 GPUs for the GX calculation (assuming :math:`N_\mathrm{sp}N_m\geq4`). On SLURM systems, use srun to launch the job:

.. code-block:: bash

  srun -n [NGPU] [/path/to/]gx [inputfile].in

where ``NGPU`` can be 1, 2, or 4 here.

Performance considerations
--------------------------

Performance will often be limited by the speed of the connection between GPUs. Typically GPUs within a single node are connected with a faster interconnect (e.g. NVLINK) than across nodes, so scaling efficiency may degrade somewhat when parallelizing across multiple nodes. For details about the scaling of the code, see Section 7 of the GX paper (Mandell et al., 2022). 
