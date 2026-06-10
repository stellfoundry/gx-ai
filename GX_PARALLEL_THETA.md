Task: Investigate and prototype theta/parallel-coordinate decomposition in GX multi-GPU parallelization.

Context:
GX currently appears to distribute multi-GPU work over species and Hermite modes only. For my target case,

  ntheta   = 96
  nperiod  = 1
  ny       = 257
  nx       = 257
  nhermite = 16
  nlaguerre= 4
  nspecies = 2

the existing decomposition over species x Hermite is limited. With nspecies=2 and nhermite=16, the maximum decomposition is 32 GPUs, but at 16 GPUs each GPU has only 2 Hermite modes, and at 32 GPUs each GPU has only 1 Hermite mode. That likely gives poor halo/compute balance in Hermite.

The large part of the data is configuration space:

  nx * ny * ntheta = 257 * 257 * 96 ≈ 6.34e6

whereas velocity/species space is only:

  nspecies * nhermite * nlaguerre = 2 * 16 * 4 = 128

So the missing useful decomposition dimension is probably theta/z/parallel coordinate, not Laguerre. nlaguerre=4 is too small to split usefully.

Goal:
Add or prototype a new process/GPU decomposition:

  P = P_species * P_hermite * P_theta

while keeping the full perpendicular plane (kx, ky) local on each GPU. Avoid decomposing x/y initially because GX uses pseudo-spectral perpendicular operations and distributed FFTs/all-to-alls would likely be much more invasive and expensive.

Desired layouts to support:
For this example problem:

  8 GPUs:   P_species=2, P_hermite=2, P_theta=2
            local Hermite = 8, local theta = 48

  16 GPUs:  P_species=2, P_hermite=2, P_theta=4
            local Hermite = 8, local theta = 24

  32 GPUs:  P_species=2, P_hermite=2, P_theta=8
            local Hermite = 8, local theta = 12

Alternative 32-GPU layout to benchmark:

  P_species=2, P_hermite=4, P_theta=4
  local Hermite = 4, local theta = 24

The intended advantage is to avoid over-decomposing Hermite while reducing memory and work per GPU through theta slabs.

Implementation idea:
Each rank/GPU owns something like:

  g_local[species_local, hermite_local + m_halos, laguerre, theta_local + theta_halos, kx, ky]

or whatever memory ordering matches current GX layout best. Keep full kx, ky and all Laguerre modes local.

Need to identify actual GX array ordering and decomposition infrastructure before changing layout.

Key code investigation steps:
1. Find the current multi-GPU decomposition setup:
   - Where ranks are assigned to species and Hermite.
   - Where local/global Hermite indices are mapped.
   - Where GPU count restrictions are enforced.
   - Where MPI communicators are created.
   Search terms:
     decomp
     nproc
     rank
     comm
     hermite
     species
     iky
     ikx
     theta
     zed
     z
     ntgrid
     ntheta

2. Find field/moment reductions:
   - Current species/Hermite decomposition likely computes partial density/current moments and all-reduces over ranks.
   - With theta decomposition, these reductions should only need to reduce over species/Hermite ranks that share the same theta slab.
   - Field arrays phi/Apar/Bpar may become theta-local if the field solve is local in theta.
   - Check whether any field solve or operator assumes global theta availability.

3. Find Hermite halo exchange:
   - Existing m-halo exchange can probably remain.
   - Make sure it happens within communicator groups that share the same species/theta slab or whatever is appropriate.
   - Avoid over-splitting Hermite; target local nhermite >= 4, preferably 8.

4. Add theta halo exchange:
   - Identify theta derivative/streaming/drift operators needing neighboring theta planes.
   - Add ghost/halo planes in theta.
   - Exchange theta halos between neighboring P_theta ranks.
   - Special care at theta boundaries: flux-tube twist-and-shift boundary conditions may couple theta endpoint communication to shifted kx modes. Do not assume simple periodic exchange until verified.
   - Internal theta slab boundaries should be ordinary halo exchange; only global endpoints need twist-and-shift logic.

5. Nonlinear/perpendicular operations:
   - Keep full x/y or kx/ky local, so existing local FFT/pseudospectral operations should remain local.
   - Need to check whether the nonlinear kernels loop over all theta planes independently. If yes, they should work on theta_local with minimal changes.
   - Avoid distributed perpendicular FFTs in this prototype.

6. Geometry arrays:
   - Geometry coefficients are probably theta-dependent.
   - Make geometry arrays theta-local or provide local views:
       bmag[theta_local], gradpar[theta_local], drift coeffs, metric coeffs, etc.
   - Check if any code assumes theta arrays are globally indexed from 0..ntheta-1.
   - Maintain global theta index mapping:
       itheta_global = theta_start[rank_theta] + itheta_local

7. Diagnostics and output:
   - Diagnostics that are functions of theta need gather or parallel write support.
   - Scalar fluxes usually require summing/reducing over local theta slabs.
   - Field snapshots/distribution output need either:
       a) gather theta slabs to rank 0, or
       b) write distributed output with correct global theta offsets.
   - Start with minimal diagnostics needed for validation.

8. Restart I/O:
   - Check whether restart files assume each rank has global theta.
   - Need to read/write theta-local chunks or gather/scatter on restart.
   - For prototype, it may be okay to initially disable restart with theta decomposition or implement simple gather/scatter.

9. Communicator design:
   Build a 3D rank topology:
      rank -> (is_rank, im_rank, itheta_rank)

   Useful communicators:
      comm_all
      comm_field_theta: ranks with same itheta_rank, reduce over species/hermite for local theta slab
      comm_m: ranks differing in im_rank, same species/theta rank, for Hermite halos
      comm_theta: ranks differing in itheta_rank, same species/hermite rank, for theta halos
      maybe comm_io: for gathering theta slabs

10. Input/API:
    Add an optional input parameter, e.g.
       nproc_theta
    or
       ntheta_parallel = true / ptheta = ...
    Then require:
       total_gpus = pspecies * phermite * ptheta
       ntheta divisible by ptheta for first prototype
    Later support uneven theta slabs if needed.

Suggested restrictions for first prototype:
  - ntheta divisible by P_theta
  - keep full kx, ky local
  - keep all Laguerre local
  - support nonlinear runs only if local perpendicular FFTs remain unchanged
  - possibly require nperiod=1 initially
  - possibly disable restart initially
  - support simple diagnostics first
  - benchmark before optimizing

Validation plan:
1. Run a small case with P_theta=1 and confirm bitwise or near-bitwise agreement with current code.
2. Run the same case with P_theta=2 and compare:
   - growth rates for linear cases
   - heat/particle flux time traces for nonlinear cases
   - field energy
   - conservation diagnostics if available
3. Test theta boundary behavior specifically:
   - compare cases with twist-and-shift active
   - ensure endpoint communication applies correct kx shift
4. Benchmark:
   - current decomposition: 2 x 4 x 1 on 8 GPUs
   - proposed:             2 x 2 x 2 on 8 GPUs
   - current:              2 x 8 x 1 on 16 GPUs
   - proposed:             2 x 2 x 4 on 16 GPUs
   - current:              2 x16 x 1 on 32 GPUs
   - proposed:             2 x 2 x 8 on 32 GPUs
   Compare wall time per timestep, memory per GPU, communication time, and scaling efficiency.

Performance hypothesis:
For nhermite=16, Hermite-only decomposition becomes unattractive beyond 8 GPUs because local Hermite width becomes 4, 2, or 1. Theta decomposition should improve scaling by preserving local Hermite width around 8 while distributing the large nx*ny*ntheta configuration-space block. It should be much less invasive than x/y decomposition because it avoids distributed perpendicular FFTs.
