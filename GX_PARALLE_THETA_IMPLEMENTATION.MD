# Parallel Theta Decomposition Implementation

This note summarizes the experimental theta-domain decomposition added to GX.
The implementation extends the existing multi-GPU split, which previously
distributed work over species and Hermite indices, so that the theta dimension
can also be split across MPI ranks/GPUs.

The new user-facing control is the expert parameter `nproc_theta`, with `ptheta`
accepted as a short alias. A value of `1` preserves the previous behavior. A
value greater than `1` divides the global theta grid into equal contiguous slabs,
with each slab then using the existing species/Hermite decomposition rules.

## High-Level Layout

The decomposition is now logically three-dimensional:

```text
rank = iproc_m + nprocs_m * (iproc_s + nprocs_s * iproc_z)
```

where:

- `iproc_z` selects the theta slab.
- `iproc_s` selects the species block within that theta slab.
- `iproc_m` selects the Hermite block within that species/theta block.

The choice to put theta outside the existing species/Hermite layout keeps the
old decomposition behavior intact inside each theta slab. That reduces the
number of call sites that need to understand theta decomposition directly and
keeps existing species/Hermite communicator semantics mostly unchanged.

## Parameter Handling

Files changed:

- `include/parameters.h`
- `src/parameters.cu`
- `docs/MultiGPU.rst`

Changes:

- Added `Parameters::nproc_theta`.
- Read `nproc_theta` from the input file, defaulting to `1`.
- Read `ptheta` as an alias, overriding `nproc_theta` if present.
- Assert that `nproc_theta` is positive.
- Store `nproc_theta` in the NetCDF expert parameter group.
- Documented theta decomposition and its current restrictions in
  `docs/MultiGPU.rst`.

Why:

- The new parameter is explicit and opt-in, so existing inputs retain their old
  behavior.
- The alias is useful for short expert input files and mirrors the existing
  compact style of some GX controls.
- Writing the value to NetCDF makes outputs self-describing.

## Grid Topology

Files changed:

- `include/grids.h`
- `src/grids.cu`

Changes:

- Added global theta size tracking with `Nz_glob`.
- Added local theta slab bounds `z_lo` and `z_up`.
- Added theta topology fields `nprocs_z` and `iproc_z`.
- Recomputed local `Nz`, `NxNycNz`, `NxNyNz`, `NxNz`, and `NycNz` after slicing
  theta.
- Added `mpcom_sm`, an MPI communicator containing ranks in the same theta slab.
- Added `mpcom_z`, an MPI communicator containing ranks with the same
  species/Hermite block across theta slabs.
- Extended `Grids::proc` to map `(iproc_m, iproc_s, iproc_z)` to a global rank.
- Required `nprocs % nproc_theta == 0`.
- Required `ntheta % nproc_theta == 0`.
- Required `nproc_theta == 1` for single-process runs.

Why:

- GX arrays are still allocated using local dimensions, so `Nz` must become the
  local slab size while `Nz_glob` preserves the physical/global theta extent.
- Keeping `z_lo`/`z_up` in `Grids` gives diagnostics, geometry, and NetCDF code a
  single source of truth for where this rank's slab sits in the global theta
  domain.
- The separate MPI communicators avoid accidentally reducing over all ranks when
  the operation should only combine species/Hermite ranks or only combine theta
  ranks.
- The divisibility assertions avoid partial slab handling. Uneven theta slabs
  would require broader indexing and I/O changes.

## NCCL Communicators

Files changed:

- `include/grids.h`
- `src/grids.cu`

Changes:

- Added `ncclComm_z` for ranks that share the same species/Hermite block and
  differ only in theta slab.
- Changed `ncclId_s`, `ncclId_m`, and `ncclId_m0` to account for independent
  communicator groups in each theta slab.
- Added `ncclId_z` for theta communicators.
- Updated `ncclComm_s` and `ncclComm_m` initialization so those communicators are
  local to a theta slab.
- Updated `ncclComm_m0` so it also stays within one theta slab.

Why:

- Existing species and Hermite NCCL collectives should not cross theta slabs.
  They operate on data that is local in theta.
- New theta halo exchanges and theta reductions need a communicator that spans
  theta slabs but keeps species/Hermite ownership fixed.
- Reusing the old full-world communicator for these operations would mix
  unrelated blocks and produce incorrect results.

## Parallel Derivatives In Theta

Files changed:

- `include/grad_parallel.h`
- `src/grad_parallel.cu`
- `src/linear.cu`

Changes:

- Added `GradParallelThetaFD`.
- `Linear_GK` and `Linear_KREHM` now select `GradParallelThetaFD` when
  `nprocs_z > 1`.
- Implemented centered finite-difference `dz`.
- Implemented centered finite-difference `dz2`.
- Added NCCL halo exchange of the first and last local theta planes.
- Added in-place safety through a temporary buffer.
- Added a pass-through `zft` for single-moment callers that expect the method to
  exist.
- Made `abs_dz` explicitly unsupported for theta decomposition.

Why:

- The existing spectral theta derivative assumes each rank owns the full theta
  line, which is no longer true after theta decomposition.
- A finite-difference derivative with one-cell halos is the smallest viable
  distributed derivative implementation.
- NCCL is already used for GPU-to-GPU collectives in GX, so using NCCL for theta
  halos avoids introducing a separate GPU-aware MPI path.
- In-place derivative calls occur in the existing code paths, so the temporary
  buffer prevents overwriting input values before neighbor differences have been
  computed.

## Unsupported Physics/Algorithm Guards

Files changed:

- `src/grids.cu`
- `src/linear.cu`
- `include/grad_parallel.h`

Changes:

- Reject theta decomposition unless `use_NCCL = true`.
- Reject restart reads.
- Disable `save_for_restart` automatically when theta decomposition is active.
- Reject `forcing_init` with nonzero `forcing_kz`.
- Reject `hyperz`, `hypercollisions_kz`, and `dealias_kz`.
- Reject `beer4+2` and `smith_par` closures.
- Make `GradParallelThetaFD::abs_dz` fail explicitly.

Why:

- The prototype only implements the communication needed by the basic finite
  difference parallel derivative.
- Restart and some diagnostics require global theta offsets in both read and
  write paths. Output writes were partly offset, but restart reads were not
  implemented end-to-end.
- `hyperz`, `hypercollisions_kz`, `dealias_kz`, `beer4+2`, and `smith_par`
  depend on spectral or absolute theta derivatives that require global theta
  transforms or a distributed equivalent of `abs_dz`.
- Nonzero `forcing_kz` assumes global Fourier-mode ownership in theta.

## Geometry

File changed:

- `src/geometry.cu`

Changes:

- Updated local theta coordinate generation to use global theta index
  `kg = z_lo + k`.
- Updated geometry-file size checks to compare against `Nz_glob`.
- Updated file/eik geometry interpolation and slicing so each rank keeps only
  its local theta slab.
- For theta-decomposed runs, compute `bgrad` from global `bmag` before slicing.
- Updated volume and flux denominator calculations to reduce across `mpcom_z`.

Why:

- Geometry is physically defined on the global theta grid. Each rank must see
  coordinates corresponding to its global theta position, not a re-centered local
  slab.
- `bgrad` needs neighbor information across theta-slab boundaries. Computing it
  globally before slicing avoids incorrect one-sided or local-periodic gradients
  at slab edges.
- Volume and flux normalization denominators integrate over theta. With theta
  split across ranks, those denominators must combine contributions across
  theta slabs with the same species/Hermite ownership.

## Reductions

Files changed:

- `include/reductions.h`
- `src/reductions.cu`

Changes:

- Added a `reduce_z` reduction axis.
- Added NCCL all-reduce over `ncclComm_z` when a reduction needs theta-global
  data.
- Removed the old assumption that a simultaneous species/Hermite reduction could
  use the full-world communicator safely.

Why:

- Some reductions now need to combine local theta slab contributions while
  preserving species/Hermite ownership.
- Once the world communicator contains independent theta slabs, full-world
  reductions can accidentally combine data from unrelated decomposition blocks.

## NetCDF And Diagnostics

Files changed:

- `include/ncdf.h`
- `src/diagnostic_classes.cu`
- `src/spectra_calc.cu`

Changes:

- Define NetCDF `theta` and `kz` dimensions using `Nz_glob`.
- Write local theta-dependent arrays with start offsets based on `z_lo`.
- Offset field, moment, spectra, and selected diagnostic writes into the global
  theta dimension.
- Write local geometry/grid data into the correct global theta slice.

Why:

- Output files should preserve the same global shape regardless of how theta is
  decomposed at runtime.
- Without `z_lo` offsets, each rank would write its local slab at theta index
  zero, overwriting other slabs or producing incorrectly ordered output.

## Solver And Field Communication

File changed:

- `src/solver.cu`

Changes:

- Changed a GK MPI fallback all-reduce to use `mpcom_sm`.
- Changed a KREHM field broadcast to use the theta-local species communicator.

Why:

- Field solves and species/Hermite reductions operate within a local theta slab.
  They should not combine data from other theta slabs unless the operation is
  explicitly theta-global.

## K-Space And Output Coordinates

Files changed:

- `src/grids.cu`
- `include/ncdf.h`
- `src/spectra_calc.cu`

Changes:

- Kept local `kz` allocation tied to local `Nz`.
- Computed output `kz` ordering using global theta indices when theta is split.
- Computed local `z_h` from global theta index.
- Computed `kz_max` from `Nz_glob` when theta decomposition is active.

Why:

- Local derivative work only owns local theta values, but output metadata and
  maximum wave-number estimates should describe the global theta grid.

## Things Not Implemented

The current implementation is intentionally limited. The following are not
implemented:

- Restart reads with theta decomposition.
- Restart writes with theta decomposition. `save_for_restart` is disabled
  automatically.
- Uneven theta decomposition where `ntheta` is not divisible by `nproc_theta`.
- Single-process runs with `nproc_theta > 1`.
- MPI-only theta halo exchange. Theta decomposition currently requires NCCL.
- Twist-and-shift endpoint communication for theta-decomposed derivatives.
- Distributed spectral `zft`/inverse `zft` across theta slabs.
- Distributed `abs_dz`.
- `hyperz` with theta decomposition.
- `hypercollisions_kz` with theta decomposition.
- `dealias_kz` with theta decomposition.
- `beer4+2` closure with theta decomposition.
- `smith_par` closure with theta decomposition.
- Forcing with nonzero `forcing_kz`.
- Runtime validation on a multi-GPU system.
- Performance tuning of halo packing, streams, overlap, or communicator
  scheduling.

Restart support is not impossible; it is just not implemented here. A complete
restart implementation would need restart reads and writes to use `Nz_glob` for
file dimensions and `z_lo` for per-rank hyperslab offsets, and would need the
restart validation logic to compare the file's global theta size against
`Nz_glob` instead of the local `Nz`.

## Validation Performed

The following static checks were run in this environment:

```bash
git diff --check
GK_SYSTEM=generic make -n obj/grids.o obj/grad_parallel.o obj/linear.o obj/solver.o obj/geometry.o obj/reductions.o obj/ncdf.o
```

The make command was a dry run. A real CUDA build and multi-GPU runtime test were
not performed because `nvcc` and GPU/MPI runtime access were not available in
this environment.
