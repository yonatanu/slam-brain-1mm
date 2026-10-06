# Data format

Release `slam_t2_1mm_v1` contains 362 scans, 343 subjects, and 19,362 slices.
The subject-disjoint partitions contain 290 training scans and 72 test scans.
There are 19,065 usable slices, including 15,288 training slices. The `usable`
flag selects slices by support fraction, estimated reference noise, and SNR.

## Files

All binary files are C-order arrays, with the first axis indexing global slice
slots. Multi-byte values are little-endian. Padding is storage only; the loader
returns the native region and active coils.

| File under `data/` | Dtype | Stored shape |
|---|---|---|
| `ksp.bin` | complex64 | `(19362,12,240,220)` |
| `mps.bin` | complex64 | `(19362,12,240,220)` |
| `gt.bin` (CG-SENSE reference images) | complex64 | `(19362,240,220)` |
| `eig.bin` | float16 | `(19362,240,220)` |
| `pe_mask.bin` | uint8, zero or one | `(362,220)` |
| `scans.parquet` | tabular | One row per scan |
| `slices.parquet` | tabular | One row per slice |
| `noise.npz` | numeric arrays | `scan<N>/{cov,W,U,noise_profile}` |

The repository also includes byte-identical copies of the release tables at
[`metadata/scans.parquet`](../metadata/scans.parquet) and
[`metadata/slices.parquet`](../metadata/slices.parquet), totaling 1.87 MB.
They can be inspected with pandas without downloading the numerical arrays.
The loader reads the copies under the extracted dataset's `data/` directory.

`stats.json` at the release root supplies `norm_const`, reconstruction settings,
and aggregate normalization/noise statistics. The eigenvalue arrays accompany
the ESPIRiT sensitivity maps. In `noise.npz`, `cov` is the input coil covariance,
`W` is the whitening transform, and `U` is the subsequent compression basis.
An empty `U` means no additional compression was needed. `noise_profile` records
the readout noise-estimation profile. These transforms act on the processing
stage's input coil basis.

## Addresses and geometry

- `scan_idx` and `stem = scan<N>` identify a scan.
- `subject_idx` groups scans from the same subject.
- `slice` identifies a zero-based slice within a scan.
- `slot` identifies its global binary row: `slot = slot0 + slice`.
- `slot0` and `n_slices` describe a scan's contiguous storage range.
- `mx`, `my` are the native readout/phase sizes; `row0`, `col0` locate them
  inside the storage canvas. `n_coils` selects the active coil channels.

For example, native k-space is
`ksp[slot, :n_coils, row0:row0+mx, col0:col0+my]`.
Spatial axes are readout (anterior to posterior) and phase (right to left).
In-plane voxel sizes are `voxel_x`, `voxel_y` in mm; FOV values are in mm.
`fov_x_mm` and `fov_y_mm` describe the processing input FOV. When a wider
readout FOV was cropped, the output readout FOV is `mx * voxel_x`; the output
phase FOV is `my * voxel_y`. The phase FOV is preserved by processing.
The nominal slice thickness is 3 mm. Each slice is an independent 2D acquisition.

The metadata includes numerical acquisition parameters (such as TE/TR, flip
angle, field strength, FOV, slice thickness/spacing, and coil counts), geometry,
split assignments, and numerical QC. TE/TR are in milliseconds, flip angles
are in degrees, and field strength is in tesla. The explicit release schema
is recorded in `manifests/metadata-v1.json`.

## Units and sampling

The acquisition mask contains acquired phase columns, including the central
calibration region. Nominal acceleration is two with a 25-line ACS region for
the usual acquisition; consult each scan's mask and `undersampled` flag.
The mask is one-dimensional and broadcasts across readout samples and coils.
Unacquired entries are exactly zero.

Processing includes noise whitening, coil compression to at most 12 channels,
and resolution cropping. The measurement noise convention is unit variance per
complex k-space sample before per-slice normalization. `scale` is the norm of
the central 24 phase lines over all readout samples and coils. Both references
and measurements are normalized by `norm_const / scale`. Maps and masks are
not rescaled. `sigma` is that normalized complex k-space noise standard deviation.
`gt_noise_std` estimates reference noise per real component in stored units.

Sensitivity maps are estimated with ESPIRiT (24 calibration samples per axis,
kernel width 6, eigenvalue crop 0.95, threshold 0.05 with a 0.02 fallback).
Their squared magnitudes sum to one on support. The complex reference uses
CG-SENSE; estimated reference noise includes the acquisition's conditioning.
The scalar noise estimate is calculated from independent noise realizations.
