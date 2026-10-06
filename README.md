# SLAM T2 data

SLAM T2 contains processed axial T2 brain MRI measurements and complex-valued
CG-SENSE reference images at approximately 1 mm in-plane resolution. The release
includes 362 scans and 19,362 slices from 343 subjects, with multicoil k-space,
sensitivity maps, acquisition masks, and acquisition/QC metadata.

The measurements retain their acquired undersampling pattern. They are whitened,
coil-compressed, and cropped in resolution. The included reference images were
reconstructed from these measurements with CG-SENSE. This repository provides
a data loader and code for running CG-SENSE reconstruction. See
[data conventions](docs/data.md) and [reconstruction](docs/reconstruction.md).

## Inspect the metadata

The release tables are included here so you can inspect acquisitions,
slice identifiers, split assignments, and QC without downloading the arrays:

| Table | Rows | Size |
|---|---:|---:|
| [metadata/scans.parquet](metadata/scans.parquet) | 362 | 54 KB |
| [metadata/slices.parquet](metadata/slices.parquet) | 19,362 | 1.82 MB |

After installation, read them from the repository root:

```python
import pandas as pd

scans = pd.read_parquet("metadata/scans.parquet")
slices = pd.read_parquet("metadata/slices.parquet")
usable_train = slices.query("split == 'train' and usable")
```

These are byte-identical copies of `data/scans.parquet` and `data/slices.parquet`
in `slam_t2_1mm_v1.tar.zst`, with hashes recorded in `manifests/data-v1.json`.
The archive includes them so the extracted dataset is self-contained. Their
column types are documented in [the metadata schema](manifests/metadata-v1.json).

## Install

```bash
mamba env create -f environment.yml
mamba activate slam-t2
```

The environment installs the package in editable mode and includes development
tools. For an existing Python 3.12+ environment, use `python -m pip install .`.

The examples run on CPU. GPU reconstruction requires a CUDA-enabled PyTorch
installation and an available CUDA device.

## Download and verify

The dataset is hosted by the [Stanford Digital Repository](https://purl.stanford.edu/tz884qn8750)
(DOI: [10.25740/tz884qn8750](https://doi.org/10.25740/tz884qn8750)), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Download the [85.13 GB archive](https://stacks.stanford.edu/file/tz884qn8750/slam_t2_1mm_v1.tar.zst)
using the setup script below.

From this repository's root, after activating the environment above, run:

```bash
python scripts/setup_data.py --destination /path/to/datasets
```

This Linux/macOS script requires Python 3.12+, `curl`, and `zstd` on your PATH;
`zstd` is included in `environment.yml`.
It downloads and verifies the archive, then extracts and verifies the dataset at
`/path/to/datasets/slam_t2_1mm_v1`. The downloaded archive is retained.

Rerun the same command after an interruption: downloads resume from the `.part`
file; extraction restarts. An existing dataset is verified without downloading
again. If a checksum fails, move the reported corrupt file or dataset directory
aside before retrying. To verify an existing dataset:

```bash
python scripts/setup_data.py --destination /path/to/datasets --verify-only
```

Allow approximately **292 GB of free space** for the archive and extracted data
(85.13 GB and 206.53 GB, respectively), plus room for reconstruction outputs.
Download, extraction, and verification can take tens of minutes, depending on
network and disk speed. Verification reads the complete data. After a successful
setup, you can remove the archive if you only need the extracted data.

For a manual download, run the following from the directory where you want to
store the data. These commands require `curl`, `zstd`, GNU `tar`, and `sha256sum`
(on macOS, use `shasum -a 256 -c` instead of `sha256sum -c`):

```bash
curl --fail --location --retry 5 --continue-at - \
  --output slam_t2_1mm_v1.tar.zst \
  https://stacks.stanford.edu/file/tz884qn8750/slam_t2_1mm_v1.tar.zst
sha256sum -c /path/to/this/repository/manifests/archive-v1.sha256
tar --zstd -xf slam_t2_1mm_v1.tar.zst
cd slam_t2_1mm_v1
sha256sum -c /path/to/this/repository/manifests/data-v1.sha256
```

Continue to extraction only if the archive checksum passes. Extraction produces
`stats.json` and a `data/` directory under `slam_t2_1mm_v1`. Pass that extracted
root to the loader. Exact sizes, the download URL, and SHA-256 are recorded in
`manifests/release-v1.json`. Code and checksum manifests are distributed in this
repository; the data archive contains the numerical data and metadata.

## Load slices and scans

```python
from slam_t2_data import SlamT2

ds = SlamT2("/path/to/slam_t2_1mm_v1")
slots = ds.select("split == 'train' and usable")
item = ds[int(slots[0])]
ksp, maps, mask = item["ksp"], item["mps"], item["mask"]
reference = item["gt"]
normalized_reference = ds.normalize(reference, item)
normalized_kspace = ds.normalize(ksp, item)
scan = ds.volume(int(item["scan_row"].scan_idx))
noise = ds.noise(int(item["scan_row"].scan_idx))
```

Arrays are NumPy memory-map views in native geometry: k-space/maps have shape
`(coils, readout, phase)` and references have shape `(readout, phase)`. Coil counts
and phase widths can differ across scans. Most references are `240 x 192`.
Normalization is explicit and uses the same factor for k-space and references.

`code/slam_t2.py` also supports `from slam_t2 import SlamT2` when that directory
is on the Python import path.

## Reconstruction example

Reference images are already included in `gt.bin`. To run CG-SENSE on a slice:

```bash
slam-t2-reconstruct --root /path/to/slam_t2_1mm_v1 \
  --slot 20 --output outputs/slot20.npz
```

This reads the released k-space, maps, and acquisition mask and writes a complex
CG-SENSE image. It reports the normal-equation residual and the difference from
the stored reference. Outputs record solver parameters, library versions, device,
thread count, and convergence. Use a new output path. Nonconvergence saves the
result for inspection and returns exit status 1. Invalid arguments return status 2.

The default solver uses zero regularization, at most 40 iterations, and relative
normal-equation tolerance `1e-5`. Results can differ slightly across devices and
library versions. Retrospective undersampling operates within the released
acquisition mask.

## Development checks

```bash
python -m pip install -e '.[dev]'
pre-commit install
python -m pytest
pre-commit run --all-files
python -m build
```

Tests run on CPU and cover indexing, storage validation,
Fourier conventions, complex adjoints, and CG solutions. Catalog checks compare
the public tables with the release checksums, schema, and subject partitions.

## Citation

If you use SLAM T2 in your research, please cite:

Urman Y, Shah Z, Kumar A, Soares BP, Setsompop K.
[Accelerating MRI With Longitudinally-Informed Latent Posterior Sampling](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70257).
*Magnetic Resonance in Medicine*. 2026;95(6):3445–3461.
