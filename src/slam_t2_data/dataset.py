"""Read processed arrays in their native geometry and stored units."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SLOT = (240, 220)
MAX_COILS = 12
LAYOUT = {
    "ksp": ((MAX_COILS, *SLOT), np.dtype("<c8")),
    "mps": ((MAX_COILS, *SLOT), np.dtype("<c8")),
    "gt": (SLOT, np.dtype("<c8")),
    "eig": (SLOT, np.dtype("<f2")),
}


def _memmap(path, shape, dtype):
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Binary size does not match the tables: {path.name}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def _index(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("index must be an integer")
    return int(value)


class SlamT2:
    """Open a release root containing ``data/`` and ``stats.json``.

    ``ds[slot]`` returns one slice; ``volume(scan_idx)`` returns a scan.
    Arrays are read-only memory-map views unless ``gt_in_memory`` is enabled.
    Spatial axes are readout, phase; normalization is explicit.
    """

    def __init__(self, root, gt_in_memory=False):
        self.root = root
        data = Path(root) / "data"
        self.scans = (
            pd.read_parquet(data / "scans.parquet")
            .sort_values("scan_idx")
            .reset_index(drop=True)
        )
        self.slices = pd.read_parquet(data / "slices.parquet").set_index(
            "slot", drop=False
        )
        self.stats = json.loads((Path(root) / "stats.json").read_text())
        self.norm_const = float(self.stats["norm_const"])
        self._validate_tables()
        n = len(self.slices)
        self._arrays = {
            name: _memmap(data / f"{name}.bin", (n, *shape), dtype)
            for name, (shape, dtype) in LAYOUT.items()
        }
        self.pe_mask = _memmap(
            data / "pe_mask.bin", (len(self.scans), SLOT[1]), np.uint8
        )
        if not np.isin(self.pe_mask, [0, 1]).all():
            raise ValueError("acquisition masks must contain only zero and one")
        for scan in self.scans.itertuples():
            start, stop = int(scan.col0), int(scan.col0 + scan.my)
            mask = self.pe_mask[int(scan.scan_idx)]
            if mask[:start].any() or mask[stop:].any():
                raise ValueError("acquisition mask extends into storage padding")
            if not mask[start:stop].any():
                raise ValueError("each scan must contain acquired phase columns")
        if gt_in_memory:
            self._arrays["gt"] = np.array(self._arrays["gt"], copy=True)

    def _validate_tables(self):
        scans, slices = self.scans, self.slices
        fields = [
            "scan_idx",
            "slot0",
            "n_slices",
            "n_coils",
            "mx",
            "my",
            "row0",
            "col0",
        ]
        values = scans[fields].to_numpy(dtype=float)
        if scans.empty or not np.isfinite(values).all():
            raise ValueError("scan dimensions must be nonempty and finite")
        if not (values == np.floor(values)).all():
            raise ValueError("scan dimensions must be integers")
        if not np.array_equal(scans.scan_idx, np.arange(len(scans))):
            raise ValueError("scan indices must be contiguous and start at zero")
        if (scans.n_slices <= 0).any() or not np.array_equal(
            scans.slot0, scans.n_slices.cumsum().shift(fill_value=0)
        ):
            raise ValueError("scan slice ranges must be contiguous and nonempty")
        if not scans.n_coils.between(1, MAX_COILS).all():
            raise ValueError("invalid coil count")
        for width, offset, total in [("mx", "row0", SLOT[0]), ("my", "col0", SLOT[1])]:
            if (
                not scans[width].between(1, total).all()
                or not (scans[offset] == total // 2 - scans[width] // 2).all()
            ):
                raise ValueError("invalid native storage geometry")
        n = int(scans.n_slices.sum())
        if not slices.index.is_unique or not np.array_equal(
            np.sort(slices.index), np.arange(n)
        ):
            raise ValueError("slice slots must cover the full storage layout once")
        expected_scans = np.repeat(scans.scan_idx, scans.n_slices.astype(int))
        ordered = slices.sort_index()
        if not np.array_equal(ordered.scan_idx, expected_scans):
            raise ValueError("slice scan indices do not match storage ranges")
        expected_slices = np.arange(n) - np.repeat(
            scans.slot0, scans.n_slices.astype(int)
        )
        if not np.array_equal(ordered["slice"], expected_slices):
            raise ValueError("slice indices do not match storage ranges")
        for field in ("n_coils", "mx", "my", "row0", "col0", "subject_idx", "split"):
            if field in scans and field in slices:
                expected = np.repeat(scans[field], scans.n_slices.astype(int))
                if not np.array_equal(ordered[field], expected):
                    raise ValueError(
                        f"slice {field} values do not match the scan table"
                    )
        if not np.isfinite(self.norm_const) or self.norm_const <= 0:
            raise ValueError("normalization constant must be finite and positive")
        if not np.isfinite(slices.scale).all() or (slices.scale <= 0).any():
            raise ValueError("slice scales must be finite and positive")

    def __len__(self):
        return len(self.slices)

    def _scan(self, scan_idx):
        scan_idx = _index(scan_idx)
        if not 0 <= scan_idx < len(self.scans):
            raise IndexError("scan index out of range")
        return self.scans.iloc[scan_idx]

    def _views(self, scan, slots):
        r, c = int(scan.row0), int(scan.col0)
        h, w, nc = int(scan.mx), int(scan.my), int(scan.n_coils)
        rows, cols = slice(r, r + h), slice(c, c + w)
        return dict(
            ksp=self._arrays["ksp"][slots, :nc, rows, cols],
            mps=self._arrays["mps"][slots, :nc, rows, cols],
            gt=self._arrays["gt"][slots, rows, cols],
            eig=self._arrays["eig"][slots, rows, cols],
            mask=self.pe_mask[int(scan.scan_idx), cols].astype(bool),
            scan_row=scan,
        )

    def __getitem__(self, slot):
        slot = _index(slot)
        if not 0 <= slot < len(self):
            raise IndexError("slice slot out of range")
        row = self.slices.loc[slot]
        result = self._views(self._scan(int(row.scan_idx)), slot)
        result.update(
            slot=slot,
            scale=float(row.scale),
            sigma=self.norm_const / float(row.scale),
            gt_noise_std=float(row.gt_noise_std),
            slice_row=row,
        )
        return result

    def volume(self, scan_idx):
        scan = self._scan(scan_idx)
        start, n = int(scan.slot0), int(scan.n_slices)
        result = self._views(scan, slice(start, start + n))
        result["slots"] = np.arange(start, start + n)
        return result

    def noise(self, scan_idx):
        scan = self._scan(scan_idx)
        prefix = f"scan{int(scan.scan_idx)}"
        with np.load(Path(self.root) / "data/noise.npz", allow_pickle=False) as archive:
            return {
                name: archive[f"{prefix}/{name}"]
                for name in ("cov", "W", "U", "noise_profile")
            }

    def select(self, query):
        """Return global slice slots matching a pandas query over slice metadata."""
        return self.slices.query(query).slot.to_numpy()

    def normalize(self, x, item):
        """Scale a reference or k-space array into normalized units."""
        return x * (self.norm_const / item["scale"])

    def denormalize(self, x, item):
        return x * (item["scale"] / self.norm_const)
