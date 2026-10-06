import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_matches_release_checksums_and_schema():
    manifest = json.loads((ROOT / "manifests/data-v1.json").read_text())
    schemas = json.loads((ROOT / "manifests/metadata-v1.json").read_text())["tables"]
    for name in ("scans", "slices"):
        path = ROOT / "metadata" / f"{name}.parquet"
        key = f"data/{name}.parquet"
        assert path.stat().st_size == manifest["files"][key]["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            manifest["files"][key]["sha256"]
        )
        schema = pq.read_schema(path)
        assert schema.metadata is None
        assert all(field.metadata is None for field in schema)
        assert {field.name: str(field.type) for field in schema} == schemas[key]
        table = pd.read_parquet(path)
        assert set(table["split"]) == {"train", "test"}
        assert (table.stem == "scan" + table.scan_idx.astype(int).astype(str)).all()
        for field in table:
            if field not in {"stem", "split"}:
                assert pd.api.types.is_numeric_dtype(table[field])


def test_catalog_addresses_and_subject_partitions():
    scans = pd.read_parquet(ROOT / "metadata/scans.parquet").set_index("scan_idx")
    slices = pd.read_parquet(ROOT / "metadata/slices.parquet").sort_values("slot")
    assert len(scans) == 362 and len(slices) == 19362
    np.testing.assert_array_equal(slices.slot, np.arange(len(slices)))
    np.testing.assert_array_equal(
        slices.slot, slices.scan_idx.map(scans.slot0) + slices["slice"]
    )
    for field in ("subject_idx", "split", "mx", "my", "row0", "col0", "n_coils"):
        np.testing.assert_array_equal(slices[field], slices.scan_idx.map(scans[field]))
    train = set(scans.query("split == 'train'").subject_idx)
    test = set(scans.query("split == 'test'").subject_idx)
    assert not train & test
    assert len(slices.query("split == 'train' and usable")) == 15288
