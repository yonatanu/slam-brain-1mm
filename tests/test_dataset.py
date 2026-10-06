import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slam_t2_data import SlamT2
from slam_t2_data.dataset import LAYOUT, SLOT


@pytest.fixture
def release(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    scans = pd.DataFrame(
        [
            dict(
                scan_idx=0,
                slot0=0,
                n_slices=2,
                n_coils=2,
                mx=4,
                my=6,
                row0=118,
                col0=107,
                subject_idx=0,
                stem="scan0",
            ),
            dict(
                scan_idx=1,
                slot0=2,
                n_slices=1,
                n_coils=3,
                mx=6,
                my=4,
                row0=117,
                col0=108,
                subject_idx=1,
                stem="scan1",
            ),
        ]
    )
    slices = pd.DataFrame(
        [
            dict(
                slot=0,
                scan_idx=0,
                slice=0,
                scale=2.0,
                gt_noise_std=1.0,
                split="train",
                usable=True,
            ),
            dict(
                slot=1,
                scan_idx=0,
                slice=1,
                scale=3.0,
                gt_noise_std=1.0,
                split="train",
                usable=False,
            ),
            dict(
                slot=2,
                scan_idx=1,
                slice=0,
                scale=4.0,
                gt_noise_std=1.2,
                split="test",
                usable=True,
            ),
        ]
    )
    scans["split"] = ["train", "test"]
    for field in ("n_coils", "mx", "my", "row0", "col0", "subject_idx"):
        slices[field] = slices.scan_idx.map(scans.set_index("scan_idx")[field])
    scans.to_parquet(folder / "scans.parquet", index=False)
    slices.iloc[::-1].to_parquet(folder / "slices.parquet", index=False)
    for name, (shape, dtype) in LAYOUT.items():
        values = np.zeros((3, *shape), dtype=dtype)
        for row in scans.itertuples():
            for slot in range(row.slot0, row.slot0 + row.n_slices):
                crop = (
                    slice(row.row0, row.row0 + row.mx),
                    slice(row.col0, row.col0 + row.my),
                )
                if name in {"ksp", "mps"}:
                    values[(slot, slice(0, row.n_coils), *crop)] = slot + 1
                else:
                    values[(slot, *crop)] = slot + 1
        values.tofile(folder / f"{name}.bin")
    mask = np.zeros((2, SLOT[1]), dtype=np.uint8)
    for row in scans.itertuples():
        mask[row.scan_idx, row.col0 : row.col0 + row.my : 2] = 1
    mask.tofile(folder / "pe_mask.bin")
    noise = {
        f"scan{i}/{key}": np.eye(2) * (i + 1)
        for i in range(2)
        for key in ["cov", "W", "U", "noise_profile"]
    }
    np.savez(folder / "noise.npz", **noise)
    (tmp_path / "stats.json").write_text(json.dumps({"norm_const": 6.0}))
    return tmp_path


def test_native_slices_volumes_and_normalization(release):
    ds = SlamT2(release)
    assert len(ds) == 3
    assert ds[0]["ksp"].shape == (2, 4, 6)
    assert ds[2]["gt"].shape == (6, 4)
    assert ds.volume(0)["ksp"].shape == (2, 2, 4, 6)
    np.testing.assert_array_equal(ds.volume(0)["slots"], [0, 1])
    np.testing.assert_array_equal(ds[0]["mask"], [1, 0, 1, 0, 1, 0])
    assert ds.select("split == 'train' and usable").tolist() == [0]
    item = ds[0]
    np.testing.assert_array_equal(ds.normalize(item["gt"], item), 3)
    np.testing.assert_array_equal(
        ds.denormalize(ds.normalize(item["gt"], item), item), item["gt"]
    )
    assert item["sigma"] == 3 and not item["ksp"].flags.writeable
    np.testing.assert_array_equal(ds.noise(1)["W"], np.eye(2) * 2)
    with pytest.raises(TypeError):
        ds[True]
    with pytest.raises(IndexError):
        ds.volume(-1)
    for slot in (-1, len(ds)):
        with pytest.raises(IndexError, match="slot"):
            ds[slot]


def test_gt_in_memory_is_an_actual_copy(release):
    ds = SlamT2(release, gt_in_memory=True)
    ds[0]["gt"][0, 0] = 9
    assert SlamT2(release)[0]["gt"][0, 0] == 1


def test_reconstruction_cli_and_compatibility_import(release):
    output = release / "reconstruction.npz"
    command = [
        sys.executable,
        "-m",
        "slam_t2_data.reconstruct",
        "--root",
        str(release),
        "--slot",
        "0",
        "--output",
        str(output),
        "--threads",
        "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["normal_residual"] < 1e-5
    with np.load(output, allow_pickle=False) as data:
        assert data["reconstruction"].shape == (4, 6)
        assert data["reconstruction"].dtype == np.complex64
        assert data["slot"] == 0
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and "output already exists" in repeated.stderr
    code = str(Path(__file__).resolve().parents[1] / "code")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from slam_t2 import SlamT2; "
            "assert SlamT2(sys.argv[2])[0]['gt'].shape == (4, 6)",
            code,
            str(release),
        ],
        check=True,
    )


@pytest.mark.parametrize("failure", ["truncated", "duplicate", "slice", "geometry"])
def test_corrupt_storage_is_rejected(release, failure):
    if failure == "truncated":
        with (release / "data/gt.bin").open("r+b") as stream:
            stream.truncate(1)
    elif failure in {"duplicate", "slice"}:
        path = release / "data/slices.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "slot" if failure == "duplicate" else "slice"] = 1
        table.to_parquet(path, index=False)
    else:
        path = release / "data/scans.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "row0"] = 0
        table.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        SlamT2(release)


@pytest.mark.parametrize(
    "field", ["n_coils", "mx", "my", "row0", "col0", "subject_idx", "split"]
)
def test_slice_scan_disagreement_is_rejected(release, field):
    path = release / "data/slices.parquet"
    table = pd.read_parquet(path)
    table.loc[0, field] = "other" if field == "split" else table.loc[0, field] + 1
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=f"slice {field}"):
        SlamT2(release)


@pytest.mark.parametrize("failure", ["nonbinary", "empty", "padding"])
def test_invalid_acquisition_masks_are_rejected(release, failure):
    path = release / "data/pe_mask.bin"
    mask = np.fromfile(path, dtype=np.uint8).reshape(2, SLOT[1])
    if failure == "empty":
        mask[0] = 0
    elif failure == "padding":
        mask[0, 0] = 1
    else:
        mask[0, 107] = 2
    mask.tofile(path)
    with pytest.raises(ValueError, match="mask|acquired"):
        SlamT2(release)


def test_cli_help_without_data(monkeypatch, capsys, tmp_path):
    from slam_t2_data.reconstruct import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["slam-t2-reconstruct", "--help"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    assert "--root" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "option,value",
    [
        ("slot", "-1"),
        ("iters", "0"),
        ("threads", "0"),
        ("tol", "nan"),
        ("lam", "inf"),
        ("lam", "-1"),
        ("tol", "0"),
    ],
)
def test_cli_rejects_invalid_arguments_before_reading_data(
    monkeypatch, capsys, tmp_path, option, value
):
    from slam_t2_data.reconstruct import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam-t2-reconstruct",
            "--root",
            str(tmp_path / "missing"),
            "--slot",
            "0",
            "--output",
            str(tmp_path / "result.npz"),
            f"--{option}",
            value,
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert option in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_failed_write_never_publishes_partial_output(tmp_path, monkeypatch):
    from slam_t2_data.reconstruct import _save_npz

    target = tmp_path / "result.npz"

    def fail(stream, **arrays):
        stream.write(b"partial")
        raise OSError("write failed")

    with monkeypatch.context() as patch:
        patch.setattr(np, "savez", fail)
        with pytest.raises(OSError, match="write failed"):
            _save_npz(target, reconstruction=np.ones((2, 2)))
    assert not list(tmp_path.iterdir())
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _save_npz(target, reconstruction=np.ones((2, 2)))
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_cli_reports_unconverged_result(release, monkeypatch, capsys):
    from slam_t2_data.reconstruct import main

    maps = np.memmap(
        release / "data/mps.bin", mode="r+", dtype="<c8", shape=(3, 12, *SLOT)
    )
    rng = np.random.default_rng(11)
    maps[0, :2, 118:122, 107:113] = rng.standard_normal(
        (2, 4, 6)
    ) + 1j * rng.standard_normal((2, 4, 6))
    maps.flush()
    target = release / "unconverged.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam-t2-reconstruct",
            "--root",
            str(release),
            "--slot",
            "0",
            "--output",
            str(target),
            "--iters",
            "1",
            "--tol",
            "1e-12",
            "--threads",
            "1",
        ],
    )
    assert main() == 1
    assert not json.loads(capsys.readouterr().out)["converged"]
    with np.load(target, allow_pickle=False) as result:
        assert not result["converged"]
        assert result["normal_residual"].item() > 1e-12
        assert result["iterations"].item() == 1
        assert result["numpy_version"].item() == np.__version__
