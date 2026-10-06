import fcntl
import hashlib
import importlib.util
import io
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "setup_data", Path(__file__).resolve().parents[1] / "scripts/setup_data.py"
)
setup_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_data)


def metadata(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def tar_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data)
            if kind == tarfile.SYMTYPE:
                info.linkname = "/outside"
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def release_fixture(tmp_path):
    payload = b"real file fixture\x00\x01"
    files = {"data/example.bin": metadata(payload)}
    raw = tar_bytes([("example/data/example.bin", payload, tarfile.REGTYPE)])
    archive = subprocess.run(
        ["zstd", "-q", "-c"], input=raw, capture_output=True, check=True
    ).stdout
    release = {
        "release": "example",
        "archive": "example.tar.zst",
        "url": "https://example.invalid/example.tar.zst",
        "extracted_bytes": len(payload),
        **metadata(archive),
    }
    return payload, files, archive, release


def test_setup_resumes_download_and_verifies_existing(
    tmp_path, monkeypatch, release_fixture
):
    payload, files, archive, release = release_fixture
    partial = tmp_path / "example.tar.zst.part"
    partial.write_bytes(archive[:10])
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        assert command[command.index("--continue-at") + 1] == "-"
        assert partial.read_bytes() == archive[:10]
        with partial.open("ab") as output:
            output.write(archive[10:])

    monkeypatch.setattr(setup_data.subprocess, "run", download)
    root = setup_data.setup(tmp_path, release, files)
    assert (root / "data/example.bin").read_bytes() == payload
    assert (tmp_path / "example.tar.zst").read_bytes() == archive
    assert not partial.exists()
    assert setup_data.setup(tmp_path, release, files) == root
    assert len(calls) == 1
    assert not list(tmp_path.glob(".slam-t2-extract-*"))


def test_complete_partial_needs_no_network(tmp_path, monkeypatch, release_fixture):
    _, files, archive, release = release_fixture
    (tmp_path / "example.tar.zst.part").write_bytes(archive)

    def no_network(*args, **kwargs):
        pytest.fail("Unexpected network request")

    monkeypatch.setattr(setup_data.subprocess, "run", no_network)
    root = setup_data.setup(tmp_path, release, files)
    setup_data.verify_dataset(root, files)


def test_failed_download_is_resumable(tmp_path, monkeypatch, release_fixture):
    _, files, archive, release = release_fixture

    def interrupted(command, **kwargs):
        (tmp_path / "example.tar.zst.part").write_bytes(archive[:10])
        raise subprocess.CalledProcessError(18, command)

    monkeypatch.setattr(setup_data.subprocess, "run", interrupted)
    with pytest.raises(subprocess.CalledProcessError):
        setup_data.setup(tmp_path, release, files)
    assert (tmp_path / "example.tar.zst.part").read_bytes() == archive[:10]
    assert not (tmp_path / "example").exists()
    assert not (tmp_path / "example.tar.zst").exists()


@pytest.mark.parametrize("filename", ["example.tar.zst", "example.tar.zst.part"])
def test_corrupt_archive_not_extracted(tmp_path, release_fixture, filename):
    _, files, archive, release = release_fixture
    damaged = bytes([archive[0] ^ 1]) + archive[1:]
    (tmp_path / filename).write_bytes(damaged)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup_data.setup(tmp_path, release, files)
    assert not (tmp_path / "example").exists()
    assert (tmp_path / filename).read_bytes() == damaged


@pytest.mark.parametrize(
    "members, error",
    [
        ([("../escape", b"abc", tarfile.REGTYPE)], "Unexpected"),
        ([("/absolute", b"abc", tarfile.REGTYPE)], "Unexpected"),
        ([("example/data/example.bin", b"", tarfile.SYMTYPE)], "Invalid"),
        ([("example/data/example.bin", b"abc", tarfile.REGTYPE)] * 2, "duplicate"),
        ([], "missing"),
        ([("example/data/example.bin", b"ab", tarfile.REGTYPE)], "Invalid"),
        ([("example/data/example.bin", b"bad", tarfile.REGTYPE)], "Corrupt"),
    ],
)
def test_rejects_invalid_archive_members(tmp_path, members, error):
    with pytest.raises(ValueError, match=error):
        setup_data.extract_members(
            io.BytesIO(tar_bytes(members)),
            tmp_path,
            "example",
            {"data/example.bin": metadata(b"abc")},
        )


def test_failed_extraction_cleans_staging(tmp_path, release_fixture):
    _, files, archive, release = release_fixture
    path = tmp_path / "example.tar.zst"
    path.write_bytes(archive)
    files["missing"] = metadata(b"")
    with pytest.raises(ValueError, match="missing"):
        setup_data.extract_archive(path, tmp_path, release, files)
    assert not (tmp_path / "example").exists()
    assert not list(tmp_path.glob(".slam-t2-extract-*"))


def test_decompression_failure_cleans_staging(tmp_path, release_fixture):
    _, files, archive, release = release_fixture
    path = tmp_path / "example.tar.zst"
    path.write_bytes(archive[:-4])
    with pytest.raises((ValueError, tarfile.TarError)):
        setup_data.extract_archive(path, tmp_path, release, files)
    assert not (tmp_path / "example").exists()
    assert not list(tmp_path.glob(".slam-t2-extract-*"))


def test_existing_invalid_dataset_is_preserved(tmp_path, release_fixture):
    _, files, _, release = release_fixture
    root = tmp_path / "example"
    root.mkdir()
    (root / "personal.txt").write_text("keep")
    with pytest.raises(ValueError, match="file list"):
        setup_data.setup(tmp_path, release, files)
    assert (root / "personal.txt").read_text() == "keep"


def test_verify_only_missing_dataset_never_downloads(tmp_path, release_fixture):
    _, files, _, release = release_fixture
    with pytest.raises(ValueError, match="dataset directory"):
        setup_data.setup(tmp_path, release, files, verify_only=True)


def test_concurrent_setup_refused(tmp_path, release_fixture):
    _, files, _, release = release_fixture
    with (tmp_path / ".slam-t2-setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another setup"):
            setup_data.setup(tmp_path, release, files)


def test_symlink_dataset_refused(tmp_path, release_fixture):
    _, files, _, release = release_fixture
    (tmp_path / "example").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="dataset directory"):
        setup_data.setup(tmp_path, release, files)


def test_insufficient_space_stops_before_download(
    tmp_path, monkeypatch, release_fixture
):
    _, files, _, release = release_fixture
    monkeypatch.setattr(
        setup_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    with pytest.raises(RuntimeError, match="Insufficient free disk space"):
        setup_data.setup(tmp_path, release, files)
    assert not (tmp_path / "example.tar.zst.part").exists()


def test_verify_rejects_changed_extracted_data(tmp_path, release_fixture):
    payload, files, archive, release = release_fixture
    (tmp_path / "example.tar.zst").write_bytes(archive)
    root = setup_data.setup(tmp_path, release, files)
    (root / "data/example.bin").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup_data.setup(tmp_path, release, files, verify_only=True)
