"""Download, verify, and extract the published SLAM T2 release (Linux/macOS)."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 8 * 1024 * 1024


def verify_file(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Incorrect size: {path}")
    print(f"Verifying {path.name}", flush=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_dataset(root, files):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a dataset directory: {root}")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError(f"Unexpected symbolic link in {root}")
    actual = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    if actual != set(files):
        raise ValueError(f"Dataset file list differs from the release manifest: {root}")
    for name, expected in files.items():
        verify_file(root / name, expected)


def download_archive(destination, release):
    archive = destination / release["archive"]
    if archive.exists() or archive.is_symlink():
        verify_file(archive, release)
        return archive
    partial = archive.with_name(archive.name + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f"Expected a regular partial download: {partial}")
    size = partial.stat().st_size if partial.exists() else 0
    if size > release["bytes"]:
        raise ValueError(f"Partial download is too large; move it aside: {partial}")
    if size < release["bytes"]:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                release["url"],
            ],
            check=True,
        )
    verify_file(partial, release)
    os.link(partial, archive)
    partial.unlink()
    return archive


def extract_members(stream, staging, release_name, files):
    expected_members = {
        f"{release_name}/{name}": value for name, value in files.items()
    }
    seen = set()
    with tarfile.open(fileobj=stream, mode="r|", bufsize=BLOCK_SIZE) as archive:
        for member in archive:
            if member.name not in expected_members or member.name in seen:
                raise ValueError(
                    f"Unexpected or duplicate archive member: {member.name}"
                )
            expected = expected_members[member.name]
            if not member.isfile() or member.size != expected["bytes"]:
                raise ValueError(f"Invalid archive member: {member.name}")
            target = staging / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Extracting and verifying {member.name}", flush=True)
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, target.open("xb") as output:
                while block := source.read(BLOCK_SIZE):
                    output.write(block)
                    digest.update(block)
            if target.stat().st_size != expected["bytes"] or (
                digest.hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Corrupt archive member: {member.name}")
            seen.add(member.name)
    if seen != set(expected_members):
        raise ValueError("Archive is missing expected files")


def extract_archive(archive, destination, release, files):
    root = destination / release["release"]
    with tempfile.TemporaryDirectory(
        prefix=".slam-t2-extract-", dir=destination
    ) as tmp:
        with subprocess.Popen(
            ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE
        ) as proc:
            try:
                extract_members(proc.stdout, Path(tmp), release["release"], files)
                while proc.stdout.read(BLOCK_SIZE):
                    pass
                if proc.wait() != 0:
                    raise ValueError("zstd decompression failed")
            finally:
                proc.stdout.close()
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"Dataset destination already exists: {root}")
        (Path(tmp) / release["release"]).rename(root)
    return root


def setup(destination, release, files, verify_only=False):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".slam-t2-setup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another setup is using this destination") from exc
        root = destination / release["release"]
        if verify_only or root.exists() or root.is_symlink():
            verify_dataset(root, files)
            return root
        for command in ("curl", "zstd"):
            if shutil.which(command) is None:
                raise RuntimeError(f"Required command is not installed: {command}")
        archive = destination / release["archive"]
        partial = archive.with_name(archive.name + ".part")
        downloaded = archive.stat().st_size if archive.exists() else 0
        if not downloaded and partial.exists():
            downloaded = partial.stat().st_size
        needed = release["extracted_bytes"] + max(0, release["bytes"] - downloaded)
        if shutil.disk_usage(destination).free < needed:
            raise RuntimeError(
                f"Insufficient free disk space; need {needed / 1e9:.2f} GB"
            )
        archive = download_archive(destination, release)
        return extract_archive(archive, destination, release, files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--verify-only", action="store_true", help="Check all extracted file hashes"
    )
    args = parser.parse_args(argv)
    release = json.loads((REPO / "manifests/release-v1.json").read_text())
    files = json.loads((REPO / "manifests/data-v1.json").read_text())["files"]
    try:
        root = setup(
            args.destination.expanduser().resolve(), release, files, args.verify_only
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as exc:
        parser.exit(1, f"Setup failed: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Setup interrupted; rerun the same command to resume.\n")
    print(f"Dataset verified and ready: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
