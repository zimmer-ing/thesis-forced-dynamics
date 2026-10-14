from __future__ import annotations

import json
import hashlib
import re
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Constants as const

RAW_ROOT = const.RAW_DATA_PATH / "NonlinearBenchmarks/_repos/nanodrone-sysid-benchmark"
SOURCE_COMMIT = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
SOURCE_URL = f"https://raw.githubusercontent.com/idsia-robotics/nanodrone-sysid-benchmark/{SOURCE_COMMIT}/data"
LFS_BATCH_URL = "https://github.com/idsia-robotics/nanodrone-sysid-benchmark.git/info/lfs/objects/batch"
POINTER_RE = re.compile(r"oid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)")
EXPECTED_FILES = tuple(
    f"train/{trajectory}_20251017_run{run}.csv"
    for trajectory in ("random", "square", "chirp") for run in (1, 2, 3, 4)
) + tuple(f"test/melon_20251017_run{run}.csv" for run in (1, 2, 3))


def expected_paths(data_root: Path | None = None) -> list[Path]:
    data_root = RAW_ROOT / "data" if data_root is None else Path(data_root)
    return [data_root / relative for relative in EXPECTED_FILES]


def parse_pointer(text: str) -> tuple[str, int] | None:
    if "https://git-lfs.github.com/spec/v1" not in text:
        return None
    match = POINTER_RE.search(text)
    if not match or int(match.group(2)) <= 0:
        raise ValueError("Could not parse LFS pointer")
    return match.group(1), int(match.group(2))


def read_pointer(path: Path) -> tuple[str, int] | None:
    with path.open("rb") as handle:
        text = handle.read(1024).decode("utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"Empty Nano-Drone CSV: {path}")
    return parse_pointer(text)


def download_pointer(relative: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    with urllib.request.urlopen(f"{SOURCE_URL}/{relative}", timeout=60) as response:
        content = response.read(1024)
    if len(content) == 1024 or parse_pointer(content.decode("utf-8")) is None:
        raise ValueError(f"Expected a Git-LFS pointer for {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    try:
        tmp.write_bytes(content)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def lfs_batch(objects: list[dict[str, int | str]]) -> dict:
    payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": objects,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        LFS_BATCH_URL,
        data=payload,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, path: Path, expected_size: int, expected_sha256: str,
             headers: dict[str, str] | None = None) -> None:
    tmp = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers=headers or {})
    digest = hashlib.sha256()
    actual_size = 0
    try:
        with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                actual_size += len(chunk)
        if actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {path}: {actual_size} != {expected_size}")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch for {path}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    data_root = RAW_ROOT / "data"
    pointers: dict[str, list[tuple[Path, int]]] = {}
    for relative, path in zip(EXPECTED_FILES, expected_paths(data_root)):
        if not path.exists():
            download_pointer(relative, path)
        pointer = read_pointer(path)
        if pointer is None:
            print(f"already-real {path}")
            continue
        oid, size = pointer
        if oid in pointers and pointers[oid][0][1] != size:
            raise ValueError(f"Conflicting sizes for LFS object {oid}")
        pointers.setdefault(oid, []).append((path, size))

    if not pointers:
        print("All 15 Nano-Drone CSV files are present; no LFS pointers remain.")
        return

    objects = [{"oid": oid, "size": files[0][1]} for oid, files in pointers.items()]
    print(f"Requesting {len(objects)} LFS objects.")
    batch = lfs_batch(objects)
    received = batch.get("objects", [])
    returned_oids = [obj.get("oid") for obj in received]
    if len(returned_oids) != len(pointers) or set(returned_oids) != set(pointers):
        raise RuntimeError("The LFS response did not contain exactly the requested objects")
    for obj in received:
        oid = obj["oid"]
        expected_size = pointers[oid][0][1]
        error = obj.get("error")
        if error:
            raise RuntimeError(f"LFS error for {oid}: {error}")
        if obj.get("size") != expected_size or not obj.get("actions", {}).get("download", {}).get("href"):
            raise RuntimeError(f"Invalid LFS download metadata for {oid}")

    for obj in received:
        action = obj["actions"]["download"]
        for path, expected_size in pointers[obj["oid"]]:
            print(f"download {path.name} ({expected_size} bytes)")
            download(action["href"], path, expected_size, obj["oid"], action.get("header"))

    print("Downloaded all Nano-Drone CSV LFS objects.")


if __name__ == "__main__":
    main()
