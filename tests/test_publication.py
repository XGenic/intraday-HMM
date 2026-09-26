"""The public exporter must keep private working material out of distributions."""

import hashlib
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_public_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_public_snapshot", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_public_archive_excludes_private_material_and_is_reproducible(tmp_path):
    public = set(exporter.ROOT_FILES) | {
        "src/example.py",
        "tests/test_example.py",
        "scripts/example.py",
        "configs/example.yaml",
        "docs/results/table.csv",
        "docs/README.md",
        ".github/workflows/ci.yml",
    }
    private = {
        ".env",
        ".env.production",
        "working-notes.md",
        "data/input.parquet",
        "runs/report.html",
        ".local/notes.md",
        ".venv/config.py",
        "src/__pycache__/example.pyc",
        "docs/.private/notes.md",
        "docs/private.key",
        "src/.env",
        "scripts/__pycache__/hidden.py",
    }
    for name in public | private:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRIVATE_SENTINEL" if name in private else name, encoding="utf-8")

    output = exporter.build(tmp_path)
    first = output.read_bytes()
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {"intraday-HMM/" + name for name in public}
        assert all(b"PRIVATE_SENTINEL" not in archive.read(name) for name in archive.namelist())
    assert (
        output.with_suffix(".zip.sha256").read_text().split()[0]
        == hashlib.sha256(first).hexdigest()
    )
    assert exporter.build(tmp_path).read_bytes() == first


def test_public_archive_requires_core_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        exporter.build(tmp_path)
    assert not (tmp_path / "dist").exists()
