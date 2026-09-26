"""Build a source-only public ZIP from an explicit allowlist, without using Git."""

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    ".env.example",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "uv.lock",
    "experiment-plan.md",
    "experiment-results.md",
)
TREES = {
    "src": {".py"},
    "tests": {".py"},
    "scripts": {".py"},
    "configs": {".yaml"},
    "docs": {".md", ".csv", ".json", ".png"},
    ".github/workflows": {".yml"},
}


def public_files(root: Path = ROOT) -> list[Path]:
    root = root.resolve()
    files = [root / name for name in ROOT_FILES]
    for directory, extensions in TREES.items():
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and path.suffix in extensions
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root / directory).parts
            )
        )
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Public file resolves outside the workspace: {path}")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build(root: Path = ROOT) -> Path:
    root = root.resolve()
    files = public_files(root)
    output = root / "dist" / "intraday-HMM-public.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            info = ZipInfo(
                "intraday-HMM/" + path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0)
            )
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"{len(files)} public files; {output.stat().st_size:,} bytes; {output}")
    return output


if __name__ == "__main__":
    build()
