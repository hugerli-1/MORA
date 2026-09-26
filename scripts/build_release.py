"""Check an allowlisted anonymous package and build a metadata-clean ZIP."""

import argparse
import hashlib
import re
import stat
from pathlib import Path
import zipfile


TOP_FILES = {"README.md", "NOTICE.md", "VALIDATION.md", "requirements.txt", "pyproject.toml", ".gitignore"}
DIRECTORIES = {"mora", "configs", "docs", "examples", "scripts", "tests", "results"}
IGNORED = {"__pycache__", ".pytest_cache", ".git", ".venv", "build", "dist"}
EXTENSIONS = {".py", ".md", ".json", ".toml", ".txt", ".csv"}
PATTERNS = {
    "private filesystem path": re.compile(r"/(?:home|Users|data|data2)/[\w.-]+/"),
    "private network address": re.compile(r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "credential assignment": re.compile(r"(?i)(?:api_key|access_token|password)\s*=\s*[\"\x27][A-Za-z0-9_/-]{12,}"),
}


def release_files(root):
    root = Path(root).resolve()
    missing = TOP_FILES - {p.name for p in root.iterdir()}
    if missing:
        raise ValueError(f"Missing release documentation: {sorted(missing)}")
    paths = []
    for entry in sorted(root.iterdir()):
        if entry.name in TOP_FILES:
            candidates = [entry]
        elif entry.name in DIRECTORIES:
            if entry.is_symlink():
                raise ValueError(f"Symlinked release directory: {entry.name}")
            candidates = sorted(entry.rglob("*"))
        else:
            continue
        for path in candidates:
            relative = path.relative_to(root)
            if any(part in IGNORED for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink in release: {relative}")
            if not path.is_file():
                continue
            if relative.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.suffix not in EXTENSIONS and path.name != ".gitignore":
                raise ValueError(f"Unexpected artifact type in release: {relative}")
            paths.append(path)
    return paths


def audit(root, paths, deny_tokens=()):
    findings = []
    for path in paths:
        relative = str(path.relative_to(root))
        content = path.read_text(encoding="utf-8")
        for name, pattern in PATTERNS.items():
            if pattern.search(content) or pattern.search(relative):
                findings.append((relative, name))
        for token in deny_tokens:
            if token.lower() in content.lower() or token.lower() in relative.lower():
                findings.append((relative, "explicitly denied identifying token"))
    if findings:
        raise ValueError(f"Anonymity audit failed: {findings}")


def build(root, output, deny_tokens=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output path; existing archives are not overwritten")
    paths = release_files(root)
    audit(root, paths, deny_tokens)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            info = zipfile.ZipInfo("MORA/" + str(path.relative_to(root)), date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or len(archive.infolist()) != len(paths):
            raise RuntimeError("ZIP integrity check failed")
    print(f"PASS: {len(paths)} files; no detected identifying paths, addresses, or credentials")
    print(f"ZIP: {output.name} ({output.stat().st_size} bytes)")
    print("SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deny-token", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        root = args.root.resolve()
        files = release_files(root)
        audit(root, files, args.deny_token)
        print(f"PASS: {len(files)} uploadable files")
    elif args.output is None:
        parser.error("--output or --check-only is required")
    else:
        build(args.root, args.output, args.deny_token)
