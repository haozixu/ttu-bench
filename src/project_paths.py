"""Checkout paths and subprocess entry points shared by experiments and tools."""
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / 'kernels'
BUILD = ROOT / 'build'


def source_path(name):
    """Resolve old flat source names, or current repo-relative source paths."""
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('source name must stay within the checkout')
    candidates = [ROOT / path]
    if len(path.parts) == 1:
        candidates += [ROOT / directory / path for directory in
                       ('src', 'tools', 'kernels', 'docs/reference')]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f'expected one source for {name}, found {len(existing)}')
    return existing[0]


def hash_sources(names):
    """Record current relative paths, including launcher/path resolution code."""
    paths = {source_path(name) for name in (*names, 'bench.py', 'project_paths.py')}
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def bench_command(name):
    """Use the same interpreter and import setup in fresh child processes."""
    name = Path(name).stem
    if not (ROOT / 'tools' / (name + '.py')).is_file():
        raise ValueError(f'unknown tool: {name}')
    return [sys.executable, str(ROOT / 'bench.py'), name]
