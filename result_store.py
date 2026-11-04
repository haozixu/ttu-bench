#!/usr/bin/env python3
"""Resolve, verify or restore byte-preserved results using results/manifest.json."""
import argparse
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent / 'results'


def catalog():
    return json.loads((ROOT / 'manifest.json').read_text())


@lru_cache(maxsize=8)
def _verified_bundle(path, size, mtime_ns, expected_hash):
    # Stat participates in the cache key: a changed bundle must be reverified.
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f'bundle hash mismatch: {path}')
    return True


def read_original(entry):
    if 'published_path' in entry:
        if entry['encoding'] == 'zip':
            path = ROOT / entry['published_path']
            stat = path.stat()
            bundle = catalog()['bundles'][entry['published_path']]
            _verified_bundle(str(path), stat.st_size, stat.st_mtime_ns, bundle['sha256'])
            with zipfile.ZipFile(path) as archive:
                data = archive.read(entry['archive_member'])
        else:
            stored = (ROOT / entry['published_path']).read_bytes()
            if hashlib.sha256(stored).hexdigest() != entry['stored_sha256']:
                raise ValueError(f'stored hash mismatch: {entry["published_path"]}')
            data = gzip.decompress(stored) if entry['encoding'] == 'gzip' else stored
    else:
        data = (ROOT / entry['local_archive_path']).read_bytes()
    if len(data) != entry['original_bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'original hash mismatch: {entry["original_path"]}')
    return data


def resolve_entry(path):
    """Load a published report by its original results-relative path or stored path.

    Absolute paths embedded in old reports are resolved at the /results/ suffix.
    Source reports themselves remain byte-for-byte unchanged.
    """
    name = str(path).replace(chr(92), '/')
    if '/results/' in name:
        name = name.split('/results/', 1)[1]
    elif name.startswith('results/'):
        name = name[len('results/'):]
    for entry in catalog()['files']:
        if name == entry['original_path'] or name in entry.get('aliases', []) or (
                entry.get('encoding') != 'zip' and name == entry.get('published_path')) or (
                entry.get('encoding') == 'zip' and name == entry['published_path'] + '#' + entry['archive_member']):
            return entry
    raise FileNotFoundError(name)


def load_result(path):
    return json.loads(read_original(resolve_entry(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    verify = sub.add_parser('verify', help='verify stored and decompressed hashes')
    verify.add_argument('--include-local', action='store_true', help='also check ignored archives and original copies')
    restore = sub.add_parser('restore', help='extract published originals under a separate directory')
    restore.add_argument('--destination', type=Path, default=ROOT / 'local' / 'restored')
    show = sub.add_parser('show', help='read an original report or an old published alias')
    show.add_argument('path')
    show.add_argument('--output', type=Path, help='write original bytes; default stdout')
    args = parser.parse_args()
    if args.action == 'show':
        data = read_original(resolve_entry(args.path))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(data)
        else:
            import sys
            sys.stdout.buffer.write(data)
        return
    count = 0
    for entry in catalog()['files']:
        if 'published_path' not in entry and not (args.action == 'verify' and args.include_local):
            continue
        data = read_original(entry)
        if args.action == 'restore':
            destination = args.destination.resolve()
            target = (destination / entry['original_path']).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('result path escapes destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError(f'refusing to overwrite differing file: {target}')
            target.write_bytes(data)
        elif args.include_local and entry.get('local_original_path'):
            if (ROOT / entry['local_original_path']).read_bytes() != data:
                raise ValueError(f'local original mismatch: {entry["original_path"]}')
        count += 1
    print(json.dumps(dict(action=args.action, verified_originals=count)))


if __name__ == '__main__':
    main()
