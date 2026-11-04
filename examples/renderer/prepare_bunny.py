#!/usr/bin/env python3
"""Prepare hash-pinned Stanford Bunny PLY files in the ignored build cache."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

from project_paths import BUILD


MANIFEST = Path(__file__).resolve().parent/'assets/stanford-bunny.json'
CACHE = BUILD/'assets/stanford-bunny'


def verify(data, record):
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('asset bytes or SHA256 differ from pinned source')


def bunny_path(level):
    metadata = json.loads(MANIFEST.read_text())
    record = metadata['levels'][level]
    path = CACHE/Path(record['member']).name
    if not path.exists():
        raise FileNotFoundError('Bunny cache missing; run python3 bench.py prepare_renderer_bunny first')
    verify(path.read_bytes(), record)
    return path, dict(name=metadata['name'],level=level,credit=metadata['credit'],source=metadata['source'],
                      terms=metadata['terms'],archive_sha256=metadata['archive_sha256'],
                      normalization_bounds=metadata['normalization_bounds'],
                      manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='existing official bunny.tar.gz; skips network')
    args = parser.parse_args()
    metadata = json.loads(MANIFEST.read_text())
    records = [metadata['readme'],*metadata['levels'].values()]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = all((CACHE/Path(record['member']).name).exists() for record in records)
    if cached:
        for record in records:
            verify((CACHE/Path(record['member']).name).read_bytes(), record)
    else:
        if args.archive:
            data = args.archive.read_bytes()
        else:
            with urllib.request.urlopen(metadata['archive_url'], timeout=45) as response:
                data = response.read(metadata['archive_bytes']+1)
        verify(data, dict(bytes=metadata['archive_bytes'],sha256=metadata['archive_sha256']))
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            selected = []
            for record in records:
                member = archive.getmember(record['member'])
                if not member.isfile() or member.size != record['bytes']:
                    raise ValueError('unexpected archive member type/size')
                original = archive.extractfile(member).read()
                verify(original,record)
                target = CACHE/Path(record['member']).name
                if target.exists() and target.read_bytes() != original:
                    raise FileExistsError(target)
                selected.append((target,original))
        for target, original in selected:
            target.write_bytes(original)
    print(json.dumps(dict(prepared=True,credit=metadata['credit'],terms=metadata['terms'],
                          files={level:str(bunny_path(level)[0]) for level in metadata['levels']})))


if __name__ == '__main__':
    main()
