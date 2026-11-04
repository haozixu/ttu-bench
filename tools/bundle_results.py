#!/usr/bin/env python3
"""Consolidate selected phase reports, retaining original bytes and old aliases.

A phase keeps its matrices, summaries, text logs and sample-events readable.
Individual gzip reports / cases become members of one deterministic ZIP.
Old compressed files are deleted after verification; no duplicate archive is kept.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from result_store import ROOT, catalog, read_original


CONTRACT = ('Retained reports preserve original bytes and embedded historical paths. '
    'Identity/gzip entries have stored hashes; ZIP bundles have physical-file hashes and per-member original hashes. '
    'Aliases resolve retained former paths. Historical matrices may reference discarded routine reports; '
    'the catalog inventories retained evidence only.')


def summarize(manifest):
    entries = manifest['files']
    paths = {e['published_path'] for e in entries if 'published_path' in e}
    manifest['summary'] = dict(original_files=len(entries), original_bytes=sum(e['original_bytes'] for e in entries),
        published_reports=sum('published_path' in e for e in entries), published_files=len(paths),
        published_bytes=sum((ROOT / p).stat().st_size for p in paths),
        local_only_files=sum('published_path' not in e for e in entries))


def bundle(phases):
    manifest = catalog()
    for phase in phases:
        if not phase.startswith('phase') or not phase[5:].isdigit():
            raise ValueError('expected phase<number>')
        selected = [e for e in manifest['files'] if e.get('published_path', '').startswith(phase + '/')
                    and (e.get('encoding') in ('gzip','zip') or '/cases/' in e['published_path']
                         or (phase == 'phase17' and Path(e['published_path']).name.startswith('final-parent-matrix-')))]
        if not selected:
            continue
        name = f'{phase}/records.zip'
        target = ROOT / name
        if target.exists() and name not in manifest.get('bundles', {}):
            raise FileExistsError(target)
        temporary = target.with_suffix('.zip.tmp')
        originals = [(e, read_original(e)) for e in selected]
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for entry, data in sorted(originals, key=lambda pair: pair[0]['original_path']):
                info = zipfile.ZipInfo(entry['original_path'], date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            for entry, data in originals:
                if archive.read(entry['original_path']) != data:
                    raise ValueError('roundtrip mismatch')
        temporary.rename(target)
        manifest.setdefault('bundles', {})[name] = dict(bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), reports=len(selected), encoding='zip-deflate')
        old_paths = []
        for entry, _ in originals:
            if entry.get('encoding') == 'zip' and entry['archive_member'] != entry['original_path']:
                entry.setdefault('aliases', []).append(entry['published_path'] + '#' + entry['archive_member'])
            if entry.get('encoding') != 'zip':
                old_paths.append(entry['published_path'])
                entry.setdefault('aliases', []).append(entry['published_path'])
            entry.update(published_path=name, encoding='zip', archive_member=entry['original_path'])
            entry.pop('stored_bytes', None)
            entry.pop('stored_sha256', None)
        summarize(manifest)
        manifest['contract'] = CONTRACT
        (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for old in old_paths:
            (ROOT / old).unlink()
        print(json.dumps(dict(phase=phase, bundled_reports=len(selected), bundle_bytes=target.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phases', nargs='+')
    bundle(parser.parse_args().phases)
