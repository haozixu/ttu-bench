#!/usr/bin/env python3
"""Publish local matrices and referenced reports as one lossless phase ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
from bundle_results import summarize
from result_store import ROOT, catalog


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--matrices',type=Path,nargs='+',required=True)
    parser.add_argument('--reports',type=Path,nargs='*',default=[])
    parser.add_argument('--readable',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    if not args.phase.startswith('phase') or not args.phase[5:].isdigit():
        parser.error('phase<number> required')
    manifest=catalog()
    target=ROOT/args.phase
    if target.exists():
        raise FileExistsError(target)
    reports=set(args.reports)
    for path in args.matrices:
        matrix=json.loads(path.read_bytes())
        for case in matrix['cases']:
            report=Path(case['file'])
            if digest(report.read_bytes())!=case['sha256']:
                raise ValueError(f'matrix reference hash mismatch: {report}')
            reports.add(report)
    all_paths=[*args.matrices,*args.readable,*sorted(reports)]
    names=[p.name for p in all_paths]
    if len(names)!=len(set(names)) or {f'{args.phase}/{n}' for n in names} & {e['original_path'] for e in manifest['files']}:
        raise ValueError('duplicate original names')
    # Read and validate all inputs before creating the publication.
    sources={p:p.read_bytes() for p in all_paths}
    for p in sources: p.resolve().relative_to(ROOT.resolve())
    target.mkdir(parents=True)
    name=f'{args.phase}/records.zip'
    archive_path=ROOT/name
    entries=[]
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(reports):
            info=zipfile.ZipInfo(path.name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            archive.writestr(info,sources[path],compresslevel=9)
    with zipfile.ZipFile(archive_path) as archive:
        for path in reports:
            if archive.read(path.name)!=sources[path]: raise ValueError('ZIP roundtrip mismatch')
    for path,data in sources.items():
        local=str(path.resolve().relative_to(ROOT.resolve()))
        entry=dict(original_path=f'{args.phase}/{path.name}',original_bytes=len(data),sha256=digest(data),
                   local_original_path=local,aliases=[local],batch=args.phase)
        if path in reports:
            entry.update(published_path=name,encoding='zip',archive_member=path.name)
        else:
            published=f'{args.phase}/{path.name}'
            (ROOT/published).write_bytes(data)
            entry.update(published_path=published,encoding='identity',stored_bytes=len(data),stored_sha256=digest(data))
        entries.append(entry)
    manifest['files'].extend(entries)
    manifest.setdefault('bundles',{})[name]=dict(bytes=archive_path.stat().st_size,
        sha256=digest(archive_path.read_bytes()),reports=len(reports),encoding='zip-deflate')
    summarize(manifest)
    manifest.setdefault('batches',[]).append(dict(name=args.phase,date=datetime.now(timezone.utc).isoformat(),
        original_files=len(entries),original_bytes=sum(e['original_bytes'] for e in entries),
        published_reports=len(entries),published_files=len(all_paths)-len(reports)+1,
        published_bytes=sum(p.stat().st_size for p in target.iterdir())))
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['batches'][-1]))


if __name__=='__main__': main()
