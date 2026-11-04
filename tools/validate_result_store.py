#!/usr/bin/env python3
"""Archive contract checks: old names, corruption rejection and safe restore."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import result_store as rs


def main():
    original_root=rs.ROOT
    manifest=rs.catalog()
    aliases=0
    for entry in manifest['files']:
        if entry.get('encoding')!='zip': continue
        for alias in entry.get('aliases',[]):
            if rs.resolve_entry(alias)!=entry: raise AssertionError(alias)
            aliases+=1
        assert rs.resolve_entry(entry['original_path'])==entry
        assert rs.resolve_entry(entry['published_path']+'#'+entry['archive_member'])==entry
    sample=next(e for e in manifest['files'] if e.get('encoding')=='zip')
    data=rs.read_original(sample)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        rs.ROOT=root
        try:
            archive=root/'sample.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('member',data)
            e=dict(original_path='nested/original.json',published_path='sample.zip',encoding='zip',archive_member='member',
                   original_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),aliases=['former.json.gz'])
            m=dict(files=[e],bundles={'sample.zip':dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest())})
            def write_manifest(): (root/'manifest.json').write_text(json.dumps(m))
            write_manifest()
            assert rs.read_original(e)==data
            assert rs.read_original(rs.resolve_entry('/old/machine/results/former.json.gz'))==data
            # A changed archive must invalidate the stat/hash cache.
            archive.write_bytes(archive.read_bytes()+b'changed')
            try: rs.read_original(e)
            except ValueError: pass
            else: raise AssertionError('corrupted bundle accepted')
            # Use a plain small file to test extraction boundaries/overwrite.
            payload=b'{}\n'
            (root/'small.json').write_bytes(payload)
            e=dict(original_path='../escape.json',published_path='small.json',encoding='identity',
                   original_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),stored_sha256=hashlib.sha256(payload).hexdigest())
            m['files']=[e]
            destination=root/'restored'
            write_manifest()
            old_argv=sys.argv
            sys.argv=['result_store.py','restore','--destination',str(destination)]
            try:
                try: rs.main()
                except ValueError: pass
                else: raise AssertionError('path traversal accepted')
                e['original_path']='original.json';write_manifest()
                destination.mkdir();(destination/'original.json').write_bytes(b'different')
                try: rs.main()
                except FileExistsError: pass
                else: raise AssertionError('differing overwrite accepted')
                (destination/'original.json').write_bytes(payload)
                rs.main()
                assert (destination/'original.json').read_bytes()==payload
            finally: sys.argv=old_argv
        finally:
            rs.ROOT=original_root
            rs._verified_bundle.cache_clear()
    print(json.dumps(dict(status='pass',aliases=aliases,corruption_rejected=True,traversal_rejected=True,differing_overwrite_rejected=True)))


if __name__=='__main__': main()
