#!/usr/bin/env python3
"""Run a TTU experiment or maintenance tool from the source checkout."""
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    commands = {path.stem: path for path in (ROOT / 'tools').glob('*.py')}
    commands['render_demo'] = ROOT / 'examples/renderer/render.py'
    commands['validate_renderer'] = ROOT / 'examples/renderer/validate.py'
    commands['compare_renderer_builds'] = ROOT / 'examples/renderer/compare_builds.py'
    commands['prepare_renderer_bunny'] = ROOT / 'examples/renderer/prepare_bunny.py'
    commands['validate_renderer_meshes'] = ROOT / 'examples/renderer/validate_meshes.py'
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python3 bench.py <command> [arguments]\n\nCommands:')
        print('\n'.join('  ' + name for name in sorted(commands)))
        return
    name = sys.argv[1].removesuffix('.py')
    if name not in commands:
        raise SystemExit(f'Unknown command: {name}. Use --help to list commands.')
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools'), str(commands[name].parent)]
    sys.argv = [str(commands[name]), *sys.argv[2:]]
    runpy.run_path(str(commands[name]), run_name='__main__')


if __name__ == '__main__':
    main()
