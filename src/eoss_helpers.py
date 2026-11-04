"""Local EoSS adapter: retain canonical bytes and explicit patch policies."""
from project_paths import ROOT as HERE
import json
import sys
from pathlib import Path

COMMUNITY = HERE.parent / 'research' / 'community'
sys.path.insert(0, str(COMMUNITY))

from eoss import paths
from eoss.binary import BinaryPatcher
from eoss.enc import control


def canonical_sm120():
    """Unmodified artifact sequence, including its bit-72 CLOSE-labelled word."""
    data = json.loads(paths.encoding_data('ttu_sm120.json').read_text())
    return b''.join(int(entry['lo'], 16).to_bytes(8, 'little') +
                    int(entry['hi'], 16).to_bytes(8, 'little')
                    for entry in data['sequences']['canonical_traversal']['instructions'])


def replace_full(image, offset, instruction):
    control.patch_at(image, offset, instruction, control.PatchPolicy.REPLACE_FULL)


def patch_slots(image, offset, sequence):
    """Use EoSS BinaryPatcher; preserve all operand and scheduling bits."""
    if len(sequence) % 16 or offset < 0 or offset + len(sequence) > len(image):
        raise ValueError('sequence must fit inside the image and contain whole SASS instructions')
    patcher = BinaryPatcher.from_bytes(image)
    for index in range(0, len(sequence), 16):
        patcher.patch_at(offset + index, sequence[index:index + 16], control.PatchPolicy.REPLACE_FULL)
    return patcher.data
