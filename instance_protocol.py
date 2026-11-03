"""Bounded decoding of the observed SM86 static-instance event.

This is a single-instance diagnostic, not a traversal continuation ABI.
"""
import math
import struct


ADDRESS_MASK = (1 << 49) - 1


def decode_instance_event(raw, as_base, as_data):
    status_class = raw[33] >> 27
    if status_class != 0x16:
        raise ValueError(f'expected observed instance class 0x16, got {status_class:#x}')
    record_address = raw[72] | (raw[73] & 0x1ffff) << 32
    offset = record_address - as_base
    if offset < 0x80 or offset % 64 or offset + 64 > len(as_data):
        raise ValueError('instance event record is outside the supplied AS or unaligned')
    matrix = list(struct.unpack_from('<12f', as_data, offset + 0x10))
    if not all(math.isfinite(value) for value in matrix):
        raise ValueError('instance matrix contains nonfinite values')
    encoded_child = struct.unpack_from('<Q', as_data, offset + 8)[0]
    child_handle = struct.unpack_from('<Q', as_data, offset - 0x20)[0]
    instance_id_word = struct.unpack_from('<I', as_data, offset - 0xc)[0]
    return dict(status_class=status_class, record_address=record_address,
                record_offset=offset, inverse_transform=matrix,
                encoded_child_root=encoded_child,
                child_root_address=encoded_child & ADDRESS_MASK,
                child_handle_candidate=child_handle,
                instance_id_candidate=instance_id_word & 0xffffff,
                instance_id_word=instance_id_word,
                record_hex=as_data[offset - 0x80:offset + 0x40].hex())


def transform_ray(ray, inverse_transform):
    origin, direction = ray[:3], ray[4:7]
    local_origin = [sum(inverse_transform[4 * axis + component] * origin[component]
                        for component in range(3)) + inverse_transform[4 * axis + 3]
                    for axis in range(3)]
    local_direction = [sum(inverse_transform[4 * axis + component] * direction[component]
                           for component in range(3)) for axis in range(3)]
    return (*local_origin, ray[3], *local_direction, ray[7])
