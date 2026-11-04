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
    # JIT [E+4] -> index-1 -> IAS instance table. Adjacency E-0xc
    # only happened to identify the right instance in the one-instance samples.
    index_word = struct.unpack_from('<I', as_data, offset + 4)[0]
    table_offset = struct.unpack_from('<Q', as_data, 0x30)[0]
    instance_count = struct.unpack_from('<I', as_data, 0x70)[0]
    if not 1 <= index_word <= instance_count:
        raise ValueError('instance event index is outside the supplied IAS table')
    metadata_offset = table_offset + (index_word - 1) * 0x80
    if metadata_offset < 0x80 or metadata_offset + 0x80 > len(as_data):
        raise ValueError('instance metadata is outside the supplied IAS')
    metadata_inverse = struct.unpack_from('<12f', as_data, metadata_offset + 0x30)
    if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(matrix, metadata_inverse)):
        raise ValueError('event matrix differs from the indexed instance metadata')
    child_handle = struct.unpack_from('<Q', as_data, metadata_offset + 0x60)[0]
    instance_id_word = struct.unpack_from('<I', as_data, metadata_offset + 0x74)[0]
    return dict(status_class=status_class, record_address=record_address,
                record_offset=offset, inverse_transform=matrix,
                encoded_child_root=encoded_child,
                child_root_address=encoded_child & ADDRESS_MASK,
                child_handle_candidate=child_handle,
                instance_index_word=index_word, instance_index=index_word - 1,
                instance_table_offset=table_offset, instance_metadata_offset=metadata_offset,
                instance_id_candidate=instance_id_word & 0xffffff,
                instance_id_word=instance_id_word,
                adjacent_id_word_unreliable=struct.unpack_from('<I', as_data, offset - 0xc)[0],
                record_hex=as_data[offset - 0x80:offset + 0x40].hex())


def transform_ray(ray, inverse_transform):
    origin, direction = ray[:3], ray[4:7]
    local_origin = [sum(inverse_transform[4 * axis + component] * origin[component]
                        for component in range(3)) + inverse_transform[4 * axis + 3]
                    for axis in range(3)]
    local_direction = [sum(inverse_transform[4 * axis + component] * direction[component]
                           for component in range(3)) for axis in range(3)]
    return (*local_origin, ray[3], *local_direction, ray[7])
