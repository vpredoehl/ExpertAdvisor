#!/usr/bin/env python3
"""Read Phase 25B-3 observer evidence without launching training or opening SQL."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def inspect(path, updates=128):
    counts = {}; shapes = set(); elements = 0; clipped = 0; preclip = None
    with path.open('rb') as source:
        def u64():
            data = source.read(8)
            if len(data) != 8: raise ValueError('truncated evidence')
            return struct.unpack('=Q', data)[0]
        while source.peek(1):
            stage = source.read(u64()).decode()
            matrices = []
            for _ in range(u64()):
                rows, cols = u64(), u64(); elements += rows * cols
                if stage == 'forward_cache':
                    shapes.add((rows, cols)); source.seek(rows * cols * 4, 1)
                else:
                    data = source.read(rows * cols * 4)
                    if len(data) != rows * cols * 4: raise ValueError('truncated matrix')
                    values = array('f'); values.frombytes(data); matrices.append(values)
            counts[stage] = counts.get(stage, 0) + 1
            if stage == 'preclip': preclip = matrices
            if stage == 'postclip':
                assert preclip is not None and len(preclip) == len(matrices) == 6
                for before, after in zip(preclip, matrices):
                    assert len(before) == len(after)
                    for index, (x, y) in enumerate(zip(before, after)):
                        assert y == max(-10.0, min(10.0, x)), ('clipping mismatch', index, x, y)
                        clipped += abs(x) > 10.0
                preclip = None
    assert counts.get('preclip') == updates and counts.get('postclip') == updates, counts
    assert counts.get('forward_cache', 0) > updates, counts
    assert shapes.issuperset({(128, 171), (128, 64), (61, 171), (61, 64)}), shapes
    return dict(stage_counts=counts, clipped_elements=clipped, observed_float_elements=elements,
                bytes=path.stat().st_size, sha256=digest(path))


def compare(left, right):
    # Hashes are accompanied by a literal byte comparison; offsets identify a
    # disagreement without treating rounded printed floats as equivalent.
    offset = 0
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1024*1024), b.read(1024*1024)
            if x != y:
                first = next((i for i, (v, w) in enumerate(zip(x, y)) if v != w), min(len(x), len(y)))
                raise AssertionError(f'bitwise mismatch at byte {offset + first}: {left} vs {right}')
            if not x: return
            offset += len(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(); p = args.directory
    evidence = {path: inspect(p / f'evidence_{path}.tensors') for path in ('metann', 'combined')}
    compare(p/'evidence_metann.tensors', p/'evidence_combined.tensors')
    compare(p/'evidence_metann.state', p/'evidence_combined.state')
    compare(p/'evidence_metann.state', p/'trial1_metann.state')
    compare(p/'evidence_metann.inputs', p/'evidence_combined.inputs')
    result = dict(paths=evidence, bitwise_equal=True, observer_on_off_equal=True,
                  numerical_differences=0, maximum_absolute_difference=0)
    (p/'numerical-equivalence.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
