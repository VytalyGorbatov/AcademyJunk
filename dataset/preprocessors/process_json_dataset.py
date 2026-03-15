from __future__ import annotations
import argparse
from codecs import decode, encode
import json
import sys
from pathlib import Path
from typing import Any, List


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build dataset with dedupe, normalization, attack flag, and categorical mappings')
    p.add_argument('-i', '--input', required=True, help='Path to input JSON file (array of objects)')
    p.add_argument('-o', '--output', required=True, help='Path to write deduplicated JSON')
    p.add_argument('--is-attack', action='store_true', help='Mark all records with is_attack=1 (else 0)')
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    try:
        with in_path.open('r', encoding='utf-8') as f:
            data: Any = json.load(f)
    except Exception as e:
        print(f'Error reading/parsing input JSON: {e}', file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print('Input JSON root must be an array', file=sys.stderr)
        return 1

    seen = set()
    deduped: list[dict[str, Any]] = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            print(f'Record at index {idx} is not an object; skipping', file=sys.stderr)
            continue

        if 'buffers_names' in rec and isinstance(rec['buffers_names'], str):
            if rec['buffers_names'] == '':
                continue

        if 'buffers' in rec and isinstance(rec['buffers'], str):
            rec['buffers'] = decode(encode(rec['buffers'], 'latin-1', 'backslashreplace'), 'unicode-escape')

        key = rec.get('buffers')
        if not isinstance(key, str):
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    # Normalization of timing fields
    def normalize_field(records: list[dict[str, Any]], field: str) -> None:
        values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
        if not values:
            return
        baseline = min(values)
        for r in records:
            val = r.get(field)
            if isinstance(val, (int, float)):
                r[field] = val - baseline

    normalize_field(deduped, 'flowstart_time')
    normalize_field(deduped, 'seconds')

    attack_value = 1 if args.is_attack else 0

    for rec in deduped:
        if rec.get('action') == 'allow':
            rec['alerted'] = 0
        else:
            rec['alerted'] = 1
        del rec['action']

        rec['is_attack'] = attack_value

    # Build mappings for string fields (excluding certain fields)
    EXCLUDE = {'buffers', 'buffer_names'}
    mappings: dict[str, dict[str, int]] = {}

    # First pass: collect per-field unique strings in order
    for rec in deduped:
        for k, v in list(rec.items()):
            if k in EXCLUDE:
                continue
            if isinstance(v, str):
                m = mappings.setdefault(k, {})
                if v not in m:
                    m[v] = len(m)

    # Second pass: replace string values with their integer ids
    for rec in deduped:
        for k, v in list(rec.items()):
            if k in EXCLUDE:
                continue
            if isinstance(v, str):
                rec[k] = mappings[k][v]

    # Assemble output object
    output_obj: dict[str, Any] = { 'dataset': deduped }
    for field, mapping in mappings.items():
        output_obj[f'{field}_mapping'] = mapping

    try:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(output_obj, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f'Error writing output JSON: {e}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
