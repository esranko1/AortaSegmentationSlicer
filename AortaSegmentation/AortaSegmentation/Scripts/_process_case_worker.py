"""
Runs process_case() for a single segmentation and writes the result to --out-json.

Launched as its own subprocess by extracting_metrics.py's run_batch(), one process per
case, so a native hang or crash in VMTK on any one case (some cases have unusually
complex topology, e.g. many branches, that vtkvmtkPolyDataNetworkExtraction can choke
on) only kills that case's subprocess -- caught via a timeout/non-zero exit code by the
parent -- instead of freezing or taking down the whole batch run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extracting_metrics import process_case  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--write-vtp', action='store_true')
    args = parser.parse_args()

    results = process_case(args.seg, args.out_dir, write_vtp=args.write_vtp)
    with open(args.out_json, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
