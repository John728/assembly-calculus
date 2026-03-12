from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyac.tasks.pointer.visualize import render_trace_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AC trace visualizations from a saved trace JSON file")
    parser.add_argument("--trace", required=True, help="Path to trace JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for generated plots")
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    for path in render_trace_visualizations(trace, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
