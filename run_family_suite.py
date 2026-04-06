from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
VALID_FAMILIES = {
    "seen_mlp",
    "unseen_mlp",
    "seen_ac",
    "unseen_ac",
}
VALID_SCALES = {"dev", "paper"}


def resolve_config_path(family: str, scale: str) -> Path:
    if family not in VALID_FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    if scale not in VALID_SCALES:
        raise ValueError(f"Unknown scale: {scale}")
    return EXPERIMENTS_DIR / f"{family}_{scale}.yaml"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {2, 3}:
        print("Usage: python run_family_suite.py <family> <scale> [--dry-run]", file=sys.stderr)
        return 2

    family, scale = args[0], args[1]
    dry_run = False
    if len(args) == 3:
        if args[2] != "--dry-run":
            print(f"Unknown option: {args[2]}", file=sys.stderr)
            return 2
        dry_run = True

    config_path = resolve_config_path(family, scale)
    command = [
        str(ROOT / "venv" / "bin" / "python"),
        str(ROOT / "run_experiment_suite.py"),
        "--config",
        str(config_path),
    ]

    if dry_run:
        print("DRY RUN:", " ".join(command))
        return 0

    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
