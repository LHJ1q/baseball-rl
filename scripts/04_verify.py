"""CLI entry point for the verification phase. Exits non-zero on any FAIL."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.verify import format_report, run_all_checks  # noqa: E402


SPLITS_DIR = REPO_ROOT / "data" / "splits"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_all_checks(SPLITS_DIR)
    print(format_report(results))
    if any(r.status == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
