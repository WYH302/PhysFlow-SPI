"""Unit assertion for the loaded LSI simulation_256 mask."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optical.spi_operator import SPIOperator


def main() -> None:
    operator = SPIOperator.from_simulation_256()
    print(json.dumps(operator.lsi_match, indent=2))


if __name__ == "__main__":
    main()
