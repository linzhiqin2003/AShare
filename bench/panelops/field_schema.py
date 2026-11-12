"""Check or regenerate the factor field catalogue from authoritative panel metadata."""

import argparse
import json
from pathlib import Path

import ashare
from ashare.factor.library import FIELDS

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    fields = json.loads(args.metadata.read_text())["fields"]
    catalogue = {name: (entry["unit"], name) for name, entry in sorted(fields.items())}
    if args.generate:
        print("FIELDS = {")
        for name, value in catalogue.items():
            print(f"    {name!r}: {value!r},")
        print("}")
    else:
        assert FIELDS == catalogue, (
            "Field catalogue differs from authoritative metadata"
        )
        print(f"{len(catalogue)} field names and units match metadata")


if __name__ == "__main__":
    main()
