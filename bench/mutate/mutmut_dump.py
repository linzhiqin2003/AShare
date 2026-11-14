"""Dump mutmut 3 results of the current working directory as JSON (run with cwd = shard workdir).

Kept as a separate script because mutmut reads its configuration from the cwd.
Output: [{"name", "path", "exit_code", "status", "seconds", "diff"}]; diffs only for mutants that
were not killed (they feed the survivor list and the equivalent-mutant matching).
"""

from __future__ import annotations

import json
import sys

from mutmut.mutation.data import SourceFileMutationData
from mutmut.mutation.diff_apply import get_diff_for_mutant
from mutmut.stats import status_by_exit_code
from mutmut.utils.file_utils import walk_mutatable_files


def main():
    rows = []
    for path in walk_mutatable_files():
        data = SourceFileMutationData(path=path)
        data.load()
        for name, code in data.exit_code_by_key.items():
            status = status_by_exit_code[code]
            row = {
                "name": name,
                "path": str(path),
                "exit_code": code,
                "status": status,
                "seconds": data.durations_by_key.get(name),
                "diff": "",
            }
            if status not in ("killed", "caught by type check"):
                try:
                    row["diff"] = get_diff_for_mutant(name, path=path)
                except Exception as exc:  # keep the row; the diff is only for humans
                    row["diff"] = f"<diff unavailable: {type(exc).__name__}>"
            rows.append(row)
    json.dump(rows, sys.stdout)


if __name__ == "__main__":
    main()
