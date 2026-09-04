"""One-time migration: recompute every SKU's train_status from its media rows.

Whole-SKU semantics (matches api.tasks.recompute_train_status):
  any media failed=1  -> 'failed'
  zero media rows     -> 'pending'
  else                -> 'completed'

Fixes stale rows left by the old batch-based status writes (e.g. SKUs that
went 'failed' during an old job whose failed media were later deleted or
re-added by the operator).

Report-only with --dry-run. Idempotent: re-running updates nothing once
statuses match. Never touches sku_media or train_job rows.

Usage:
    uv run python scripts/recompute_train_status.py [--db sku_match.db] [--dry-run]
"""

import argparse
import sqlite3
import sys


def compute_status(failed_flags: list[int]) -> str:
    if any(failed_flags):
        return "failed"
    if not failed_flags:
        return "pending"
    return "completed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default="sku_match.db", help="SQLite database path")
    parser.add_argument("--dry-run", action="store_true", help="Report without updating")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    skus = conn.execute("SELECT sku_id, train_status FROM sku ORDER BY sku_id").fetchall()
    print(f"SKUs: {len(skus)}")

    changes: list[tuple[str, str, str]] = []  # (sku_id, old, new)
    for sku_id, old_status in skus:
        flags = [
            row[0]
            for row in conn.execute(
                "SELECT failed FROM sku_media WHERE sku_id = ?", (sku_id,)
            ).fetchall()
        ]
        new_status = compute_status(flags)
        if new_status != old_status:
            changes.append((sku_id, old_status, new_status))

    for sku_id, old_status, new_status in changes:
        marker = "would fix" if args.dry_run else "fixing"
        print(f"  {marker}: {sku_id}: {old_status} -> {new_status}")
    if not changes:
        print("All train_status values already consistent.")

    if args.dry_run:
        print(f"[dry-run] {len(changes)} row(s) would be updated.")
    elif changes:
        conn.executemany(
            "UPDATE sku SET train_status = ? WHERE sku_id = ?",
            [(new, sku_id) for sku_id, _, new in changes],
        )
        conn.commit()
        print(f"Updated {len(changes)} row(s).")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
