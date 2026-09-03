"""One-time migration: split detection_diff into detections_added / detections_removed.

For every recognition_log row, recomputes the two split metrics from the stored
authoritative JSON (final_result_json vs ai_result_json) and writes them to the
new columns. Semantics match the fix endpoint:

  added   = entries in final with source="manual" (every fix-add is stamped
            at creation and history was backfilled with source flags)
  removed = original entries no longer present in final
            (max(0, len(original) - survivors))

Rows without final_result_json (never fixed) get 0/0. Corrupt JSON rows are
skipped and reported. Never touches user_correction_json.

Optionally drops the superseded detection_diff column afterwards (default;
requires SQLite >= 3.35). Idempotent: safe to re-run.

Usage:
    uv run python scripts/split_detection_diff.py [--db sku_match.db] [--dry-run] \
        [--keep-old-column]
"""

import argparse
import json
import sqlite3
import sys

TABLE = "recognition_log"


def compute_split_metrics(original_dets: list, final_dets: list) -> tuple[int, int]:
    """Return (added, removed) comparing final vs original detection lists."""
    added = sum(1 for d in final_dets if d.get("source") == "manual")
    survivors = len(final_dets) - added
    removed = max(0, len(original_dets) - survivors)
    return added, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default="sku_match.db", help="SQLite database path")
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    parser.add_argument(
        "--keep-old-column",
        action="store_true",
        help="Do not drop the superseded detection_diff column",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE})").fetchall()}
        if not cols:
            print(f"ERROR: table {TABLE} not found in {args.db}")
            return 1

        if not args.dry_run:
            for col in ("detections_added", "detections_removed"):
                if col not in cols:
                    conn.execute(
                        f"ALTER TABLE {TABLE} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0"
                    )
                    print(f"added column: {TABLE}.{col}")

        rows = conn.execute(
            f"SELECT id, ai_result_json, final_result_json, detection_diff FROM {TABLE}"  # noqa: S608
            if "detection_diff" in cols
            else f"SELECT id, ai_result_json, final_result_json FROM {TABLE}"  # noqa: S608
        ).fetchall()

        updated = skipped = 0
        sum_added = sum_removed = 0
        for row in rows:
            row_id, ai_json, final_json = row[0], row[1], row[2]
            if not final_json:
                added, removed = 0, 0
            else:
                try:
                    original = json.loads(ai_json or "{}").get("detections", [])
                    final_dets = json.loads(final_json).get("detections", [])
                    if not isinstance(original, list) or not isinstance(final_dets, list):
                        raise ValueError("detections is not a list")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"  row {row_id}: SKIPPED corrupt JSON ({e})")
                    skipped += 1
                    continue
                added, removed = compute_split_metrics(original, final_dets)
            sum_added += added
            sum_removed += removed
            cur = conn.execute(
                f"SELECT detections_added, detections_removed FROM {TABLE} WHERE id = ?",  # noqa: S608
                (row_id,),
            ).fetchone() if not args.dry_run else None
            if cur is not None and cur == (added, removed):
                continue  # already correct
            if not args.dry_run:
                conn.execute(
                    f"UPDATE {TABLE} SET detections_added = ?, detections_removed = ? "  # noqa: S608
                    "WHERE id = ?",
                    (added, removed, row_id),
                )
            updated += 1

        if not args.dry_run:
            conn.commit()

        # Drop superseded column (SQLite >= 3.35)
        dropped = False
        if (
            not args.dry_run
            and not args.keep_old_column
            and "detection_diff" in cols
            and sqlite3.sqlite_version_info >= (3, 35)
        ):
            conn.execute(f"ALTER TABLE {TABLE} DROP COLUMN detection_diff")  # noqa: S608
            conn.commit()
            dropped = True

        print(
            f"\nrows={len(rows)} updated={updated} skipped={skipped} "
            f"sum_added={sum_added} sum_removed={sum_removed} "
            f"old_column_dropped={dropped}{' (dry-run)' if args.dry_run else ''}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
