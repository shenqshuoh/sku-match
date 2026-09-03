"""Backfill `source: "model"` on existing detection entries in stored recognition logs.

Adds the source flag to every detection entry lacking one, in both ai_result_json
(original detect output) and final_result_json (corrected view). Entries that
already carry a source (e.g. "manual" from the add fixType) are preserved as-is.
user_correction_json is never touched (it holds fix submissions, not detections).

Idempotent: safe to re-run. Usage:
    uv run python scripts/backfill_source_flag.py [--db sku_match.db] [--dry-run]
"""

import argparse
import json
import sqlite3

MODEL_SOURCE = "model"


def backfill_result_json(raw: str | None) -> tuple[str | None, int, int]:
    """Fill source on detections inside one result JSON blob.

    Returns (new_raw, filled_count, manual_count). new_raw is None when the
    column is empty or nothing changed (caller skips the UPDATE then).
    """
    if not raw:
        return None, 0, 0
    data = json.loads(raw)
    detections = data.get("detections") if isinstance(data, dict) else None
    if not isinstance(detections, list):
        return None, 0, 0
    filled = manual = 0
    changed = False
    for det in detections:
        if not isinstance(det, dict):
            continue
        existing = det.get("source")
        if existing == MODEL_SOURCE:
            continue
        if existing is None:
            det["source"] = MODEL_SOURCE
            filled += 1
            changed = True
        else:
            manual += 1  # preserve "manual" (or any explicit non-model value)
    if not changed:
        return None, filled, manual
    return json.dumps(data, ensure_ascii=False), filled, manual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="sku_match.db", help="Path to the SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, task_id, ai_result_json, final_result_json FROM recognition_log"
    ).fetchall()

    stats = {"rows": len(rows), "ai_filled": 0, "final_filled": 0, "manual_kept": 0, "errors": 0}
    updates: list[tuple[str, str | None, str | None, int]] = []

    for row in rows:
        try:
            new_ai, f1, m1 = backfill_result_json(row["ai_result_json"])
            new_final, f2, m2 = backfill_result_json(row["final_result_json"])
        except (json.JSONDecodeError, TypeError):
            stats["errors"] += 1
            print(f"  SKIP (corrupt JSON) id={row['id']} taskId={row['task_id']}")
            continue
        stats["ai_filled"] += f1
        stats["final_filled"] += f2
        stats["manual_kept"] += m1 + m2
        if new_ai is not None or new_final is not None:
            updates.append((row["task_id"], new_ai, new_final, row["id"]))

    for task_id, new_ai, new_final, row_id in updates:
        if not args.dry_run:
            conn.execute(
                "UPDATE recognition_log SET ai_result_json = COALESCE(?, ai_result_json), "
                "final_result_json = COALESCE(?, final_result_json) WHERE id = ?",
                (new_ai, new_final, row_id),
            )

    if not args.dry_run:
        conn.commit()
    conn.close()

    print(
        f"rows={stats['rows']}  ai_result filled={stats['ai_filled']}  "
        f"final_result filled={stats['final_filled']}  "
        f"manual kept={stats['manual_kept']}  errors={stats['errors']}  "
        f"{'(dry-run, nothing written)' if args.dry_run else 'committed'}"
    )


if __name__ == "__main__":
    main()
