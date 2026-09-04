"""One-time migration: enrich stored userCorrection entries with detailed fields.

Every entry in user_correction_json gains server-resolved values by replaying
the correction list in order against ai_result_json (the immutable original),
using the same mutation rules as the fix endpoint:

  reassign   -> {fixType, itemId, roiRect (bbox at fix time), skuId (new),
                 skuIdOld (skuId in the view right before this fix)}
  remove     -> {fixType, itemId, roiRect (bbox at removal), skuId (at removal)}
  adjust-roi -> {fixType, itemId, roiRect (new), roiRectOld (pre-fix), skuId}
  add        -> {fixType, itemId (server-assigned), roiRect, skuId}

Add ids follow the monotonic never-reuse rule (high-water over original ids
and every id ever assigned to an add). For entries that already carry an int
itemId (previously enriched), the id is trusted and the replay continues from
it — so the script is idempotent.

After replay, the end-state view is verified against the stored
final_result_json detections (projection: itemId/bbox/classId/className/
detectionConf/skuId/source — fields the fix logic writes or carries verbatim).
Any mismatch (corrupt JSON, lost history from the pre-incremental latest-wins
era, hand-edited data) skips the row with a report — stored data is never
guessed. Rows are only rewritten when the replay reproduces the stored state.

Legacy note: rows fixed under the pre-incremental latest-wins semantics store
only the LAST submission's items; replaying them onto the original reproduces
the stored final state, so they are enriched — but at-fix-time field values
derive from the original view (earlier lost submissions cannot be recovered).

Usage:
    uv run python scripts/enrich_user_correction.py [--db sku_match.db] [--dry-run]
"""

import argparse
import copy
import json
import sqlite3
import sys

TABLE = "recognition_log"

# Detection fields the fix logic writes or carries verbatim — used for the
# end-state check. skuName/matchScore/skuDistribution/matchedVectorTags are
# excluded (indexer lookups / model outputs not reproducible in replay).
_COMPARE_FIELDS = ("itemId", "bbox", "classId", "className", "detectionConf", "skuId", "source")


def _det_projection(dets: list) -> list:
    return [tuple(d.get(f) for f in _COMPARE_FIELDS) for d in dets]


def replay_and_enrich(original_dets: list, entries: list) -> tuple[list, list] | None:
    """Replay entries onto a copy of the original view.

    Returns (enriched_entries, final_view) or None when the entries cannot be
    replayed consistently (unknown shape, unresolvable itemId).
    """
    view = copy.deepcopy(original_dets)
    ever_ids = {d.get("itemId") for d in original_dets if isinstance(d.get("itemId"), int)}
    enriched = []

    for e in entries:
        if not isinstance(e, dict):
            return None
        ftype = e.get("fixType")
        if ftype == "add":
            sku_id = e.get("skuId") or ""
            roi = list(e["roiRect"]) if isinstance(e.get("roiRect"), list) else []
            iid = e.get("itemId")
            if not isinstance(iid, int):
                # Legacy add (itemId never logged): assign the next monotonic id
                iid = (max(ever_ids) + 1) if ever_ids else 1
            ever_ids.add(iid)
            view.append({
                "itemId": iid,
                "bbox": roi,
                "classId": None,
                "className": None,
                "detectionConf": None,
                "skuId": sku_id,
                "skuName": None,
                "matchScore": None,
                "skuDistribution": None,
                "matchedVectorTags": None,
                "source": "manual",
            })
            enriched.append({"fixType": "add", "itemId": iid, "roiRect": roi, "skuId": sku_id})
            continue

        iid = e.get("itemId")
        if not isinstance(iid, int):
            return None
        det = next((d for d in view if d.get("itemId") == iid), None)
        if det is None:
            return None
        if ftype == "remove":
            enriched.append({
                "fixType": "remove",
                "itemId": iid,
                "roiRect": list(det.get("bbox") or []),
                "skuId": det.get("skuId", ""),
            })
            view.remove(det)
        elif ftype == "reassign":
            sku_id = e.get("skuId") or ""
            enriched.append({
                "fixType": "reassign",
                "itemId": iid,
                "roiRect": list(det.get("bbox") or []),
                "skuId": sku_id,
                "skuIdOld": det.get("skuId", ""),
            })
            det["skuId"] = sku_id
        elif ftype == "adjust-roi":
            roi = list(e["roiRect"]) if isinstance(e.get("roiRect"), list) else []
            enriched.append({
                "fixType": "adjust-roi",
                "itemId": iid,
                "roiRect": roi,
                "roiRectOld": list(det.get("bbox") or []),
                "skuId": det.get("skuId", ""),
            })
            det["bbox"] = roi
        else:
            return None
    return enriched, view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default="sku_match.db", help="SQLite database path")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        f"SELECT task_id, ai_result_json, final_result_json, user_correction_json "
        f"FROM {TABLE} WHERE user_correction_json IS NOT NULL"
    ).fetchall()

    stats = {"rows": len(rows), "updated": 0, "unchanged": 0, "skipped": 0}
    type_counts: dict[str, int] = {}
    for task_id, ai_js, fin_js, uc_js in rows:
        try:
            original = json.loads(ai_js)
            entries = json.loads(uc_js)
        except json.JSONDecodeError as exc:
            print(f"SKIP {task_id}: corrupt JSON ({exc})")
            stats["skipped"] += 1
            continue
        if not isinstance(original, dict) or not isinstance(entries, list):
            print(f"SKIP {task_id}: unexpected payload shape")
            stats["skipped"] += 1
            continue
        original_dets = original.get("detections")
        if not isinstance(original_dets, list):
            print(f"SKIP {task_id}: original has no detections list")
            stats["skipped"] += 1
            continue
        if fin_js:
            try:
                stored_final = json.loads(fin_js)
            except json.JSONDecodeError as exc:
                print(f"SKIP {task_id}: corrupt final_result_json ({exc})")
                stats["skipped"] += 1
                continue
        else:
            stored_final = None

        result = replay_and_enrich(original_dets, entries)
        if result is None:
            print(f"SKIP {task_id}: correction list not replayable (lost/invalid history)")
            stats["skipped"] += 1
            continue
        enriched, view = result

        if stored_final is not None and _det_projection(view) != _det_projection(
            stored_final.get("detections", [])
        ):
            print(f"SKIP {task_id}: replay end-state != stored final_result_json")
            stats["skipped"] += 1
            continue

        if json.dumps(enriched, ensure_ascii=False) == json.dumps(entries, ensure_ascii=False):
            stats["unchanged"] += 1
            continue
        for e in enriched:
            type_counts[e["fixType"]] = type_counts.get(e["fixType"], 0) + 1
        if not args.dry_run:
            conn.execute(
                f"UPDATE {TABLE} SET user_correction_json = ? WHERE task_id = ?",
                (json.dumps(enriched, ensure_ascii=False), task_id),
            )
        stats["updated"] += 1

    if not args.dry_run:
        conn.commit()
    conn.close()

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}rows={stats['rows']} updated={stats['updated']} "
          f"unchanged={stats['unchanged']} skipped={stats['skipped']}")
    print(f"{prefix}entries enriched by type: {type_counts or '{}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
