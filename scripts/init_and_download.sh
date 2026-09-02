#!/usr/bin/env bash
set -euo pipefail

# Reinitialise DB + Chroma, add SKUs from SKUDB/, then download processed crops.
# SKUDB layout: $REF_DIR/<sku_id>/*.jpg  with  $REF_DIR/sku_names.csv  (sku_id,sku_name).
# The folder name IS the sku_id; sku_name comes from sku_names.csv (fallback: folder name).
# Usage: bash scripts/init_and_download.sh [--test N | -t N | -T]
#   -t, --test N   Process only the first N SKU folders (useful for quick test runs).
#   -T             Shorthand for -t 20.

API_URL="http://localhost:8000"
API_KEY="4ELxhB_knxPmcYfOo4n4VtOmiXAx0WonzgI142VXrIA"
REF_DIR="/opt/SKUDB"
NAME_CSV="$REF_DIR/sku_names.csv"   # sku_id -> sku_name mapping (UTF-8 CSV, optional)
OUT_DIR="/opt/SKUDB_processed"
DB_PATH="/root/sku-match/sku_match.db"
CHROMA_PATH="/root/sku-match/chroma_data"
CDN_DOMAIN="https://vr.jihaihotpot.com/"
ORIGIN_DOMAIN="http://iovip-z2.qiniuio.com"

# --- Argument parsing ---
TEST_LIMIT=0  # 0 = process all SKUs
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--test)
            TEST_LIMIT="${2:?Error: -t/--test requires a number}"
            shift 2
            ;;
        -T)
            TEST_LIMIT=20
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/init_and_download.sh [-t N | --test N | -T]"
            echo "  -t, --test N   Process only the first N SKU folders"
            echo "  -T             Shorthand for -t 20"
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'" >&2
            exit 1
            ;;
    esac
done

if [ "$TEST_LIMIT" -gt 0 ]; then
    echo "*** TEST MODE: processing first $TEST_LIMIT SKUs only ***"
fi

echo "=== Step 1: Stop API ==="
systemctl stop sku-match || true
sleep 2

echo "=== Step 2: Wipe DB + Chroma + Patches + Colors + Log ==="
rm -f "$DB_PATH"
rm -rf "$CHROMA_PATH"
rm -rf /root/sku-match/data/patches
rm -rf /root/sku-match/data/colors
> /tmp/sku-match-api.log

echo "=== Step 3: Start API ==="
systemctl start sku-match
sleep 5

for i in $(seq 1 20); do
    if curl -sf "$API_URL/health" > /dev/null; then
        echo "API is up."
        break
    fi
    echo "Waiting for API... ($i/20)"
    sleep 2
done
if ! curl -sf "$API_URL/health" > /dev/null; then
    echo "ERROR: API did not start. Check: systemctl status sku-match"
    exit 1
fi

echo "=== Step 4: Add SKUs from SKUDB folders ==="

# Load sku_id -> sku_name mapping from CSV (utf-8-sig strips BOM; csv module handles
# any commas in names). name_map stays in scope for the step 6 prefix matching.
declare -A name_map=()
if [ -f "$NAME_CSV" ]; then
    while IFS=$'\t' read -r sid sname; do
        [ -n "$sid" ] && name_map["$sid"]="$sname"
    done < <(python3 -c "
import csv, sys
with open('$NAME_CSV', encoding='utf-8-sig', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header
    for row in reader:
        if len(row) >= 2:
            sys.stdout.write(row[0] + '\t' + row[1] + '\n')
")
    echo "Loaded ${#name_map[@]} sku names from $NAME_CSV"
else
    echo "WARNING: $NAME_CSV not found; falling back to folder name as sku_name"
fi

sku_count=0
for folder in $(ls -1 "$REF_DIR" | sort); do
    dirpath="$REF_DIR/$folder"
    [ -d "$dirpath" ] || continue

    sku_count=$((sku_count + 1))
    if [ "$TEST_LIMIT" -gt 0 ] && [ "$sku_count" -gt "$TEST_LIMIT" ]; then
        echo "Test limit reached ($TEST_LIMIT SKUs). Stopping."
        break
    fi

    # sku_name from the CSV (fallback: folder name); folder's numeric prefix is
    # stripped because the API now auto-numbers every submitted (bare) skuId.
    # On a fresh DB with sequentially-numbered folders, renumbering reproduces
    # identical final ids (sorted order, starting at 100001).
    sku_id="$(printf '%s' "$folder" | sed -E 's/^[0-9]+_//')"
    sku_name="${name_map[$folder]:-$folder}"
    train_job_id="job_${folder}_$(date +%s)"

    # Read ignore list
    ignore_file="$dirpath/ignore.txt"
    declare -A ignore_set
    if [ -f "$ignore_file" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && ignore_set["$line"]=1
        done < "$ignore_file"
    fi

    files_json="["
    first=true
    for img in $(find "$dirpath" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort); do
        fname=$(basename "$img")
        if [ "${ignore_set[$fname]+isset}" ]; then
            echo "  SKIP $fname (ignored)"
            continue
        fi
        if [ "$first" = true ]; then
            first=false
        else
            files_json+=","
        fi
        files_json+="\"$img\""
    done
    files_json+="]"
    unset ignore_set

    if [ "$files_json" = "[]" ]; then
        echo "SKIP $sku_id ($sku_name): no images"
        continue
    fi

    img_count=$(echo "$files_json" | tr ',' '\n' | wc -l | tr -d ' ')
    echo "Adding $sku_id: $sku_name ($img_count images)..."

    response=$(curl -sf -X POST "$API_URL/api/v1/goods/sku/new" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"skuId\":\"$sku_id\",\"skuName\":\"$sku_name\",\"files\":$files_json,\"trainJobId\":\"$train_job_id\"}" \
    ) || {
        echo "FAIL $sku_id: $response"
        continue
    }
    echo "  -> $response"
done

echo ""
echo "=== Step 5: Wait for embedding tasks ==="

# Event-driven output: print a line only when a SKU reaches a terminal state.
# Heartbeat every ~30s so the terminal isn't silent.
declare -A seen=()
done_count=0
total_skus=0
poll_max=120

for i in $(seq 1 "$poll_max"); do
    # Fetch all SKUs. Output: "total<TAB>processing" then one line per terminal SKU:
    # "status<TAB>sku_id<TAB>sku_name<TAB>media_count"
    output=$(curl -sf "$API_URL/api/v1/goods/sku/list?page=1&size=9999" \
        -H "X-API-Key: $API_KEY" | python3 -c "
import sys, json
data = json.load(sys.stdin)
skus = data.get('data', {}).get('list', [])

def terminal(st):
    st = st or ''
    return st in ('completed', 'failed')

processing = sum(1 for s in skus if not terminal(s.get('trainStatus')))
print(f'{len(skus)}\t{processing}')
for s in skus:
    st = s.get('trainStatus') or ''
    if terminal(st):
        print(f'{st}\t{s[\"skuId\"]}\t{s.get(\"skuName\", \"\")}\t{len(s.get(\"medias\", []))}')
" 2>/dev/null)

    IFS=$'\t' read -r total_skus processing <<< "$(echo "$output" | head -1)"

    # Print only newly-terminal SKUs
    new=0
    while IFS=$'\t' read -r st sid sname mcount; do
        [ -z "$st" ] && continue
        [ -n "${seen[$sid]+isset}" ] && continue
        seen[$sid]=1
        done_count=$((done_count + 1))
        new=$((new + 1))
        if [ "$st" = "completed" ]; then
            echo "  [$done_count/$total_skus] ✓ $sid  ($mcount imgs)"
        else
            echo "  [$done_count/$total_skus] ✗ $sid  $st"
        fi
    done < <(echo "$output" | tail -n +2)

    if [ "${processing:-0}" -eq 0 ] 2>/dev/null; then
        echo "All embedding tasks completed ($done_count/$total_skus)."
        break
    fi

    # Heartbeat every 6 polls (~30s) when nothing new happened
    if [ "$new" -eq 0 ] && [ $((i % 6)) -eq 0 ]; then
        echo "  [$done_count/$total_skus done, $processing still embedding, poll $i/$poll_max]"
    fi

    sleep 5
done

echo ""
echo "=== Embedding Results ==="
curl -sf "$API_URL/api/v1/goods/sku/list?page=1&size=9999" \
    -H "X-API-Key: $API_KEY" | python3 -c "
import sys, json
data = json.load(sys.stdin)
skus = data.get('data', {}).get('list', [])

def terminal(st):
    st = st or ''
    return st in ('completed', 'failed')

success = [s for s in skus if (s.get('trainStatus') or '') == 'completed']
failed  = [s for s in skus if (s.get('trainStatus') or '') == 'failed']
pending = [s for s in skus if not terminal(s.get('trainStatus'))]

if success:
    print(f'  ✓ SUCCESS:  {len(success):3d} SKUs')
if failed:
    print(f'  ✗ FAILED:   {len(failed):3d} SKUs')
    for s in failed:
        imgs = len(s.get('medias', []))
        print(f'      {s[\"skuId\"]:<24s}  {s.get(\"trainStatus\", \"\")}  ({imgs} imgs)  {s.get(\"skuName\", \"\")}')
if pending:
    print(f'  ⏳ PENDING:  {len(pending):3d} SKUs (not yet completed)')
    for s in pending:
        print(f'      {s[\"skuId\"]:<24s}  {s.get(\"skuName\", \"\")}')
print()
print(f'  Total: {len(skus)} SKUs')
"

echo ""
echo "=== Step 6: Download processed crops from Qiniu ==="
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Processed-crop CDN URLs exist only in the API log ("Uploaded crop_... -> <cdn>"),
# so we grep it. sku_id now contains an underscore (e.g. 100014_nfsqkqs), so positional
# awk splitting is ambiguous; instead match the crop filename prefix against the known
# folder names in name_map above. On a standard sequential rebuild the API-assigned
# ids equal the folder names, so output goes to $OUT_DIR/$folder.
total=0
ok=0
fail=0
while IFS= read -r line; do
    # Log line (possibly prefixed by timestamp/level): "... Uploaded crop_<sku_id>_<media_id>_<rand>.jpg -> https://<cdn>/..."
    cropname="${line#*Uploaded }"
    cropname="${cropname%% -> *}"
    cdn_url="${line#* -> }"

    sku_id=""
    for id in "${!name_map[@]}"; do
        if [[ "$cropname" == crop_${id}_* ]]; then
            sku_id="$id"
            break
        fi
    done
    sku_id="${sku_id:-unknown}"

    mkdir -p "$OUT_DIR/$sku_id"

    key=${cdn_url#$CDN_DOMAIN}
    origin_url="$ORIGIN_DOMAIN/$key"
    outfile="$OUT_DIR/$sku_id/$cropname"

    total=$((total + 1))
    if curl -sf -o "$outfile" -H "Host: vr.jihaihotpot.com" "$origin_url" 2>/dev/null; then
        ok=$((ok + 1))
    else
        echo "FAIL: $cdn_url"
        fail=$((fail + 1))
    fi

    if [ $((total % 20)) -eq 0 ]; then
        echo "  Downloaded $ok/$total..."
    fi
done < <(grep 'Uploaded.*crop_' /tmp/sku-match-api.log)

echo ""
echo "Done: $ok downloaded, $fail failed out of $total"
echo "Saved to $OUT_DIR/"
for d in "$OUT_DIR"/*/; do
    name=$(basename "$d")
    count=$(ls -1 "$d" | wc -l | tr -d ' ')
    echo "  $name: $count images"
done
