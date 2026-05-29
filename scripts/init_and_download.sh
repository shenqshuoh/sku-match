#!/usr/bin/env bash
set -euo pipefail

# Reinitialise DB + Chroma, add SKUs from reference/, then download processed crops.
# Usage: bash scripts/init_and_download.sh

API_URL="http://localhost:8000"
API_KEY="4ELxhB_knxPmcYfOo4n4VtOmiXAx0WonzgI142VXrIA"
REF_DIR="/opt/drink_imgs/reference"
OUT_DIR="/opt/drink_imgs/reference_processed"
DB_PATH="/root/sku-match/sku_match.db"
CHROMA_PATH="/root/sku-match/chroma_data"
CDN_DOMAIN="https://vr.jihaihotpot.com/"
ORIGIN_DOMAIN="http://iovip-z2.qiniuio.com"

echo "=== Step 1: Stop API ==="
ssh sku-match-jihai-gpu "systemctl stop sku-match || true"
sleep 2

echo "=== Step 2: Wipe DB + Chroma ==="
rm -f "$DB_PATH"
rm -rf "$CHROMA_PATH"

echo "=== Step 3: Start API ==="
ssh sku-match-jihai-gpu "systemctl start sku-match"
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
    echo "ERROR: API did not start. Check: ssh sku-match-jihai-gpu 'systemctl status sku-match'"
    exit 1
fi

echo "=== Step 4: Add SKUs from reference folders ==="
sku_counter=1
for folder in $(ls -1 "$REF_DIR" | sort); do
    dirpath="$REF_DIR/$folder"
    [ -d "$dirpath" ] || continue

    name_file="$dirpath/name.txt"
    if [ -f "$name_file" ]; then
        sku_name=$(cat "$name_file" | tr -d '\n')
    else
        sku_name="$folder"
    fi

    sku_id=$(printf "t%02d" $sku_counter)
    train_job_id="job_${sku_id}_$(date +%s)"

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
        echo "SKIP $sku_id ($folder): no images"
        sku_counter=$((sku_counter + 1))
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
        sku_counter=$((sku_counter + 1))
        continue
    }
    echo "  -> $response"
    sku_counter=$((sku_counter + 1))
done

echo ""
echo "=== Step 5: Wait for embedding tasks ==="
for i in $(seq 1 120); do
    pending=$(curl -sf "$API_URL/api/v1/goods/sku/list?page=1&pageSize=50" \
        -H "X-API-Key: $API_KEY" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pending = [s['skuId'] for s in data.get('data',{}).get('list',[]) if s.get('trainStatus') not in ('SUCCESS','FAILED')]
if pending:
    print(','.join(pending))
else:
    print('')
" 2>/dev/null)
    if [ -z "$pending" ]; then
        echo "All embedding tasks completed."
        break
    fi
    echo "  Still pending: $pending ($i/120)"
    sleep 5
done

echo ""
echo "=== Final SKU status ==="
curl -sf "$API_URL/api/v1/goods/sku/list?page=1&pageSize=50" \
    -H "X-API-Key: $API_KEY" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('data',{}).get('list',[]):
    media_count = len(s.get('medias',[]))
    print(f\"  {s['skuId']:4s}  {s['trainStatus']:10s}  {media_count:3d} medias  {s['skuName']}\")
"

echo ""
echo "=== Step 6: Download processed crops from Qiniu ==="
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Build sku_id -> folder name mapping from media URLs
mapping=$(curl -sf "$API_URL/api/v1/goods/sku/list?page=1&pageSize=50" \
    -H "X-API-Key: $API_KEY" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data['data']['list']:
    sid = s['skuId']
    for m in s['medias']:
        url = m['mediaUrl']
        parts = url.split('/reference/')[-1].split('/')
        folder = parts[0] if parts else sid
        print(f'{sid}\t{folder}')
        break
")

total=0
ok=0
fail=0
while IFS= read -r line; do
    cdn_url=$(echo "$line" | grep -o 'https://[^ ]*')
    # Extract sku_id from filename: crop_{sku_id}_{media_id}_{rand}.jpg
    filename=$(basename "$cdn_url")
    sku_id=$(echo "$filename" | awk -F'_' '{print $2}')

    folder=$(echo "$mapping" | awk -v sid="$sku_id" '$1 == sid {print $2; exit}')
    folder=${folder:-$sku_id}

    mkdir -p "$OUT_DIR/$folder"

    key=${cdn_url#$CDN_DOMAIN}
    origin_url="$ORIGIN_DOMAIN/$key"
    outfile="$OUT_DIR/$folder/$(basename "$cdn_url")"

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
