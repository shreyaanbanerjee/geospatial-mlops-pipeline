#!/usr/bin/env bash
set -e
# usage: ./serve/test_curl.sh before.tif after.tif
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 before.tif after.tif"
  exit 1
fi
BEFORE="$1"
AFTER="$2"
URL="http://localhost:8000/predict"
curl -s -X POST "$URL" -F "before=@${BEFORE}" -F "after=@${AFTER}" -F "tile_size=256" -F "overlap=32" | jq .
