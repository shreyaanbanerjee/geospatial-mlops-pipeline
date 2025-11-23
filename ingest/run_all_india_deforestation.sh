#!/usr/bin/env bash
set -euo pipefail

PROJECT=""
SERVICE_ACCOUNT=""
KEY_FILE=""
BEFORE_START=""
BEFORE_END=""
AFTER_START=""
AFTER_END=""
DEST=""
DRIVE_FOLDER="EO_Exports"
GCS_BUCKET=""
GCS_PREFIX=""
CLOUD_PCT=60
SCALE=10
CRS="EPSG:4326"
AOI_GLOB='aoi/india_*.geojson'
DRY_RUN=false

usage() {
  grep '^#' "$0" | sed -e 's/^# //'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --key-file) KEY_FILE="$2"; shift 2 ;;
    --before) BEFORE_START="$2"; BEFORE_END="$3"; shift 3 ;;
    --after) AFTER_START="$2"; AFTER_END="$3"; shift 3 ;;
    --dest) DEST="$2"; shift 2 ;;
    --drive-folder) DRIVE_FOLDER="$2"; shift 2 ;;
    --gcs-bucket) GCS_BUCKET="$2"; shift 2 ;;
    --gcs-prefix) GCS_PREFIX="$2"; shift 2 ;;
    --cloud-pct) CLOUD_PCT="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --crs) CRS="$2"; shift 2 ;;
    --aoi-glob) AOI_GLOB="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then echo "[ERR] --project required"; exit 1; fi
if [[ -z "$BEFORE_START" || -z "$AFTER_START" ]]; then echo "[ERR] --before and --after required"; exit 1; fi
if [[ "$DEST" != "drive" && "$DEST" != "gcs" ]]; then echo "[ERR] --dest must be drive|gcs"; exit 1; fi
if [[ "$DEST" == "drive" && -z "$DRIVE_FOLDER" ]]; then echo "[ERR] --drive-folder required for drive"; exit 1; fi
if [[ "$DEST" == "gcs" && -z "$GCS_BUCKET" ]]; then echo "[ERR] --gcs-bucket required for gcs"; exit 1; fi
if [[ -n "$SERVICE_ACCOUNT" && -z "$KEY_FILE" ]]; then echo "[ERR] --key-file required when using --service-account"; exit 1; fi

RUN_PY="python3 ingest/gee_ingest.py"

COMMON=( --project "$PROJECT" --cloud-pct "$CLOUD_PCT" --scale "$SCALE" --crs "$CRS" )
if [[ -n "$SERVICE_ACCOUNT" ]]; then
  COMMON+=( --service-account "$SERVICE_ACCOUNT" --key-file "$KEY_FILE" )
fi

shopt -s nullglob
AOIS=( $AOI_GLOB )
if [[ ${#AOIS[@]} -eq 0 ]]; then echo "[WARN] No AOIs found with $AOI_GLOB"; exit 0; fi
echo "[INFO] Found ${#AOIS[@]} AOIs."

i=0
for AOI in "${AOIS[@]}"; do
  ((i++))
  NAME="$(basename "$AOI" .geojson)"
  CMD=( $RUN_PY "${COMMON[@]}" --aoi "$AOI" --before "$BEFORE_START" "$BEFORE_END" --after "$AFTER_START" "$AFTER_END" --name "$NAME" )
  if [[ "$DEST" == "drive" ]]; then
    CMD+=( --drive-folder "$DRIVE_FOLDER" )
  else
    CMD+=( --gcs-bucket "$GCS_BUCKET" --gcs-prefix "$GCS_PREFIX/$NAME" )
  fi
  echo "[$i/${#AOIS[@]}] ${CMD[*]}"
  if $DRY_RUN; then
    continue
  fi
  LOG="logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p logs
  "${CMD[@]}" 2>&1 | tee "$LOG"
  echo "[INFO] Completed $NAME (log: $LOG)"
done

echo "[DONE] All AOIs processed."
