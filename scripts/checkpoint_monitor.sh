#!/bin/bash
# checkpoint_monitor.sh
#
# Monitors training checkpoints, uploads each to Google Drive via rclone as
# soon as it's confirmed saved, then destroys the vast.ai instance after the
# final checkpoint is uploaded.
#
# Usage:
#   chmod +x checkpoint_monitor.sh
#   ./checkpoint_monitor.sh 2>&1 | tee -a /workspace/infra_setup/scripts/checkpoint_monitor.log

set -uo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_BASE="/workspace/openpi/checkpoints/waypoint_joint_libero_sg_03"
TRAIN_LOG="/workspace/openpi/logs/waypoint_joint_libero_sg_031.log"
REMOTE_NAME="gdrive"
GDRIVE_DEST="workspace/checkpoints/waypoint_joint_libero_sg_03"
TOTAL_STEPS=1200
SAVE_INTERVAL=100
CHECK_INTERVAL=60          # seconds between scans
UPLOAD_MAX_RETRIES=3

# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect vast.ai credentials
# ─────────────────────────────────────────────────────────────────────────────
detect_vast_instance_id() {
    # Primary: VAST_CONTAINERLABEL env var → "C.<instance_id>"
    if [[ -n "${VAST_CONTAINERLABEL:-}" ]]; then
        local id="${VAST_CONTAINERLABEL#C.}"
        if [[ "$id" =~ ^[0-9]+$ ]]; then
            echo "$id"
            return 0
        fi
    fi
    # Fallback: first running instance from vastai CLI
    if command -v vastai &>/dev/null; then
        vastai show instances --raw 2>/dev/null \
            | python3 -c "
import sys, json
for r in json.load(sys.stdin):
    if r.get('actual_status') == 'running':
        print(r['id']); break
" 2>/dev/null
        return 0
    fi
    return 1
}

detect_vast_api_key() {
    local key_file="$HOME/.vast_api_key"
    if [[ -f "$key_file" ]]; then
        cat "$key_file" 2>/dev/null | tr -d '[:space:]'
    fi
}

VAST_INSTANCE_ID=$(detect_vast_instance_id)
VAST_API_KEY=$(detect_vast_api_key)

if [[ -z "$VAST_INSTANCE_ID" ]]; then
    echo "FATAL: could not detect vast.ai instance ID" >&2
    exit 1
fi
if [[ -z "$VAST_API_KEY" ]]; then
    echo "WARNING: could not detect vast.ai API key (will rely on vastai CLI)" >&2
fi

UPLOADED_TRACKER="/workspace/infra_setup/scripts/.uploaded_steps"
touch "$UPLOADED_TRACKER"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

is_step_uploaded()   { grep -qx "$1" "$UPLOADED_TRACKER" 2>/dev/null; }
mark_step_uploaded() { echo "$1" >> "$UPLOADED_TRACKER"; }

is_save_confirmed() {
    grep -q "Saved checkpoint step ${1} " "$TRAIN_LOG" 2>/dev/null
}

is_checkpoint_complete() {
    local d="$1"
    [[ -f "$d/model.safetensors" ]] && [[ -f "$d/metadata.pt" ]]
}

is_training_running() {
    pgrep -f "train_waypoint_joint.py" >/dev/null 2>&1
}

# ─────────────────────────────────────────────────────────────────────────────
# Upload one checkpoint step (with retries + verification)
# ─────────────────────────────────────────────────────────────────────────────
upload_checkpoint() {
    local step="$1"
    local src="$CHECKPOINT_BASE/$step"
    local dst="${REMOTE_NAME}:${GDRIVE_DEST}/${step}"

    if ! is_checkpoint_complete "$src"; then
        log "SKIP step $step — files incomplete"
        return 1
    fi

    local local_bytes
    local_bytes=$(du -sb "$src" 2>/dev/null | awk '{print $1}')
    log "UPLOAD START step $step  (${local_bytes} bytes)  →  $dst"

    local attempt=0
    while (( attempt < UPLOAD_MAX_RETRIES )); do
        attempt=$((attempt + 1))
        log "  attempt $attempt/$UPLOAD_MAX_RETRIES ..."

        rclone copy "$src" "$dst" \
            --drive-chunk-size 128M \
            --transfers 8 \
            --retries 5 \
            --stats-one-line \
            -v 2>&1

        local rc=$?
        if [[ $rc -ne 0 ]]; then
            log "  rclone exited with code $rc"
            (( attempt < UPLOAD_MAX_RETRIES )) && sleep 30
            continue
        fi

        local remote_bytes
        remote_bytes=$(rclone size "$dst" --json 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('bytes',0))" 2>/dev/null)

        if [[ "$local_bytes" == "$remote_bytes" ]]; then
            log "  VERIFIED step $step  (local=$local_bytes  remote=$remote_bytes)"
            mark_step_uploaded "$step"
            return 0
        fi

        log "  SIZE MISMATCH step $step  (local=$local_bytes  remote=${remote_bytes:-?})"
        (( attempt < UPLOAD_MAX_RETRIES )) && sleep 30
    done

    log "  GAVE UP on step $step after $UPLOAD_MAX_RETRIES attempts"
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Destroy vast.ai instance (CLI first, API fallback)
# ─────────────────────────────────────────────────────────────────────────────
destroy_instance() {
    log "DESTROY vast.ai instance $VAST_INSTANCE_ID ..."

    if command -v vastai &>/dev/null; then
        vastai destroy instance "$VAST_INSTANCE_ID" 2>&1
        if [[ $? -eq 0 ]]; then
            log "  Destroyed via vastai CLI"
            return 0
        fi
        log "  vastai CLI failed, trying API fallback ..."
    fi

    if [[ -z "${VAST_API_KEY:-}" ]]; then
        log "  ERROR: no API key — cannot destroy instance!"
        return 1
    fi

    local resp http_code body
    resp=$(curl -s -w "\n%{http_code}" --request DELETE \
        "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/" \
        --header "Authorization: Bearer ${VAST_API_KEY}")
    http_code=$(echo "$resp" | tail -1)
    body=$(echo "$resp" | sed '$d')
    log "  API response HTTP $http_code: $body"

    if [[ "$http_code" -ge 200 ]] && [[ "$http_code" -lt 300 ]]; then
        log "  Destroyed via API"
        return 0
    fi

    log "  ERROR: could not destroy instance!"
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Scan & upload any un-uploaded checkpoints whose save is confirmed
# Returns 0 if at least one new checkpoint was uploaded.
# ─────────────────────────────────────────────────────────────────────────────
scan_and_upload() {
    local uploaded_any=1   # 1 = false in bash exit-code sense

    for step_dir in "$CHECKPOINT_BASE"/*/; do
        [[ -d "$step_dir" ]] || continue
        local step
        step=$(basename "$step_dir")
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        is_step_uploaded "$step" && continue

        if is_save_confirmed "$step"; then
            upload_checkpoint "$step" && uploaded_any=0
        fi
    done

    return $uploaded_any
}

# ─────────────────────────────────────────────────────────────────────────────
# Final sweep: upload everything that exists (no log-confirmation needed,
# since the process already exited).
# ─────────────────────────────────────────────────────────────────────────────
final_sweep() {
    log "Running final sweep ..."
    for step_dir in "$CHECKPOINT_BASE"/*/; do
        [[ -d "$step_dir" ]] || continue
        local step
        step=$(basename "$step_dir")
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        is_step_uploaded "$step" && continue

        log "  Final sweep: step $step"
        upload_checkpoint "$step"
    done

    # Also upload wandb_run_id.txt
    local wandb_file="$CHECKPOINT_BASE/wandb_run_id.txt"
    if [[ -f "$wandb_file" ]]; then
        log "  Uploading wandb_run_id.txt ..."
        rclone copy "$wandb_file" "${REMOTE_NAME}:${GDRIVE_DEST}/" --retries 3 2>&1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    log "╔══════════════════════════════════════════════════════╗"
    log "║   Checkpoint Monitor & Upload Service               ║"
    log "╚══════════════════════════════════════════════════════╝"
    log "  Checkpoint dir : $CHECKPOINT_BASE"
    log "  Google Drive   : ${REMOTE_NAME}:${GDRIVE_DEST}"
    log "  Train log      : $TRAIN_LOG"
    log "  Total steps    : $TOTAL_STEPS (save every $SAVE_INTERVAL)"
    log "  Check interval : ${CHECK_INTERVAL}s"
    log "  Vast instance  : $VAST_INSTANCE_ID  (auto-detected from \$VAST_CONTAINERLABEL)"
    log "  Vast API key   : ${VAST_API_KEY:0:8}...  (auto-detected from ~/.vast_api_key)"
    log "  Tracker file   : $UPLOADED_TRACKER"
    log ""

    # Pre-existing uploaded steps (from a previous run of this script)
    if [[ -s "$UPLOADED_TRACKER" ]]; then
        log "  Previously uploaded: $(sort -n "$UPLOADED_TRACKER" | tr '\n' ' ')"
    fi
    log "──────────────────────────────────────────────────────"

    # ── Main loop ──
    while true; do
        scan_and_upload

        if ! is_training_running; then
            log "Training process is no longer running."
            sleep 30   # let any final I/O flush
            final_sweep

            # Summary
            log "══════════════════════════════════════════════════════"
            log "  Upload complete."
            if [[ -s "$UPLOADED_TRACKER" ]]; then
                log "  Uploaded steps: $(sort -n "$UPLOADED_TRACKER" | tr '\n' ' ')"
            fi
            log "══════════════════════════════════════════════════════"

            log "Proceeding to destroy vast.ai instance ..."
            destroy_instance
            break
        fi

        log "Training running … next scan in ${CHECK_INTERVAL}s"
        sleep "$CHECK_INTERVAL"
    done

    log "Monitor finished. Goodbye."
}

main
