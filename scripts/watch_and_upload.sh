#!/bin/bash
# Watch for a model checkpoint file to appear, upload to remote storage via rclone,
# verify the upload, then automatically destroy the vast.ai instance.
#
# Usage:
#   Edit the variables below, then run:
#     chmod +x watch_and_upload.sh
#     nohup ./watch_and_upload.sh > watch_and_upload.log 2>&1 &

# --- Configuration ---
WATCH_FILE="/workspace/<experiment_dir>/<checkpoint_step>/<target_file>"  # file to wait for before uploading
UPLOAD_DIR="/workspace/<experiment_dir>"                                   # entire directory to upload
REMOTE_DEST="<rclone_remote>:<remote_path>/<experiment_dir>"
CHECK_INTERVAL=60
VAST_API_KEY="<your_vast_ai_api_key>"
VAST_INSTANCE_ID="<your_instance_id>"
# ---------------------

echo "[$(date)] Starting monitor for ${WATCH_FILE}"

# Phase 1: Wait for target model file to appear
while true; do
    if [ -f "${WATCH_FILE}" ]; then
        echo "[$(date)] Found ${WATCH_FILE}!"
        break
    fi
    echo "[$(date)] ${WATCH_FILE} not found yet. Checking again in ${CHECK_INTERVAL}s..."
    sleep ${CHECK_INTERVAL}
done

# Phase 2: Upload to remote storage
echo "[$(date)] Starting rclone upload of ${UPLOAD_DIR}..."
rclone copy "${UPLOAD_DIR}" "${REMOTE_DEST}" -P --drive-chunk-size 128M --transfers 8 --retries 5
RCLONE_EXIT=$?

if [ ${RCLONE_EXIT} -ne 0 ]; then
    echo "[$(date)] ERROR: rclone upload failed with exit code ${RCLONE_EXIT}"
    exit 1
fi
echo "[$(date)] rclone upload command finished (exit code: ${RCLONE_EXIT})"

# Phase 3: Verify upload by comparing file counts and sizes
echo "[$(date)] Verifying upload with rclone check..."
rclone check "${UPLOAD_DIR}" "${REMOTE_DEST}" --one-way 2>&1
CHECK_EXIT=$?

if [ ${CHECK_EXIT} -eq 0 ]; then
    echo "[$(date)] Verification PASSED: all local files exist on remote with matching sizes/hashes."
else
    echo "[$(date)] WARNING: rclone check reported differences (exit code: ${CHECK_EXIT})."
    echo "[$(date)] Attempting rclone size comparison as fallback..."

    LOCAL_SIZE=$(rclone size "${UPLOAD_DIR}" --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('bytes',0))" 2>/dev/null)
    REMOTE_SIZE=$(rclone size "${REMOTE_DEST}" --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('bytes',0))" 2>/dev/null)

    echo "[$(date)] Local size:  ${LOCAL_SIZE} bytes"
    echo "[$(date)] Remote size: ${REMOTE_SIZE} bytes"

    if [ -n "${LOCAL_SIZE}" ] && [ -n "${REMOTE_SIZE}" ] && [ "${LOCAL_SIZE}" -eq "${REMOTE_SIZE}" ]; then
        echo "[$(date)] Size match confirmed. Proceeding."
    else
        echo "[$(date)] ERROR: Size mismatch or could not verify. Aborting instance destroy."
        exit 1
    fi
fi

# Phase 4: Destroy vast.ai instance
echo "[$(date)] Destroying vast.ai instance ${VAST_INSTANCE_ID}..."
DESTROY_RESPONSE=$(curl -s -w "\n%{http_code}" --request DELETE \
    --url "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/" \
    --header "Authorization: Bearer ${VAST_API_KEY}")

HTTP_CODE=$(echo "${DESTROY_RESPONSE}" | tail -1)
BODY=$(echo "${DESTROY_RESPONSE}" | head -n -1)

echo "[$(date)] Response (HTTP ${HTTP_CODE}): ${BODY}"

if [ "${HTTP_CODE}" -ge 200 ] && [ "${HTTP_CODE}" -lt 300 ]; then
    echo "[$(date)] vast.ai instance destroyed successfully."
else
    echo "[$(date)] WARNING: Failed to destroy vast.ai instance (HTTP ${HTTP_CODE})."
fi

echo "[$(date)] All done."
