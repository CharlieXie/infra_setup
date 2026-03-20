#!/bin/bash
# vast_instance_info.sh
#
# Auto-detects vast.ai API key and current instance ID, then optionally
# destroys the instance.
#
# Usage:
#   ./vast_instance_info.sh              # show detected info only
#   ./vast_instance_info.sh --destroy    # show info + destroy (with confirmation)

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─────────────────────────────────────────────────────────────────────────────
# Detect vast.ai API key
#   The vastai CLI stores it in ~/.vast_api_key
# ─────────────────────────────────────────────────────────────────────────────
detect_api_key() {
    local key_file="$HOME/.vast_api_key"
    if [[ -f "$key_file" ]]; then
        cat "$key_file" 2>/dev/null | tr -d '[:space:]'
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Detect current vast.ai instance ID
#
#   Method 1 (primary):  VAST_CONTAINERLABEL env var  →  "C.<instance_id>"
#   Method 2 (fallback): Parse from `vastai show instances` using the
#                         machine's IP / hostname.
# ─────────────────────────────────────────────────────────────────────────────
detect_instance_id() {
    # Method 1: environment variable set by vast.ai runtime
    if [[ -n "${VAST_CONTAINERLABEL:-}" ]]; then
        local id="${VAST_CONTAINERLABEL#C.}"
        if [[ "$id" =~ ^[0-9]+$ ]]; then
            echo "$id"
            return 0
        fi
    fi

    # Method 2: vastai CLI — pick the first "running" instance
    if command -v vastai &>/dev/null; then
        local id
        id=$(vastai show instances --raw 2>/dev/null \
            | python3 -c "
import sys, json
rows = json.load(sys.stdin)
for r in rows:
    if r.get('actual_status') == 'running':
        print(r['id']); break
" 2>/dev/null)
        if [[ -n "$id" ]]; then
            echo "$id"
            return 0
        fi
    fi

    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Destroy instance via vastai CLI, with API fallback
# ─────────────────────────────────────────────────────────────────────────────
destroy_instance() {
    local instance_id="$1"
    local api_key="$2"

    echo ""
    echo -e "${YELLOW}>>> Destroying instance ${BOLD}${instance_id}${NC} ${YELLOW}...${NC}"

    # Try CLI first
    if command -v vastai &>/dev/null; then
        local output
        output=$(vastai destroy instance "$instance_id" 2>&1)
        local rc=$?
        echo "    vastai CLI output: $output  (exit code: $rc)"
        if [[ $rc -eq 0 ]]; then
            echo -e "${GREEN}>>> Instance destroyed successfully via CLI${NC}"
            return 0
        fi
        echo -e "${YELLOW}    CLI failed, trying API fallback ...${NC}"
    fi

    # API fallback
    if [[ -z "$api_key" ]]; then
        echo -e "${RED}>>> ERROR: no API key — cannot destroy via API${NC}"
        return 1
    fi

    local resp http_code body
    resp=$(curl -s -w "\n%{http_code}" --request DELETE \
        "https://console.vast.ai/api/v0/instances/${instance_id}/" \
        --header "Authorization: Bearer ${api_key}")
    http_code=$(echo "$resp" | tail -1)
    body=$(echo "$resp" | sed '$d')
    echo "    API response HTTP ${http_code}: ${body}"

    if [[ "$http_code" -ge 200 ]] && [[ "$http_code" -lt 300 ]]; then
        echo -e "${GREEN}>>> Instance destroyed via API${NC}"
        return 0
    fi

    echo -e "${RED}>>> ERROR: could not destroy instance (HTTP ${http_code})${NC}"
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║   vast.ai Instance Info & Destroy Tool      ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    # --- Detect API key ---
    echo -e "${BOLD}[1] API Key${NC}  (source: ~/.vast_api_key)"
    local api_key
    api_key=$(detect_api_key)
    if [[ -n "$api_key" ]]; then
        local masked="${api_key:0:8}...${api_key: -8}"
        echo -e "    ${GREEN}Found:${NC} ${masked}  (${#api_key} chars)"
    else
        echo -e "    ${RED}NOT FOUND${NC}"
    fi
    echo ""

    # --- Detect instance ID ---
    echo -e "${BOLD}[2] Instance ID${NC}  (source: \$VAST_CONTAINERLABEL env var)"
    local instance_id
    instance_id=$(detect_instance_id)
    if [[ -n "$instance_id" ]]; then
        echo -e "    ${GREEN}Found:${NC} ${instance_id}"
        echo -e "    ${CYAN}VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-<not set>}${NC}"
    else
        echo -e "    ${RED}NOT FOUND${NC}"
    fi
    echo ""

    # --- Cross-verify with vastai CLI ---
    echo -e "${BOLD}[3] Cross-verification${NC}  (vastai show instances)"
    if command -v vastai &>/dev/null; then
        vastai show instances 2>/dev/null
    else
        echo -e "    ${YELLOW}vastai CLI not available${NC}"
    fi
    echo ""

    # --- Destroy if requested ---
    if [[ "${1:-}" == "--destroy" ]]; then
        if [[ -z "$instance_id" ]]; then
            echo -e "${RED}Cannot destroy: instance ID not detected${NC}"
            exit 1
        fi

        echo -e "${RED}${BOLD}WARNING: You are about to DESTROY instance ${instance_id}!${NC}"
        echo -e "${RED}This will terminate all running processes on this machine.${NC}"
        echo ""
        read -r -p "Type the instance ID to confirm: " confirm
        if [[ "$confirm" != "$instance_id" ]]; then
            echo -e "${YELLOW}>>> Aborted (input did not match instance ID)${NC}"
            exit 0
        fi

        destroy_instance "$instance_id" "$api_key"
    else
        echo -e "${CYAN}To destroy this instance, run:${NC}"
        echo -e "  ${BOLD}$0 --destroy${NC}"
    fi

    echo ""
}

main "$@"
