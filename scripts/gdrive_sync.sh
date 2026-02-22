#!/bin/bash
# gdrive_sync.sh - Interactive Google Drive sync tool
#
# Features:
#   - Auto-installs rclone if missing
#   - Guides through Google Drive OAuth token setup
#   - Interactive file/directory browser for both local and remote
#   - Preserves directory name when copying directories
#
# Usage:
#   chmod +x gdrive_sync.sh
#   ./gdrive_sync.sh

set -uo pipefail

REMOTE_NAME="gdrive"
RCLONE_CONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"

# Transfer tuning
TRANSFERS=8               # parallel file transfers
DRIVE_CHUNK_SIZE="128M"   # chunk size for Google Drive multipart uploads (larger = faster for big files)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Globals set by browse functions
SELECTED_PATH=""
SELECTED_IS_DIR=false

# ─────────────────────────────────────────────────────────────────────────────
# Install rclone
# ─────────────────────────────────────────────────────────────────────────────
install_rclone() {
    echo -e "${YELLOW}>>> 正在安装 rclone...${NC}"
    curl -fsSL https://rclone.org/install.sh | sudo bash
    echo -e "${GREEN}>>> rclone 安装完成${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Configure Google Drive remote
# ─────────────────────────────────────────────────────────────────────────────
configure_gdrive() {
    echo ""
    echo -e "${CYAN}${BOLD}=== 配置 Google Drive 授权 ===${NC}"
    echo ""
    echo -e "由于此服务器无法打开浏览器，需要在${BOLD}本地有浏览器的机器${NC}上完成授权。"
    echo ""
    echo -e "请在本地机器上执行以下步骤："
    echo -e "  1. 安装 rclone（若未安装）: https://rclone.org/install/"
    echo -e "  2. 运行以下命令："
    echo ""
    echo -e "     ${GREEN}${BOLD}rclone authorize \"drive\"${NC}"
    echo ""
    echo -e "  3. 浏览器会自动打开 Google 授权页面，登录并授权"
    echo -e "  4. 授权成功后终端会输出一段 JSON token，例如："
    echo -e "     ${YELLOW}{\"access_token\":\"...\",\"token_type\":\"Bearer\",\"refresh_token\":\"...\",\"expiry\":\"...\"}${NC}"
    echo -e "  5. 复制完整的 JSON 字符串（从 { 到 } 全部复制）"
    echo ""
    read -r -p "请粘贴 token (完整 JSON): " token

    if [[ -z "$token" ]]; then
        echo -e "${RED}错误: token 不能为空，退出${NC}"
        exit 1
    fi

    mkdir -p "$(dirname "$RCLONE_CONF")"

    # Remove existing gdrive config block if any
    if grep -q "^\[$REMOTE_NAME\]" "$RCLONE_CONF" 2>/dev/null; then
        echo -e "${YELLOW}>>> 检测到已有 [$REMOTE_NAME] 配置，将覆盖...${NC}"
        # Use python to safely remove the section
        python3 - "$RCLONE_CONF" "$REMOTE_NAME" <<'PYEOF'
import sys, configparser
conf_file, section = sys.argv[1], sys.argv[2]
config = configparser.RawConfigParser()
config.read(conf_file)
if config.has_section(section):
    config.remove_section(section)
with open(conf_file, 'w') as f:
    config.write(f)
PYEOF
    fi

    # Append new config
    cat >> "$RCLONE_CONF" << EOF

[$REMOTE_NAME]
type = drive
scope = drive
token = $token
EOF

    echo ""
    echo -e "${YELLOW}>>> 正在测试连接...${NC}"
    if rclone lsd "${REMOTE_NAME}:" --max-depth 1 &>/dev/null; then
        echo -e "${GREEN}>>> 连接成功！Google Drive 远程 '${REMOTE_NAME}' 已配置完毕${NC}"
    else
        echo -e "${RED}>>> 连接失败，token 可能不正确，请重试${NC}"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Local filesystem browser
# Sets: SELECTED_PATH, SELECTED_IS_DIR
# ─────────────────────────────────────────────────────────────────────────────
browse_local() {
    local current_path
    current_path="$(realpath "${1:-/workspace}" 2>/dev/null || echo "${1:-/workspace}")"

    while true; do
        echo ""
        echo -e "${BLUE}┌─── 本地文件浏览器 ─────────────────────────────────────┐${NC}"
        echo -e "${BLUE}│${NC} ${BOLD}$current_path${NC}"
        echo -e "${BLUE}└────────────────────────────────────────────────────────┘${NC}"
        echo -e "  ${GREEN}[ 0]${NC}  ✓  选择当前目录: $(basename "$current_path")"
        [[ "$current_path" != "/" ]] && echo -e "  ${YELLOW}[ b]${NC}  ↑  返回上级目录"
        echo ""

        # Collect dirs and files (ignore hidden)
        local -a dirs=() files=()
        while IFS= read -r -d $'\0' entry; do
            dirs+=("$(basename "$entry")")
        done < <(find "$current_path" -maxdepth 1 -mindepth 1 -type d ! -name '.*' -print0 2>/dev/null | sort -z)

        while IFS= read -r -d $'\0' entry; do
            files+=("$(basename "$entry")")
        done < <(find "$current_path" -maxdepth 1 -mindepth 1 -type f ! -name '.*' -print0 2>/dev/null | sort -z)

        local idx=1
        local dir_count=${#dirs[@]}

        if [[ $dir_count -gt 0 ]]; then
            echo -e "  ${CYAN}📁 目录:${NC}"
            for d in "${dirs[@]}"; do
                printf "  ${YELLOW}[%2d]${NC}  📁  %s\n" "$idx" "$d"
                idx=$(( idx + 1 ))
            done
        fi

        if [[ ${#files[@]} -gt 0 ]]; then
            echo -e "  ${CYAN}📄 文件:${NC}"
            for f in "${files[@]}"; do
                printf "  ${YELLOW}[%2d]${NC}  📄  %s\n" "$idx" "$f"
                idx=$(( idx + 1 ))
            done
        fi

        local total=$(( idx - 1 ))
        echo ""
        read -r -p "请输入选择 [0=选择当前目录, b=返回, 数字=进入/选择]: " choice

        case "$choice" in
            0)
                SELECTED_PATH="$current_path"
                SELECTED_IS_DIR=true
                return 0
                ;;
            b|B)
                [[ "$current_path" != "/" ]] && current_path="$(dirname "$current_path")"
                ;;
            '' | *[!0-9]*)
                echo -e "${RED}无效输入${NC}"
                ;;
            *)
                if [[ $choice -ge 1 && $choice -le $total ]]; then
                    if [[ $choice -le $dir_count ]]; then
                        current_path="$current_path/${dirs[$((choice-1))]}"
                    else
                        SELECTED_PATH="$current_path/${files[$((choice-1-dir_count))]}"
                        SELECTED_IS_DIR=false
                        return 0
                    fi
                else
                    echo -e "${RED}输入超出范围 (1-$total)${NC}"
                fi
                ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Google Drive browser
# Sets: SELECTED_PATH, SELECTED_IS_DIR
# ─────────────────────────────────────────────────────────────────────────────
browse_remote() {
    local current_path="${1:-}"

    while true; do
        local display="${REMOTE_NAME}:${current_path}"

        echo ""
        echo -e "${BLUE}┌─── Google Drive 浏览器 ────────────────────────────────┐${NC}"
        echo -e "${BLUE}│${NC} ${BOLD}$display${NC}"
        echo -e "${BLUE}└────────────────────────────────────────────────────────┘${NC}"
        echo -e "  ${GREEN}[ 0]${NC}  ✓  选择当前目录: ${current_path:-/}"
        [[ -n "$current_path" ]] && echo -e "  ${YELLOW}[ b]${NC}  ↑  返回上级目录"
        echo -e "  ${CYAN}  正在加载目录内容...${NC}"

        local -a dirs=() files=()
        while IFS= read -r line; do
            [[ -n "$line" ]] && dirs+=("$line")
        done < <(rclone lsd "${REMOTE_NAME}:${current_path}" 2>/dev/null | awk '{print $NF}' | sort)

        while IFS= read -r line; do
            [[ -n "$line" ]] && files+=("$line")
        done < <(rclone lsf "${REMOTE_NAME}:${current_path}" --files-only 2>/dev/null | sort)

        # Overwrite "loading..." line
        printf "\033[1A\033[2K"

        local idx=1
        local dir_count=${#dirs[@]}

        if [[ $dir_count -gt 0 ]]; then
            echo -e "  ${CYAN}📁 目录:${NC}"
            for d in "${dirs[@]}"; do
                printf "  ${YELLOW}[%2d]${NC}  📁  %s\n" "$idx" "$d"
                idx=$(( idx + 1 ))
            done
        fi

        if [[ ${#files[@]} -gt 0 ]]; then
            echo -e "  ${CYAN}📄 文件:${NC}"
            for f in "${files[@]}"; do
                printf "  ${YELLOW}[%2d]${NC}  📄  %s\n" "$idx" "$f"
                idx=$(( idx + 1 ))
            done
        fi

        local total=$(( idx - 1 ))
        echo ""
        read -r -p "请输入选择 [0=选择当前目录, b=返回, 数字=进入/选择]: " choice

        case "$choice" in
            0)
                SELECTED_PATH="${REMOTE_NAME}:${current_path}"
                SELECTED_IS_DIR=true
                return 0
                ;;
            b|B)
                if [[ -n "$current_path" ]]; then
                    current_path="$(dirname "$current_path")"
                    [[ "$current_path" == "." ]] && current_path=""
                fi
                ;;
            '' | *[!0-9]*)
                echo -e "${RED}无效输入${NC}"
                ;;
            *)
                if [[ $choice -ge 1 && $choice -le $total ]]; then
                    if [[ $choice -le $dir_count ]]; then
                        local sel="${dirs[$((choice-1))]}"
                        current_path="${current_path:+$current_path/}$sel"
                    else
                        local sel="${files[$((choice-1-dir_count))]}"
                        SELECTED_PATH="${REMOTE_NAME}:${current_path:+$current_path/}$sel"
                        SELECTED_IS_DIR=false
                        return 0
                    fi
                else
                    echo -e "${RED}输入超出范围 (1-$total)${NC}"
                fi
                ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Execute rclone copy with proper directory handling
# Usage: run_copy <src> <src_is_dir> <dst_dir>
# If src is a dir, copies to dst_dir/<dirname>/ (preserves dir name)
# If src is a file, copies into dst_dir/
# ─────────────────────────────────────────────────────────────────────────────
run_copy() {
    local src="$1"
    local src_is_dir="$2"
    local dst_dir="$3"

    # Strip trailing slashes for consistent basename extraction
    src="${src%/}"
    dst_dir="${dst_dir%/}"

    local src_name
    # For remote paths like "gdrive:foo/bar", basename of the part after ":"
    if [[ "$src" == *:* ]]; then
        src_name="$(basename "${src#*:}")"
    else
        src_name="$(basename "$src")"
    fi

    local final_dst
    if [[ "$src_is_dir" == "true" ]]; then
        final_dst="${dst_dir}/${src_name}"
    else
        final_dst="${dst_dir}"
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  源:     ${GREEN}${src}${NC}"
    echo -e "  目标:   ${GREEN}${final_dst}${NC}"
    if [[ "$src_is_dir" == "true" ]]; then
        echo -e "  说明:   复制目录，目标中将包含目录名 '${BOLD}${src_name}${NC}'"
    else
        echo -e "  说明:   复制单个文件"
    fi
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -r -p "确认执行？[y/N] " confirm

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}>>> 已取消${NC}"
        return 0
    fi

    echo ""
    rclone copy "$src" "$final_dst" --progress \
        --transfers="$TRANSFERS" \
        --drive-chunk-size="$DRIVE_CHUNK_SIZE"
    echo ""
    echo -e "${GREEN}>>> 完成！${NC}"
    echo -e "    目标路径: ${BOLD}${final_dst}${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Download: Google Drive → Local
# ─────────────────────────────────────────────────────────────────────────────
do_download() {
    echo ""
    echo -e "${CYAN}${BOLD}=== ⬇  从 Google Drive 下载 ===${NC}"

    echo -e "\n【步骤 1/2】选择 Google Drive 上的源"
    echo -e "  · 按数字导航进入目录"
    echo -e "  · 按 ${GREEN}[0]${NC} 选择当前目录（下载整个目录）"
    echo -e "  · 按数字选择文件（下载单个文件）"
    browse_remote ""
    local src="$SELECTED_PATH"
    local src_is_dir="$SELECTED_IS_DIR"

    echo -e "\n【步骤 2/2】选择本地目标目录"
    echo -e "  · 按数字导航"
    echo -e "  · 按 ${GREEN}[0]${NC} 选择当前目录作为目标"
    browse_local "/workspace"
    local dst_dir="$SELECTED_PATH"

    run_copy "$src" "$src_is_dir" "$dst_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# Upload: Local → Google Drive
# ─────────────────────────────────────────────────────────────────────────────
do_upload() {
    echo ""
    echo -e "${CYAN}${BOLD}=== ⬆  上传到 Google Drive ===${NC}"

    echo -e "\n【步骤 1/2】选择本地源"
    echo -e "  · 按数字导航进入目录"
    echo -e "  · 按 ${GREEN}[0]${NC} 选择当前目录（上传整个目录）"
    echo -e "  · 按数字选择文件（上传单个文件）"
    browse_local "/workspace"
    local src="$SELECTED_PATH"
    local src_is_dir="$SELECTED_IS_DIR"

    echo -e "\n【步骤 2/2】选择 Google Drive 上的目标目录"
    echo -e "  · 按数字导航"
    echo -e "  · 按 ${GREEN}[0]${NC} 选择当前目录作为目标"
    browse_remote ""
    local dst_dir="$SELECTED_PATH"

    run_copy "$src" "$src_is_dir" "$dst_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main menu loop
# ─────────────────────────────────────────────────────────────────────────────
main_menu() {
    while true; do
        echo ""
        echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}${BOLD}║        Google Drive 同步工具             ║${NC}"
        echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════╝${NC}"
        echo -e "  ${YELLOW}[1]${NC}  ⬇  从 Google Drive 下载到本地"
        echo -e "  ${YELLOW}[2]${NC}  ⬆  从本地上传到 Google Drive"
        echo -e "  ${YELLOW}[q]${NC}  ✗  退出"
        echo ""
        read -r -p "请选择操作: " choice

        case "$choice" in
            1) do_download ;;
            2) do_upload ;;
            q|Q)
                echo -e "${GREEN}再见！${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选择，请输入 1、2 或 q${NC}"
                ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${CYAN}${BOLD}Google Drive 同步工具${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # 1. Install rclone if missing
    if ! command -v rclone &>/dev/null; then
        echo -e "${YELLOW}>>> 未检测到 rclone${NC}"
        install_rclone
    else
        echo -e "${GREEN}>>> rclone 已安装: $(rclone --version | head -1)${NC}"
    fi

    # 2. Configure Google Drive remote if not present
    if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE_NAME}:$"; then
        echo -e "${YELLOW}>>> 未找到 Google Drive 配置 (remote: '${REMOTE_NAME}')，开始配置...${NC}"
        configure_gdrive
    else
        echo -e "${GREEN}>>> Google Drive 远程 '${REMOTE_NAME}' 已配置${NC}"
    fi

    # 3. Enter interactive menu
    main_menu
}

main "$@"
