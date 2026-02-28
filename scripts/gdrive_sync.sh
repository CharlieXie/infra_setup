#!/bin/bash
# gdrive_sync.sh - Google Drive sync tool
#
# Features:
#   - Auto-installs rclone if missing
#   - Guides through Google Drive OAuth token setup
#   - Direct path input for upload/download (no interactive browser)
#   - Preserves directory name when copying directories
#
# Usage:
#   chmod +x gdrive_sync.sh
#   ./gdrive_sync.sh                                        # interactive menu
#   ./gdrive_sync.sh download <gdrive路径> <本地目录>        # 直接下载
#   ./gdrive_sync.sh upload   <本地路径>   <gdrive目录>      # 直接上传
#
# Examples:
#   ./gdrive_sync.sh download "MyFolder/dataset.zip" "/workspace/data"
#   ./gdrive_sync.sh download "MyFolder/subdir"      "/workspace"
#   ./gdrive_sync.sh upload   "/workspace/data"      "Backups/data"
#   ./gdrive_sync.sh upload   "/workspace/file.zip"  "Backups"

set -uo pipefail

REMOTE_NAME="gdrive"
RCLONE_CONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"

# Transfer tuning
TRANSFERS=8               # parallel file transfers
DRIVE_CHUNK_SIZE="128M"   # chunk size for Google Drive multipart uploads

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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
# Detect whether a remote path is a directory
# ─────────────────────────────────────────────────────────────────────────────
is_remote_dir() {
    local path="$1"
    # Append "/" and try to list — succeeds only if path is a directory
    rclone lsf "${REMOTE_NAME}:${path}/" --max-depth 1 &>/dev/null 2>&1
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

    src="${src%/}"
    dst_dir="${dst_dir%/}"

    local src_name
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
# Args: [gdrive_path] [local_dir]
# ─────────────────────────────────────────────────────────────────────────────
do_download() {
    local gdrive_path="${1:-}"
    local local_dir="${2:-}"

    echo ""
    echo -e "${CYAN}${BOLD}=== ⬇  从 Google Drive 下载 ===${NC}"
    echo ""

    if [[ -z "$gdrive_path" ]]; then
        echo -e "  Google Drive 路径示例:"
        echo -e "    ${YELLOW}MyFolder/dataset.zip${NC}   （单个文件）"
        echo -e "    ${YELLOW}MyFolder/subdir${NC}         （整个目录）"
        echo ""
        read -r -p "请输入 Google Drive 源路径: " gdrive_path
    fi

    if [[ -z "$gdrive_path" ]]; then
        echo -e "${RED}错误: 路径不能为空${NC}"
        return 1
    fi

    if [[ -z "$local_dir" ]]; then
        echo ""
        echo -e "  本地目标目录示例: ${YELLOW}/workspace/data${NC}"
        echo ""
        read -r -p "请输入本地目标目录: " local_dir
    fi

    if [[ -z "$local_dir" ]]; then
        echo -e "${RED}错误: 本地目录不能为空${NC}"
        return 1
    fi

    local src="${REMOTE_NAME}:${gdrive_path}"
    local src_is_dir=false

    echo -e "\n${CYAN}>>> 正在检测远程路径类型...${NC}"
    if is_remote_dir "$gdrive_path"; then
        src_is_dir=true
        echo -e "    类型: ${BLUE}目录${NC}"
    else
        echo -e "    类型: ${BLUE}文件${NC}"
    fi

    run_copy "$src" "$src_is_dir" "$local_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# Upload: Local → Google Drive
# Args: [local_path] [gdrive_dir]
# ─────────────────────────────────────────────────────────────────────────────
do_upload() {
    local local_path="${1:-}"
    local gdrive_dir="${2:-}"

    echo ""
    echo -e "${CYAN}${BOLD}=== ⬆  上传到 Google Drive ===${NC}"
    echo ""

    if [[ -z "$local_path" ]]; then
        echo -e "  本地路径示例:"
        echo -e "    ${YELLOW}/workspace/data/file.zip${NC}   （单个文件）"
        echo -e "    ${YELLOW}/workspace/data${NC}             （整个目录）"
        echo ""
        read -r -p "请输入本地源路径: " local_path
    fi

    if [[ -z "$local_path" ]]; then
        echo -e "${RED}错误: 路径不能为空${NC}"
        return 1
    fi

    if [[ ! -e "$local_path" ]]; then
        echo -e "${RED}错误: 本地路径不存在: $local_path${NC}"
        return 1
    fi

    if [[ -z "$gdrive_dir" ]]; then
        echo ""
        echo -e "  Google Drive 目标目录示例: ${YELLOW}Backups/myproject${NC}"
        echo ""
        read -r -p "请输入 Google Drive 目标目录: " gdrive_dir
    fi

    if [[ -z "$gdrive_dir" ]]; then
        echo -e "${RED}错误: Google Drive 目录不能为空${NC}"
        return 1
    fi

    local src_is_dir=false
    if [[ -d "$local_path" ]]; then
        src_is_dir=true
        echo -e "    类型: ${BLUE}目录${NC}"
    else
        echo -e "    类型: ${BLUE}文件${NC}"
    fi

    local dst="${REMOTE_NAME}:${gdrive_dir}"
    run_copy "$local_path" "$src_is_dir" "$dst"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main menu loop (循环模式)
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
# Startup mode selector — shown when no CLI args are provided
# ─────────────────────────────────────────────────────────────────────────────
choose_mode() {
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║             Google Drive 同步工具  — 启动向导            ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  请选择使用方式："
    echo ""
    echo -e "  ${YELLOW}[1]${NC}  交互菜单   — 选择操作后逐步引导输入路径"
    echo -e "  ${YELLOW}[2]${NC}  直接输入   — 现在输入源地址和目标地址立即执行"
    echo ""
    echo -e "  ${BLUE}也可直接传参运行（跳过此菜单）：${NC}"
    echo -e "    ${GREEN}$(basename "$0") download${NC}  <gdrive路径>  <本地目录>"
    echo -e "    ${GREEN}$(basename "$0") upload${NC}    <本地路径>    <gdrive目录>"
    echo ""
    echo -e "  路径示例："
    echo -e "    ${YELLOW}gdrive路径${NC}  →  ${CYAN}MyFolder/dataset.zip${NC}  或  ${CYAN}MyFolder/subdir${NC}"
    echo -e "    ${YELLOW}本地路径${NC}    →  ${CYAN}/workspace/data/file.zip${NC}  或  ${CYAN}/workspace/data${NC}"
    echo ""
    echo -e "  ${YELLOW}[q]${NC}  退出"
    echo ""

    while true; do
        read -r -p "请选择 [1/2/q]: " mode
        case "$mode" in
            1)
                main_menu
                return
                ;;
            2)
                echo ""
                echo -e "  请选择操作："
                echo -e "  ${YELLOW}[1]${NC}  ⬇  下载 (Google Drive → 本地)"
                echo -e "  ${YELLOW}[2]${NC}  ⬆  上传 (本地 → Google Drive)"
                echo ""
                read -r -p "请选择 [1/2]: " op
                case "$op" in
                    1) do_download ;;
                    2) do_upload ;;
                    *) echo -e "${RED}无效选择${NC}" ; continue ;;
                esac
                return
                ;;
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

    # 3. Handle command-line arguments or show mode selector
    if [[ $# -ge 1 ]]; then
        case "$1" in
            download)
                do_download "${2:-}" "${3:-}"
                ;;
            upload)
                do_upload "${2:-}" "${3:-}"
                ;;
            *)
                echo -e "${RED}未知命令: $1${NC}"
                echo ""
                echo -e "用法:"
                echo -e "  $0                                       # 启动向导"
                echo -e "  $0 download <gdrive路径> <本地目录>       # 直接下载"
                echo -e "  $0 upload   <本地路径>   <gdrive目录>     # 直接上传"
                exit 1
                ;;
        esac
    else
        choose_mode
    fi
}

main "$@"
