#!/bin/bash
# Claude Code Local — Doctor (macOS and Linux; Windows has scripts/doctor.ps1)
# Checks the core install and every add-on you have installed, and says what to fix.
#
# Usage: bash scripts/doctor.sh [--fix] [--report] [--bench]
#   --fix      safe repairs only: restart your own service that is down, recreate a
#              missing symlink. Never deletes anything.
#   --report   print a privacy-safe block to paste into a bug report (no usernames,
#              home paths, hostnames, IPs, tokens or env values) instead of the full output
#   --bench    also run a 100-token benchmark on the recommended model (macOS)
#
# Exit code: 0 when nothing failed, 1 when at least one check failed.
# Add-ons that were never installed show as "not installed" and are never errors.
#
# Test/override knobs: MLX_PORT (4000), BROKER_PORT (9223), DOCK_PORT (8899),
# FIAOS_PORT (9000), CCL_HOME, MLX_VENV, NARRATE_DIR/NARRATE_DATA, STP_DIR, FIAOS_DIR,
# MOONSTONE_DIR, FAILOVER_DIR, CCL_DOCTOR_LAUNCHCTL, CCL_DOCTOR_OS.

DOCTOR_VERSION="2"
MIN_CLAUDE_VERSION="2.0.0"
DOCTOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$DOCTOR_DIR")"
OS="${CCL_DOCTOR_OS:-$(uname -s)}"
ARCH="$(uname -m)"

MODE_FIX=0; MODE_REPORT=0; MODE_BENCH=0
RESULTS=""          # lines of "status|id|message"
N_PASS=0; N_WARN=0; N_FAIL=0; N_SKIP=0

# ── Colors ────────────────────────────────────────────────────
setup_colors() {
  if [ -t 1 ] && [ "$MODE_REPORT" != 1 ]; then
    C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_CYAN=$'\033[36m'
  else
    C_BOLD=""; C_DIM=""; C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""
  fi
}

say()  { [ "$MODE_REPORT" = 1 ] || echo "$@"; }
info() { say "${C_DIM}·${C_RESET} $*"; }
head_() { say ""; say "${C_BOLD}$*${C_RESET}"; }

# record <pass|warn|fail|skip> <id> <message>
record() {
  RESULTS="${RESULTS}$1|$2|$3
"
  case "$1" in
    pass) N_PASS=$((N_PASS + 1)); say "${C_GREEN}✓${C_RESET} $3" ;;
    warn) N_WARN=$((N_WARN + 1)); say "${C_YELLOW}!${C_RESET} $3" ;;
    fail) N_FAIL=$((N_FAIL + 1)); say "${C_RED}✗${C_RESET} $3" ;;
    skip) N_SKIP=$((N_SKIP + 1)); say "${C_DIM}-${C_RESET} $3" ;;
  esac
}

# $HOME -> ~ for anything we print
tilde() { case "$1" in "$HOME"*) printf '%s' "~${1#"$HOME"}" ;; *) printf '%s' "$1" ;; esac; }

# ── Privacy: redact() reads stdin, writes the same text with nothing personal ──
_sed_escape() { printf '%s' "$1" | sed -e 's/[][\.*^$|/&(){}+?-]/\\&/g'; }

redact() {
  local home user host short
  home="${HOME:-}"; user="${CCL_DOCTOR_USER:-$(id -un 2>/dev/null)}"
  host="${CCL_DOCTOR_HOST:-$(hostname 2>/dev/null)}"; short="${host%%.*}"
  local args=()
  # 1. exact values of this machine
  [ -n "$home" ] && [ "$home" != "/" ] && args+=(-e "s|$(_sed_escape "$home")|~|g")
  [ -n "$host" ] && args+=(-e "s|$(_sed_escape "$host")|<host>|g")
  [ ${#short} -ge 4 ] && args+=(-e "s|$(_sed_escape "$short")|<host>|g")
  # 2. any other home path style, then the bare user name as a whole word
  args+=(-e 's#/Users/[^/[:space:]]+#~#g' -e 's#/home/[^/[:space:]]+#~#g'
         -e 's#[A-Za-z]:\\Users\\[^\\[:space:]]+#~#g')
  if [ ${#user} -ge 2 ]; then
    local u; u="$(_sed_escape "$user")"
    args+=(-e "s#(^|[^A-Za-z0-9_])$u([^A-Za-z0-9_]|\$)#\\1<user>\\2#g"
           -e "s#(^|[^A-Za-z0-9_])$u([^A-Za-z0-9_]|\$)#\\1<user>\\2#g")
  fi
  # 3. network identity and secrets
  args+=(
    -e 's#[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}#<email>#g'
    -e 's#[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}#<ip>#g'
    -e 's#([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}#<mac>#g'
    -e 's#([0-9A-Fa-f]{0,4}:){3,7}[0-9A-Fa-f]{0,4}#<ip>#g'
    -e 's#[A-Za-z0-9-]+\.(local|lan|home|internal|corp)([^A-Za-z0-9]|$)#<host>\2#g'
    -e 's#(sk|pk|rk)-[A-Za-z0-9_-]{8,}#<token>#g'
    -e 's#(ghp|gho|ghu|ghs|ghr|github_pat|hf|xox[abprs])[_-][A-Za-z0-9_-]{8,}#<token>#g'
    -e 's#AKIA[0-9A-Z]{12,}#<token>#g'
    -e 's#([Bb]earer|[Bb]asic)[[:space:]]+[A-Za-z0-9._~+/=-]+#\1 <token>#g'
    -e 's#([A-Za-z_]*([Kk][Ee][Yy]|[Tt][Oo][Kk][Ee][Nn]|[Ss][Ee][Cc][Rr][Ee][Tt]|[Pp][Aa][Ss][Ss][Ww]([Oo][Rr])?[Dd])[A-Za-z_]*)[[:space:]]*[:=][[:space:]]*[^[:space:]]+#\1=<redacted>#g'
    -e 's#(^|[^A-Za-z0-9_])([A-Z][A-Z0-9_]{2,})=[^[:space:]]+#\1\2=<redacted>#g'
    -e 's#[A-Za-z0-9+=_]{32,}#<token>#g'
  )
  sed -E "${args[@]}"
}

# ── Small helpers ─────────────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }

# 0 when $1 >= $2 (dotted numbers)
ver_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN{
    n=split(a,x,"."); m=split(b,y,"."); k=(n>m?n:m);
    for(i=1;i<=k;i++){ xi=x[i]+0; yi=y[i]+0; if(xi>yi) exit 0; if(xi<yi) exit 1 }
    exit 0 }'
}

first_version() { grep -Eo '[0-9]+(\.[0-9]+)+' | head -1; }

# HTTP status code of a local URL, "000" when nothing answers
http_code() {
  if have curl; then
    local c; c="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null)"
    echo "${c:-000}"

  else
    local port; port="$(echo "$1" | sed -E 's#^[a-z]+://[^:/]+:([0-9]+).*#\1#')"
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then echo 200; else echo 000; fi
  fi
}
answers() { [ "$(http_code "http://127.0.0.1:$1${2:-/}")" != "000" ]; }

# sysctl lives in /usr/sbin, which a non-login shell's PATH may lack: then memory read as 0 GB and
# the doctor recommended the 2B tier on any Mac. Call it by its full path when it is there.
SYSCTL="$( [ -x /usr/sbin/sysctl ] && echo /usr/sbin/sysctl || echo sysctl )"

total_ram_gb() {
  if [ "$OS" = Darwin ]; then
    "$SYSCTL" -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}'
  else
    awk '/^MemTotal:/ {print int(($2+524288)/1048576)}' /proc/meminfo 2>/dev/null
  fi
}

cpu_name() {
  if [ "$OS" = Darwin ]; then "$SYSCTL" -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon"
  else awk -F: '/model name/ {gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null; fi
}

gpu_name() {
  if [ "$OS" = Darwin ]; then cpu_name; return; fi
  if have nvidia-smi; then
    local g; g="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    [ -n "$g" ] && { echo "$g"; return; }
  fi
  if have lspci; then
    local l; l="$(lspci 2>/dev/null | grep -iE 'vga|3d|display' | head -1 | sed -E 's/^[^:]+: //; s/^[^:]+: //')"
    [ -n "$l" ] && { echo "$l"; return; }
  fi
  echo "$(cpu_name) (no discrete GPU found)"
}

# ── Services we may restart (the user's own LaunchAgent / systemd user unit) ──
service_unit() {   # $1 = key -> prints label (mac) or unit name (linux), empty when none
  local pat
  case "$1" in
    server)    pat='mlxserver|claude-code-local|ccl' ;;
    broker)    pat='browser-broker|browser-agent' ;;
    moonstone) pat='moonstone' ;;
    remote)    pat='fiaos\.server|fiaos$|FiaOS' ;;
    *) return 0 ;;
  esac
  if [ "$OS" = Darwin ]; then
    local f
    for f in "$HOME"/Library/LaunchAgents/*.plist; do
      [ -e "$f" ] || continue
      f="$(basename "$f" .plist)"
      echo "$f" | grep -qiE "$pat" || continue
      echo "$f" | grep -qiE 'update|tunnel|secbump|caffeinate' && continue
      echo "$f"; return 0
    done
  elif have systemctl; then
    systemctl --user list-unit-files --no-legend 2>/dev/null | awk '{print $1}' | grep -iE "$pat" \
      | grep -viE 'update|tunnel' | head -1
  fi
}

restart_service() {   # $1 = key; returns 0 when a restart was issued
  local unit; unit="$(service_unit "$1")"
  [ -n "$unit" ] || return 1
  if [ "$OS" = Darwin ]; then
    local lc="${CCL_DOCTOR_LAUNCHCTL:-launchctl}" uid; uid="$(id -u)"
    "$lc" kickstart -k "gui/$uid/$unit" >/dev/null 2>&1 && return 0
    "$lc" bootstrap "gui/$uid" "$HOME/Library/LaunchAgents/$unit.plist" >/dev/null 2>&1
  else
    systemctl --user restart "$unit" >/dev/null 2>&1
  fi
}

wait_up() {   # $1 port, $2 path; waits up to ~10s
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do answers "$1" "$2" && return 0; sleep 1; done
  return 1
}

# service_check <id> <label> <port> <path> <key> [1 = a down service is a failure, 0 = warning]
service_check() {
  local id="$1" label="$2" port="$3" path="$4" key="$5" required="${6:-1}"
  if answers "$port" "$path"; then
    record pass "$id" "$label answering on port $port"
    return
  fi
  local hint=""
  if [ "$MODE_FIX" = 1 ]; then
    if restart_service "$key"; then
      if wait_up "$port" "$path"; then
        record pass "$id" "$label was down; restarted your own service, now answering on port $port (fixed)"
        return
      fi
      hint=" (restarted it but it is still not answering)"
    else
      hint=" (no service of yours to restart; start it by hand)"
    fi
  elif [ -n "$(service_unit "$key")" ]; then
    hint=" (run with --fix to restart it)"
  fi
  if [ "$required" = 1 ]; then record fail "$id" "$label not answering on port $port$hint"
  else record warn "$id" "$label not answering on port $port$hint"; fi
}

# link_check <id> <link path> <target> : recreate a missing symlink with --fix
link_check() {
  local id="$1" link="$2" target="$3"
  if [ -e "$link" ]; then record pass "$id" "$(tilde "$link") present"; return; fi
  if [ ! -e "$target" ]; then
    record fail "$id" "$(tilde "$link") missing and so is what it points to (re-run the installer)"; return
  fi
  if [ "$MODE_FIX" = 1 ]; then
    mkdir -p "$(dirname "$link")"
    [ -L "$link" ] && rm -f "$link"      # only ever a dangling symlink, never a real file
    if ln -s "$target" "$link" 2>/dev/null; then
      record pass "$id" "$(tilde "$link") was missing; recreated the symlink (fixed)"; return
    fi
  fi
  record fail "$id" "$(tilde "$link") missing or broken$([ "$MODE_FIX" = 1 ] || echo ' (run with --fix to recreate it)')"
}

# ── Core ──────────────────────────────────────────────────────
ccl_home() { echo "${CCL_HOME:-$HOME/.cache/claude-code-local}"; }

check_claude() {
  local bin="" v c
  if have claude; then bin="claude"
  else for c in "$HOME/.local/bin/claude" "$HOME/.claude/local/claude"; do [ -x "$c" ] && bin="$c" && break; done
  fi
  if [ -z "$bin" ]; then
    record fail claude "claude CLI not found (install Claude Code, then re-run)"; return
  fi
  v="$("$bin" --version 2>/dev/null | first_version)"
  CLAUDE_VER="${v:-unknown}"
  if [ -z "$v" ]; then record warn claude "claude CLI found but its version could not be read"
  elif ver_ge "$v" "$MIN_CLAUDE_VERSION"; then record pass claude "claude CLI $v (need $MIN_CLAUDE_VERSION or newer)"
  else record fail claude "claude CLI $v is older than $MIN_CLAUDE_VERSION (update Claude Code)"; fi
}

check_python() {
  if ! have python3; then record fail python "python3 not found"; return; fi
  PY_VER="$(python3 --version 2>&1 | first_version)"
  if ! ver_ge "$PY_VER" 3.9; then record fail python "Python $PY_VER is too old (need 3.9+)"; return; fi
  if [ "$OS" = Darwin ]; then
    MLX_VENV="${MLX_VENV:-$HOME/.local/mlx-server}"
    if [ ! -x "$MLX_VENV/bin/python" ]; then
      record fail venv "MLX virtualenv missing (run: bash setup.sh)"; return
    fi
    if "$MLX_VENV/bin/python" -c "import mlx_lm" 2>/dev/null; then
      MLX_VER="$("$MLX_VENV/bin/python" -m pip show mlx-lm 2>/dev/null | awk '/^Version:/ {print $2}')"
      record pass venv "Python $PY_VER, MLX virtualenv OK, mlx-lm ${MLX_VER:-installed}"
    else
      record fail venv "MLX virtualenv exists but mlx-lm is not installed (run: bash setup.sh)"
    fi
  else
    if python3 -c "import venv, ssl" 2>/dev/null; then record pass python "Python $PY_VER with venv and ssl"
    else record fail python "Python $PY_VER lacks venv or ssl (Debian/Ubuntu: apt install python3-venv)"; fi
  fi
}

check_models_mac() {
  local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub" n=0 d
  if [ -d "$hub" ]; then for d in "$hub"/models--*; do [ -d "$d" ] && n=$((n + 1)); done; fi
  MODELS_FOUND=$n
  if [ "$n" -gt 0 ]; then record pass models "$n model(s) in the Hugging Face cache"
  else record fail models "no models downloaded yet (run: bash setup.sh)"; fi
}

json_get() {   # json_get <file> <key>
  python3 -I - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
PY
}

check_models_linux() {
  local h cfg; h="$(ccl_home)"; cfg="$h/config.json"
  if [ ! -f "$cfg" ]; then
    record fail models "no model installed (run the installer, or: python3 -I portable/ccl.py install)"
    record fail llama "llama.cpp not installed yet"; return
  fi
  have python3 || { record fail models "cannot read the install config without python3"; return; }
  local gguf mm bin tag
  gguf="$(json_get "$cfg" model_gguf)"; mm="$(json_get "$cfg" mmproj)"
  bin="$(json_get "$cfg" server_bin)"; tag="$(json_get "$cfg" llama_tag)"
  LLAMA_TAG="$tag"
  if [ -f "$gguf" ] && [ -f "$mm" ]; then
    record pass models "model files present ($(json_get "$cfg" model_name), $(json_get "$cfg" quant))"
  elif [ -f "$gguf" ]; then record warn models "model present but its vision projector file is missing (re-run the installer)"
  else record fail models "model file missing (re-run the installer to download it again)"; fi
  if [ ! -x "$bin" ]; then record fail llama "llama-server binary missing (re-run the installer)"; return; fi
  # Present is not enough: on stock Ubuntu 24.04 it could not start because libgomp.so.1 was
  # missing. ldd lists every shared library it can't find as "not found".
  local missing=""
  if have ldd; then
    missing="$(ldd "$bin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ' | sed 's/ $//')"
  fi
  if [ -n "$missing" ]; then
    local hint=" (install the system package that provides it)"
    case "$missing" in *libgomp*) hint=" (Debian/Ubuntu: sudo apt-get install libgomp1)" ;; esac
    record fail llama "llama-server can't start, missing libraries: $missing$hint"
  else
    record pass llama "llama-server present (build ${tag:-unknown}, $(json_get "$cfg" accel))"
  fi
}

check_core() {
  head_ "Core"
  check_claude
  check_python
  if [ "$OS" = Darwin ]; then check_models_mac; else check_models_linux; fi

  local free; free="$(df -k "$HOME" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}')"
  DISK_FREE_GB="${free:-0}"
  if [ "$DISK_FREE_GB" -ge 10 ]; then record pass disk "${DISK_FREE_GB} GB free"
  else record warn disk "only ${DISK_FREE_GB} GB free (models need 10 GB or more)"; fi

  # Server on MLX_PORT. With a service installed it should be up; otherwise it starts
  # on demand with claude-local, so down is only a warning.
  local required=0
  [ -n "$(service_unit server)" ] && required=1
  service_check server "model server" "${MLX_PORT:-4000}" /health server "$required"

  # Symlinks the installers create
  if [ "$OS" = Darwin ] && [ -d "$HOME/.local/mlx-native-server" ]; then
    link_check server-link "$HOME/.local/mlx-native-server/server.py" "$REPO_DIR/proxy/server.py"
  fi
  check_keepgoing
}

# keep going: the command and the shell rc line. setup.sh (macOS) and portable/setup-linux.sh
# (Linux) install both; an older install has neither, so missing is a warning, never a failure.
check_keepgoing() {
  local link="$HOME/.local/bin/keepgoing"
  if [ -e "$link" ] || [ -L "$link" ] || [ "$MODE_FIX" = 1 ]; then
    link_check keepgoing-link "$link" "$REPO_DIR/bin/keepgoing"
  else
    record warn keepgoing-link "keep going not installed: $(tilde "$link") missing (run with --fix to add it)"
  fi
  local rc found=""
  for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
    if [ -f "$rc" ] && grep -q "scripts/keepgoing-shell.sh" "$rc"; then found="$rc"; break; fi
  done
  if [ -n "$found" ]; then
    record pass keepgoing-shell "keep going shell line in $(tilde "$found")"
  else
    record warn keepgoing-shell "no keep going line in ~/.zshrc or ~/.bashrc (re-run the installer to add it)"
  fi
  if [ "$OS" != Darwin ] && ! have python3; then
    record fail keepgoing-python "keep going needs python3 on PATH"
  fi
}

# ── Add-ons: detection mirrors scripts/addons.sh (addon_clone_dir) ──
browser_dir() {
  if [ -n "$BROWSER_BROKER_DIR" ]; then echo "$BROWSER_BROKER_DIR"
  elif [ ! -d "$HOME/.browser-agent/.git" ] && [ -d "$HOME/.browser-broker/.git" ]; then echo "$HOME/.browser-broker"
  else echo "$HOME/.browser-agent"; fi
}
addon_dir() {
  case "$1" in
    voice)     echo "${NARRATE_DIR:-$HOME/NarrateClaude}" ;;
    browser)   browser_dir ;;
    phone)     echo "${STP_DIR:-$HOME/claude-screen-to-phone}" ;;
    remote)    echo "${FIAOS_DIR:-$HOME/FiaOS}" ;;
    moonstone) echo "${MOONSTONE_DIR:-$HOME/moonstone}" ;;
    failover)  echo "${FAILOVER_DIR:-$HOME/.local/share/claude-failover}" ;;
  esac
}
# 0 when the add-on is installed: its clone folder exists, or its own command is on PATH
addon_installed() {
  local d; d="$(addon_dir "$1")"
  [ -d "$d" ] && return 0
  case "$1" in
    voice)    have narrate-claude && return 0 ;;
    failover) have agent-llm && return 0 ;;
    phone)    [ -f "$HOME/.claude/imessage-toggle.sh" ] && return 0 ;;
  esac
  return 1
}
ADDONS_LIST="voice browser moonstone remote phone failover"

check_voice() {
  local data="${NARRATE_DATA:-$HOME/.local/share/narrateclaude}" dir; dir="$(addon_dir voice)"
  if have narrate-claude; then record pass voice-cmd "narrate-claude on PATH"
  elif [ -f "$dir/narrative-claude.sh" ]; then
    link_check voice-cmd "${KIT_BIN_DIR:-$HOME/.local/bin}/narrate-claude" "$dir/narrative-claude.sh"
  else record fail voice-cmd "narrate-claude not on PATH (re-run the voice installer)"; fi
  if ls "$data"/models/ggml-*.bin >/dev/null 2>&1; then record pass voice-whisper "whisper model present"
  else record fail voice-whisper "no whisper model in the voice data folder (re-run the voice installer)"; fi
  if have speak || [ -x "$data/venv/bin/piper" ]; then
    if ls "$data"/models/*.onnx >/dev/null 2>&1; then record pass voice-piper "Piper voice and speak command present"
    else record warn voice-piper "speak/piper present but no Piper voice file found"; fi
  else record fail voice-piper "Piper (speak) not found (re-run the voice installer)"; fi
}

check_addons() {
  head_ "Add-ons"
  local a
  for a in $ADDONS_LIST; do
    if [ "$a" = phone ] && [ "$OS" != Darwin ]; then record skip phone "phone: macOS only"; continue; fi
    if ! addon_installed "$a"; then
      # a service already answering on its port also counts as installed
      case "$a" in
        browser)   answers "${BROKER_PORT:-9223}" /health || { record skip "$a" "$a: not installed"; continue; } ;;
        moonstone) answers "${DOCK_PORT:-${MOONSTONE_PORT:-8899}}" / || { record skip "$a" "$a: not installed"; continue; } ;;
        *) record skip "$a" "$a: not installed"; continue ;;
      esac
    fi
    case "$a" in
      voice) check_voice ;;
      browser) service_check browser "browser broker" "${BROKER_PORT:-9223}" /health broker 1 ;;
      moonstone) service_check moonstone "Moonstone" "${DOCK_PORT:-${MOONSTONE_PORT:-8899}}" / moonstone 1 ;;
      remote) service_check remote "FiaOS" "${FIAOS_PORT:-9000}" / remote 1 ;;
      phone)
        local miss="" f
        for f in imessage-toggle.sh imessage-send.sh imessage-ask.sh; do [ -f "$HOME/.claude/$f" ] || miss="$miss $f"; done
        if [ -z "$miss" ]; then record pass phone "screen-to-phone scripts present"
        else record fail phone "screen-to-phone scripts missing:$miss (re-run its installer)"; fi ;;
      failover)
        if have agent-llm; then record pass failover "agent-llm on PATH"
        elif [ -f "$(addon_dir failover)/bin/agent-llm" ]; then
          link_check failover "${KIT_BIN_DIR:-$HOME/.local/bin}/agent-llm" "$(addon_dir failover)/bin/agent-llm"
        else record fail failover "agent-llm not on PATH (re-run the failover installer)"; fi ;;
    esac
  done
}

# ── Mac model advice (the original doctor verdict) ────────────
mac_verdict() {
  local mem="$1"
  head_ "🎯 Verdict: what to actually run"
  if [ "$mem" -lt 16 ]; then
    info "${mem} GB → small-Mac tier 8gb: Qwen 3.5 2B (8-bit, ~2.7 GB, sees images), 16K context (docs/SMALL-MACS.md)."
    RECOMMENDED="mlx-community/Qwen3.5-2B-MLX-8bit"; EXPECTED_TPS="~60"
  elif [ "$mem" -lt 32 ]; then
    info "${mem} GB → small-Mac tier 16gb: Qwen 3.5 4B (bf16, ~9 GB, sees images), 16K context (docs/SMALL-MACS.md). Hermes 4 14B 4-bit (text only, issue #54) also fits."
    RECOMMENDED="mlx-community/Qwen3.5-4B-bf16"; EXPECTED_TPS="~20"
  elif [ "$mem" -lt 64 ]; then
    info "${mem} GB → Gemma 4 12B (4-bit, ~11 GB). With 48 GB you can also run Gemma 4 31B 4-bit (~18 GB) if you close other apps."
    RECOMMENDED="divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text"; EXPECTED_TPS="?"
  elif [ "$mem" -lt 96 ]; then
    info "${mem} GB → Gemma 4 31B (4-bit, ~18 GB). Agent-12: 11/12 easy, 8/8 hard."
    RECOMMENDED="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"; EXPECTED_TPS="~26"
  else
    info "${mem} GB → Qwen 3.8 27B (8-bit, ~29 GB), with Gemma 4 31B for speed. Headroom for any MLX model."
    RECOMMENDED="lmstudio-community/Qwen3.8-27B-MLX-8bit"; EXPECTED_TPS="~18"
  fi
  if [ "${MODELS_FOUND:-0}" -eq 0 ]; then info "Next step: bash setup.sh   # downloads $RECOMMENDED and creates a launcher"; fi
}

run_bench() {
  head_ "⏱  Benchmark"
  local hf="${HF_HOME:-$HOME/.cache/huggingface}"
  if [ "$OS" != Darwin ] || [ ! -d "$MLX_VENV" ]; then info "Skipping: needs the MLX install on a Mac."; return; fi
  if [ -z "$RECOMMENDED" ] || [ ! -d "$hf/hub/models--$(echo "$RECOMMENDED" | sed 's|/|--|g')" ]; then
    info "Skipping: the recommended model is not downloaded yet."; return
  fi
  info "Running 100-token generation on $RECOMMENDED ..."
  local out tps
  out=$("$MLX_VENV/bin/python" -m mlx_lm.generate --model "$RECOMMENDED" --prompt "Write one short sentence." --max-tokens 100 2>&1 | tail -3)
  tps=$(echo "$out" | grep -oE 'Generation: *[0-9.]+ tokens-per-sec' | grep -oE '[0-9.]+' | head -1)
  if [ -n "$tps" ]; then say "${C_GREEN}✓${C_RESET} Measured: ${tps} tok/s (vs ${EXPECTED_TPS} expected for your tier)"
  else info "Run finished; could not parse tok/s."; fi
}

# ── Report ────────────────────────────────────────────────────
print_report() {
  local osname kernel commit failed warned passed skipped
  if [ "$OS" = Darwin ]; then osname="macOS $(sw_vers -productVersion 2>/dev/null)"
  else osname="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-Linux}")"; fi
  kernel="$(uname -r)"
  commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  ccl_ver="$(head -n1 "$REPO_DIR/VERSION" 2>/dev/null || echo unknown)"
  failed="$(echo "$RESULTS" | awk -F'|' '$1=="fail" {sub(/^fail\|[^|]*\|/,""); print "  - " $0}')"
  warned="$(echo "$RESULTS" | awk -F'|' '$1=="warn" {sub(/^warn\|[^|]*\|/,""); print "  - " $0}')"
  passed="$(echo "$RESULTS" | awk -F'|' '$1=="pass" {print $2}' | tr '\n' ' ')"
  skipped="$(echo "$RESULTS" | awk -F'|' '$1=="skip" {print $2}' | tr '\n' ' ')"
  {
    echo "claude-code-local doctor report (doctor v$DOCTOR_VERSION, claude-code-local $ccl_ver, repo $commit)"
    echo "OS:        $osname ($ARCH, kernel $kernel)"
    echo "Chip/GPU:  $(gpu_name)"
    echo "RAM:       $(total_ram_gb) GB"
    echo "Versions:  claude ${CLAUDE_VER:-none}, python ${PY_VER:-none}${MLX_VER:+, mlx-lm $MLX_VER}${LLAMA_TAG:+, llama.cpp $LLAMA_TAG}"
    echo "Summary:   $N_PASS ok, $N_WARN warnings, $N_FAIL failed, $N_SKIP not installed"
    echo "Failed:"
    if [ -n "$failed" ]; then echo "$failed"; else echo "  (none)"; fi
    echo "Warnings:"
    if [ -n "$warned" ]; then echo "$warned"; else echo "  (none)"; fi
    echo "Passed:        ${passed:-(none)}"
    echo "Not installed: ${skipped:-(none)}"
  } | redact
}

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}"; }

main() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --fix) MODE_FIX=1 ;;
      --report) MODE_REPORT=1 ;;
      --bench) MODE_BENCH=1 ;;
      -h|--help) usage; return 0 ;;
      *) echo "unknown option: $arg" >&2; usage >&2; return 2 ;;
    esac
  done
  setup_colors

  if [ "$OS" != Darwin ] && [ "$OS" != Linux ]; then
    echo "This doctor runs on macOS and Linux. On Windows use scripts\\doctor.ps1." >&2; return 2
  fi

  say ""
  say "${C_CYAN}╔══════════════════════════════════════════════════╗${C_RESET}"
  say "${C_CYAN}║${C_RESET}   ${C_BOLD}Claude Code Local — Doctor${C_RESET}                   ${C_CYAN}║${C_RESET}"
  say "${C_CYAN}╚══════════════════════════════════════════════════╝${C_RESET}"
  if [ "$MODE_FIX" = 1 ]; then say "${C_DIM}--fix: safe repairs only (restart your own service, recreate a symlink). Nothing is deleted.${C_RESET}"; fi

  local mem; mem="$(total_ram_gb)"; mem="${mem:-0}"
  head_ "🖥  This machine"
  info "OS:      $OS $ARCH"
  info "Chip:    $(gpu_name)"
  info "Memory:  ${mem} GB"

  if [ "$OS" = Darwin ] && [ "$ARCH" != arm64 ]; then
    record fail platform "macOS needs Apple Silicon (M1 or later); this is $ARCH"
  elif [ "$OS" = Linux ] && [ "$ARCH" != x86_64 ] && [ "$ARCH" != aarch64 ] && [ "$ARCH" != arm64 ]; then
    record fail platform "unsupported CPU architecture $ARCH"
  fi

  check_core
  check_addons

  if [ "$MODE_REPORT" != 1 ]; then
    if [ "$OS" = Darwin ]; then mac_verdict "$mem"
    elif have python3 && [ -f "$REPO_DIR/portable/ccl.py" ]; then
      head_ "🎯 Verdict: what to actually run"
      python3 -I "$REPO_DIR/portable/ccl.py" detect 2>/dev/null | sed 's/^/  /'
    fi
    if [ "$MODE_BENCH" = 1 ]; then run_bench; fi
    say ""
    say "Result: ${C_GREEN}$N_PASS ok${C_RESET}, ${C_YELLOW}$N_WARN warnings${C_RESET}, ${C_RED}$N_FAIL failed${C_RESET}, $N_SKIP not installed"
    if [ "$N_FAIL" -gt 0 ]; then say "${C_DIM}Filing a bug? Run: bash scripts/doctor.sh --report   and paste the block.${C_RESET}"; fi
    say ""
  else
    print_report
  fi
  [ "$N_FAIL" -eq 0 ]
}

# Run only when executed, so tests can `source` this file for its functions.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
