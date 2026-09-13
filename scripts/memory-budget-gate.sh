#!/usr/bin/env bash
# memory-budget-gate.sh — Warn or exit if peak RSS exceeds budget
#
# Issue #3766: the gate must watch the WHOLE child process tree spawned by
# --command (cargo, rustc, ld, test runners), not the monitoring shell
# itself. It walks the tree from the command's PID every poll, sums RSS in
# KB, and enforces thresholds actively: an --exit or system-headroom breach
# kills the child tree immediately instead of waiting for the command to
# finish (that is the "before the OOM killer" part of the contract).
#
# Usage: ./scripts/memory-budget-gate.sh --warn GB [--exit GB] \
#            [--headroom GB] [--command "CMD"]
#
# Options:
#   --warn GB       Warning threshold in GB (default: 10)
#   --exit GB       Exit threshold in GB (default: unlimited); breach kills the
#                   child tree and exits 2 immediately
#   --headroom GB   Abort when system-wide MemAvailable + SwapFree drops below
#                   GB (default: 2; 0 disables). Linux-only (/proc/meminfo);
#                   skipped gracefully elsewhere.
#   --command CMD   Command to run while monitoring memory
#
# Exit codes: 0 = OK (below warn), 1 = warning (above warn),
#             2 = exceeded exit threshold (child tree killed),
#             3 = wrapped command failed,
#             4 = system headroom breached (child tree killed)

set -euo pipefail

WARN_GB=10
EXIT_GB=""
HEADROOM_GB=2
COMMAND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warn) WARN_GB="$2"; shift 2 ;;
    --exit) EXIT_GB="$2"; shift 2 ;;
    --headroom) HEADROOM_GB="$2"; shift 2 ;;
    --command) COMMAND="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

WARN_KB=$((WARN_GB * 1024 * 1024))
EXIT_KB=""
if [[ -n "$EXIT_GB" ]]; then
  EXIT_KB=$((EXIT_GB * 1024 * 1024))
fi
HEADROOM_KB=$((HEADROOM_GB * 1024 * 1024))

MAX_RSS_KB=0
BREACH=""
CMD_PID=""

echo "Memory budget gate"
echo "Warning threshold: ${WARN_GB} GB"
if [[ -n "$EXIT_KB" ]]; then
  echo "Exit threshold: ${EXIT_GB} GB"
fi
if [[ "$HEADROOM_GB" -gt 0 ]]; then
  echo "System headroom floor: ${HEADROOM_GB} GB (MemAvailable + SwapFree)"
fi

ps_rss_kb() {
  local rss
  rss=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' ' || true)
  [[ -z "$rss" ]] && rss=0
  echo "$rss"
}

meminfo_kb() {
  # Second column of the named /proc/meminfo field, in KB.
  local val
  val=$(awk -v f="$1" '$1 == f ":" { print $2; exit }' /proc/meminfo 2>/dev/null || true)
  [[ -z "$val" ]] && val=""
  echo "$val"
}

# Sum RSS over the whole process subtree rooted at $1 (BFS via pgrep -P).
# Guards: a visited set against cycles, and no descent below PID 1 — if the
# root dies mid-walk its orphans reparent to init and a naive walk would
# otherwise sum the entire system.
tree_rss_kb() {
  local root="$1"
  if [[ "$root" -eq 1 ]] || ! kill -0 "$root" 2>/dev/null; then
    ps_rss_kb "$root"
    return
  fi
  local -A seen=()
  local queue=("$root")
  local total=0 pid rss child children
  while [[ ${#queue[@]} -gt 0 ]]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ -n "${seen[$pid]:-}" ]] && continue
    seen[$pid]=1
    rss=$(ps_rss_kb "$pid")
    total=$((total + rss))
    [[ "$pid" -eq 1 ]] && continue
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
      queue+=("$child")
    done
  done
  echo "$total"
}

kill_tree() {
  local root="$1"
  local descendants
  descendants=$(pgrep -P "$root" 2>/dev/null || true)
  local d
  for d in $descendants; do
    kill_tree "$d"
  done
  kill "$root" 2>/dev/null || true
}

system_headroom_kb() {
  local avail swap
  avail=$(meminfo_kb MemAvailable)
  if [[ -z "$avail" ]]; then
    echo "-1"
    return
  fi
  swap=$(meminfo_kb SwapFree)
  [[ -z "$swap" ]] && swap=0
  echo $((avail + swap))
}

on_signal() {
  if [[ -n "$CMD_PID" ]]; then
    kill_tree "$CMD_PID"
  fi
  echo "Memory budget gate interrupted; child tree killed" >&2
  exit 130
}
trap on_signal INT TERM

monitor_memory() {
  local cmd_pid="$1"
  while kill -0 "$cmd_pid" 2>/dev/null; do
    local tree_rss headroom_kb
    tree_rss=$(tree_rss_kb "$cmd_pid")
    if [[ "$tree_rss" -gt "$MAX_RSS_KB" ]]; then
      MAX_RSS_KB=$tree_rss
    fi
    if [[ -n "$EXIT_KB" ]] && [[ "$tree_rss" -gt "$EXIT_KB" ]]; then
      BREACH="exit"
      echo "" >&2
      echo "EXIT THRESHOLD BREACH: child tree RSS ${tree_rss} KB > ${EXIT_KB} KB; killing tree rooted at PID ${cmd_pid}" >&2
      kill_tree "$cmd_pid"
      return
    fi
    if [[ "$HEADROOM_GB" -gt 0 ]]; then
      headroom_kb=$(system_headroom_kb)
      if [[ "$headroom_kb" -ge 0 ]] && [[ "$headroom_kb" -lt "$HEADROOM_KB" ]]; then
        BREACH="headroom"
        echo "" >&2
        echo "HEADROOM BREACH: system MemAvailable + SwapFree ${headroom_kb} KB < floor ${HEADROOM_KB} KB; killing tree rooted at PID ${cmd_pid}" >&2
        kill_tree "$cmd_pid"
        return
      fi
    fi
    sleep 0.2
  done
}

CHILD_STATUS=0
if [[ -n "$COMMAND" ]]; then
  eval "$COMMAND" &
  CMD_PID=$!

  monitor_memory "$CMD_PID"

  set +e
  wait "$CMD_PID"
  CHILD_STATUS=$?
  set -e
  CMD_PID=""
  if [[ "$CHILD_STATUS" -ne 0 ]] && [[ -z "$BREACH" ]]; then
    echo "Wrapped command failed with exit code ${CHILD_STATUS}" >&2
  fi
fi

trap - INT TERM

MAX_RSS_GB=$(echo "scale=2; $MAX_RSS_KB / 1024 / 1024" | bc)
echo ""
echo "Peak RSS: ${MAX_RSS_GB} GB"
if [[ "$CHILD_STATUS" -ne 0 ]]; then
  echo "Wrapped command exit code: ${CHILD_STATUS}"
fi

if [[ "$BREACH" == "exit" ]]; then
  echo "ERROR: Peak RSS ${MAX_RSS_GB} GB exceeds exit threshold ${EXIT_GB} GB" >&2
  exit 2
fi

if [[ "$BREACH" == "headroom" ]]; then
  echo "ERROR: System memory headroom dropped below ${HEADROOM_GB} GB" >&2
  exit 4
fi

if [[ "$CHILD_STATUS" -ne 0 ]]; then
  exit 3
fi

if [[ "$MAX_RSS_KB" -gt "$WARN_KB" ]]; then
  echo "WARNING: Peak RSS ${MAX_RSS_GB} GB exceeds warning threshold ${WARN_GB} GB" >&2
  exit 1
fi

echo "Memory budget check passed (${MAX_RSS_GB} GB < ${WARN_GB} GB)"
exit 0
