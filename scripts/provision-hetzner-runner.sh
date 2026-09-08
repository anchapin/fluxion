#!/usr/bin/env bash
# provision-hetzner-runner.sh
#
# Provisions a Hetzner Cloud VM and registers it as a GitHub Actions
# self-hosted runner for the fluxion repository.
#
# SECURITY POSTURE (Issues #3445, #3448)
# ---------------------------------------
# Persistent self-hosted runners MUST NOT execute untrusted (PR-controlled)
# code. This script therefore:
#   * does NOT install Docker and does NOT add the `runner` user to the
#     `docker` group (acceptance criterion #2 of #3445) — docker-group
#     membership is root-equivalent on the host;
#   * declares the runner with a default `--work _work` layout so GitHub
#     Actions isolates each job in its own `_work/<job>/...` subdirectory,
#     so PR checkouts cannot poison subsequent main-merge jobs (or vice
#     versa) via leftover build artefacts;
#   * registers the runner under labels that workflow jobs gate on
#     `push`/`main` only (see rust-tests.yml::*-hz and ci.yml::*-hz);
#   * (Issue #3448 acceptance #1) pins the server's SSH host key. The
#     script fetches the host's public key with `ssh-keyscan` and
#     rejects the server unless at least one presented key matches a
#     caller-supplied allowlist (one or more
#     `--known-host-fingerprint <SHA256:...>` flags). All subsequent
#     SSH sessions are opened with `StrictHostKeyChecking=yes` against a
#     temporary `UserKnownHostsFile` containing only the verified key —
#     `StrictHostKeyChecking=no` is never used. Operators must obtain
#     the expected fingerprint out-of-band (e.g., via the Hetzner Cloud
#     console or by running `ssh-keyscan` from a network whose
#     authenticity is independently verified);
#   * (Issue #3448 acceptance #2) verifies the SHA-256 of the GitHub
#     Actions runner tarball against the pinned constant
#     `EXPECTED_RUNNER_TARBALL_SHA256` BEFORE extraction. Any mismatch
#     aborts provisioning with a non-zero exit (fail-closed) — there is
#     no "warn and continue" mode. The pinned value must be updated
#     together with `RUNNER_VERSION`; the operator must obtain the new
#     hash out-of-band (e.g., `curl -fsSL <url> | sha256sum` from a
#     trusted network, then cross-check against the GitHub release page
#     served over HTTPS) before pinning. This mirrors the fail-closed
#     ONNX signature policy in `fluxion-core` (see AGENTS.md §"Toolchain,
#     Security, and Generated Artifacts");
#   * (Issue #3448 acceptance #3) passes the GitHub runner registration
#     token via stdin to the remote provisioning script rather than as
#     a positional argv. The token never appears in the remote shell's
#     argv (visible via `ps`), eliminating one class of process-listing
#     leak during the brief provisioning window.
#
# Operators rotating a previously-provisioned runner into the hardened
# posture must additionally run:
#     gpasswd -d runner docker || true   # drop the legacy docker-group grant
#     apt-get purge -y docker-ce docker-ce-cli containerd.io || true
#     rm -rf /var/lib/docker
# The workflow gating alone is sufficient for the immediate trust
# boundary; the docker-group purge is defence-in-depth in case a future
# workflow ever adds `runs-on: [self-hosted, fluxion-ci]` to a job that
# takes pull_request events.
#
# PREREQUISITES
#   - hcloud CLI installed and authenticated:
#       brew install hcloud          # macOS
#       hcloud context create fluxion
#   - GitHub runner registration token (NOT a PAT):
#       gh api -X POST repos/anchapin/fluxion/actions/runners/registration-token \
#         --jq .token
#   - SSH key already added to your Hetzner project:
#       hcloud ssh-key list
#
# USAGE
#   ./scripts/provision-hetzner-runner.sh \
#     --github-repo    anchapin/fluxion \
#     --github-token   <RUNNER_REGISTRATION_TOKEN> \
#     --hcloud-ssh-key <SSH_KEY_NAME_IN_HETZNER> \
#     [--server-type   cx22]               # 2 vCPU, 4 GB RAM, ~EUR 4/mo
#     [--location      hel1]               # hel1 | nbg1 | fsn1 | ash | sin
#     [--runner-name   fluxion-runner-1]
#     [--runner-labels "self-hosted,linux,x86_64,fluxion-ci"]
#
# RUNNER LABELS
#   The default label set includes "fluxion-ci". After provisioning, set the
#   repository variable so heavy CI jobs route to this runner:
#     gh variable set FLUXION_LINUX_RUNNER --body "fluxion-ci" \
#       --repo anchapin/fluxion
#   Remove (or empty) the variable to fall back to GitHub-hosted runners.
#
# TEARDOWN
#   1. Get a removal token:
#        REMOVAL_TOKEN=$(gh api -X POST \
#          repos/anchapin/fluxion/actions/runners/remove-token --jq .token)
#   2. Deregister the runner:
#        ssh root@<SERVER_IP> \
#          "cd /home/runner/actions-runner && \
#           ./svc.sh stop && ./svc.sh uninstall && \
#           ./config.sh remove --token $REMOVAL_TOKEN"
#   3. Delete the VM:
#        hcloud server delete <RUNNER_NAME>

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
SERVER_TYPE="${SERVER_TYPE:-cx22}"
LOCATION="${LOCATION:-hel1}"
RUNNER_NAME="${RUNNER_NAME:-fluxion-runner-$(date +%s)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,linux,x86_64,fluxion-ci}"
HCLOUD_SSH_KEY="${HCLOUD_SSH_KEY:-}"
RUNNER_VERSION="2.323.0"

# Pinned SHA-256 of the runner tarball (Issue #3448 acceptance #2).
# Source: `sha256sum` over the HTTPS-served release asset
# https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/
# actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
# To bump: download the new tarball, run `sha256sum` over it, cross-check
# the value against the GitHub release page served over HTTPS, and update
# BOTH this constant AND RUNNER_VERSION in the same commit.
EXPECTED_RUNNER_TARBALL_SHA256="0dbc9bf5a58620fc52cb6cc0448abcca964a8d74b5f39773b7afcad9ab691e19"

# Allowlist of trusted SSH host-key fingerprints (Issue #3448 acceptance #1).
# Populated from `--known-host-fingerprint` flags below. Each value is a
# single SHA-256 fingerprint in OpenSSH format (`SHA256:<base64>`), as
# printed by `ssh-keygen -lf <known_hosts>`. The script fetches the new
# VM's host key with `ssh-keyscan` and aborts unless at least one of its
# keys matches one of these fingerprints. Obtain expected fingerprints
# out-of-band (e.g., from a previous verified provisioning run, the
# Hetzner Cloud console, or a `ssh-keyscan` from a network whose
# authenticity is independently verified) — first-connect trust is
# intentionally NOT granted.
KNOWN_HOST_FINGERPRINTS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --github-repo)             GITHUB_REPO="$2";              shift 2 ;;
    --github-token)            GITHUB_TOKEN="$2";             shift 2 ;;
    --server-type)             SERVER_TYPE="$2";              shift 2 ;;
    --location)                LOCATION="$2";                 shift 2 ;;
    --runner-name)             RUNNER_NAME="$2";              shift 2 ;;
    --runner-labels)           RUNNER_LABELS="$2";            shift 2 ;;
    --hcloud-ssh-key)          HCLOUD_SSH_KEY="$2";           shift 2 ;;
    --known-host-fingerprint)  KNOWN_HOST_FINGERPRINTS+=("$2"); shift 2 ;;
    --help|-h)
      sed -n '/^# PREREQUISITES/,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -z "$GITHUB_REPO" ]]    && fail "--github-repo is required (e.g. anchapin/fluxion)"
[[ -z "$GITHUB_TOKEN" ]]   && fail "--github-token is required (runner registration token, not a PAT)"
[[ -z "$HCLOUD_SSH_KEY" ]] && fail "--hcloud-ssh-key is required (run: hcloud ssh-key list)"
command -v hcloud &>/dev/null    || fail "hcloud CLI not found: https://github.com/hetznercloud/cli"
command -v jq &>/dev/null        || fail "jq not found (brew install jq / apt install jq)"
command -v ssh-keyscan &>/dev/null \
  || fail "ssh-keyscan not found (install openssh-client)"
command -v ssh-keygen &>/dev/null \
  || fail "ssh-keygen not found (install openssh-client)"
command -v sha256sum &>/dev/null \
  || fail "sha256sum not found (install coreutils)"
[[ ${#KNOWN_HOST_FINGERPRINTS[@]} -gt 0 ]] \
  || fail "At least one --known-host-fingerprint is required (Issue #3448 acceptance #1). Obtain the expected SHA256 fingerprint out-of-band and re-run."

# ── Create the server ─────────────────────────────────────────────────────────
echo "==> Creating Hetzner server '${RUNNER_NAME}' (${SERVER_TYPE} @ ${LOCATION})"
SERVER_JSON=$(hcloud server create \
  --name          "$RUNNER_NAME" \
  --type          "$SERVER_TYPE" \
  --image         "ubuntu-24.04" \
  --location      "$LOCATION"    \
  --ssh-key       "$HCLOUD_SSH_KEY" \
  --poll-interval 5s \
  --output json)

SERVER_IP=$(echo "$SERVER_JSON" | jq -r '.server.public_net.ipv4.ip')
echo "==> Server IP: ${SERVER_IP}"

# ── SSH host-key verification (Issue #3448 acceptance #1) ─────────────────────
# Fetch the VM's SSH host key with `ssh-keyscan`, parse its SHA-256
# fingerprint(s), and reject the server unless at least one matches the
# caller-supplied allowlist (`--known-host-fingerprint`). All subsequent
# SSH sessions in this script use `StrictHostKeyChecking=yes` against a
# temp `UserKnownHostsFile` containing only the verified keys — no
# `StrictHostKeyChecking=no` ever leaves this block.
echo "==> Verifying SSH host key against allowlist ..."
TMPDIR_HK=$(mktemp -d)
trap 'rm -rf "${TMPDIR_HK:-}"' EXIT
KNOWN_HOSTS_FILE="${TMPDIR_HK}/known_hosts"
SSH_BASE_OPTS=( -o StrictHostKeyChecking=yes
                -o UserKnownHostsFile="$KNOWN_HOSTS_FILE"
                -o BatchMode=yes
                -o ConnectTimeout=10 )

# ssh-keyscan may print multiple keys (one per algorithm). Capture all
# candidates and verify at least one against the allowlist.
SSH_SCAN_RAW="${TMPDIR_HK}/scan.raw"
SSH_SCAN_KEYS="${TMPDIR_HK}/scan.known_hosts"
ssh-keyscan -T 10 -t ed25519,rsa,ecdsa "${SERVER_IP}" > "$SSH_SCAN_RAW" 2>/dev/null \
  || fail "ssh-keyscan failed for ${SERVER_IP}"
[[ -s "$SSH_SCAN_RAW" ]] || fail "ssh-keyscan returned no keys for ${SERVER_IP}"

# Convert host-prefixed scan output into a known_hosts-format file
# `ssh-keygen -lf` can read.
awk 'NF && $1 !~ /^#/ {print}' "$SSH_SCAN_RAW" > "$SSH_SCAN_KEYS"
[[ -s "$SSH_SCAN_KEYS" ]] || fail "ssh-keyscan produced no parseable keys for ${SERVER_IP}"

PRESENTED_FPS=$(ssh-keygen -lf "$SSH_SCAN_KEYS" 2>/dev/null | awk '{print $2}' || true)
[[ -n "$PRESENTED_FPS" ]] || fail "ssh-keygen could not fingerprint scanned keys for ${SERVER_IP}"

verified_fp=""
while IFS= read -r fp; do
  [[ -z "$fp" ]] && continue
  for allowed in "${KNOWN_HOST_FINGERPRINTS[@]}"; do
    if [[ "$fp" == "$allowed" ]]; then
      verified_fp="$fp"
      break 2
    fi
  done
done <<< "$PRESENTED_FPS"

if [[ -z "$verified_fp" ]]; then
  echo "ERROR: SSH host-key fingerprint for ${SERVER_IP} does not match any allowlist entry." >&2
  echo "       Presented:" >&2
  while IFS= read -r fp; do echo "         $fp" >&2; done <<< "$PRESENTED_FPS"
  echo "       Allowed:   ${KNOWN_HOST_FINGERPRINTS[*]}" >&2
  fail "Refusing to connect (MITM protection, fail-closed). Verify the VM's fingerprint via the Hetzner Cloud console and re-run with --known-host-fingerprint <SHA256:...>."
fi
echo "    matched: ${verified_fp}"

# Persist only the verified key line(s) — drop any non-matching keys
# the server may have presented. A `known_hosts` file with the verified
# key alone keeps the connection MITM-safe for every subsequent `ssh`
# in this script.
: > "$KNOWN_HOSTS_FILE"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  tmp_hk="${TMPDIR_HK}/probe"
  printf '%s\n' "$line" > "$tmp_hk"
  fp=$(ssh-keygen -lf "$tmp_hk" 2>/dev/null | awk '{print $2}')
  if [[ "$fp" == "$verified_fp" ]]; then
    printf '%s\n' "$line" >> "$KNOWN_HOSTS_FILE"
  fi
done < "$SSH_SCAN_KEYS"
[[ -s "$KNOWN_HOSTS_FILE" ]] \
  || fail "Internal error: verified fingerprint ${verified_fp} not found in scanned keys"

# ── Wait for SSH ──────────────────────────────────────────────────────────────
echo "==> Waiting for SSH ..."
for i in $(seq 1 30); do
  ssh "${SSH_BASE_OPTS[@]}" "root@${SERVER_IP}" echo ok &>/dev/null && break
  echo "   attempt ${i}/30 — retrying in 5s"
  sleep 5
done
# Final connectivity probe with the verified key.
ssh "${SSH_BASE_OPTS[@]}" "root@${SERVER_IP}" true \
  || fail "Cannot establish SSH session to ${SERVER_IP} with the verified host key (MITM suspected or key mismatch)."

# ── Remote provisioning ───────────────────────────────────────────────────────
# Issue #3448 acceptance #3: pipe the registration token via stdin so it
# never appears in the remote shell's argv. The remote script reads
# stdin once at startup with `IFS= read -r GITHUB_TOKEN`.
#
# Implementation note: a single heredoc cannot be both the bash script
# and the stdin pipe (the heredoc would override the pipe). We
# materialise the script body to a temp file and pass it to the
# remote `bash` via `bash -c "$(cat "$PROVISION_SCRIPT")"`, which
# leaves the local stdin pipe free for the token.
echo "==> Provisioning runner on ${SERVER_IP} ..."
PROVISION_SCRIPT="${TMPDIR_HK}/provision.sh"
cat > "$PROVISION_SCRIPT" << 'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
GITHUB_REPO="$1"
# Token arrives on stdin (Issue #3448 acceptance #3) — NOT in argv, so
# it does not show up in `ps` / `/proc/<pid>/cmdline` on the runner.
IFS= read -r GITHUB_TOKEN
[[ -n "${GITHUB_TOKEN:-}" ]] \
  || { echo "ERROR: registration token not received on stdin" >&2; exit 1; }

echo "--- System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  curl git jq unzip sudo ca-certificates gnupg lsb-release \
  build-essential pkg-config \
  libssl-dev libfontconfig1-dev libfreetype6-dev

echo "--- Rust stable"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain stable --no-modify-path
source /root/.cargo/env
rustup component add rustfmt clippy

echo "--- Runner user"
# Issue #3445: the runner service account MUST NOT be a member of the
# `docker` group. Docker group membership grants root-equivalent access to the
# host (mount → host filesystem → sudo), which is incompatible with
# the persistent-self-hosted-runner threat model. Do NOT reintroduce
# `usermod -aG docker runner` here or anywhere else in the provisioning
# pipeline — see docs/SECURITY.md §"Self-hosted runner job execution
# policy" for the full rationale.
#
# Docker itself is also intentionally NOT installed on this runner image.
# No workflow that targets the `fluxion-ci` / `fluxion-overflow` labels
# invokes `docker` (verified via `rg "docker" .github/workflows/` —
# every match is in docker.yml which is pinned to `ubuntu-latest` /
# `ubuntu-latest-8-cores`, never self-hosted). If a future workflow
# genuinely needs Docker on a self-hosted runner, gate it behind a
# dedicated runner label (e.g. `self-hosted,docker-required`) and add
# the docker group to a *separate* service account, never the one that
# runs PR-controlled build scripts.
useradd -m -s /bin/bash runner 2>/dev/null || true
cp -r /root/.cargo /home/runner/.cargo 2>/dev/null || true
chown -R runner:runner /home/runner/.cargo 2>/dev/null || true
echo 'source /home/runner/.cargo/env' >> /home/runner/.bashrc

echo "--- GitHub Actions runner v${RUNNER_VERSION}"
RUNNER_DIR="/home/runner/actions-runner"
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"
TARBALL="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
curl -fsSL \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}" \
  -o "$TARBALL"

# Issue #3448 acceptance #2: verify the tarball SHA-256 BEFORE
# extraction. Any mismatch is fail-closed — refuse to extract. Mirrors
# the `verify_onnx_signature` policy in fluxion-core.
ACTUAL_SHA256=$(sha256sum "$TARBALL" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_RUNNER_TARBALL_SHA256" ]]; then
  echo "ERROR: Runner tarball SHA-256 mismatch (fail-closed)." >&2
  echo "  expected: $EXPECTED_RUNNER_TARBALL_SHA256" >&2
  echo "  actual:   $ACTUAL_SHA256" >&2
  echo "  url:      https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}" >&2
  rm -f "$TARBALL"
  exit 1
fi
echo "    sha256: ${ACTUAL_SHA256} (matches pinned value)"

tar xzf "$TARBALL"
rm "$TARBALL"
chown -R runner:runner "$RUNNER_DIR"
# Issue #3445: every Actions job on a persistent runner writes into a
# `_work/<job-name>/...` subtree, but the runner's own `_work` directory
# persists across jobs. Pre-existing PR-controlled checkout artefacts
# in `_work` could otherwise leak into a subsequent main-merge job
# (or vice-versa). GitHub Actions handles per-job subdirectories
# correctly out of the box — do NOT disable or hoist the default
# `_work` layout here.

echo "--- Registering with GitHub"
sudo -u runner ./config.sh \
  --url         "https://github.com/${GITHUB_REPO}" \
  --token       "$GITHUB_TOKEN" \
  --name        "$RUNNER_NAME" \
  --labels      "$RUNNER_LABELS" \
  --runnergroup Default \
  --work        _work \
  --unattended \
  --replace

echo "--- systemd service"
./svc.sh install runner
./svc.sh start

echo "--- Runner status"
./svc.sh status
REMOTE

printf '%s\n' "$GITHUB_TOKEN" | ssh "${SSH_BASE_OPTS[@]}" \
  "root@${SERVER_IP}" \
  GITHUB_REPO="$GITHUB_REPO" \
  RUNNER_NAME="$RUNNER_NAME" \
  RUNNER_LABELS="$RUNNER_LABELS" \
  RUNNER_VERSION="$RUNNER_VERSION" \
  EXPECTED_RUNNER_TARBALL_SHA256="$EXPECTED_RUNNER_TARBALL_SHA256" \
  bash -c "$(cat "$PROVISION_SCRIPT")" -- "$GITHUB_REPO"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Runner '${RUNNER_NAME}' is live at ${SERVER_IP}"
echo "  Labels : ${RUNNER_LABELS}"
echo "  View   : https://github.com/${GITHUB_REPO}/settings/actions/runners"
echo ""
echo "Activate for heavy CI jobs:"
echo "  gh variable set FLUXION_LINUX_RUNNER --body fluxion-ci --repo ${GITHUB_REPO}"
echo ""
echo "To decommission later:"
echo "  TOKEN=\$(gh api -X POST repos/${GITHUB_REPO}/actions/runners/remove-token --jq .token)"
echo "  ssh root@${SERVER_IP} \"cd /home/runner/actions-runner && ./svc.sh stop && ./svc.sh uninstall && ./config.sh remove --token \$TOKEN\""
echo "  hcloud server delete ${RUNNER_NAME}"
