#!/usr/bin/env bash
# provision-hetzner-runner.sh
#
# Provisions a Hetzner Cloud VM and registers it as a GitHub Actions
# self-hosted runner for the fluxion repository.
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

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --github-repo)    GITHUB_REPO="$2";    shift 2 ;;
    --github-token)   GITHUB_TOKEN="$2";   shift 2 ;;
    --server-type)    SERVER_TYPE="$2";    shift 2 ;;
    --location)       LOCATION="$2";       shift 2 ;;
    --runner-name)    RUNNER_NAME="$2";    shift 2 ;;
    --runner-labels)  RUNNER_LABELS="$2";  shift 2 ;;
    --hcloud-ssh-key) HCLOUD_SSH_KEY="$2"; shift 2 ;;
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
command -v hcloud &>/dev/null || fail "hcloud CLI not found: https://github.com/hetznercloud/cli"
command -v jq &>/dev/null    || fail "jq not found (brew install jq / apt install jq)"

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

# ── Wait for SSH ──────────────────────────────────────────────────────────────
echo "==> Waiting for SSH ..."
for i in $(seq 1 30); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
      -o BatchMode=yes "root@${SERVER_IP}" echo ok &>/dev/null && break
  echo "   attempt ${i}/30 — retrying in 5s"
  sleep 5
done

# ── Remote provisioning ───────────────────────────────────────────────────────
echo "==> Provisioning runner on ${SERVER_IP} ..."
ssh -o StrictHostKeyChecking=no "root@${SERVER_IP}" bash -s -- \
  "$GITHUB_REPO" "$GITHUB_TOKEN" "$RUNNER_NAME" "$RUNNER_LABELS" "$RUNNER_VERSION" \
  << 'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
GITHUB_REPO="$1"
GITHUB_TOKEN="$2"
RUNNER_NAME="$3"
RUNNER_LABELS="$4"
RUNNER_VERSION="$5"

echo "--- System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  curl git jq unzip sudo ca-certificates gnupg lsb-release \
  build-essential pkg-config \
  libssl-dev libfontconfig1-dev libfreetype6-dev

echo "--- Docker"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io
systemctl enable --now docker

echo "--- Rust stable"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain stable --no-modify-path
source /root/.cargo/env
rustup component add rustfmt clippy

echo "--- Runner user"
useradd -m -s /bin/bash runner 2>/dev/null || true
usermod -aG docker runner
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
tar xzf "$TARBALL"
rm "$TARBALL"
chown -R runner:runner "$RUNNER_DIR"

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
