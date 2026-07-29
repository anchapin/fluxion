# Self-Hosted Runners on Hetzner Cloud

This guide covers provisioning cheap Hetzner Cloud VMs as GitHub Actions
self-hosted runners for fluxion's heavy CI jobs, and explains the automatic
fallback to GitHub-hosted runners when no self-hosted runner is available.

## Overview

Fluxion uses a **tiered CI strategy** (see [Phase 4 PR](https://github.com/anchapin/fluxion/pull/788)):

| Trigger | Runner | Jobs |
|---|---|---|
| PR branch push | GitHub-hosted (free) | `fmt`, `clippy`, `test-pr` (ubuntu-latest) |
| Main merge | Self-hosted preferred, GH-hosted fallback | `test-full` (3-platform), `build-release`, `integration-tests`, `python-examples` |

The routing is controlled by a single repository variable:

```
FLUXION_LINUX_RUNNER=fluxion-ci   # routes Linux jobs to self-hosted runners
# (unset or empty)                 # falls back to ubuntu-latest GitHub-hosted
```

**Security note:** Self-hosted runners are only used for `push` events on
`main`, never for `pull_request`. This is safe for public repos — forked PRs
cannot execute code on your self-hosted infrastructure.

---

## Prerequisites

1. **Hetzner Cloud account** — [console.hetzner.cloud](https://console.hetzner.cloud)
2. **hcloud CLI** installed and authenticated:
   ```bash
   brew install hcloud          # macOS
   # or: https://github.com/hetznercloud/cli/releases
   hcloud context create fluxion
   # paste your Hetzner API token when prompted
   ```
3. **SSH key** added to your Hetzner project:
   ```bash
   hcloud ssh-key create --name my-key --public-key-from-file ~/.ssh/id_ed25519.pub
   hcloud ssh-key list   # confirm it appears
   ```
4. **GitHub CLI** (`gh`) for obtaining the registration token:
   ```bash
   brew install gh
   gh auth login
   ```

---

## Provisioning a runner

### 1. Get a runner registration token

Registration tokens are single-use and expire after 1 hour:

```bash
REG_TOKEN=$(gh api -X POST \
  repos/anchapin/fluxion/actions/runners/registration-token \
  --jq .token)
echo "$REG_TOKEN"   # keep this, you'll pass it to the script
```

### 2. Run the provisioning script

```bash
chmod +x scripts/provision-hetzner-runner.sh

scripts/provision-hetzner-runner.sh \
  --github-repo    anchapin/fluxion \
  --github-token   "$REG_TOKEN" \
  --hcloud-ssh-key my-key
```

The script will:
- Create a **cx22** VM (2 vCPU, 4 GB RAM, Ubuntu 24.04) at ~€4/month
- Install: `git`, `docker`, `build-essential`, `libssl-dev`, `libfontconfig1-dev`, Rust stable
- Download the GitHub Actions runner agent (v2.323.0)
- Register it with labels `self-hosted,linux,x86_64,fluxion-ci`
- Start it as a **systemd service** (auto-restarts on reboot)

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--server-type` | `cx22` | Hetzner server type. `cx32` (4 vCPU/8GB) for heavier builds |
| `--location` | `hel1` | Data center: `hel1`\|`nbg1`\|`fsn1` (EU), `ash` (US-East), `sin` (Asia) |
| `--runner-name` | `fluxion-runner-<ts>` | Unique name shown in GitHub UI |
| `--runner-labels` | `self-hosted,linux,x86_64,fluxion-ci` | Comma-separated labels |

### 3. Activate the runner

Once provisioning completes, flip the repository variable:

```bash
gh variable set FLUXION_LINUX_RUNNER --body "fluxion-ci" \
  --repo anchapin/fluxion
```

Verify the runner is online:

```
https://github.com/anchapin/fluxion/settings/actions/runners
```

The next push to `main` will route Linux jobs to your self-hosted runner.
GitHub-hosted runners handle macOS and Windows automatically.

---

## Scaling to multiple runners

Run the script multiple times with different `--runner-name` values.
All runners with the `fluxion-ci` label join the same pool — GitHub
distributes jobs across them automatically:

```bash
for i in 1 2 3; do
  REG_TOKEN=$(gh api -X POST \
    repos/anchapin/fluxion/actions/runners/registration-token --jq .token)
  scripts/provision-hetzner-runner.sh \
    --github-repo    anchapin/fluxion \
    --github-token   "$REG_TOKEN" \
    --hcloud-ssh-key my-key \
    --runner-name    "fluxion-runner-${i}"
done
```

Three cx22 nodes run in parallel for ~€12/month total.

---

## GPU Runners (CUDA / ONNX Runtime)

The `cuda-parity` CI job (issue #1895) requires a runner with an NVIDIA GPU
to execute the live CPU-vs-CUDA parity tests in
`tests/surrogate_backend_parity.rs`. Without a GPU the job continues on
error (advisory, non-blocking) because the test self-skips when CUDA EP
is unavailable.

### Requirements

| Component | Minimum version | Notes |
|---|---|---|
| NVIDIA GPU | Kepler-era or newer | Pascal (P100) or newer recommended for good performance |
| Driver | CUDA 11.8 / driver 515+ | Must support the CUDA toolkit version below |
| CUDA Toolkit | 11.8 | Match the ONNX Runtime CUDA EP build |
| cuDNN | 8.x | Required by ONNX Runtime CUDA EP |
| ONNX Runtime | Built with CUDA EP | Enable via `--features cuda --features ort` at build time |

### Verifying GPU availability on the runner

```bash
nvidia-smi
# Expected output: GPU list with model name, driver version, CUDA version

# Check available GPU memory
nvidia-smi --query-gpu=memory.free,memory.total --format=csv
```

### Runner labels for GPU jobs

GPU jobs use the same `FLUXION_LINUX_RUNNER` pool. If your runner has an
NVIDIA GPU, it can run both CPU and GPU CI jobs. The `cuda-parity` job
automatically skips when no GPU is detected (via `cuda_ep_available()` in
the test), so a heterogeneous pool works correctly.

To provision a GPU-enabled runner, run the standard provisioning script on
a machine with an NVIDIA GPU (bare metal or GPU cloud instance):

```bash
REG_TOKEN=$(gh api -X POST \
  repos/anchapin/fluxion/actions/runners/registration-token --jq .token)

# Provision on a GPU node (cx42-flex or similar Hetzner GPU instance)
# Note: Hetzner's cloud does NOT offer GPU instances — use a dedicated
# GPU cloud provider (e.g., Lambda Labs, AWS p3/p4, GCP T4/A100) or a
# bare-metal GPU machine. The labels remain the same.
scripts/provision-hetzner-runner.sh \
  --github-repo    anchapin/fluxion \
  --github-token   "$REG_TOKEN" \
  --hcloud-ssh-key my-key \
  --runner-name    "fluxion-runner-gpu-1" \
  --runner-labels  "self-hosted,linux,x86_64,fluxion-ci,gpu"
```

The runner will automatically handle both CPU-only and GPU CI jobs. GPU
jobs that cannot acquire a GPU will continue on error rather than fail.

### ONNX Runtime CUDA EP notes

- ONNX Runtime must be compiled with CUDA EP support (`--features ort,cuda`)
- The CUDA EP requires matching CUDA toolkit + driver versions
- Issue #1285: a committed ONNX surrogate model is needed for full
  per-timestep CSV parity artifact upload. Until then, the parity test
  validates wiring and error-path semantics only.

---

## Fallback behaviour

The `FLUXION_LINUX_RUNNER` variable controls routing for all heavy Linux jobs:

```yaml
# ci.yml and rust-tests.yml — relevant jobs use this expression:
runs-on: ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}
```

| `FLUXION_LINUX_RUNNER` value | Where jobs run |
|---|---|
| `fluxion-ci` | Your self-hosted Hetzner runner(s) |
| unset or empty | GitHub-hosted `ubuntu-latest` (free for public repo) |

To fall back immediately (e.g. runner is down for maintenance):

```bash
gh variable delete FLUXION_LINUX_RUNNER --repo anchapin/fluxion
# or set it to ubuntu-latest explicitly:
gh variable set FLUXION_LINUX_RUNNER --body "ubuntu-latest" --repo anchapin/fluxion
```

---

## Security hardening

- **Ephemeral jobs only on main:** `pull_request` jobs always use GitHub-hosted
  runners. Forked PRs never touch your self-hosted machines.
- **Dedicated runner user:** The script runs the agent as a non-root `runner`
  user with Docker group membership.
- **No persistent secrets on disk:** Use GitHub Actions secrets (`${{ secrets.X }}`),
  not files baked into the image.
- **Firewall:** By default Hetzner VMs expose all ports. Add a firewall rule:
  ```bash
  hcloud firewall create --name fluxion-runner-fw
  hcloud firewall add-rule fluxion-runner-fw \
    --direction in --protocol tcp --port 22 --source-ip 0.0.0.0/0
  hcloud firewall apply-to-resource fluxion-runner-fw \
    --type server --server <RUNNER_NAME>
  ```
  The runner polls GitHub outbound — no inbound ports beyond SSH are needed.
- **Periodic OS updates:** SSH in weekly and run `apt-get upgrade -y`, or
  use unattended-upgrades.

---

## Maintenance

### Check runner status

```bash
ssh root@<SERVER_IP> \
  "cd /home/runner/actions-runner && ./svc.sh status"
```

### Update the runner agent

GitHub will notify you when a new runner version is available. To update:

```bash
# 1. Stop the service
ssh root@<SERVER_IP> \
  "cd /home/runner/actions-runner && ./svc.sh stop"

# 2. Get a new registration token and re-run the provisioning script
#    with --runner-name matching the existing server, or update manually:
ssh root@<SERVER_IP> bash << 'EOF'
cd /home/runner/actions-runner
NEW_VERSION=2.323.0   # update to latest
curl -fsSL https://github.com/actions/runner/releases/download/v${NEW_VERSION}/actions-runner-linux-x64-${NEW_VERSION}.tar.gz \
  | tar xz
./svc.sh start
EOF
```

---

## Decommissioning a runner

```bash
# 1. Remove-token (single-use, 1h expiry)
REMOVAL_TOKEN=$(gh api -X POST \
  repos/anchapin/fluxion/actions/runners/remove-token --jq .token)

# 2. Deregister and stop the agent
ssh root@<SERVER_IP> \
  "cd /home/runner/actions-runner && \
   ./svc.sh stop && \
   ./svc.sh uninstall && \
   ./config.sh remove --token $REMOVAL_TOKEN"

# 3. Delete the VM
hcloud server delete <RUNNER_NAME>

# 4. If this was your last runner, clear the repo variable
gh variable delete FLUXION_LINUX_RUNNER --repo anchapin/fluxion
```

---

## Hetzner Overflow Pool (Autoscaling — Issue #2133)

The **overflow pool** is a separate autoscaling system that activates when
GitHub-hosted runners are saturated (job queue wait >5 minutes).  It is
**independent** from the static `fluxion-ci` runner pool provisioned above.

### How it works: the probe job pattern

GitHub Actions has no native "fallback runner" concept.  The workaround is a
**probe job pattern**:

1. A lightweight **probe job** starts on `ubuntu-latest` with a 5-minute
   `timeout-minutes`.  It immediately succeeds (acquires a GH runner or fails
   fast).
2. If a GH runner is available, the probe finishes in seconds → the primary
   CI job runs on GH-hosted runners.
3. If GH runners are saturated, the probe waits in queue.  After 5 minutes
   without a runner, GitHub cancels it (result = `cancelled`) → the overflow
   CI job runs on the Hetzner autoscaling pool (`self-hosted,fluxion-overflow`).
4. Only **one** of the two paths runs per workflow run — never both.

```
ubuntu-latest ──(probe succeeds)──▶  CI on GH-hosted
    │
    └──(probe times out/cancelled)──▶  CI on self-hosted,fluxion-overflow
```

### Overflow vs. static pool

| Pool | Labels | Purpose | Routing |
|---|---|---|---|
| Static `fluxion-ci` | `self-hosted,fluxion-ci` | Primary for main-merge jobs | `FLUXION_LINUX_RUNNER \|\| ubuntu-latest` |
| Overflow (autoscaling) | `self-hosted,fluxion-overflow` | Fallback when GH runners saturated | Probe pattern (GH primary, Hetzner fallback) |

The two pools are independent and coexist.  Static runners continue to operate
as before.  The overflow pool only spins up when GH runners are saturated.

### Deploying the autoscaling service

The overflow pool is managed by
[testflows/github-hetzner-runners](https://github.com/testflows/TestFlows-GitHub-Hetzner-Runners),
a Python service that polls GitHub Actions API and creates/deletes Hetzner VMs
on demand.

#### 1. Install

```bash
pip3 install testflows.github.hetzner.runners
```

#### 2. Set environment variables

```bash
export GITHUB_TOKEN=ghp_...        # Classic token with workflow scope
export GITHUB_REPOSITORY=anchapin/fluxion
export HETZNER_TOKEN=...           # From console.hetzner.cloud
```

#### 3. Deploy (always-on)

```bash
github-hetzner-runners cloud deploy
```

#### 4. Configure via `~/.config/github-hetzner-runners/config.yml`

```yaml
max-runners: 4                     # maximum concurrent runners
labels: self-hosted,fluxion-overflow  # must match workflow job `runs-on`
runner-server-type: cx22           # CX23 for heavy builds (~€0.0096/hr)
recycle-without-rebuild: on        # powered-off servers are reused
```

> **Note:** `scale-up-delay` is NOT needed — the 5-minute probe timeout
> handles routing; VMs are created on demand for jobs that reach the overflow
> pool.

#### Cost

| Runner | Specs | Price |
|---|---|---|
| GitHub-hosted | 2-core Linux | $0.006/min (free for public repos) |
| Hetzner CX22 overflow | 2 vCPU, 4 GB | ~€0.006/hr (~$0.0001/min) |

Typical 15-minute job:
- GH-hosted (free tier): **$0.00**
- Hetzner overflow: ~€0.0015 (1 hr minimum billing)

Since GH-hosted is free, overflow only runs when GH is saturated — cost is
minimal.  Monitor spend via the github-hetzner-runners embedded dashboard.

### Repository variables

No new repository variables are required for the overflow pool.  The probe
pattern is self-contained in the workflow files and routes based on probe
results, not on variables.

### Updating existing workflows

The reusable workflow `.github/workflows/ci-steps.yml` contains the common CI
steps and accepts a `runner-label` input.  Caller workflows use the probe
pattern:

```yaml
jobs:
  # Probe: Try GH runner, timeout after 5 min
  my-job-gh-probe:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - run: echo "GH runner acquired"

  # If probe succeeds → run CI on GH
  my-job-gh:
    needs: my-job-gh-probe
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: echo "CI on GH"

  # If probe fails (timeout/cancelled) → run CI on Hetzner overflow
  my-job-hz:
    needs: my-job-gh-probe
    runs-on: [self-hosted,fluxion-overflow]
    if: |
      needs.my-job-gh-probe.result == 'failure'
      || needs.my-job-gh-probe.result == 'cancelled'
    steps:
      - uses: actions/checkout@v4
      - run: echo "CI on Hetzner overflow"
```

The following workflows use the probe pattern:

| Workflow | Jobs |
|---|---|
| `rust-tests.yml` | `energy-conservation-*`, `fmt-*`, `clippy-*`, `known-issues-stale-*`, `ashrae-cases-cycle-*` |
| `ci.yml` | `python-examples-*`, `integration-tests-*` |

Jobs using `FLUXION_LINUX_RUNNER || ubuntu-latest` (e.g., `test-full`,
`build-release`, `cuda-smoke`, `cuda-parity`) are unaffected — they already
route to the static self-hosted pool as primary with GH as fallback.

### Verification

After deploying github-hetzner-runners, verify runners appear in GitHub:

```
https://github.com/anchapin/fluxion/settings/actions/runners
```

Look for runners with labels `self-hosted,fluxion-overflow`.  They will
auto-register when the overflow pool creates a VM and deregister when the VM
is destroyed.

### Cost controls

- `max-runners: 4` — prevents runaway parallelism on the overflow pool
- `recycle-without-rebuild: on` — powered-off VMs are reused (no rebuild cost)
- The static `fluxion-ci` pool continues to handle the majority of main-merge
  jobs; overflow only activates when GH runners are saturated

