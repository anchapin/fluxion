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
