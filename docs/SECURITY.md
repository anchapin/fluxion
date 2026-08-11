# Security

Security policy, accepted-risk advisories, and hardening guides for Fluxion.
Covers vulnerability reporting, the `cargo audit` ignore list, and the GitHub
Actions supply-chain / least-privilege baseline enforced across all workflows.

## Reporting vulnerabilities

Email security@fluxion.org (PGP key on request) for any suspected security
issue. Do **not** open a public GitHub issue for unreported vulnerabilities.

## Accepted-risk advisories

The advisories below are suppressed in `.cargo/audit.toml` with a documented
justification. Each entry includes the rationale and the conditions under
which it should be revisited.

### `RUSTSEC-2026-0192` — `ttf-parser` (unmaintained)

| Field | Value |
| --- | --- |
| Crate | `ttf-parser` v0.20.0 |
| Type | unmaintained (no security vulnerability) |
| Parents | `plotters` v0.3.7 (default `ttf` feature) → `font-kit` v0.14.x → `ttf-parser` |
| Fluxion use site | `src/validation/report.rs` (chart caption rendering via `("sans-serif", 50).into_font()`) |
| Issue | [#1458](https://github.com/anchapin/fluxion/issues/1458) |
| First seen | 2026-06-29 (advisory publication) |

#### Why ignored

- **Not a vulnerability.** Rustsec type is `unmaintained`, not a CVE. There is
  no patched version (latest published release, v0.25.1 from Nov 2024, is
  also unmaintained and carries the same advisory). [Advisory
  link](https://rustsec.org/advisories/RUSTSEC-2026-0192.html).
- **Depth-2 transitive.** Fluxion does not import `ttf-parser` directly. The
  crate enters via `plotters`'s default `ttf` feature, which pulls
  `font-kit` → `ttf-parser`. Confirmed via
  `cargo tree -i ttf-parser`:
  ```
  ttf-parser v0.20.0
  └── plotters v0.3.7
      └── fluxion v1.0.0
  ```
- **No mainstream successor.** The advisory recommends `skrifa` (Google
  Fonts "oxidize" / fontations). `skrifa` is **not** adopted by `plotters`
  0.3.7; switching requires either waiting for an upstream `plotters`
  release or wiring it ourselves.

#### Why not removed

There is no `ttf-parser` version that is currently maintained, so no version
bump can clear the warning. Two removal paths were considered and deferred:

1. **Switch `plotters` to its `ab_glyph` font feature.**
   `plotters` already supports a `ttf`-free font path (`ab_glyph` +
   `register_font`). However, `ab_glyph` does not auto-load system fonts, so
   the existing `("sans-serif", 50).into_font()` call sites in
   `src/validation/report.rs` would require bundling a TTF/OTF font file
   (with license review) and adding a `register_font` call. This exceeds the
   S/<=4h low-risk scope of issue #1458.
2. **Replace `plotters` entirely.** Out of scope: `plotters` is used
   only for two placeholder chart generators that are currently inert
   (not invoked from any test or production code path). Larger refactor
   belongs in a dedicated cleanup issue.

#### Revisit when

- `ttf-parser` ships a release after June 2026 (advisory entry updated to
  "patched" with a non-zero version range), **or**
- `plotters` 0.3.x or 0.4 ships a release that drops or replaces its
  `font-kit`/`ttf-parser` font backend, **or**
- Fluxion removes the inactive chart-placeholder code in
  `src/validation/report.rs` and the `ttf`/`bitmap_encoder` `plotters`
  features can be safely dropped from the root `Cargo.toml`, removing the
  `ttf-parser` entry from all three lockfiles.

Tracking issue: [#1458](https://github.com/anchapin/fluxion/issues/1458).

---

### `RUSTSEC-2024-0436` — `paste` (unmaintained)

| Field | Value |
| --- | --- |
| Crate | `paste` |
| Type | unmaintained (no security vulnerability) |
| Parents | `nalgebra`, `statrs`, `faer` (transitive) |
| Resolution | No replacement released. `paste` `1.x` is widely used and
  baked into the Rust numerical ecosystem. |

### `RUSTSEC-2026-0177` — PyO3 0.22 `Sync` bound (PR-B)

| Field | Value |
| --- | --- |
| Crate | `pyo3` 0.22.6 |
| Type | memory-safety (PyCFunction::new_closure missing `Sync` bound) |
| Affects | pyo3 0.15.0..<0.29.0 |
| Fixed in | pyo3 ≥ 0.29.0 |
| Fluxion use site | `Cargo.toml:205` pins `pyo3 = "0.22"`; used by `src/python/` and `src/bin/fluxion-py/` |
| Tracking | [Issue #2553](https://github.com/anchapin/fluxion/issues/2553) (PR-B) |
| Time-bound | **2026-12-31** — if migration has not landed by this date, REMOVE this ignore from `.cargo/audit.toml` so cargo audit surfaces the failure and forces the work to be prioritized |

#### Why ignored

- **Actively known vulnerability.** PyO3 0.22.6 is missing the `Sync` bound
  on `PyCFunction::new_closure` (GHSA-chgr-c6px-7xpp). Patched in
  pyo3 ≥ 0.29.0.
- **Migration is non-trivial.** pyo3 0.22 → 0.29 spans 7 minor versions and
  includes breaking API changes:
  - `Bound<'py, T>` migration (replaces the old `&PyAny` / `&PyTuple` API)
  - `IntoPy` / `FromPyObject` rewrites
  - GIL-refs removal (the `Python::with_gil` / `PyGil::acquire` API is gone)
  - `abi3-py310` → `abi3-py312` (or higher) abi bump
  - `pyo3-macros` and `pyo3-build-config` version alignment
  - `src/python/` and `src/bin/fluxion-py/` must be rewritten against the new API
- **Out of scope** for issue #2553's "remove stale ignores" mandate.

#### Why not removed

Removing this ignore would make `cargo audit` exit non-zero on every PR
until the PyO3 migration lands, blocking all development. The
**time-bound** in `.cargo/audit.toml` (target 2026-12-31) is the
explicit deadline: after that date, the ignore must be removed and the
PyO3 migration becomes a release blocker.

#### Revisit when

- pyo3 ≥ 0.29.0 migration is complete and `Cargo.toml` pins the new version, **or**
- 2026-12-31 arrives without migration — at which point REMOVE the ignore.

### `RUSTSEC-2026-0098`, `RUSTSEC-2026-0099` — remediated transitively in #2553 (PR-A)

Both rustls-webpki name-constraint advisories (URI names, wildcard DNS) are
patched in `rustls-webpki >= 0.103.12`. Fluxion's `Cargo.lock` resolves
`rustls-webpki` to `0.103.13` (transitively via `rustls 0.23.37` →
`hyper-rustls 0.27.7` → `reqwest 0.12.28`). The previous ignore entries
were stale suppressions; they were removed in PR-A so that `cargo audit`
now verifies the fix on every PR.

---

## Audit configuration

The file `.cargo/audit.toml` is the canonical `cargo audit` configuration
(per the [rustsec example](https://github.com/rustsec/rustsec/blob/main/cargo-audit/audit.toml.example)).
An older `audit.toml` at the repository root is a historical artifact from
before the convention was adopted and is **not** read by `cargo audit` 0.22+.

To reproduce locally:

```bash
cargo audit           # unmaintained warnings are informational, no failure
cargo audit -D unmaintained  # strictest gate (used in CI if/when enforced)
```

The CI workflow at `.github/workflows/security.yml` runs `cargo audit` on every
pull request and weekly on the main branch.

---

## GitHub Actions least-privilege & supply-chain baseline

**Tracking issue:** [#2526](https://github.com/anchapin/fluxion/issues/2526).

Every `.github/workflows/*.yml` workflow follows the rules below. The policy is
enforced by review; `actionlint` and the SHA-pinning audit below surface
regressions.

### 1. Explicit `permissions:` on every workflow

Each workflow declares a top-level `permissions:` block. The default is the
minimum that lets the workflow do its job — for the majority of build / test /
lint / validation workflows that is:

```yaml
permissions:
  contents: read
```

A job that needs more escalates **only** itself via a job-level `permissions:`
block. Workflows never rely on the repository-wide default token permissions
(which historically granted `contents: write` to every run).

| Scope | Granted to | Example |
| --- | --- | --- |
| `contents: read` | every workflow (default) | `rust-tests.yml`, `security.yml` |
| `contents: write` | jobs that `git commit`/`git push` | benchmark baseline updates in `ashrae_benchmark_harness.yml`, `tdqs_regression.yml`; gh-pages push in `performance_dashboard.yml` |
| `packages: write` | jobs pushing to ghcr.io | `docker.yml` build/merge jobs |
| `pull-requests: write` | jobs posting PR comments | `github-script` comment steps in `mutation-testing.yml`, `ripr-preflight.yml` |
| `issues: write` | jobs opening/commenting issues | `nightly_validation.yml`, `known-issues-stale.yml` |
| `security-events: write` | SARIF upload | Trivy job in `docker.yml` |
| `id-token: write` | OIDC only (trusted publishing / AWS) | PyPI publish jobs in `pypi-release.yml`; all jobs in `cloud_campaign.yml` |

### 2. `actions/github-script` scoping

GitHub Actions **does not support `permissions:` at the step level** (confirmed
via `actionlint` — it rejects an inline `permissions:` key under a `step`).
Therefore each `actions/github-script` invocation is constrained by the
**job-level** `permissions:` of the job that contains it, which is set to the
minimal scope the script actually needs (e.g. `pull-requests: write` to post a
single PR comment). Every `actions/github-script` use is pinned to a fixed
commit SHA, so the action itself cannot exfiltrate or widen the token.

### 3. PyPI release — OIDC trusted publishing

`pypi-release.yml` keeps `contents: read` at workflow level. The `publish` and
`publish-test` jobs grant **only** `id-token: write` (plus `contents: read`)
at the job level — the minimal set for PyPI trusted publishing. They never
receive `contents: write`, because publishing pushes to PyPI, not to the repo.

### 4. AWS — OIDC federation, no static secrets

`cloud_campaign.yml` formerly injected long-lived static keys
(`secrets.AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`).
It now assumes an IAM role via OIDC using
`aws-actions/configure-aws-credentials@<SHA>` with short-lived STS credentials.
No static AWS secret is referenced by any workflow.

**Required repository configuration** (Settings → Secrets and variables →
Actions → **Variables**, not Secrets):

- `FLUXION_AWS_CAMPAIGN_ROLE_ARN` — IAM role ARN to assume. Until this is set,
  the configure-aws-credentials step fails fast rather than falling back to
  static keys.

The role's trust policy must permit `sts:AssumeRoleWithWebIdentity` for this
repository's GitHub OIDC provider, restricted to the `main` branch:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Federated": "arn:aws:iam::<ACCOUNT>:oidc-provider/token.actions.githubusercontent.com" },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:anchapin/fluxion:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

The role's own permission policy must be scoped to exactly the S3/SNS resources
the campaign scripts touch (least privilege), e.g. `s3:PutObject` /
`s3:GetObject` on `arn:aws:s3:::<bucket>/*` only.

### 5. Action pinning

- **All third-party actions are pinned to a 40-char commit SHA** with the
  human-readable tag in a trailing comment (`@<sha>  # vN`). Tags are
  mutable and are never used as the resolved ref.
- **First-party `actions/*` actions are also SHA-pinned** per this repo's
  convention (e.g. `actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4`).
- The only tag-ref that remained (`disk-space.yml` → `actions/checkout@v4`) was
  pinned in this change. `git ls-remote <repo> refs/tags/<tag>` resolves a tag
  to its commit SHA before pinning.

### 6. Adding a new workflow — checklist

1. Add a top-level `permissions:` block. Start from `contents: read` and add a
   scope only if a step provably needs it.
2. If a job posts a PR comment, opens an issue, pushes a commit/tag, publishes a
   package, uploads SARIF, or assumes a cloud role — give **that job only** the
   extra `permissions:` it needs.
3. Pin every `uses:` to a commit SHA (resolve via `git ls-remote`).
4. Run `actionlint .github/workflows/<file>.yml` and
   `python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" <file>`
   before committing.
5. Never reference static cloud keys; use OIDC (`id-token: write` +
   `role-to-assume`) and document the required repo variable here.

## Production deploy checklist

Hardening controls that MUST be verified before a `fluxion-rest` instance is
exposed to untrusted traffic. Each item maps to an enforced control (code or
config) so it can be checked mechanically, not just by process.

- **Header redaction on TraceLayer spans (Issue #2504).**
  The `tower_http` `TraceLayer` span is built by `SafeHeaderMakeSpan`
  (`src/api/server.rs`), which records only an explicit allow-list of safe
  request headers — `x-request-id`, `content-type`, `user-agent`. Credential
  headers (`Authorization`, `Cookie`, `x-api-key`, AWS Sig V4 `x-amz-*`) are
  omitted by construction; there is no deny-list to keep in sync. The previous
  `DefaultMakeSpan::new().include_headers(true)` recorded *every* request
  header, leaking bearer tokens and session cookies into structured logs
  (OWASP A09:2021). Regression test `tracelayer_does_not_log_credentials`
  asserts neither the credential header names nor their values appear in span
  output. Do **not** revert to `include_headers(true)`; do **not** widen
  `SAFE_HEADER_ALLOWLIST` to include any credential-bearing header. If you add
  a subscriber that exports span fields to an observability backend, this
  allow-list is what bounds what leaves the process.
- **Auth mode (Issue #2505).** Set `FLUXION_REST_AUTH=token|tls` for any
  network-reachable bind; `off` is refused for `0.0.0.0` release builds unless
  `FLUXION_REST_ALLOW_INSECURE=1`.
- **CORS (Issue #2505).** `FLUXION_REST_CORS_ORIGINS` must be an explicit
  origin allow-list (never permissive).
- **Rate limiting (Issue #2505).** Tune `FLUXION_REST_RATE_LIMIT_RPS` /
  `FLUXION_REST_RATE_LIMIT_BURST` to the deployment; defaults are `100`/`1000`.
- **TLS for telemetry sinks.** `fluxion-twin` MQTT is TLS-only by default
  (`mqtts://`, port 8883); plaintext requires `FLUXION_MQTT_ALLOW_INSECURE`.
