//! Model integrity: checksums, manifests, signatures, and path validation.
//!
//! SHA-256 hashing, sha256sum-manifest parsing, the `FLUXION_ONNX_MODEL_SIGNATURE`
//! override, `O_NOFOLLOW` verified model loading, and model-version validation.

use crate::util::sha256_hex::{decode_sha256_hex, sha256_hex};
use sha2::{Digest, Sha256};
use std::path::Path;
use subtle::ConstantTimeEq;

/// Errors produced when constructing or validating a model version string.
///
/// Issue #1335: typed error so callers can distinguish a malformed semver
/// from a hash mismatch or a registry miss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionError {
    /// The version string is not strict semver (e.g. "3.1" or "v3").
    InvalidSemver(String),
    /// The version is a syntactically valid semver but is the forbidden
    /// placeholder "0.0.0".
    PlaceholderVersion(String),
    /// The SHA-256 hex string is not 64 lowercase/uppercase hex characters.
    InvalidHash(String),
    /// The ONNX opset version is unsupported (must be in `1..=17`).
    UnsupportedOpset(u32),
}

impl std::fmt::Display for VersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionError::InvalidSemver(v) => {
                write!(f, "invalid semver version '{}': expected strict MAJOR.MINOR.PATCH (e.g. '3.1.0'); pre-release/build identifiers are allowed", v)
            }
            VersionError::PlaceholderVersion(v) => {
                write!(f, "forbidden placeholder version '{}': '0.0.0' is reserved for the default-constructed metadata and is not a valid release identifier", v)
            }
            VersionError::InvalidHash(h) => {
                write!(
                    f,
                    "invalid SHA-256 hash '{}': expected 64 hexadecimal characters",
                    h
                )
            }
            VersionError::UnsupportedOpset(op) => {
                write!(
                    f,
                    "unsupported ONNX opset {}: supported range is 1..=17",
                    op
                )
            }
        }
    }
}

impl std::error::Error for VersionError {}

/// Strict-semver validator for surrogate model version strings.
///
/// Accepts `MAJOR.MINOR.PATCH` with optional `[-prerelease]` and `[+build]`,
/// where each numeric component is `0..=999` and the `v` prefix is rejected.
/// Returns `Ok(())` for valid semver (including `0.0.0` syntactically — the
/// placeholder check is enforced separately by [`VersionError::PlaceholderVersion`]).
pub fn validate_semver(version: &str) -> Result<(), VersionError> {
    if version.is_empty() || version.len() > 64 {
        return Err(VersionError::InvalidSemver(version.to_string()));
    }
    let (core, _pre_build) = match version.split_once('-') {
        Some((c, rest)) => (c, Some(rest)),
        None => match version.split_once('+') {
            Some((c, rest)) => (c, Some(rest)),
            None => (version, None),
        },
    };
    let mut parts = core.split('.');
    let major = parts
        .next()
        .ok_or_else(|| VersionError::InvalidSemver(version.to_string()))?;
    let minor = parts
        .next()
        .ok_or_else(|| VersionError::InvalidSemver(version.to_string()))?;
    let patch = parts
        .next()
        .ok_or_else(|| VersionError::InvalidSemver(version.to_string()))?;
    if parts.next().is_some() {
        return Err(VersionError::InvalidSemver(version.to_string()));
    }
    let is_numeric_component = |s: &str| {
        !s.is_empty()
            && s.len() <= 3
            && s.chars().all(|c| c.is_ascii_digit())
            && (s.len() == 1 || !s.starts_with('0'))
    };
    if !(is_numeric_component(major) && is_numeric_component(minor) && is_numeric_component(patch))
    {
        return Err(VersionError::InvalidSemver(version.to_string()));
    }
    Ok(())
}

/// Validate a SHA-256 hex string (64 lowercase or uppercase hex chars).
pub fn validate_sha256_hex(hash: &str) -> Result<(), VersionError> {
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(VersionError::InvalidHash(hash.to_string()));
    }
    Ok(())
}

/// Pinned metadata for a surrogate ONNX model release.
///
/// Issue #1335: the registry stores one `ModelVersion` per release. The
/// `model_sha256` matches the bytes of the ONNX file; the
/// `training_data_hash` matches a content hash of the training set manifest
/// (a CI-managed file outside this repo).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelVersion {
    /// Strict semver version (e.g. "3.1.0").
    pub version: String,
    /// Lowercase hex SHA-256 of the `.onnx` file.
    pub model_sha256: String,
    /// ONNX opset version used to export the model (1..=17 per ADR-0004).
    pub onnx_opset_version: u32,
    /// Lowercase hex SHA-256 of the training data manifest.
    pub training_data_hash: String,
    /// ISO-8601 date when the model was trained (UTC).
    pub trained_on: String,
    /// Free-form one-line summary of the training set.
    pub training_data_summary: String,
    /// Minimum fraction of held-out EnergyPlus hourly zone-temperature
    /// predictions that must remain within the validation tolerance.
    ///
    /// The release value is established from ASHRAE 140 Case 950 hourly output,
    /// which is generated by EnergyPlus 25.2.0 with the Golden, Colorado TMY3
    /// weather file and is excluded from surrogate training. Validation compares
    /// every predicted hourly zone temperature with the committed reference and
    /// counts samples within the gate tolerance; model acceptance requires the
    /// observed fraction to meet or exceed this non-zero threshold.
    pub expected_accuracy: f64,
    /// Absolute path or relative path under the model store; ONNX files
    /// themselves are never committed to git (see ADR-0004).
    pub model_path: String,
}

impl ModelVersion {
    /// Build a `ModelVersion` from raw fields, validating all invariants.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        version: &str,
        model_sha256: &str,
        onnx_opset_version: u32,
        training_data_hash: &str,
        trained_on: &str,
        training_data_summary: &str,
        expected_accuracy: f64,
        model_path: &str,
    ) -> Result<Self, VersionError> {
        if version == "0.0.0" {
            return Err(VersionError::PlaceholderVersion(version.to_string()));
        }
        validate_semver(version)?;
        validate_sha256_hex(model_sha256)?;
        validate_sha256_hex(training_data_hash)?;
        if onnx_opset_version == 0 || onnx_opset_version > 17 {
            return Err(VersionError::UnsupportedOpset(onnx_opset_version));
        }
        Ok(ModelVersion {
            version: version.to_string(),
            model_sha256: model_sha256.to_ascii_lowercase(),
            onnx_opset_version,
            training_data_hash: training_data_hash.to_ascii_lowercase(),
            trained_on: trained_on.to_string(),
            training_data_summary: training_data_summary.to_string(),
            expected_accuracy,
            model_path: model_path.to_string(),
        })
    }

    /// Parse the version entry out of one JSON object (registry file shape).
    pub fn from_json(value: &serde_json::Value) -> Result<Self, VersionError> {
        let version = value
            .get("version")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| VersionError::InvalidSemver("<missing>".to_string()))?;
        let model_sha256 = value
            .get("model_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| VersionError::InvalidHash("<missing>".to_string()))?;
        let onnx_opset_version = value
            .get("onnx_opset_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or(VersionError::UnsupportedOpset(0))? as u32;
        let training_data_hash = value
            .get("training_data_hash")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| VersionError::InvalidHash("<missing>".to_string()))?;
        let trained_on = value
            .get("trained_on")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let training_data_summary = value
            .get("training_data_summary")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let expected_accuracy = value
            .get("expected_accuracy")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let model_path = value
            .get("model_path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        ModelVersion::new(
            version,
            model_sha256,
            onnx_opset_version,
            training_data_hash,
            trained_on,
            training_data_summary,
            expected_accuracy,
            model_path,
        )
    }
}

/// In-memory registry of pinned surrogate model versions.
///
/// Loaded from `tests/surrogate_models/registry.json` (see ADR-0004). The
/// `.onnx` files themselves are not in git; this registry only carries
/// hashes and metadata so that `load_version` can validate before opening
/// the session.
#[derive(Clone, Debug, Default)]
pub struct ModelRegistry {
    pub versions: Vec<ModelVersion>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        ModelRegistry::default()
    }

    pub fn from_versions(versions: Vec<ModelVersion>) -> Self {
        ModelRegistry { versions }
    }

    /// Parse a registry from its JSON representation.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("registry JSON parse error: {}", e))?;
        let arr = value
            .get("versions")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "registry must contain a top-level 'versions' array".to_string())?;
        let mut versions = Vec::with_capacity(arr.len());
        for (i, entry) in arr.iter().enumerate() {
            let v = ModelVersion::from_json(entry)
                .map_err(|e| format!("registry entry #{}: {}", i, e))?;
            versions.push(v);
        }
        Ok(ModelRegistry { versions })
    }

    pub fn lookup(&self, version: &str) -> Option<&ModelVersion> {
        self.versions.iter().find(|v| v.version == version)
    }

    pub fn latest(&self) -> Option<&ModelVersion> {
        self.versions.last()
    }

    pub fn len(&self) -> usize {
        self.versions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

/// Compute the lowercase hex SHA-256 digest of a file's bytes.
///
/// Streams the file through the hasher in 1 MiB chunks (Issue #3592) so a
/// hostile writer that grows the file between the size pre-check in
/// `validate_model_path_in_dir` (line ~3900) and the read cannot OOM the
/// verifier. After each chunk the cumulative byte count is compared to
/// `MAX_MODEL_SIZE_BYTES`; if the cap is exceeded the function short-
/// circuits with the same `model file exceeds size limit` error used by
/// `validate_model_path_in_dir`, matching the message format the rest of
/// the verifier already produces. Steady-state memory is bounded at one
/// 1 MiB buffer regardless of file size.
pub fn compute_file_sha256(path: &Path) -> Result<String, String> {
    use std::io::Read;
    if !path.exists() {
        return Err(format!("file not found: {}", path.display()));
    }
    let mut file = std::fs::File::open(path).map_err(|e| format!("read failed: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024]; // 1 MiB streaming chunk
    let mut total: u64 = 0;
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err(format!("read failed: {}", e)),
        };
        total = total.saturating_add(n as u64);
        if total > MAX_MODEL_SIZE_BYTES {
            return Err(format!(
                "model file exceeds size limit ({} bytes)",
                MAX_MODEL_SIZE_BYTES
            ));
        }
        hasher.update(&buf[..n]);
    }
    Ok(sha256_hex(hasher.finalize()))
}

/// Compute the lowercase hex SHA-256 of a byte slice (for in-memory checks).
pub fn compute_bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    sha256_hex(hasher.finalize())
}

/// Compare a claimed SHA-256 against the file's actual SHA-256.
///
/// Issue #3655 / CWE-208 — the previous implementation used
/// `str::eq_ignore_ascii_case`, which short-circuits on the first
/// non-matching byte and therefore leaks per-byte timing information about
/// how many leading bytes of a candidate digest match. For a SHA-256
/// integrity tag this is *not* an extractable secret under typical use, but
/// it is the wrong primitive for any tag/MAC comparison and the rest of
/// the Rust security ecosystem treats it as a smell.
///
/// The replacement decodes both hex inputs to 32-byte digests via
/// [`crate::util::sha256_hex::decode_sha256_hex`] (which itself is constant
/// in its visit pattern over the input length) and then compares the bytes
/// with [`subtle::ConstantTimeEq::ct_eq`]. `subtle` is the same audited
/// crate used elsewhere in the Rust crypto ecosystem for HMAC, AEAD, and
/// signature-tag checks, and its XOR-OR accumulator is documented to be
/// written so the LLVM backend cannot fold it into a branch. Upper- and
/// lower-case hex continue to compare equal (the decoder accepts both) so
/// the public contract of this function is unchanged for existing callers.
pub fn validate_hash(expected: &str, actual: &str) -> Result<(), String> {
    let expected_bytes = decode_sha256_hex(expected).ok_or_else(|| {
        format!(
            "SHA-256 mismatch: expected {}, got {} (expected is not a valid 64-char hex SHA-256)",
            expected, actual
        )
    })?;
    let actual_bytes = decode_sha256_hex(actual).ok_or_else(|| {
        format!(
            "SHA-256 mismatch: expected {}, got {} (actual is not a valid 64-char hex SHA-256)",
            expected, actual
        )
    })?;
    if expected_bytes.ct_eq(&actual_bytes).into() {
        Ok(())
    } else {
        Err(format!(
            "SHA-256 mismatch: expected {}, got {}",
            expected, actual
        ))
    }
}

/// Maximum permitted ONNX model file size: 256 MiB (Issue #2529).
pub const MAX_MODEL_SIZE_BYTES: u64 = 256 * 1024 * 1024;

/// Default allow-list directory for ONNX models when `FLUXION_MODEL_DIR` is
/// unset (Issue #2529). Mirrors the conventional location used by
/// `FLUXION_ONNX_MODEL`'s default (`models/surrogate_zone_thermal.onnx`).
const DEFAULT_MODEL_DIR: &str = "models";

/// Name of the environment variable that overrides the SHA-256 manifest
/// check for rotated models (Issue #2906). When set to a 64-char lowercase
/// or uppercase hex SHA-256 digest, that digest is used as the authoritative
/// expected hash INSTEAD of the manifest at `<model>.sha256`. The env override
/// exists so operators can rotate a model without first waiting for the
/// manifest to be updated in the repo; once the manifest is updated, unset
/// the variable.
pub const ENV_ONNX_MODEL_SIGNATURE: &str = "FLUXION_ONNX_MODEL_SIGNATURE";

/// Verify the integrity of an ONNX model against a SHA-256 manifest
/// (Issue #2906).
///
/// Back-compat wrapper around [`open_and_verify_onnx`]: preserves the
/// original `Result<(), String>` signature while routing through the
/// `O_NOFOLLOW` single-handle path that closes the TOCTOU window
/// (Issue #3573).
///
/// Behaviour, in resolution order:
/// 1. If `FLUXION_ONNX_MODEL_SIGNATURE` is set to a 64-char hex SHA-256,
///    use it as the authoritative expected digest.
/// 2. Otherwise, look for a manifest file at `<model>.sha256` (the standard
///    `sha256sum` output format). If present, parse it and look for an
///    entry whose filename matches the model basename.
/// 3. If neither source provides a digest, fail-closed with a message
///    referencing Issue #3209.
/// 4. Compute the SHA-256 of the buffered model bytes and compare with
///    the resolved expected digest. Mismatch returns `Err` (fail-closed)
///    so a poisoned or bit-flipped model cannot influence ASHRAE 140
///    results.
///
/// The manifest format mirrors `sha256sum` output:
///
/// ```text
/// # comment lines start with '#'; blank lines are ignored
/// <64-hex-digest>  <relative-path>
/// ```
///
/// Both single- and double-space separators are accepted, as is the
/// `sha256sum -b` "binary mode" `*` prefix on the path. The entry whose
/// filename equals the model basename **exactly** wins (no
/// `ends_with` matching either direction); see [`parse_sha256_manifest`]
/// for the empty-filename rules. Issue #3582.
pub fn verify_onnx_signature(model_path: &Path) -> Result<(), String> {
    open_and_verify_onnx(model_path).map(|_verified_bytes| ())
}

/// Open `model_path` once with `O_NOFOLLOW` semantics, hash the bytes
/// via SHA-256, verify the digest against the manifest / env-var, and
/// return the verified byte buffer.
///
/// This is the single point at which the ONNX model leaves the
/// filesystem (Issue #3573). Callers MUST pass the returned bytes to
/// `ort::session::SessionBuilder::commit_from_memory` rather than
/// re-resolving the path for session instantiation — re-opening the
/// path would re-introduce the TOCTOU window between the verify-read
/// and the load-read.
///
/// Defences (each independent, defense-in-depth):
///
/// 1. `O_NOFOLLOW` open: refuses to follow a symlink at the final path
///    component, so the kernel pins the open to a specific inode even
///    if the directory entry is replaced (mv'd over) before the read
///    returns. Hard-coded per-platform values rather than a `libc` dep
///    just for one constant.
///
/// 2. Size consistency check: the handle's `metadata().len()` is
///    captured before the read and re-checked after. A writer who
///    modifies the same inode (not just the directory entry) between
///    those two reads will surface as a size mismatch and the load
///    fails closed.
///
/// 3. SHA-256 over the buffered bytes: the manifest / env-var digest
///    must match. A single, byte-exact read is fed into both the
///    hasher and (via `commit_from_memory`) into the session loader, so
///    there is no possibility of "verify saw A but session loaded B"
///    once this function returns `Ok`.
pub fn open_and_verify_onnx(model_path: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;

    // ----- (1) open with O_NOFOLLOW on Unix ----------------------------
    #[cfg(unix)]
    let mut file = {
        use std::os::unix::fs::OpenOptionsExt;

        // O_NOFOLLOW values per fcntl(2). Hard-coded to avoid pulling
        // in `libc` as a direct dependency just for this one constant.
        #[cfg(target_os = "linux")]
        const O_NOFOLLOW: i32 = 0o400_000;
        #[cfg(any(
            target_os = "macos",
            target_os = "freebsd",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos",
        ))]
        const O_NOFOLLOW: i32 = 0x0100;
        #[cfg(any(target_os = "openbsd", target_os = "netbsd"))]
        const O_NOFOLLOW: i32 = 0x0200;
        // Conservative fallback: no symlink-rejection on unknown Unix
        // targets. The size-consistency check still catches in-place
        // modifications; only symlink swaps slip through.
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "freebsd",
            target_os = "openbsd",
            target_os = "netbsd",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos",
        )))]
        const O_NOFOLLOW: i32 = 0;

        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW)
            .open(model_path)
            .map_err(|e| {
                // ELOOP = the trailing component was a symlink and
                // O_NOFOLLOW tripped; surface that as the named
                // defence.
                let detail = if e.raw_os_error() == Some(40 /* ELOOP on Linux */)
                    || e.raw_os_error() == Some(62 /* ELOOP on macOS/BSD */)
                {
                    "O_NOFOLLOW rejected a symlink at the model path (Issue #3573)"
                } else {
                    "check permissions and that the file exists"
                };
                format!(
                    "failed to open ONNX model at {}: {e} ({detail})",
                    model_path.display()
                )
            })?
    };

    #[cfg(not(unix))]
    let mut file = std::fs::File::open(model_path)
        .map_err(|e| format!("failed to open ONNX model at {}: {e}", model_path.display()))?;

    // ----- (2) size consistency check (pre) ----------------------------
    let initial_size = file
        .metadata()
        .map_err(|e| format!("failed to stat ONNX model {}: {e}", model_path.display()))?
        .len();

    if initial_size > MAX_MODEL_SIZE_BYTES {
        return Err(format!(
            "ONNX model {} exceeds maximum size: {} bytes (limit {})",
            model_path.display(),
            initial_size,
            MAX_MODEL_SIZE_BYTES
        ));
    }

    // ----- (3) single read into a Vec<u8> ------------------------------
    let mut bytes = Vec::with_capacity(initial_size as usize);
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read ONNX model {}: {e}", model_path.display()))?;

    // ----- (4) size consistency check (post) ---------------------------
    // If the file grew or shrank during the read, refuse. This catches
    // the writer-into-same-inode class of attack that O_NOFOLLOW alone
    // cannot detect (the inode is bound to our handle but its contents
    // can still be modified by anyone holding write access).
    let post_size = file
        .metadata()
        .map_err(|e| format!("failed to re-stat ONNX model {}: {e}", model_path.display()))?
        .len();
    if post_size != initial_size || bytes.len() as u64 != initial_size {
        return Err(format!(
            "ONNX model {} changed size during read (initial={} read={} post={}). \
             This indicates a TOCTOU write-into-same-inode attack; refusing to load \
             (fail-closed, Issue #3573).",
            model_path.display(),
            initial_size,
            bytes.len(),
            post_size
        ));
    }

    // ----- (5) digest resolution (manifest OR env-var override) -------
    // Issue #3590: the `FLUXION_ONNX_MODEL_SIGNATURE` env override silently
    // re-routes the integrity check to a different authoritative digest. We
    // remember whether that branch was taken here so the post-compare warn
    // (below) can echo the override hash, the manifest hash it bypassed (if
    // any), and the model path, and so the
    // `fluxion_surrogate_load_outcome_total{outcome="override_accepted"}`
    // counter can be bumped for production dashboards.
    let env_override: Option<String>;
    let expected = match std::env::var(ENV_ONNX_MODEL_SIGNATURE)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        Some(digest) => {
            validate_sha256_hex(&digest).map_err(|_| {
                format!(
                    "{ENV_ONNX_MODEL_SIGNATURE}={digest:?} is not a valid 64-char hex SHA-256 \
                     digest; refusing to load (fail-closed, see Issue #2906)"
                )
            })?;
            env_override = Some(digest.to_ascii_lowercase());
            env_override.clone().unwrap()
        }
        None => {
            env_override = None;
            match read_manifest_hash(model_path) {
                Some(Ok(hash)) => hash,
                Some(Err(e)) => return Err(e),
                None => {
                    return Err(format!(
                        "no SHA-256 manifest at {} and {ENV_ONNX_MODEL_SIGNATURE} unset; \
                         integrity verification impossible (fail-closed, Issue #3209). \
                         Ship a <model>.sha256 alongside the .onnx file or set \
                         {ENV_ONNX_MODEL_SIGNATURE}=<hex-digest> for rotated models.",
                        manifest_path_for(model_path).display()
                    ));
                }
            }
        }
    };

    // ----- (6) SHA-256 of the bytes we actually read ------------------
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = sha256_hex(hasher.finalize());

    validate_hash(&expected, &actual).map_err(|e| {
        format!(
            "ONNX model integrity verification FAILED for {} ({e}). This may indicate \
             model poisoning (bit-flip, tampered registry mirror, malicious build \
             step). Refusing to load (fail-closed, Issue #2906). To override for a \
             rotated model, set {ENV_ONNX_MODEL_SIGNATURE}=<hex-digest> to the new \
             expected digest, regenerate the manifest, and unset the override.",
            model_path.display()
        )
    })?;

    // Issue #3590 — audit/observability for `FLUXION_ONNX_MODEL_SIGNATURE`.
    // Every successful load that used the env override emits a `tracing::warn!`
    // (target `fluxion::ai::surrogate::signature_override`) and bumps the
    // `fluxion_surrogate_load_outcome_total{outcome="override_accepted"}`
    // counter so production dashboards can detect stale overrides before
    // they authorise a model the operator never intended. We also surface
    // the manifest hash (if any) the override is bypassing, so an
    // operator reviewing logs can see "the env var replaced manifest
    // <hash> for <path> with <override>" without having to recompute it
    // manually. A load that failed validation never reaches this line
    // (the `?` above short-circuits), so the counter is incremented
    // exactly once per successful override use, matching the CI
    // assertion in `verify_onnx_signature_env_override_emits_warn_and_counter`.
    if let Some(override_hash) = env_override.as_ref() {
        // Manifest hash, if any, the override is bypassing. We do NOT
        // surface a malformed-manifest error here — the override is the
        // authoritative digest, so a broken or missing manifest is
        // silently accepted on this path (that's the whole point of
        // #2906). We only record what was there for the operator's
        // audit log.
        let manifest_hash: Option<String> = match read_manifest_hash(model_path) {
            Some(Ok(h)) => Some(h),
            _ => None,
        };
        let manifest_hash_display: &str = manifest_hash.as_deref().unwrap_or("<none>");
        tracing::warn!(
            target: "fluxion::ai::surrogate::signature_override",
            FLUXION_ONNX_MODEL_SIGNATURE = %override_hash,
            manifest_hash = %manifest_hash_display,
            model_path = %model_path.display(),
            "FLUXION_ONNX_MODEL_SIGNATURE env override accepted for ONNX model load; \
             unset the variable once the rotated manifest is committed \
             (see docs/AGENTS.md goal #5 and Issue #3590)",
        );
        metrics::counter!(
            "fluxion_surrogate_load_outcome_total",
            "outcome" => "override_accepted",
        )
        .increment(1);
    }

    // Bytes are returned by value; the caller (`with_gpu_backend`,
    // `with_multi_device`) wraps them in an `Arc<Vec<u8>>` and feeds
    // them to `commit_from_memory`, eliminating the second filesystem
    // read that previously caused the TOCTOU window (Issue #3573).
    Ok(bytes)
}

/// Return the conventional path of the SHA-256 manifest for `model_path`
/// (i.e. `<model_path>.sha256`).
fn manifest_path_for(model_path: &Path) -> std::path::PathBuf {
    let mut s = model_path.as_os_str().to_owned();
    s.push(".sha256");
    std::path::PathBuf::from(s)
}

/// Try to read and parse a SHA-256 manifest at `<model>.sha256`. Returns:
/// - `Some(Ok(hash))` if a matching entry is found,
/// - `Some(Err(e))` if the manifest exists but is malformed, or
/// - `None` if the manifest file does not exist.
fn read_manifest_hash(model_path: &Path) -> Option<Result<String, String>> {
    let manifest_path = manifest_path_for(model_path);
    let contents = match std::fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            return Some(Err(format!(
                "failed to read SHA-256 manifest at {}: {e}",
                manifest_path.display()
            )))
        }
    };
    Some(parse_sha256_manifest(&manifest_path, &contents, model_path))
}

/// Parse `sha256sum`-format manifest text and return the hash for the
/// entry whose filename matches `model_path`. Accepts `# comment` lines,
/// blank lines, single- or double-space separators, and the binary-mode
/// `*` prefix on the filename.
///
/// ## Issue #3582 — strict filename binding
///
/// The previous implementation matched the entry's filename to the
/// model basename with `filename.ends_with(expected_basename) ||
/// expected_basename.ends_with(filename)`, and short-circuited on the
/// first manifest entry that had no filename. That had two
/// consequences:
///
/// 1. A manifest entry for `x.onnx` was accepted for any model whose
///    basename ended with `x.onnx` (e.g. `evilx.onnx`,
///    `attackx.onnx`). The bytes-hash check is the next line of
///    defence, but the per-file binding is part of the contract on
///    its own: AGENTS.md goal #5 (fail-closed) and `verify_onnx_signature`
///    being the single point at which an ONNX model leaves the
///    filesystem (#3573) both depend on the manifest entry naming the
///    exact file being verified.
///
/// 2. The first manifest entry with no filename authorised every model
///    that did not find a later basename-matching entry. A multi-entry
///    manifest that starts with `<hash>` (no filename) would bind to
///    whichever model was loaded first.
///
/// The rules below restore the per-file contract:
///
/// - The manifest entry's filename is reduced to its **basename**
///   (last path component) and compared to `expected_basename` exactly.
///   The previous `ends_with`/`starts_with` heuristic accepted
///   `evilx.onnx` for an entry naming `x.onnx`; the basename-equality
///   rule rejects that. The basename rule still accepts legitimate
///   `sha256sum`-style entries whose path includes a relative directory
///   prefix (e.g. `assets/dummy_surrogate.onnx` for model
///   `dummy_surrogate.onnx`).
/// - An empty `filename` is honoured only when the manifest holds
///   exactly one entry AND `manifest_path.file_stem()` equals
///   `expected_basename` (i.e. the manifest file is unambiguously named
///   for this model). Otherwise the bare hash is rejected with a
///   fail-closed error referencing Issue #3582.
pub(crate) fn parse_sha256_manifest(
    manifest_path: &Path,
    contents: &str,
    model_path: &Path,
) -> Result<String, String> {
    let expected_basename = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if expected_basename.is_empty() {
        return Err(format!(
            "cannot derive basename from model path {}",
            model_path.display()
        ));
    }

    // Two-pass: collect the first exact-match and the first
    // empty-filename entry, then decide. Issue #3582. Returning
    // eagerly on the first match would let a bare-hash entry earlier
    // in the manifest shadow an exact match later, and would let a
    // suffix-matching entry shadow an exact basename somewhere below.
    let mut exact_match: Option<String> = None;
    let mut empty_filename_hash: Option<String> = None;
    let mut entry_count: usize = 0;

    for (lineno, raw) in contents.lines().enumerate() {
        let line = raw.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // sha256sum layout: `<hash><space><space-or-*><path>` or
        // `<hash><space><path>` (text mode). Split on first run of
        // whitespace; first field is the hash, remainder is the filename.
        let mut split = line.splitn(2, |c: char| c.is_whitespace());
        let hash = split.next().unwrap_or("").trim();
        let filename_raw = split.next().unwrap_or("").trim();
        // sha256sum -b prefixes the path with `*`; strip it.
        let filename_with_prefix = filename_raw
            .strip_prefix('*')
            .unwrap_or(filename_raw)
            .trim();
        if hash.len() != 64 {
            // Could be a malformed line; surface a precise error rather
            // than silently skipping.
            return Err(format!(
                "{}:{} malformed entry (hash must be 64 hex chars, got {:?})",
                manifest_path.display(),
                lineno + 1,
                hash
            ));
        }
        if let Err(e) = validate_sha256_hex(hash) {
            return Err(format!(
                "{}:{} hash {:?} is not valid hex: {:?}",
                manifest_path.display(),
                lineno + 1,
                hash,
                e
            ));
        }
        entry_count += 1;
        if filename_with_prefix.is_empty() {
            // Remember the bare hash; honour it only at the end if the
            // preconditions below are met (Issue #3582).
            if empty_filename_hash.is_none() {
                empty_filename_hash = Some(hash.to_ascii_lowercase());
            }
            continue;
        }
        // Reduce the entry's filename to its basename before comparing.
        // `sha256sum` writes the path as it was passed in (often a
        // relative path like `assets/dummy_surrogate.onnx`); the binding
        // contract is per-file, so directory prefixes are not part of
        // the identity. Issue #3582: comparing full-path strings here
        // would break legitimate `sha256sum` manifests, and comparing
        // raw `ends_with` either-direction is exactly the bug being
        // fixed. `Path::file_name()` strips any leading directory
        // components and never panics on inputs without a separator.
        let manifest_basename = std::path::Path::new(filename_with_prefix)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if manifest_basename == expected_basename {
            // Strict basename equality (Issue #3582). `x.onnx` does NOT
            // match `evilx.onnx`, `attackx.onnx`, etc.
            if exact_match.is_none() {
                exact_match = Some(hash.to_ascii_lowercase());
            }
        }
        // Else: basename does not equal the model basename — ignore.
        // The hash still has to match the bytes; we just refuse to
        // bind the wrong filename.
    }

    if let Some(hash) = exact_match {
        return Ok(hash);
    }
    if let Some(hash) = empty_filename_hash {
        if entry_count == 1 {
            // Single-entry bare-hash manifest — only safe to honour
            // when the manifest file is unambiguously for THIS model
            // (i.e. `manifest_path.file_stem() == expected_basename`).
            // In normal usage `manifest_path` is built by
            // `manifest_path_for(model_path)` and this condition holds
            // automatically; the check is defence-in-depth against
            // a manifest file being misnamed or being loaded for a
            // different model.
            let manifest_named_for_this_model =
                manifest_path.file_stem().and_then(|s| s.to_str()) == Some(expected_basename);
            if manifest_named_for_this_model {
                return Ok(hash);
            }
        }
        return Err(format!(
            "{}: empty-filename entry cannot be bound to model {} \
             (basename {:?}); manifest has {} entries and \
             manifest_path.file_stem() does not equal the model basename \
             (Issue #3582)",
            manifest_path.display(),
            model_path.display(),
            expected_basename,
            entry_count,
        ));
    }
    Err(format!(
        "{}: no entry found for model {} (basename {:?})",
        manifest_path.display(),
        model_path.display(),
        expected_basename
    ))
}

/// Validates a user-supplied model path against the security policy from
/// Issue #2529 before it reaches the ONNX runtime. Reads the allow-list
/// directory from the `FLUXION_MODEL_DIR` environment variable (default
/// `models/`).
///
/// On success returns the canonicalised absolute path. All error messages
/// are deliberately generic and omit the raw user-supplied path so that
/// attacker-controlled input is never reflected back to the Python caller
/// (closes the error oracle).
pub fn validate_model_path(p: &str) -> Result<std::path::PathBuf, String> {
    let dir = std::env::var("FLUXION_MODEL_DIR").unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    validate_model_path_in_dir(p, std::path::Path::new(&dir))
}

/// Parameterised core of [`validate_model_path`]. Accepts an explicit
/// allow-list directory so it can be unit-tested without racing on the
/// process-wide `FLUXION_MODEL_DIR` env var.
///
/// Checks, in order:
/// 1. `Path::new(p).is_file()` — existence (follows symlinks like the rest
///    of `std::fs`).
/// 2. `symlink_metadata()` refuses a symlink at the user-supplied path
///    component (Issue #3651) — the same policy as
///    `validate_epw_path_in_dir` (`fluxion-core/src/weather/epw.rs`).
/// 3. extension == `onnx`.
/// 4. canonicalised path is inside `allowed_dir` (component-wise
///    `starts_with` on canonical paths — blocks `..` traversal and symlinks
///    that escape the allow-list).
/// 5. file size ≤ [`MAX_MODEL_SIZE_BYTES`].
pub fn validate_model_path_in_dir(
    p: &str,
    allowed_dir: &Path,
) -> Result<std::path::PathBuf, String> {
    let raw = Path::new(p);
    if !raw.is_file() {
        return Err("model file not found".to_string());
    }
    // Refuse symlinks (Issue #3651). canonicalize() resolves symlinks to
    // their targets, so checking symlink_metadata() here also rejects a
    // symlink that points to a file *inside* the allow-list (belt-and-braces
    // against future TOCTOU or symlink-swap attacks) — the same mechanism,
    // ordering, and message pattern as `validate_epw_path_in_dir`. This is
    // the only gate on platforms where the `O_NOFOLLOW` open fallback is a
    // no-op (`const O_NOFOLLOW: i32 = 0`).
    let link_meta = std::fs::symlink_metadata(raw)
        .map_err(|_| "failed to read model file metadata".to_string())?;
    if link_meta.file_type().is_symlink() {
        return Err("model file path may not be a symbolic link".to_string());
    }
    if raw
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        != Some("onnx".to_string())
    {
        return Err("invalid model file extension (expected .onnx)".to_string());
    }
    let canonical_model =
        std::fs::canonicalize(raw).map_err(|_| "failed to canonicalize model path".to_string())?;
    let canonical_dir = std::fs::canonicalize(allowed_dir)
        .map_err(|_| "allowed model directory not found".to_string())?;
    if !canonical_model.starts_with(&canonical_dir) {
        return Err("model path outside allowed directory".to_string());
    }
    let size = std::fs::metadata(&canonical_model)
        .map_err(|_| "failed to read model file metadata".to_string())?
        .len();
    if size > MAX_MODEL_SIZE_BYTES {
        return Err(format!(
            "model file exceeds size limit ({} bytes)",
            MAX_MODEL_SIZE_BYTES
        ));
    }
    Ok(canonical_model)
}
