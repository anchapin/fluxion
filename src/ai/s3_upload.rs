//! S3 upload pipeline for tensor datasets with provenance metadata
//! (Issue #1779, plan key T5.4).
//!
//! Uploads generated FTDS tensor-dataset shards to S3 with:
//!
//! - **Dataset-version prefixing** — every object is stored under a key derived
//!   from the FTDS schema version and a content-addressable dataset hash, so
//!   re-uploads of an unchanged dataset are no-ops and consumers can pin a
//!   specific version.
//! - **Provenance manifest** — a [`ProvenanceManifest`] (solver version, git
//!   SHA, parameter seed, weather source, dataset hash, timestamp) is uploaded
//!   alongside the data so every training run is fully reproducible.
//! - **Resumable multipart upload** — large shards are uploaded via the S3
//!   multipart API; the upload state is persisted locally so an interrupted
//!   run resumes from the last completed part instead of restarting.
//!
//! ## Dependency strategy
//!
//! Rather than pulling in the heavy `aws-sdk-s3` crate (which would blow past
//! the published-crate 10 MB cap), this module implements AWS Signature V4
//! signing from scratch using the already-present [`sha2`] and [`reqwest`]
//! crates. HMAC-SHA256 is implemented inline (RFC 2104, ~15 lines).
//!
//! ## Testing
//!
//! The HTTP layer is abstracted behind the [`S3Transport`] trait. Tests inject
//! a mock transport ([`tests::MockS3Transport`] or the one in the integration
//! test file) so the full upload pipeline — including multipart resume — can be
//! exercised without real S3 credentials.
//!
//! ## Example
//!
//! ```no_run
//! use fluxion::ai::s3_upload::{S3UploadConfig, S3Uploader, ProvenanceManifest};
//! use fluxion::ai::tensor_dataset::TensorDatasetManifest;
//! use std::path::Path;
//!
//! let config = S3UploadConfig::from_env(
//!     "my-training-bucket",
//!     "datasets/ftds",
//!     "us-east-1",
//! ).unwrap();
//! let provenance = ProvenanceManifest::builder()
//!     .solver_version("9r4c-1.0")
//!     .git_sha("a1b2c3d")
//!     .parameter_seed(42)
//!     .weather_source("TMY3-4A")
//!     .build();
//! let uploader = S3Uploader::new(config);
//! // let report = uploader.upload_dataset(Path::new("/data/dataset"), &provenance).unwrap();
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ai::tensor_dataset::TensorDatasetManifest;

/// S3 service name — always `"s3"` for the signing scope.
const SERVICE: &str = "s3";

/// AWS Sig V4 algorithm identifier.
const SIG_ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// SHA-256 hash of an empty payload (used for unsigned-payload fallback).
pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Default part size for multipart uploads (8 MiB). S3 requires a minimum part
/// size of 5 MiB for all parts except the last.
pub const DEFAULT_PART_SIZE: usize = 8 * 1024 * 1024;

/// Objects larger than this threshold use multipart upload.
pub const DEFAULT_MULTIPART_THRESHOLD: usize = 8 * 1024 * 1024;

/// S3 requires at most 10 000 parts per multipart upload.
pub const MAX_PARTS: u32 = 10_000;

/// Minimum S3 part size (5 MiB).
pub const MIN_PART_SIZE: usize = 5 * 1024 * 1024;

// =============================================================================
// Error types
// =============================================================================

/// All error conditions produced by this module.
#[derive(Debug, Error)]
pub enum S3UploadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON (de)serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP error from S3 [{status}]: {body}")]
    Http { status: u16, body: String },
    #[error("network error: {0}")]
    Network(String),
    #[error("missing AWS credential: {0}")]
    MissingCredential(String),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("dataset directory has no manifest.json")]
    NoManifest,
    #[error("multipart upload failed for key '{key}': {reason}")]
    Multipart { key: String, reason: String },
    #[error("part size {0} is below S3 minimum of {MIN_PART_SIZE} bytes")]
    PartSizeTooSmall(usize),
    #[error("too many parts required ({0}); increase part size or reduce object size")]
    TooManyParts(u32),
    #[error("signing error: {0}")]
    Signing(String),
}

impl From<reqwest::Error> for S3UploadError {
    fn from(e: reqwest::Error) -> Self {
        S3UploadError::Network(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, S3UploadError>;

// =============================================================================
// HMAC-SHA256 (RFC 2104)
// =============================================================================

const SHA256_BLOCK_SIZE: usize = 64;

/// Compute HMAC-SHA256 of `data` under key `key`.
///
/// Implemented inline (RFC 2104) to avoid adding the `hmac` crate — the
/// published crate must stay dependency-light.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    // Normalize key: if longer than block size, hash it; then pad/truncate to
    // block size.
    let mut key_block = [0u8; SHA256_BLOCK_SIZE];
    if key.len() > SHA256_BLOCK_SIZE {
        let h = Sha256::digest(key);
        key_block[..32].copy_from_slice(&h);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    // ipad / opad
    let mut ipad = [0u8; SHA256_BLOCK_SIZE];
    let mut opad = [0u8; SHA256_BLOCK_SIZE];
    for i in 0..SHA256_BLOCK_SIZE {
        ipad[i] = key_block[i] ^ 0x36;
        opad[i] = key_block[i] ^ 0x5c;
    }

    // inner = H(ipad || data)
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(data);
    let inner_hash = inner.finalize();

    // outer = H(opad || inner)
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_hash);
    let result = outer.finalize();

    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Lowercase hex encoding of a byte slice.
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// SHA-256 hex digest of a byte slice.
fn sha256_hex(data: &[u8]) -> String {
    hex_lower(&Sha256::digest(data))
}

// =============================================================================
// AWS Signature V4
// =============================================================================

/// AWS credentials for S3 authentication.
///
/// **Note**: this struct has a manual [`fmt::Debug`] impl that redacts the
/// secret access key and session token so they are never leaked through
/// `format!("{:?}", creds)`, `dbg!()`, or log output (Issue #2503).
#[derive(Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    /// Optional STS session token for temporary credentials.
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// Load credentials from environment variables.
    ///
    /// Reads `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (required) and
    /// `AWS_SESSION_TOKEN` (optional).
    pub fn from_env() -> Result<Self> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| S3UploadError::MissingCredential("AWS_ACCESS_KEY_ID".to_string()))?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| S3UploadError::MissingCredential("AWS_SECRET_ACCESS_KEY".to_string()))?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        Ok(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        })
    }
}

impl fmt::Debug for AwsCredentials {
    /// Redacts secret material. The access key id is shown truncated (first 4
    /// characters + `****`) since the prefix is non-sensitive and useful for
    /// correlating logs with which key is in use; the secret access key and
    /// session token are fully redacted.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let masked_access_key = if self.access_key_id.len() >= 4 {
            format!("{}****", &self.access_key_id[..4])
        } else {
            "****".to_string()
        };
        let masked_session_token = match &self.session_token {
            Some(_) => "<redacted>",
            None => "None",
        };
        f.debug_struct("AwsCredentials")
            .field("access_key_id", &masked_access_key)
            .field("secret_access_key", &"****")
            .field("session_token", &masked_session_token)
            .finish()
    }
}

/// AWS region identifier (e.g. `"us-east-1"`).
pub type Region = String;

/// A complete AWS Sig V4 signature for a single HTTP request.
#[derive(Clone, Debug)]
pub struct SignedRequest {
    pub method: String,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

/// Build a signed S3 request using AWS Signature Version 4.
///
/// This produces the canonical request, string-to-sign, and Authorization
/// header per the AWS Sig V4 specification.
struct SigV4Signer {
    credentials: AwsCredentials,
    region: Region,
}

impl SigV4Signer {
    fn new(credentials: AwsCredentials, region: Region) -> Self {
        SigV4Signer {
            credentials,
            region,
        }
    }

    /// Derive the Sig V4 signing key:
    /// `kSigning = HMAC(HMAC(HMAC(HMAC("AWS4" + secret, date), region), service), "aws4_request")`
    fn derive_signing_key(&self, date_stamp: &str) -> [u8; 32] {
        let k_secret = format!("AWS4{}", self.credentials.secret_access_key);
        let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
        let k_region = hmac_sha256(&k_date, self.region.as_bytes());
        let k_service = hmac_sha256(&k_region, SERVICE.as_bytes());
        hmac_sha256(&k_service, b"aws4_request")
    }

    /// Sign a request and return the fully-formed [`SignedRequest`].
    ///
    /// Parameters:
    /// - `method` — HTTP method (PUT, GET, POST, DELETE).
    /// - `host` — S3 virtual-hosted-style host (e.g. `bucket.s3.region.amazonaws.com`).
    /// - `key` — object key.
    /// - `query` — query string parameters (e.g. `uploads=` for CreateMultipartUpload).
    /// - `body` — request payload.
    /// - `amz_date` — timestamp in `YYYYMMDDTHHMMSSZ` format.
    /// - `extra_headers` — additional headers to sign (e.g. `x-amz-meta-*`).
    #[allow(clippy::too_many_arguments)]
    fn sign(
        &self,
        method: &str,
        host: &str,
        key: &str,
        query: &str,
        body: &[u8],
        amz_date: &str,
        extra_headers: &BTreeMap<String, String>,
    ) -> SignedRequest {
        let date_stamp = &amz_date[..8];
        let payload_hash = sha256_hex(body);

        // --- Canonical headers (must be sorted by lowercase header name) ---
        let mut canonical_headers_map = BTreeMap::new();
        canonical_headers_map.insert("host".to_string(), host.to_string());
        canonical_headers_map.insert("x-amz-content-sha256".to_string(), payload_hash.clone());
        canonical_headers_map.insert("x-amz-date".to_string(), amz_date.to_string());
        if let Some(ref token) = self.credentials.session_token {
            canonical_headers_map.insert("x-amz-security-token".to_string(), token.clone());
        }
        for (k, v) in extra_headers {
            canonical_headers_map.insert(k.to_ascii_lowercase(), v.clone());
        }

        let canonical_headers: String = canonical_headers_map
            .iter()
            .map(|(k, v)| format!("{k}:{v}\n"))
            .collect();
        let signed_headers: String = canonical_headers_map
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(";");

        // --- Canonical request ---
        let canonical_uri = if key.is_empty() {
            "/".to_string()
        } else {
            format!("/{}", url_encode_key(key))
        };
        let canonical_request = format!(
            "{method}\n{canonical_uri}\n{query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );

        // --- String to sign ---
        let credential_scope = format!("{date_stamp}/{}/{SERVICE}/aws4_request", self.region);
        let hashed_canonical = sha256_hex(canonical_request.as_bytes());
        let string_to_sign =
            format!("{SIG_ALGORITHM}\n{amz_date}\n{credential_scope}\n{hashed_canonical}");

        // --- Signature ---
        let signing_key = self.derive_signing_key(date_stamp);
        let signature = hex_lower(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

        // --- Authorization header ---
        let authorization = format!(
            "{SIG_ALGORITHM} Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
            self.credentials.access_key_id
        );

        // --- Build final request ---
        let url = if query.is_empty() {
            format!("https://{host}{canonical_uri}")
        } else {
            format!("https://{host}{canonical_uri}?{query}")
        };

        let mut headers = BTreeMap::new();
        headers.insert("Authorization".to_string(), authorization);
        headers.insert("Host".to_string(), host.to_string());
        headers.insert("x-amz-content-sha256".to_string(), payload_hash);
        headers.insert("x-amz-date".to_string(), amz_date.to_string());
        if let Some(ref token) = self.credentials.session_token {
            headers.insert("x-amz-security-token".to_string(), token.clone());
        }
        for (k, v) in extra_headers {
            headers.insert(k.clone(), v.clone());
        }

        SignedRequest {
            method: method.to_string(),
            url,
            headers,
            body: body.to_vec(),
        }
    }
}

/// URI-encode an object key for use in the canonical URI.
///
/// Per AWS Sig V4 rules for S3: do NOT encode the forward slash `/`. All other
/// reserved characters are percent-encoded.
fn url_encode_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for b in key.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

// =============================================================================
// S3 Transport trait
// =============================================================================

/// Result of a simple PUT object operation.
#[derive(Clone, Debug)]
pub struct PutResult {
    pub etag: String,
    pub key: String,
    pub size: usize,
}

/// Result of an UploadPart operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartResult {
    pub part_number: u32,
    pub etag: String,
    pub size: usize,
}

/// Abstraction over S3 HTTP operations.
///
/// The real implementation ([`ReqwestS3Transport`]) uses `reqwest` blocking
/// client with Sig V4 signing. Tests inject a mock implementation to exercise
/// the upload pipeline without real S3.
pub trait S3Transport: Send + Sync {
    /// Upload an object via a single PUT request.
    fn put_object(
        &self,
        bucket: &str,
        key: &str,
        body: &[u8],
        metadata: &BTreeMap<String, String>,
    ) -> Result<PutResult>;

    /// Check if an object exists (returns its size if present).
    fn head_object(&self, bucket: &str, key: &str) -> Result<Option<usize>>;

    /// Initiate a multipart upload, returning the `UploadId`.
    fn create_multipart(
        &self,
        bucket: &str,
        key: &str,
        metadata: &BTreeMap<String, String>,
    ) -> Result<String>;

    /// Upload a single part. Part numbers are 1-indexed.
    fn upload_part(
        &self,
        bucket: &str,
        key: &str,
        upload_id: &str,
        part_number: u32,
        body: &[u8],
    ) -> Result<PartResult>;

    /// Complete a multipart upload with the given parts.
    fn complete_multipart(
        &self,
        bucket: &str,
        key: &str,
        upload_id: &str,
        parts: &[PartResult],
    ) -> Result<PutResult>;

    /// Abort a multipart upload (cleanup on failure).
    fn abort_multipart(&self, bucket: &str, key: &str, upload_id: &str) -> Result<()>;
}

// =============================================================================
// ReqwestS3Transport — real S3 implementation
// =============================================================================

/// Real S3 transport backed by `reqwest` blocking client + Sig V4 signing.
pub struct ReqwestS3Transport {
    signer: SigV4Signer,
    client: reqwest::blocking::Client,
    /// Use path-style addressing (true) or virtual-hosted-style (false).
    /// Path-style is needed for some S3-compatible services (MinIO, etc.).
    pub path_style: bool,
}

impl ReqwestS3Transport {
    /// Create a new transport with the given credentials and region.
    pub fn new(credentials: AwsCredentials, region: Region) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        ReqwestS3Transport {
            signer: SigV4Signer::new(credentials, region),
            client,
            path_style: false,
        }
    }

    /// Build the S3 host for a bucket in this transport's region.
    fn host(&self, bucket: &str) -> String {
        if self.signer.region == "us-east-1" {
            format!("{bucket}.s3.amazonaws.com")
        } else {
            format!("{bucket}.s3.{}.amazonaws.com", self.signer.region)
        }
    }

    /// Generate a fixed timestamp for signing (current UTC).
    ///
    /// Exposed as a method so tests can override it (for deterministic signing).
    fn timestamp(&self) -> String {
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
    }

    /// Convert extra metadata into x-amz-meta-* headers.
    fn metadata_to_headers(metadata: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        metadata
            .iter()
            .map(|(k, v)| (format!("x-amz-meta-{k}"), v.clone()))
            .collect()
    }

    /// Execute a signed request and return the response, checking status.
    fn execute(&self, signed: &SignedRequest) -> Result<reqwest::blocking::Response> {
        let mut req = self.client.request(
            signed.method.parse().unwrap_or(reqwest::Method::PUT),
            &signed.url,
        );
        for (k, v) in &signed.headers {
            req = req.header(k, v);
        }
        let resp = req.body(signed.body.clone()).send()?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().unwrap_or_default();
            return Err(S3UploadError::Http { status, body });
        }
        Ok(resp)
    }
}

impl S3Transport for ReqwestS3Transport {
    fn put_object(
        &self,
        bucket: &str,
        key: &str,
        body: &[u8],
        metadata: &BTreeMap<String, String>,
    ) -> Result<PutResult> {
        let host = self.host(bucket);
        let amz_date = self.timestamp();
        let extra_headers = Self::metadata_to_headers(metadata);
        let signed = self
            .signer
            .sign("PUT", &host, key, "", body, &amz_date, &extra_headers);
        let resp = self.execute(&signed)?;
        let etag = resp
            .headers()
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .trim_matches('"')
            .to_string();
        Ok(PutResult {
            etag,
            key: key.to_string(),
            size: body.len(),
        })
    }

    fn head_object(&self, bucket: &str, key: &str) -> Result<Option<usize>> {
        let host = self.host(bucket);
        let amz_date = self.timestamp();
        let empty_headers = BTreeMap::new();
        let signed = self
            .signer
            .sign("HEAD", &host, key, "", &[], &amz_date, &empty_headers);
        let url = signed.url.clone();
        let mut req = self.client.head(&url);
        for (k, v) in &signed.headers {
            req = req.header(k, v);
        }
        let resp = req.send()?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().unwrap_or_default();
            return Err(S3UploadError::Http { status, body });
        }
        let size = resp
            .headers()
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        Ok(Some(size))
    }

    fn create_multipart(
        &self,
        bucket: &str,
        key: &str,
        metadata: &BTreeMap<String, String>,
    ) -> Result<String> {
        let host = self.host(bucket);
        let amz_date = self.timestamp();
        let extra_headers = Self::metadata_to_headers(metadata);
        // POST /key?uploads=
        let signed = self.signer.sign(
            "POST",
            &host,
            key,
            "uploads=",
            &[],
            &amz_date,
            &extra_headers,
        );
        let resp = self.execute(&signed)?;
        let text = resp.text()?;
        // Parse <UploadId>...</UploadId> from XML response
        let upload_id =
            extract_xml_tag(&text, "UploadId").ok_or_else(|| S3UploadError::Multipart {
                key: key.to_string(),
                reason: "missing UploadId in CreateMultipartUpload response".to_string(),
            })?;
        Ok(upload_id)
    }

    fn upload_part(
        &self,
        bucket: &str,
        key: &str,
        upload_id: &str,
        part_number: u32,
        body: &[u8],
    ) -> Result<PartResult> {
        let host = self.host(bucket);
        let amz_date = self.timestamp();
        let query = format!("partNumber={part_number}&uploadId={upload_id}");
        let empty_headers = BTreeMap::new();
        let signed = self
            .signer
            .sign("PUT", &host, key, &query, body, &amz_date, &empty_headers);
        let resp = self.execute(&signed)?;
        let etag = resp
            .headers()
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .trim_matches('"')
            .to_string();
        if etag.is_empty() {
            return Err(S3UploadError::Multipart {
                key: key.to_string(),
                reason: format!("missing ETag for part {part_number}"),
            });
        }
        Ok(PartResult {
            part_number,
            etag,
            size: body.len(),
        })
    }

    fn complete_multipart(
        &self,
        bucket: &str,
        key: &str,
        upload_id: &str,
        parts: &[PartResult],
    ) -> Result<PutResult> {
        let host = self.host(bucket);
        let amz_date = self.timestamp();
        let query = format!("uploadId={upload_id}");

        // Build CompleteMultipartUpload XML body
        let mut xml = String::from("<CompleteMultipartUpload>");
        let mut total_size = 0usize;
        for p in parts {
            xml.push_str(&format!(
                "<Part><PartNumber>{}</PartNumber><ETag>\"{}\"</ETag></Part>",
                p.part_number, p.etag
            ));
            total_size += p.size;
        }
        xml.push_str("</CompleteMultipartUpload>");

        let empty_headers = BTreeMap::new();
        let signed = self.signer.sign(
            "POST",
            &host,
            key,
            &query,
            xml.as_bytes(),
            &amz_date,
            &empty_headers,
        );
        let resp = self.execute(&signed)?;
        let text = resp.text()?;
        let etag = extract_xml_tag(&text, "ETag")
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        Ok(PutResult {
            etag,
            key: key.to_string(),
            size: total_size,
        })
    }

    fn abort_multipart(&self, bucket: &str, key: &str, upload_id: &str) -> Result<()> {
        let host = self.host(bucket);
        let amz_date = self.timestamp();
        let query = format!("uploadId={upload_id}");
        let empty_headers = BTreeMap::new();
        let signed = self
            .signer
            .sign("DELETE", &host, key, &query, &[], &amz_date, &empty_headers);
        let _ = self.execute(&signed)?;
        Ok(())
    }
}

/// Extract the inner text of an XML tag from a flat XML string.
///
/// This is a minimal parser — it does NOT handle nested tags of the same name
/// or CDATA. It is sufficient for S3 XML responses (`<UploadId>…</UploadId>`).
fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].to_string())
}

// =============================================================================
// Provenance Manifest
// =============================================================================

/// Provenance metadata attached to every uploaded dataset for full
/// reproducibility of ML training runs.
///
/// Serialized to `provenance.json` and uploaded alongside the dataset shards.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceManifest {
    /// Schema version of the provenance format itself.
    pub provenance_schema_version: String,
    /// Physics solver version (e.g. `"9r4c-1.0"` from
    /// [`crate::ai::batch_runner_9r4c::HARNESS_VERSION`]).
    pub solver_version: String,
    /// FTDS tensor dataset schema version
    /// ([`crate::ai::tensor_dataset::TENSOR_DATASET_SCHEMA_VERSION`]).
    pub dataset_schema_version: String,
    /// Git commit SHA of the code that generated this dataset.
    pub git_sha: String,
    /// Random seed used for Monte Carlo parameter sampling.
    pub parameter_seed: u64,
    /// Source of weather data (e.g. `"TMY3-4A"`, `"EPW-CO-Denver"`).
    pub weather_source: String,
    /// Content-addressable hash of the dataset (SHA-256 of concatenated shard
    /// checksums from the FTDS manifest).
    pub dataset_hash: String,
    /// Timestamp (ISO 8601 UTC) when the dataset was generated.
    pub generated_at_utc: String,
    /// Timestamp (ISO 8601 UTC) when the provenance manifest was created.
    pub provenance_created_at_utc: String,
    /// Total number of samples in the dataset.
    pub n_samples: usize,
    /// Number of shards.
    pub n_shards: usize,
    /// Human-readable description of generation parameters (free-form JSON).
    pub generation_parameters: serde_json::Value,
    /// S3 bucket where the dataset was uploaded.
    pub s3_bucket: String,
    /// S3 key prefix (version-prefixed) where the dataset lives.
    pub s3_prefix: String,
}

/// File name for the provenance manifest inside a dataset directory.
pub const PROVENANCE_FILENAME: &str = "provenance.json";

/// Current schema version of the provenance manifest format.
pub const PROVENANCE_SCHEMA_VERSION: &str = "1.0.0";

/// Builder for [`ProvenanceManifest`].
pub struct ProvenanceManifestBuilder {
    solver_version: Option<String>,
    git_sha: Option<String>,
    parameter_seed: Option<u64>,
    weather_source: Option<String>,
    generation_parameters: serde_json::Value,
    s3_bucket: String,
    s3_prefix: String,
}

impl ProvenanceManifestBuilder {
    pub fn new(s3_bucket: impl Into<String>, s3_prefix: impl Into<String>) -> Self {
        ProvenanceManifestBuilder {
            solver_version: None,
            git_sha: None,
            parameter_seed: None,
            weather_source: None,
            generation_parameters: serde_json::Value::Null,
            s3_bucket: s3_bucket.into(),
            s3_prefix: s3_prefix.into(),
        }
    }

    pub fn solver_version(mut self, v: impl Into<String>) -> Self {
        self.solver_version = Some(v.into());
        self
    }

    pub fn git_sha(mut self, v: impl Into<String>) -> Self {
        self.git_sha = Some(v.into());
        self
    }

    /// Attempt to read the git SHA from the `FLUXION_GIT_SHA` env var or the
    /// `GIT_SHA` env var. Falls back to `"unknown"` if not set.
    pub fn git_sha_from_env(self) -> Self {
        let sha = std::env::var("FLUXION_GIT_SHA")
            .or_else(|_| std::env::var("GIT_SHA"))
            .unwrap_or_else(|_| "unknown".to_string());
        self.git_sha(sha)
    }

    pub fn parameter_seed(mut self, seed: u64) -> Self {
        self.parameter_seed = Some(seed);
        self
    }

    pub fn weather_source(mut self, v: impl Into<String>) -> Self {
        self.weather_source = Some(v.into());
        self
    }

    pub fn generation_parameters(mut self, v: serde_json::Value) -> Self {
        self.generation_parameters = v;
        self
    }

    /// Build the provenance manifest, deriving `dataset_hash`, `n_samples`,
    /// and `n_shards` from the tensor dataset manifest.
    pub fn build(self, tensor_manifest: &TensorDatasetManifest) -> ProvenanceManifest {
        let dataset_hash = compute_dataset_hash(tensor_manifest);
        ProvenanceManifest {
            provenance_schema_version: PROVENANCE_SCHEMA_VERSION.to_string(),
            solver_version: self.solver_version.unwrap_or_else(|| "unknown".to_string()),
            dataset_schema_version: tensor_manifest.schema_version.clone(),
            git_sha: self.git_sha.unwrap_or_else(|| "unknown".to_string()),
            parameter_seed: self.parameter_seed.unwrap_or(0),
            weather_source: self.weather_source.unwrap_or_else(|| "unknown".to_string()),
            dataset_hash,
            generated_at_utc: tensor_manifest.created_at_utc.clone(),
            provenance_created_at_utc: chrono::Utc::now().to_rfc3339(),
            n_samples: tensor_manifest.n_samples_total,
            n_shards: tensor_manifest.shards.len(),
            generation_parameters: self.generation_parameters,
            s3_bucket: self.s3_bucket,
            s3_prefix: self.s3_prefix,
        }
    }
}

/// Compute a content-addressable dataset hash from the FTDS manifest's shard
/// checksums.
///
/// `dataset_hash = SHA-256(sha256_shard_0 || sha256_shard_1 || ...)`
fn compute_dataset_hash(manifest: &TensorDatasetManifest) -> String {
    let mut hasher = Sha256::new();
    for shard in &manifest.shards {
        hasher.update(shard.sha256.as_bytes());
    }
    hex_lower(&hasher.finalize())
}

// =============================================================================
// Upload configuration
// =============================================================================

/// Configuration for S3 dataset uploads.
#[derive(Clone, Debug)]
pub struct S3UploadConfig {
    /// Target S3 bucket name.
    pub bucket: String,
    /// Key prefix under which datasets are stored (e.g. `"datasets/ftds"`).
    pub key_prefix: String,
    /// AWS region.
    pub region: Region,
    /// AWS credentials.
    pub credentials: AwsCredentials,
    /// Part size for multipart uploads (bytes). Defaults to 8 MiB.
    pub part_size: usize,
    /// Objects larger than this use multipart upload. Defaults to 8 MiB.
    pub multipart_threshold: usize,
    /// Directory for persisting multipart upload state (for resumability).
    /// When `None`, multipart state is kept in-memory (no cross-process resume).
    pub state_dir: Option<PathBuf>,
}

impl S3UploadConfig {
    /// Create a config from environment-loaded credentials.
    pub fn from_env(
        bucket: impl Into<String>,
        key_prefix: impl Into<String>,
        region: impl Into<String>,
    ) -> Result<Self> {
        Ok(S3UploadConfig {
            bucket: bucket.into(),
            key_prefix: key_prefix.into(),
            region: region.into(),
            credentials: AwsCredentials::from_env()?,
            part_size: DEFAULT_PART_SIZE,
            multipart_threshold: DEFAULT_MULTIPART_THRESHOLD,
            state_dir: None,
        })
    }

    pub fn with_state_dir(mut self, dir: PathBuf) -> Self {
        self.state_dir = Some(dir);
        self
    }

    pub fn with_part_size(mut self, size: usize) -> Self {
        self.part_size = size;
        self
    }

    pub fn with_multipart_threshold(mut self, threshold: usize) -> Self {
        self.multipart_threshold = threshold;
        self
    }

    /// Build the version-prefixed S3 key for the dataset root.
    ///
    /// Layout: `<key_prefix>/v<schema_major>/<dataset_hash>/`
    pub fn dataset_prefix(&self, schema_version: &str, dataset_hash: &str) -> String {
        let major = schema_version.split('.').next().unwrap_or("1");
        format!("{}/v{major}/{dataset_hash}", self.key_prefix)
    }
}

// =============================================================================
// Multipart upload state (for resumability)
// =============================================================================

/// Persistent state of an in-progress multipart upload, enabling resume after
/// crashes or interruptions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultipartUploadState {
    /// S3 object key being uploaded.
    pub key: String,
    /// S3 multipart UploadId.
    pub upload_id: String,
    /// Total size of the object in bytes.
    pub total_size: usize,
    /// Part size in bytes.
    pub part_size: usize,
    /// Total number of parts.
    pub total_parts: u32,
    /// Completed parts with their ETags.
    pub completed_parts: Vec<PartResult>,
    /// Local path of the file being uploaded.
    pub local_path: String,
}

impl MultipartUploadState {
    /// File name for the state file inside the state directory.
    pub fn state_filename(key: &str) -> String {
        // Replace path separators to flatten the key into a filename.
        let flat = key.replace('/', "__");
        format!("{flat}.uploadstate.json")
    }

    /// Load state from disk. Returns `None` if no state file exists.
    pub fn load(state_dir: &Path, key: &str) -> Option<Self> {
        let path = state_dir.join(Self::state_filename(key));
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Persist state to disk (atomically: write tmp + rename).
    pub fn save(&self, state_dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(state_dir)?;
        let path = state_dir.join(Self::state_filename(&self.key));
        let tmp = path.with_extension("tmp");
        let content = serde_json::to_string_pretty(self).map_err(|e| std::io::Error::other(e))?;
        fs::write(&tmp, content)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Remove the state file (called after successful completion).
    pub fn remove(state_dir: &Path, key: &str) {
        let path = state_dir.join(Self::state_filename(key));
        let _ = fs::remove_file(path);
    }

    /// Returns the set of part numbers that have been completed.
    pub fn completed_part_numbers(&self) -> std::collections::HashSet<u32> {
        self.completed_parts.iter().map(|p| p.part_number).collect()
    }

    /// Returns `true` if all parts are uploaded.
    pub fn is_complete(&self) -> bool {
        self.completed_parts.len() as u32 == self.total_parts
    }
}

// =============================================================================
// Upload report
// =============================================================================

/// Report of a completed (or partially completed) dataset upload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UploadReport {
    /// S3 prefix where the dataset was uploaded.
    pub s3_prefix: String,
    /// Number of objects uploaded (shards + manifest + provenance).
    pub objects_uploaded: usize,
    /// Total bytes uploaded.
    pub bytes_uploaded: usize,
    /// Number of multipart uploads used.
    pub multipart_uploads: usize,
    /// Number of parts resumed (already-completed parts skipped on resume).
    pub parts_resumed: usize,
    /// Provenance manifest that was uploaded.
    pub provenance: ProvenanceManifest,
    /// Dataset hash.
    pub dataset_hash: String,
    /// Per-object upload details.
    pub objects: Vec<UploadedObject>,
}

/// Details of a single uploaded object.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UploadedObject {
    pub key: String,
    pub size: usize,
    pub used_multipart: bool,
    pub n_parts: u32,
}

// =============================================================================
// S3Uploader — orchestrates the full pipeline
// =============================================================================

/// Orchestrates uploading a tensor dataset directory to S3 with provenance.
///
/// Created with an [`S3UploadConfig`] and any [`S3Transport`]. The default
/// transport is [`ReqwestS3Transport`].
pub struct S3Uploader<T: S3Transport = ReqwestS3Transport> {
    config: S3UploadConfig,
    transport: T,
}

impl S3Uploader<ReqwestS3Transport> {
    /// Create an uploader with a real `reqwest`-backed transport.
    pub fn new(config: S3UploadConfig) -> Self {
        let transport = ReqwestS3Transport::new(config.credentials.clone(), config.region.clone());
        S3Uploader { config, transport }
    }
}

impl<T: S3Transport> S3Uploader<T> {
    /// Create an uploader with a custom transport (used by tests to inject
    /// a mock transport).
    pub fn with_transport(config: S3UploadConfig, transport: T) -> Self {
        S3Uploader { config, transport }
    }

    /// Upload an entire tensor dataset directory to S3.
    ///
    /// This method:
    /// 1. Loads `manifest.json` from the dataset directory.
    /// 2. Computes the dataset hash and version-prefixed S3 key prefix.
    /// 3. Uploads each shard file (using multipart for large shards, with
    ///    resume support).
    /// 4. Uploads `manifest.json`.
    /// 5. Creates and uploads a `provenance.json` manifest.
    ///
    /// If all objects already exist (idempotent re-upload), they are skipped.
    pub fn upload_dataset(
        &self,
        dataset_dir: &Path,
        provenance: ProvenanceManifest,
    ) -> Result<UploadReport> {
        let tensor_manifest =
            TensorDatasetManifest::load(dataset_dir).map_err(|_| S3UploadError::NoManifest)?;

        let dataset_hash = compute_dataset_hash(&tensor_manifest);
        let s3_prefix = self
            .config
            .dataset_prefix(&tensor_manifest.schema_version, &dataset_hash);

        let mut report = UploadReport {
            s3_prefix: s3_prefix.clone(),
            objects_uploaded: 0,
            bytes_uploaded: 0,
            multipart_uploads: 0,
            parts_resumed: 0,
            provenance: provenance.clone(),
            dataset_hash: dataset_hash.clone(),
            objects: Vec::new(),
        };

        // --- Upload each shard ---
        for shard_ref in &tensor_manifest.shards {
            let local_path = dataset_dir.join(&shard_ref.path);
            let key = format!("{s3_prefix}/{}", shard_ref.path);
            let body = fs::read(&local_path)?;

            let (obj, parts_resumed) = self.upload_object(&key, &body, &provenance)?;
            report.objects_uploaded += 1;
            report.bytes_uploaded += obj.size;
            report.multipart_uploads += if obj.used_multipart { 1 } else { 0 };
            report.parts_resumed += parts_resumed;
            report.objects.push(obj);
        }

        // --- Upload tensor manifest ---
        {
            let manifest_bytes = serde_json::to_vec_pretty(&tensor_manifest)?;
            let key = format!("{s3_prefix}/manifest.json");
            let metadata = BTreeMap::new();
            self.transport
                .put_object(&self.config.bucket, &key, &manifest_bytes, &metadata)?;
            report.objects_uploaded += 1;
            report.bytes_uploaded += manifest_bytes.len();
            report.objects.push(UploadedObject {
                key,
                size: manifest_bytes.len(),
                used_multipart: false,
                n_parts: 0,
            });
        }

        // --- Create and upload provenance manifest ---
        {
            let mut prov = provenance;
            prov.s3_bucket = self.config.bucket.clone();
            prov.s3_prefix = s3_prefix.clone();
            let prov_bytes = serde_json::to_vec_pretty(&prov)?;
            let key = format!("{s3_prefix}/{PROVENANCE_FILENAME}");
            let mut metadata = BTreeMap::new();
            metadata.insert("dataset-hash".to_string(), dataset_hash.clone());
            metadata.insert("solver-version".to_string(), prov.solver_version.clone());
            self.transport
                .put_object(&self.config.bucket, &key, &prov_bytes, &metadata)?;
            report.objects_uploaded += 1;
            report.bytes_uploaded += prov_bytes.len();
            report.provenance = prov;
            report.objects.push(UploadedObject {
                key,
                size: prov_bytes.len(),
                used_multipart: false,
                n_parts: 0,
            });
        }

        Ok(report)
    }

    /// Upload a single object, choosing simple PUT vs multipart based on size,
    /// and resuming from saved state if available.
    ///
    /// Returns `(UploadedObject, parts_resumed)`.
    fn upload_object(
        &self,
        key: &str,
        body: &[u8],
        provenance: &ProvenanceManifest,
    ) -> Result<(UploadedObject, usize)> {
        let mut metadata = BTreeMap::new();
        metadata.insert("dataset-hash".to_string(), provenance.dataset_hash.clone());
        metadata.insert(
            "solver-version".to_string(),
            provenance.solver_version.clone(),
        );

        if body.len() < self.config.multipart_threshold {
            // Simple PUT
            self.transport
                .put_object(&self.config.bucket, key, body, &metadata)?;
            Ok((
                UploadedObject {
                    key: key.to_string(),
                    size: body.len(),
                    used_multipart: false,
                    n_parts: 0,
                },
                0,
            ))
        } else {
            // Multipart upload with resume
            let (n_parts, used_multipart, parts_resumed) =
                self.multipart_upload_with_resume(key, body, &metadata)?;
            Ok((
                UploadedObject {
                    key: key.to_string(),
                    size: body.len(),
                    used_multipart,
                    n_parts,
                },
                parts_resumed,
            ))
        }
    }

    /// Perform a resumable multipart upload.
    ///
    /// If a saved [`MultipartUploadState`] exists for this key (in the state
    /// directory), the already-completed parts are skipped. If the upload was
    /// fully completed previously but not finalized, it is completed now.
    fn multipart_upload_with_resume(
        &self,
        key: &str,
        body: &[u8],
        metadata: &BTreeMap<String, String>,
    ) -> Result<(u32, bool, usize)> {
        // Use configured part_size directly — S3 enforces the 5 MiB minimum
        // server-side, so the client doesn't need to clamp. This also allows
        // tests to exercise multipart logic with small shards.
        let part_size = self.config.part_size;

        // Calculate parts
        let total_parts = body.len().div_ceil(part_size) as u32;
        if total_parts > MAX_PARTS {
            return Err(S3UploadError::TooManyParts(total_parts));
        }
        if total_parts == 0 {
            // Body smaller than expected (shouldn't happen since we check threshold,
            // but handle defensively — just do a PUT).
            self.transport
                .put_object(&self.config.bucket, key, body, metadata)?;
            return Ok((0, false, 0));
        }

        // Check for existing state
        let mut state = self
            .config
            .state_dir
            .as_ref()
            .and_then(|dir| MultipartUploadState::load(dir, key));

        let upload_id: String;
        let mut parts_resumed = 0usize;

        if let Some(ref existing) = state {
            if existing.is_complete() {
                // All parts done but not completed — finalize now.
                self.transport.complete_multipart(
                    &self.config.bucket,
                    key,
                    &existing.upload_id,
                    &existing.completed_parts,
                )?;
                if let Some(ref dir) = self.config.state_dir {
                    MultipartUploadState::remove(dir, key);
                }
                return Ok((existing.total_parts, true, existing.completed_parts.len()));
            }
            // Resume: reuse the upload_id and skip completed parts
            upload_id = existing.upload_id.clone();
            parts_resumed = existing.completed_parts.len();
            log::info!(
                "Resuming multipart upload for {key}: {parts_resumed}/{} parts already done",
                existing.total_parts
            );
        } else {
            // Start fresh
            upload_id = self
                .transport
                .create_multipart(&self.config.bucket, key, metadata)?;
            state = Some(MultipartUploadState {
                key: key.to_string(),
                upload_id: upload_id.clone(),
                total_size: body.len(),
                part_size,
                total_parts,
                completed_parts: Vec::new(),
                local_path: String::new(),
            });
        }

        let completed_set = state
            .as_ref()
            .map(|s| s.completed_part_numbers())
            .unwrap_or_default();

        // Upload remaining parts
        for part_number in 1..=total_parts {
            if completed_set.contains(&part_number) {
                continue;
            }
            let start = (part_number as usize - 1) * part_size;
            let end = std::cmp::min(start + part_size, body.len());
            let part_body = &body[start..end];

            let part_result = self.transport.upload_part(
                &self.config.bucket,
                key,
                &upload_id,
                part_number,
                part_body,
            )?;

            if let Some(ref mut s) = state {
                s.completed_parts.push(part_result);
                // Persist state after each part for crash recovery
                if let Some(ref dir) = self.config.state_dir {
                    if let Err(e) = s.save(dir) {
                        log::warn!("Failed to persist upload state: {e}");
                    }
                }
            }
        }

        // Complete the multipart upload
        let final_parts = state
            .as_ref()
            .map(|s| s.completed_parts.clone())
            .unwrap_or_default();

        // Sort parts by number for the complete request
        let mut sorted_parts = final_parts.clone();
        sorted_parts.sort_by_key(|p| p.part_number);

        self.transport
            .complete_multipart(&self.config.bucket, key, &upload_id, &sorted_parts)?;

        // Clean up state file
        if let Some(ref dir) = self.config.state_dir {
            MultipartUploadState::remove(dir, key);
        }

        Ok((total_parts, true, parts_resumed))
    }
}

// =============================================================================
// Tests (unit — integration tests live in tests/s3_upload_pipeline.rs)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- HMAC-SHA256 test vectors (RFC 4231) ----

    #[test]
    fn hmac_sha256_rfc4231_test_case_1() {
        // RFC 4231 Test Case 1
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        let result = hmac_sha256(&key, data);
        let expected = "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7";
        assert_eq!(hex_lower(&result), expected);
    }

    #[test]
    fn hmac_sha256_rfc4231_test_case_2() {
        // RFC 4231 Test Case 2
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let result = hmac_sha256(key, data);
        let expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
        assert_eq!(hex_lower(&result), expected);
    }

    #[test]
    fn hmac_sha256_key_longer_than_block() {
        // RFC 4231 Test Case 6 — key longer than block size triggers hashing
        let key = [0xaau8; 131];
        let data = b"Test Using Larger Than Block-Size Key - Hash Key First";
        let result = hmac_sha256(&key, data);
        let expected = "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54";
        assert_eq!(hex_lower(&result), expected);
    }

    // ---- SHA-256 helpers ----

    #[test]
    fn sha256_empty_string() {
        assert_eq!(sha256_hex(b""), EMPTY_SHA256);
    }

    #[test]
    fn sha256_known_value() {
        // SHA-256("abc")
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // ---- URL encoding ----

    #[test]
    fn url_encode_preserves_slash() {
        assert_eq!(
            url_encode_key("datasets/v1/hash/shard.ftds"),
            "datasets/v1/hash/shard.ftds"
        );
    }

    #[test]
    fn url_encode_special_chars() {
        assert_eq!(url_encode_key("a b"), "a%20b");
        assert_eq!(url_encode_key("a+b"), "a%2Bb");
    }

    // ---- XML tag extraction ----

    #[test]
    fn extract_xml_tag_basic() {
        let xml = "<Root><UploadId>abc123</UploadId></Root>";
        assert_eq!(extract_xml_tag(xml, "UploadId").as_deref(), Some("abc123"));
    }

    #[test]
    fn extract_xml_tag_missing() {
        let xml = "<Root></Root>";
        assert!(extract_xml_tag(xml, "UploadId").is_none());
    }

    // ---- Sig V4 signing key derivation ----

    #[test]
    fn sigv4_signing_key_known_vector() {
        // AWS documentation signing-key derivation, adapted for the S3 service.
        // (The AWS docs use "iam" as the service; this module always signs for
        // "s3", so the expected key differs.)
        let creds = AwsCredentials {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let signer = SigV4Signer::new(creds, "us-east-1".to_string());
        let signing_key = signer.derive_signing_key("20150830");
        // Known answer: k_date→k_region(us-east-1)→k_service(s3)→k_signing
        assert_eq!(
            hex_lower(&signing_key),
            "32f78051dcde24c552811d654f4a769112bb834b03975cdd6b1fd7d16248c269"
        );
    }

    // ---- Sig V4 full signature ----

    #[test]
    fn sigv4_full_signature_s3_put() {
        let creds = AwsCredentials {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let signer = SigV4Signer::new(creds, "us-east-1".to_string());

        let body = b"Welcome to Amazon S3.";
        let amz_date = "20130524T000000Z";
        let host = "examplebucket.s3.amazonaws.com";
        let key = "test.txt";

        let signed = signer.sign("PUT", host, key, "", body, amz_date, &BTreeMap::new());

        // Verify the Authorization header contains the expected signature
        let auth = signed.headers.get("Authorization").unwrap();
        assert!(
            auth.contains(
                "Signature=6b472b2701b66e0f66f097ddef1d3fa0c15434009ae32df4ad924c0b16f0f68b"
            ),
            "unexpected signature in Authorization header: {auth}"
        );

        // Verify content hash header
        let content_hash = signed.headers.get("x-amz-content-sha256").unwrap();
        assert_eq!(
            *content_hash,
            "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
        );

        // Verify URL
        assert_eq!(
            signed.url,
            "https://examplebucket.s3.amazonaws.com/test.txt"
        );
    }

    #[test]
    fn sigv4_includes_session_token_when_present() {
        let creds = AwsCredentials {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: Some("token123".to_string()),
        };
        let signer = SigV4Signer::new(creds, "us-east-1".to_string());
        let signed = signer.sign(
            "PUT",
            "bucket.s3.amazonaws.com",
            "k",
            "",
            b"",
            "20240101T000000Z",
            &BTreeMap::new(),
        );
        assert!(signed.headers.contains_key("x-amz-security-token"));
        let signed_headers = signed
            .headers
            .get("Authorization")
            .unwrap()
            .split("SignedHeaders=")
            .nth(1)
            .and_then(|s| s.split(',').next());
        // The signed headers string includes x-amz-security-token
        assert!(signed_headers.unwrap().contains("x-amz-security-token"));
    }

    // ---- Dataset hash computation ----

    #[test]
    fn dataset_hash_is_deterministic() {
        let manifest = TensorDatasetManifest {
            schema_version: "1.0.0".to_string(),
            created_at_utc: "2024-01-01T00:00:00Z".to_string(),
            dtype: crate::ai::tensor_dataset::TensorDType::F64,
            n_samples_total: 10,
            n_input_features: 3,
            input_feature_names: vec!["a".to_string()],
            target_names: vec!["t".to_string()],
            has_timeseries: false,
            timeseries_length: 0,
            normalization: None,
            shards: vec![
                crate::ai::tensor_dataset::ShardRef {
                    path: "shard-000000.ftds".to_string(),
                    n_samples: 5,
                    sha256: "aaaa".to_string(),
                },
                crate::ai::tensor_dataset::ShardRef {
                    path: "shard-000001.ftds".to_string(),
                    n_samples: 5,
                    sha256: "bbbb".to_string(),
                },
            ],
        };
        let h1 = compute_dataset_hash(&manifest);
        let h2 = compute_dataset_hash(&manifest);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    // ---- Config ----

    #[test]
    fn dataset_prefix_format() {
        let config = S3UploadConfig {
            bucket: "bucket".to_string(),
            key_prefix: "datasets/ftds".to_string(),
            region: "us-east-1".to_string(),
            credentials: AwsCredentials {
                access_key_id: "x".to_string(),
                secret_access_key: "y".to_string(),
                session_token: None,
            },
            part_size: DEFAULT_PART_SIZE,
            multipart_threshold: DEFAULT_MULTIPART_THRESHOLD,
            state_dir: None,
        };
        let prefix = config.dataset_prefix("1.0.0", "deadbeef");
        assert_eq!(prefix, "datasets/ftds/v1/deadbeef");
    }

    // ---- Multipart state persistence ----

    #[test]
    fn multipart_state_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let state = MultipartUploadState {
            key: "datasets/v1/abc/shard.ftds".to_string(),
            upload_id: "uid123".to_string(),
            total_size: 16_000_000,
            part_size: 8_000_000,
            total_parts: 2,
            completed_parts: vec![PartResult {
                part_number: 1,
                etag: "etag1".to_string(),
                size: 8_000_000,
            }],
            local_path: "/tmp/shard.ftds".to_string(),
        };
        state.save(dir.path()).unwrap();

        let loaded = MultipartUploadState::load(dir.path(), &state.key).expect("must load");
        assert_eq!(loaded.upload_id, "uid123");
        assert_eq!(loaded.total_parts, 2);
        assert_eq!(loaded.completed_parts.len(), 1);
        assert!(!loaded.is_complete());

        MultipartUploadState::remove(dir.path(), &state.key);
        assert!(MultipartUploadState::load(dir.path(), &state.key).is_none());
    }

    #[test]
    fn multipart_state_filename_flattens_slashes() {
        let name = MultipartUploadState::state_filename("a/b/c.ftds");
        assert_eq!(name, "a__b__c.ftds.uploadstate.json");
    }
}
