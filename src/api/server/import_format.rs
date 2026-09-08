// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `POST /v1/import/{osm|gbxml|idf|epjson}` endpoint and the
//! [`tempfile_for_bytes`] helper. Decomposed from the legacy `server.rs`
//! so the import path is one focused submodule (Issue #3457 / #3543 —
//! module-size ratchet).

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;
use tempfile::NamedTempFile;

use crate::api::schema::SimulationSchemaV1;
use crate::api::server::api_error::ApiError;
use crate::api::server::state::AppState;
use crate::interop::{gbxml, osm};
use crate::io::idf::{IdfFile, IdfParser};

/// Wire-level representation of an import failure. Returned with HTTP 4xx so
/// clients can present the same error string the CLI prints.
#[derive(Debug, Clone, Serialize)]
pub struct ImportResponse {
    pub schema_id: String,
    pub schema: SimulationSchemaV1,
}

/// Import a file from one of the supported external formats. The body is the
/// raw file bytes; the path parameter selects the decoder.
pub async fn import_format(
    State(state): State<AppState>,
    Path(fmt): Path<String>,
    body: axum::body::Bytes,
) -> Result<Json<ImportResponse>, ApiError> {
    let fmt = fmt.to_ascii_lowercase();
    let schema = match fmt.as_str() {
        "osm" => {
            let tmp = tempfile_for_bytes(&body, "osm")?;
            osm::import_osm(tmp.path()).map_err(|e| ApiError::ImportFailed(e.to_string()))?
        }
        "gbxml" => {
            let tmp = tempfile_for_bytes(&body, "gbxml")?;
            gbxml::import_gbxml(tmp.path()).map_err(|e| ApiError::ImportFailed(e.to_string()))?
        }
        "idf" => {
            let body_str = std::str::from_utf8(&body)
                .map_err(|e| ApiError::ImportFailed(format!("invalid UTF-8 in IDF body: {e}")))?;
            let idf: IdfFile = IdfParser::from_str(body_str)
                .map_err(|e| ApiError::ImportFailed(format!("IDF parse error: {e}")))?;
            let schema = SimulationSchemaV1::try_from(&idf)
                .map_err(|e| ApiError::ImportFailed(format!("IDF conversion error: {e}")))?;
            schema
        }
        "epjson" => {
            let body_str = std::str::from_utf8(&body).map_err(|e| {
                ApiError::ImportFailed(format!("invalid UTF-8 in epJSON body: {e}"))
            })?;
            let idf: IdfFile = IdfParser::from_epjson_str(body_str)
                .map_err(|e| ApiError::ImportFailed(format!("epJSON parse error: {e}")))?;
            let schema = SimulationSchemaV1::try_from(&idf)
                .map_err(|e| ApiError::ImportFailed(format!("IDF conversion error: {e}")))?;
            schema
        }
        other => return Err(ApiError::UnsupportedFormat(other.to_string())),
    };

    let id = state.store(schema.clone()).await;
    Ok(Json(ImportResponse {
        schema_id: id,
        schema,
    }))
}

/// Persist `bytes` to a uniquely-named, owner-only temp file and return
/// the [`NamedTempFile`] handle. The file is removed when the handle is
/// dropped.
///
/// Security (Issue #2556): the previous implementation built the path as
/// `fluxion-import-{nanos}.{ext}` under `std::env::temp_dir()` and opened
/// it with `std::fs::File::create` — a predictable name plus `O_CREAT`
/// without `O_EXCL`. On a multi-tenant host, an unprivileged co-tenant
/// that could predict (or race) the same nanosecond could pre-create the
/// path as a symlink to e.g. `/etc/passwd`.
pub fn tempfile_for_bytes(bytes: &[u8], ext: &str) -> Result<NamedTempFile, ApiError> {
    use std::io::Write;

    let suffix = format!(".{ext}");
    let mut tmp = {
        let mut builder = tempfile::Builder::new();
        builder
            .prefix("fluxion-import-")
            .suffix(&suffix)
            .rand_bytes(16);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            builder.permissions(std::fs::Permissions::from_mode(0o600));
        }
        builder
            .tempfile()
            .map_err(|e| ApiError::ImportFailed(format!("temp file create: {e}")))?
    };

    tmp.as_file_mut()
        .write_all(bytes)
        .map_err(|e| ApiError::ImportFailed(format!("temp file write: {e}")))?;
    tmp.as_file_mut()
        .sync_all()
        .map_err(|e| ApiError::ImportFailed(format!("temp file sync: {e}")))?;

    let meta = std::fs::symlink_metadata(tmp.path())
        .map_err(|e| ApiError::ImportFailed(format!("temp file stat: {e}")))?;
    let ft = meta.file_type();
    if !ft.is_file() || ft.is_symlink() {
        return Err(ApiError::ImportFailed(format!(
            "temp file is not a regular file (file_type={ft:?})"
        )));
    }

    Ok(tmp)
}
