// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FFI safety regression tests for the FMI 2.0 extern "C" shims.
//!
//! Regression test for GitHub issue #2555: `ffd_fmu2SetReal` and
//! `ffd_fmu2GetReal` previously reconstructed slices from raw pointers
//! (`std::slice::from_raw_parts(vr, nvr)` and the `_mut` variant) with no
//! length cap and no alignment check. A malformed FMU master supplying a
//! maliciously large `nvr`, a misaligned `u32`/`f64` pointer, or a null
//! component could trigger undefined behaviour — out-of-bounds reads, writes
//! past the end of the heap, or platform-specific misaligned-access faults.
//!
//! These tests call the public `unsafe extern "C"` surface directly with
//! adversarial inputs and assert each shim returns
//! `fmi2Status::fmi2Error` rather than panicking, segfaulting, or reading
//! uninitialised memory.

use fluxion::interop::fmi::{
    ffd_fmu2FreeInstance, ffd_fmu2GetReal, ffd_fmu2Instantiate, ffd_fmu2SetReal, fmi2Status,
    Fmi2Component, FMI2_MAX_VALUE_REFERENCES,
};

/// Build a fresh FMU instance via the public C entry point.
fn instantiate() -> Fmi2Component {
    unsafe {
        ffd_fmu2Instantiate(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            0,
            None,
            std::ptr::null_mut(),
        )
    }
}

// -----------------------------------------------------------------------------
// SetReal — null/bounds rejection
// -----------------------------------------------------------------------------

#[test]
fn set_real_rejects_null_component() {
    let status = unsafe {
        ffd_fmu2SetReal(
            std::ptr::null_mut(),
            std::ptr::null::<u32>(),
            0,
            std::ptr::null::<f64>(),
        )
    };
    assert_eq!(status, fmi2Status::fmi2Error);
}

#[test]
fn set_real_rejects_null_vr() {
    let c = instantiate();
    let value: *const f64 = std::ptr::null();
    let status = unsafe { ffd_fmu2SetReal(c, std::ptr::null(), 1, value) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_rejects_null_value() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let status = unsafe { ffd_fmu2SetReal(c, vr_storage.as_ptr(), 1, std::ptr::null()) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_rejects_nvr_above_cap() {
    let c = instantiate();
    let oversized = FMI2_MAX_VALUE_REFERENCES + 1;
    let vr_storage = vec![0u32; oversized];
    let value_storage = vec![0.0f64; oversized];
    let status =
        unsafe { ffd_fmu2SetReal(c, vr_storage.as_ptr(), oversized, value_storage.as_ptr()) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_rejects_huge_nvr() {
    // 1 GiB worth of f64 would be the obvious OOB attack; cap must trigger
    // long before that without ever touching the heap.
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let value_storage = [0.0f64; 1];
    let status = unsafe {
        ffd_fmu2SetReal(
            c,
            vr_storage.as_ptr(),
            usize::MAX / 8,
            value_storage.as_ptr(),
        )
    };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_rejects_misaligned_vr() {
    let c = instantiate();
    // 32 bytes of storage; offset by 1 byte to guarantee u32 misalignment.
    let storage = vec![0u8; 32];
    let misaligned_vr = unsafe { storage.as_ptr().add(1) }.cast::<u32>();
    let value_storage = [0.0f64; 1];
    let status = unsafe { ffd_fmu2SetReal(c, misaligned_vr, 1, value_storage.as_ptr()) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_rejects_misaligned_value() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let storage = vec![0u8; 16];
    let misaligned_value = unsafe { storage.as_ptr().add(1) }.cast::<f64>();
    let status = unsafe { ffd_fmu2SetReal(c, vr_storage.as_ptr(), 1, misaligned_value) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_accepts_zero_nvr() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let value_storage = [0.0f64; 1];
    // Documented no-op when nvr == 0; the slice is empty so alignment is
    // irrelevant, but the contract still requires non-null pointers.
    let status = unsafe { ffd_fmu2SetReal(c, vr_storage.as_ptr(), 0, value_storage.as_ptr()) };
    assert_eq!(status, fmi2Status::fmi2OK);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn set_real_accepts_valid_input() {
    let c = instantiate();
    // vr=1 maps to `inlet_air_temperature`, an FFD-defined input. 293.15 K
    // is a physically plausible value that flows through `set_real` OK.
    let vr_storage = [1u32; 1];
    let value_storage = [293.15f64; 1];
    let status = unsafe { ffd_fmu2SetReal(c, vr_storage.as_ptr(), 1, value_storage.as_ptr()) };
    assert_eq!(status, fmi2Status::fmi2OK);
    unsafe { ffd_fmu2FreeInstance(c) };
}

// -----------------------------------------------------------------------------
// GetReal — null/bounds rejection (mirrors SetReal)
// -----------------------------------------------------------------------------

#[test]
fn get_real_rejects_null_component() {
    let status = unsafe {
        ffd_fmu2GetReal(
            std::ptr::null_mut(),
            std::ptr::null::<u32>(),
            0,
            std::ptr::null_mut::<f64>(),
        )
    };
    assert_eq!(status, fmi2Status::fmi2Error);
}

#[test]
fn get_real_rejects_null_vr() {
    let c = instantiate();
    let value: *mut f64 = std::ptr::null_mut();
    let status = unsafe { ffd_fmu2GetReal(c, std::ptr::null(), 1, value) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_rejects_null_value() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let status = unsafe { ffd_fmu2GetReal(c, vr_storage.as_ptr(), 1, std::ptr::null_mut()) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_rejects_nvr_above_cap() {
    let c = instantiate();
    let oversized = FMI2_MAX_VALUE_REFERENCES + 1;
    let vr_storage = vec![0u32; oversized];
    let mut value_storage = vec![0.0f64; oversized];
    let status = unsafe {
        ffd_fmu2GetReal(
            c,
            vr_storage.as_ptr(),
            oversized,
            value_storage.as_mut_ptr(),
        )
    };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_rejects_huge_nvr() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let mut value_storage = [0.0f64; 1];
    let status = unsafe {
        ffd_fmu2GetReal(
            c,
            vr_storage.as_ptr(),
            usize::MAX / 8,
            value_storage.as_mut_ptr(),
        )
    };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_rejects_misaligned_vr() {
    let c = instantiate();
    let storage = vec![0u8; 32];
    let misaligned_vr = unsafe { storage.as_ptr().add(1) }.cast::<u32>();
    let mut value_storage = [0.0f64; 1];
    let status = unsafe { ffd_fmu2GetReal(c, misaligned_vr, 1, value_storage.as_mut_ptr()) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_rejects_misaligned_value() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let storage = vec![0u8; 16];
    let misaligned_value = unsafe { storage.as_ptr().add(1) }.cast::<f64>();
    let status = unsafe { ffd_fmu2GetReal(c, vr_storage.as_ptr(), 1, misaligned_value.cast_mut()) };
    assert_eq!(status, fmi2Status::fmi2Error);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_accepts_zero_nvr() {
    let c = instantiate();
    let vr_storage = [0u32; 1];
    let mut value_storage = [0.0f64; 1];
    let status = unsafe { ffd_fmu2GetReal(c, vr_storage.as_ptr(), 0, value_storage.as_mut_ptr()) };
    assert_eq!(status, fmi2Status::fmi2OK);
    unsafe { ffd_fmu2FreeInstance(c) };
}

#[test]
fn get_real_accepts_valid_output() {
    let c = instantiate();
    // Output vr > num_inputs (9); vr=10 maps to zone_air_temperatures[0].
    let vr_storage = [10u32; 1];
    let mut value_storage = [0.0f64; 1];
    let status = unsafe { ffd_fmu2GetReal(c, vr_storage.as_ptr(), 1, value_storage.as_mut_ptr()) };
    assert_eq!(status, fmi2Status::fmi2OK);
    unsafe { ffd_fmu2FreeInstance(c) };
}

// -----------------------------------------------------------------------------
// FreeInstance — defensive null handling
// -----------------------------------------------------------------------------

#[test]
fn free_instance_tolerates_null() {
    // Box::from_raw(null) would be UB; the existing guard makes it safe.
    unsafe { ffd_fmu2FreeInstance(std::ptr::null_mut()) };
}
