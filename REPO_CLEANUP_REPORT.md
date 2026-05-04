# Repository Cleanup Report - fluxion

Generated: 2026-05-04T10:01:24.555115

## Summary

- Duplicate file groups: 524
- Cache directories: 0
- Temp files: 0
- Empty files: 599
- Empty directories: 191
- Suspicious directories: 0
- Missing .gitignore entries: 3
- **Total: 1314**

## Duplicate Files

### Hash: 9eaed0581bf5...
- `tests/validation/benchmark_report.rs`
- `.sdd/worktrees/backend-fcb41780/tests/benchmark_report_validation.rs`
- `.sdd/worktrees/backend-fcb41780/tests/validation/benchmark_report.rs`

### Hash: 58e6e986c88b...
- `.automaker/features/issue-573-cdd82820-3d28-4c8f-85aa-d67cb8865b52/feature.json.bak1`
- `.automaker/features/issue-573-cdd82820-3d28-4c8f-85aa-d67cb8865b52/feature.json.bak3`
- `.automaker/features/issue-573-cdd82820-3d28-4c8f-85aa-d67cb8865b52/feature.json.bak2`
- `.sdd/worktrees/backend-fcb41780/.automaker/features/issue-573-cdd82820-3d28-4c8f-85aa-d67cb8865b52/feature.json`
- `.sdd/worktrees/backend-fcb41780/.automaker/features/issue-573-cdd82820-3d28-4c8f-85aa-d67cb8865b52/feature.json.bak1`
- ... and 2 more

### Hash: ec47ce806e6c...
- `target/release/build/rayon-core-5e16e2929457f8ef/invoked.timestamp`
- `target/release/build/openssl-sys-323919f6527b2acd/invoked.timestamp`
- `target/release/build/rustix-289c104b298133bf/invoked.timestamp`
- `target/release/build/pulp-b85979b329c49e41/invoked.timestamp`
- `target/release/build/zerocopy-c4eff2cd97fe1042/invoked.timestamp`
- ... and 2875 more

### Hash: 43de5140285c...
- `target/release/build/getrandom-b9b43c0bbdd4c1c1/output`
- `target/release/build/nano-gemm-c64-448b82dcfc219265/output`
- `target/release/build/getrandom-7aceb8a75860db6c/output`
- `target/release/build/nano-gemm-f64-5c310715fdf5cd10/output`
- `target/release/build/nano-gemm-c32-b24d9e342abcdfc1/output`
- ... and 33 more

### Hash: b26d9caad53f...
- `target/release/build/libc-eb82498fbcf0dbb7/output`
- `target/debug/build/libc-acf125aee15ac43b/output`
- `target/debug/build/libc-95779b78f257bfb2/output`
- `.sdd/worktrees/backend-fcb41780/target/debug/build/libc-acf125aee15ac43b/output`
- `examples/target/debug/build/libc-c2677e94c632b46d/output`

### Hash: 65c899bcfdb8...
- `target/release/build/zerocopy-6f7563d5525b3cbd/output`
- `target/debug/build/zerocopy-3985508579920a52/output`
- `target/debug/build/zerocopy-7eb3f355baaea47d/output`
- `target/debug/build/zerocopy-9b04a6927dd0aca4/output`
- `.sdd/worktrees/backend-fcb41780/target/debug/build/zerocopy-9b04a6927dd0aca4/output`

### Hash: f481c94a65ec...
- `target/release/build/icu_properties_data-5fc5e16d28f4c73c/output`
- `target/debug/build/icu_normalizer_data-d67b7f2521675ee4/output`
- `target/debug/build/icu_properties_data-74139cca2d2fa957/output`
- `target/debug/build/icu_normalizer_data-cfb14313a121aabe/output`
- `target/debug/build/icu_properties_data-100c125cc291380b/output`
- ... and 4 more

### Hash: 3ed0329cffa2...
- `target/release/deps/test_session_pool-e17a9085a7f59486.hyper-a2e1a4c17b856adf.hyper.e8ee0f149934e106-cgu.3.rcgu.o.rcgu.o`
- `target/release/deps/integration_batch_oracle-56e625e4f58aff18.hyper-a2e1a4c17b856adf.hyper.e8ee0f149934e106-cgu.3.rcgu.o.rcgu.o`
- `target/release/deps/generate_delta_config-3ecc9da54c50468b.hyper-a2e1a4c17b856adf.hyper.e8ee0f149934e106-cgu.3.rcgu.o.rcgu.o`
- `target/release/examples/test_6r2c-4a86021a2c371a5e.hyper-a2e1a4c17b856adf.hyper.e8ee0f149934e106-cgu.3.rcgu.o.rcgu.o`

### Hash: 7b727b7d6113...
- `target/release/deps/test_weather_interpolation-e7003094be8dee51.tokio-8d3c313ac505d39e.tokio.795ab06487d695d3-cgu.3.rcgu.o.rcgu.o`
- `target/release/deps/generate_delta_config-3ecc9da54c50468b.tokio-8d3c313ac505d39e.tokio.795ab06487d695d3-cgu.3.rcgu.o.rcgu.o`
- `target/release/deps/test_session_pool-e17a9085a7f59486.tokio-8d3c313ac505d39e.tokio.795ab06487d695d3-cgu.3.rcgu.o.rcgu.o`
- `target/release/deps/test_thermal_mass_integration-fce779276a3af1f3.tokio-8d3c313ac505d39e.tokio.795ab06487d695d3-cgu.3.rcgu.o.rcgu.o`
- `target/release/deps/test_demand_response_comprehensive-9eb970cea569183a.tokio-8d3c313ac505d39e.tokio.795ab06487d695d3-cgu.3.rcgu.o.rcgu.o`
- ... and 1 more

### Hash: e98cc47c6889...
- `target/release/deps/integration_batch_oracle-56e625e4f58aff18.ring-0744ff5b27fe7f64.ring.e2c4188d9b46d11-cgu.2.rcgu.o.rcgu.o`
- `target/release/deps/generate_delta_config-3ecc9da54c50468b.ring-0744ff5b27fe7f64.ring.e2c4188d9b46d11-cgu.2.rcgu.o.rcgu.o`
- `target/release/deps/test_session_pool-e17a9085a7f59486.ring-0744ff5b27fe7f64.ring.e2c4188d9b46d11-cgu.2.rcgu.o.rcgu.o`
- `target/release/examples/test_6r2c-4a86021a2c371a5e.ring-0744ff5b27fe7f64.ring.e2c4188d9b46d11-cgu.2.rcgu.o.rcgu.o`

## Unnecessary Items

### Large Files
- `data/reference/ashrae140/series_195.csv`
- `target/release/fluxion`
- `target/release/run_ashrae_validation`
- `target/release/libfluxion.so`
- `target/release/run_multi_zone_validation`
- `target/release/export_csv`
- `target/release/libfluxion.rlib`
- `target/release/run_cross_validation`
- `target/release/deps/throughput_benchmark-5b54356c29c0a26f`
- `target/release/deps/test_result_aggregation-f5999c5c909908d9`
- `target/release/deps/performance_integration_test-866125aa4e607fe8`
- `target/release/deps/test_parallel_validation-639c26f13c7f2f84`
- `target/release/deps/solar_distribution_tests-fb70e6c789f60b33`
- `target/release/deps/test_modular_surrogates-787d91a07d716199`
- `target/release/deps/test_constants_integration-c5f3b690d843dd05`
- `target/release/deps/test_validator_core-cc951f948bc9dc7d`
- `target/release/deps/test_shared_batch_service-7eb37681fa3090d3`
- `target/release/deps/run_ashrae_validation-f1a38c182121a104`
- `target/release/deps/test_interzone_conductance-5de04070abc45a39`
- `target/release/deps/lib_batch_oracle-8a3550cd5f51c12b`
- `target/release/deps/test_thermal_mass_integration-fce779276a3af1f3`
- `target/release/deps/fluxion-35aab0a908f9475f`
- `target/release/deps/ctf_coefficient_validation-7f1a1e46f63fb711`
- `target/release/deps/libnalgebra-3dac5f5f6f77312a.rmeta`
- `target/release/deps/libfluxion.so`
- `target/release/deps/libprivate_gemm_x86-4acf1651e79fa723.rlib`
- `target/release/deps/export_csv-e403c3727a7ec84f`
- `target/release/deps/libsimba-dd2032c1013a3118.rlib`
- `target/release/deps/validator_config-63c4f04944d4e62f`
- `target/release/deps/integration_batch_oracle-56e625e4f58aff18`
- ... and 606 more

## Empty Files

- `validate_logs.txt`
- `validate_full_log.txt`
- `api/tests/__init__.py`
- `.jules/bolt.md`
- `tools/__init__.py`
- `tools/compliance_agent/tests/__init__.py`
- `target/release/.cargo-lock`
- `target/release/build/object-f512dba44aa1bcd6/stderr`
- `target/release/build/rayon-core-5e16e2929457f8ef/stderr`
- `target/release/build/openssl-sys-323919f6527b2acd/stderr`
- `target/release/build/rustix-289c104b298133bf/stderr`
- `target/release/build/pulp-b85979b329c49e41/stderr`
- `target/release/build/zerocopy-c4eff2cd97fe1042/stderr`
- `target/release/build/libc-6754ff4b367e1cfe/stderr`
- `target/release/build/rustls-51f64ac122d4c95f/stderr`
- `target/release/build/rustls-51f64ac122d4c95f/output`
- `target/release/build/yeslogic-fontconfig-sys-8f2ea2311909b79d/stderr`
- `target/release/build/getrandom-b9b43c0bbdd4c1c1/stderr`
- `target/release/build/syn-e34e92f5bdeb645c/stderr`
- `target/release/build/parking_lot_core-f6d2f188586ea942/stderr`
- `target/release/build/generic-array-aeabcff5c6711df2/stderr`
- `target/release/build/httparse-51bbf189946076dd/stderr`
- `target/release/build/serde_json-e79122dcc8f2e50d/stderr`
- `target/release/build/libc-eb82498fbcf0dbb7/stderr`
- `target/release/build/zerocopy-6f7563d5525b3cbd/stderr`
- `target/release/build/serde-d06fa69ddc8879a3/stderr`
- `target/release/build/nano-gemm-c64-448b82dcfc219265/stderr`
- `target/release/build/proc-macro2-18909ccb974bda1a/stderr`
- `target/release/build/num-traits-9d0a60373b63ed35/stderr`
- `target/release/build/rstest_macros-cb0eac470027dff2/stderr`
- ... and 569 more

## Empty Directories

- `.claude/skills/`
- `runs/`
- `target/release/incremental/`
- `target/release/build/object-f512dba44aa1bcd6/out/`
- `target/release/build/rayon-core-5e16e2929457f8ef/out/`
- `target/release/build/openssl-sys-323919f6527b2acd/out/`
- `target/release/build/zerocopy-c4eff2cd97fe1042/out/`
- `target/release/build/libc-6754ff4b367e1cfe/out/`
- `target/release/build/rustls-51f64ac122d4c95f/out/`
- `target/release/build/yeslogic-fontconfig-sys-8f2ea2311909b79d/out/`
- `target/release/build/getrandom-b9b43c0bbdd4c1c1/out/`
- `target/release/build/syn-e34e92f5bdeb645c/out/`
- `target/release/build/parking_lot_core-f6d2f188586ea942/out/`
- `target/release/build/generic-array-aeabcff5c6711df2/out/`
- `target/release/build/httparse-51bbf189946076dd/out/`
- `target/release/build/serde_json-e79122dcc8f2e50d/out/`
- `target/release/build/libc-eb82498fbcf0dbb7/out/`
- `target/release/build/zerocopy-6f7563d5525b3cbd/out/`
- `target/release/build/proc-macro2-18909ccb974bda1a/out/`
- `target/release/build/num-traits-9d0a60373b63ed35/out/`
- `target/release/build/rstest_macros-cb0eac470027dff2/out/`
- `target/release/build/anyhow-cf7f4238169fff42/out/`
- `target/release/build/ort-sys-0f3f784050f276b4/out/`
- `target/release/build/native-tls-f4a3e71f7ba3f813/out/`
- `target/release/build/getrandom-7aceb8a75860db6c/out/`
- `target/release/build/pathfinder_simd-fd231278b7fdff04/out/`
- `target/release/build/thiserror-bce988e38db52980/out/`
- `target/release/build/httparse-a5caf1794d38a2fc/out/`
- `target/release/build/openssl-1a65e2b103a18bfc/out/`
- `target/release/build/icu_normalizer_data-6184f7ee95182060/out/`
- ... and 161 more

## Missing .gitignore Entries

- `.ruff_cache/`
- `node_modules/`
- `nohup.out`
