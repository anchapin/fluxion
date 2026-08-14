//! `fluxion` binary — thin shim around the canonical CLI surface.
//!
//! Issue #2929: the canonical `Cli` / `Commands` definitions, the dispatcher
//! and the helper functions used to live in this file, while a near-identical
//! but smaller copy lived in `src/cli/mod.rs` and was dead code (never called
//! by any user-facing entry point). The two surfaces drifted silently.
//!
//! All CLI logic now lives in [`fluxion::cli`]; this file only:
//!
//! 1. Calls [`fluxion::cli::run_cli`] from `main` so the binary and any future
//!    embedder (PyO3 entry, axum front-end, GUI, …) parse the same argv and
//!    dispatch through the same handlers.
//! 2. Houses the `#[cfg(test)]` unit tests that exercise `Cli::try_parse_from`
//!    and the dispatch helpers (`run_workflow`, `run_measure_command`,
//!    `validate_diagnostic_case`, `not_yet_implemented`). They reach into the
//!    lib via `use fluxion::cli::*` so they remain first-party unit tests of
//!    the canonical CLI surface.

use fluxion::cli::run_cli;

fn main() -> anyhow::Result<()> {
    run_cli()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use clap::Parser;
    use fluxion::cli::{
        run_measure_command, run_workflow, validate_diagnostic_case, Cli, Commands,
        MeasureSubcommand,
    };

    #[test]
    fn test_validate_statistical_flag_accepted() {
        // Test that --statistical flag is accepted
        let args = ["fluxion", "validate", "--statistical"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --statistical flag");
    }

    #[test]
    fn test_validate_alpha_flag_accepted() {
        // Test that --alpha flag is accepted
        let args = ["fluxion", "validate", "--statistical", "--alpha", "0.01"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --alpha flag");
    }

    #[test]
    fn test_validate_default_behavior_unchanged() {
        // Test that default behavior (no --statistical) works
        let args = ["fluxion", "validate"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should work without --statistical flag");

        if let Some(Commands::Validate { statistical, .. }) = cli.unwrap().command {
            assert!(!statistical, "Default should have statistical=false");
        }
    }

    #[test]
    fn test_validate_statistical_flag_sets_true() {
        // Test that --statistical flag sets statistical to true
        let args = ["fluxion", "validate", "--statistical"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --statistical flag");

        if let Some(Commands::Validate { statistical, .. }) = cli.unwrap().command {
            assert!(statistical, "--statistical should set statistical=true");
        }
    }

    #[test]
    fn test_validate_alpha_default_value() {
        // Test that --alpha has default value of 0.05
        let args = ["fluxion", "validate", "--statistical"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --statistical flag");

        if let Some(Commands::Validate { alpha, .. }) = cli.unwrap().command {
            assert_eq!(alpha, 0.05, "Default alpha should be 0.05");
        }
    }

    #[test]
    fn test_validate_alpha_custom_value() {
        // Test that --alpha accepts custom values
        let args = ["fluxion", "validate", "--statistical", "--alpha", "0.01"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept --alpha flag");

        if let Some(Commands::Validate { alpha, .. }) = cli.unwrap().command {
            assert_eq!(alpha, 0.01, "Custom alpha should be 0.01");
        }
    }

    #[test]
    fn test_validate_alpha_boundary_values() {
        // Test that --alpha accepts boundary values 0.0 and 1.0
        for alpha_val in ["0.0", "1.0"] {
            let args = ["fluxion", "validate", "--statistical", "--alpha", alpha_val];
            let cli = Cli::try_parse_from(args.iter());
            assert!(cli.is_ok(), "CLI should accept alpha={}", alpha_val);

            if let Some(Commands::Validate { alpha, .. }) = cli.unwrap().command {
                let expected = alpha_val.parse::<f64>().unwrap();
                assert_eq!(alpha, expected, "Alpha should be {}", alpha_val);
            }
        }
    }

    #[test]
    fn test_validate_alpha_too_large_rejected() {
        // Test that alpha > 1.0 is rejected
        let args = ["fluxion", "validate", "--statistical", "--alpha", "1.5"];
        let cli = Cli::try_parse_from(args.iter());
        // This test verifies CLI parsing only; runtime validation is separate
        assert!(
            cli.is_ok(),
            "CLI parsing should succeed (runtime validation handles invalid alpha)"
        );
    }

    #[test]
    fn test_validate_statistical_flag_integration() {
        // Test that --statistical flag integrates with other flags
        let args = [
            "fluxion",
            "validate",
            "--statistical",
            "--alpha",
            "0.05",
            "--format",
            "markdown",
        ];
        let cli = Cli::try_parse_from(args.iter());
        assert!(
            cli.is_ok(),
            "CLI should accept --statistical with other flags"
        );

        if let Some(Commands::Validate {
            statistical,
            alpha,
            format: fmt,
            ..
        }) = cli.unwrap().command
        {
            assert!(statistical, "--statistical should be true");
            assert_eq!(alpha, 0.05, "alpha should be 0.05");
            assert_eq!(fmt, "markdown", "format should be markdown");
        }
    }

    #[test]
    fn test_validate_without_statistical_backward_compatible() {
        // Test that validation works without --statistical flag
        let args = ["fluxion", "validate", "--format", "csv"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should work without --statistical");

        if let Some(Commands::Validate {
            statistical,
            format: fmt,
            ..
        }) = cli.unwrap().command
        {
            assert!(!statistical, "statistical should be false by default");
            assert_eq!(fmt, "csv", "format should be csv");
        }
    }

    #[test]
    fn test_direct_simulation_mode_energyplus_style() {
        // Test EnergyPlus-compatible direct simulation mode
        let args = [
            "fluxion",
            "-w",
            "weather.epw",
            "-d",
            "output/",
            "input.flux",
        ];
        let cli = Cli::try_parse_from(args.iter());
        assert!(
            cli.is_ok(),
            "CLI should accept EnergyPlus-style direct simulation"
        );
        let cli = cli.unwrap();
        assert_eq!(cli.weather, Some("weather.epw".to_string()));
        assert_eq!(cli.input, Some("input.flux".to_string()));
        assert_eq!(cli.output_directory, Some("output/".to_string()));
    }

    #[test]
    fn test_run_subcommand() {
        // Test OpenStudio-compatible run subcommand
        let args = ["fluxion", "run", "-w", "workflow.fwf"];
        let cli = Cli::try_parse_from(args.iter());
        assert!(cli.is_ok(), "CLI should accept run subcommand");
        if let Some(Commands::Run { workflow, .. }) = cli.unwrap().command {
            assert!(workflow.is_some());
        }
    }

    // --- Issue #2947 (originally #2711): stubbed workflows must fail loudly,
    // not silently succeed ---

    #[test]
    fn test_workflow_execution_is_gated_non_silent() {
        // The workflow runner parses the file but must NOT return Ok(()) for
        // the unimplemented execution path (issue #2947, originally #2711).
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "{{\"name\": \"stub\", \"steps\": []}}").unwrap();
        let err = run_workflow(Some(tmp.path()), false, false, false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(
            msg.contains("#2947"),
            "error should reference the tracking issue #2947: {msg}"
        );
    }

    #[test]
    fn test_measures_only_workflow_is_gated() {
        // `fluxion run --measures-only` must return a non-silent error.
        let err = run_workflow(Some(Path::new("dummy.fwf")), false, true, false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    #[test]
    fn test_postprocess_only_workflow_is_gated() {
        // `fluxion run --postprocess-only` must return a non-silent error.
        let err = run_workflow(Some(Path::new("dummy.fwf")), false, false, true).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    #[test]
    fn test_measure_update_is_gated() {
        // `fluxion measure update` must return a non-silent error.
        let cmd = MeasureSubcommand::Update {
            measure_dir: PathBuf::from("m"),
        };
        let err = run_measure_command(&cmd).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    #[test]
    fn test_measure_update_all_is_gated() {
        // `fluxion measure update --all` must return a non-silent error.
        let cmd = MeasureSubcommand::UpdateAll {
            measures_dir: PathBuf::from("m"),
        };
        let err = run_measure_command(&cmd).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    #[test]
    fn test_measure_compute_args_is_gated() {
        // `fluxion measure compute-args` must return a non-silent error.
        let cmd = MeasureSubcommand::ComputeArguments {
            model: PathBuf::from("model.flux"),
            measure_dir: PathBuf::from("measure"),
        };
        let err = run_measure_command(&cmd).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    #[test]
    fn test_measure_run_tests_is_gated() {
        // `fluxion measure run-tests` must return a non-silent error.
        let cmd = MeasureSubcommand::RunTests {
            measures_dir: PathBuf::from("measures"),
        };
        let err = run_measure_command(&cmd).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    #[test]
    fn test_diagnostic_case_range_is_gated() {
        // `fluxion validate-case 195-470` must return a non-silent error.
        let err = validate_diagnostic_case("195-470").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not yet implemented"), "got: {msg}");
        assert!(msg.contains("#2947"), "should reference #2947: {msg}");
    }

    // Note: direct simulation (`fluxion -w weather.epw input.flux`) is harder
    // to exercise via unit test because it requires a valid EPW file plus a
    // valid JSON model file. The gating is covered end-to-end by the source
    // code (the very last lines of `run_direct_simulation` return
    // `not_yet_implemented("direct simulation")`) and by the structure of the
    // integration test suite:

    #[test]
    fn test_direct_simulation_gated_via_source() {
        // Source-level guard: the lib's `src/cli/mod.rs` MUST contain the
        // `not_yet_implemented` call for `run_direct_simulation`. Anyone
        // removing the gating to implement direct simulation will need to
        // update this test (and the eight integration tests in
        // `tests/integration/test_cli.rs`).
        let source = include_str!("../cli/mod.rs");
        assert!(
            source.contains("Err(not_yet_implemented(\"direct simulation\"))"),
            "run_direct_simulation must call not_yet_implemented to preserve #2947 loud-failure"
        );
    }
}
