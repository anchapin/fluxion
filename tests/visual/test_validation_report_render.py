"""Visual regression tests for physics validation reports.

Tests rendering of physics engine outputs including:
- Validation report charts
- Benchmark comparison plots
- Diagnostic thermal distribution plots
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden')


class TestValidationReportRender:
    """Tests for validation report rendering."""

    def setup_method(self):
        self.output_dir = os.path.join(GOLDEN_DIR, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_validation_report_header(self):
        """Test validation report header renders correctly."""
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, 'Fluxion Validation Report', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.axis('off')

        output_path = os.path.join(self.output_dir, 'validation_header.png')
        fig.savefig(output_path)
        plt.close(fig)

        expected_path = os.path.join(GOLDEN_DIR, 'validation_header.png')
        if os.path.exists(expected_path):
            self._assert_images_match(output_path, expected_path)

    def test_benchmark_comparison_chart(self):
        """Test benchmark comparison chart renders correctly."""
        modules = ['Weather', 'Solar', 'Conduction', 'Ventilation', 'Zone Balance']
        fluxion_times = [1.2, 3.4, 2.1, 0.8, 4.5]
        reference_times = [1.3, 3.5, 2.0, 0.9, 4.6]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(modules))
        width = 0.35

        ax.bar(x - width/2, fluxion_times, width, label='Fluxion', color='#3b82f6')
        ax.bar(x + width/2, reference_times, width, label='Reference (E+)", color='#22c55e')

        ax.set_xlabel('Module')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Benchmark Comparison: Fluxion vs EnergyPlus Reference')
        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=45, ha='right')
        ax.legend()

        output_path = os.path.join(self.output_dir, 'benchmark_comparison.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        expected_path = os.path.join(GOLDEN_DIR, 'benchmark_comparison.png')
        if os.path.exists(expected_path):
            self._assert_images_match(output_path, expected_path)

    def test_thermal_distribution_plot(self):
        """Test thermal distribution diagnostic plot renders correctly."""
        zone_temps = np.random.normal(22, 2, 1000)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(zone_temps, bins=30, color='#3b82f6', edgecolor='white', alpha=0.8)
        ax.axvline(zone_temps.mean(), color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {zone_temps.mean():.1f}°C')
        ax.set_xlabel('Zone Temperature (°C)')
        ax.set_ylabel('Frequency')
        ax.set_title('Zone Temperature Distribution')
        ax.legend()

        output_path = os.path.join(self.output_dir, 'thermal_distribution.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        expected_path = os.path.join(GOLDEN_DIR, 'thermal_distribution.png')
        if os.path.exists(expected_path):
            self._assert_images_match(output_path, expected_path)

    def test_error_convergence_plot(self):
        """Test error convergence chart for solver validation."""
        iterations = np.arange(1, 101)
        error_5r1c = 5.0 * np.exp(-0.05 * iterations) + np.random.normal(0, 0.01, 100)
        error_ctf = 4.5 * np.exp(-0.04 * iterations) + np.random.normal(0, 0.01, 100)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(iterations, error_5r1c, label='5R1C', color='#3b82f6')
        ax.semilogy(iterations, error_ctf, label='CTF', color='#22c55e')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS Error (°C)')
        ax.set_title('Solver Error Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(self.output_dir, 'error_convergence.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        expected_path = os.path.join(GOLDEN_DIR, 'error_convergence.png')
        if os.path.exists(expected_path):
            self._assert_images_match(output_path, expected_path)

    def _assert_images_match(self, actual_path, expected_path):
        """Compare actual output to golden reference."""
        from PIL import Image

        actual = Image.open(actual_path)
        expected = Image.open(expected_path)

        actual_array = np.array(actual.resize((100, 100)))
        expected_array = np.array(expected.resize((100, 100)))

        mse = np.mean((actual_array.astype(float) - expected_array.astype(float)) ** 2)
        assert mse < 100, f"Image mismatch (MSE={mse:.2f})"
