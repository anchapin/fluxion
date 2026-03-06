#!/usr/bin/env python3
"""
Demo script for the Geometry Ingestion Pipeline (Issue #448)

This script demonstrates the automated geometry extraction pipeline that:
1. Extracts building geometry from PDF/CAD files using VLMs
2. Converts to CTA tensors
3. Provides zero-copy handoff to Rust core

Usage:
    python3 demo_geometry_pipeline.py [--vlm-provider mock|ollama|openai]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# Import the geometry extraction pipeline
from tools.geometry_extraction import (
    GeometryIngestionPipeline,
    BuildingGeometry,
    ThermalZone,
    Wall,
    Window,
    Door,
    Point2D,
)


def create_sample_geometry() -> BuildingGeometry:
    """Create a sample building geometry for demonstration."""
    # Create a simple two-zone building
    geometry = BuildingGeometry()
    
    # Add walls (counter-clockwise, starting from bottom-left)
    # Exterior walls
    geometry.walls.append(Wall(
        id="wall_1",
        start_point=Point2D(0.0, 0.0),
        end_point=Point2D(10.0, 0.0),
        height=2.4,
        thickness=0.2
    ))
    geometry.walls.append(Wall(
        id="wall_2",
        start_point=Point2D(10.0, 0.0),
        end_point=Point2D(10.0, 8.0),
        height=2.4,
        thickness=0.2
    ))
    geometry.walls.append(Wall(
        id="wall_3",
        start_point=Point2D(10.0, 8.0),
        end_point=Point2D(0.0, 8.0),
        height=2.4,
        thickness=0.2
    ))
    geometry.walls.append(Wall(
        id="wall_4",
        start_point=Point2D(0.0, 8.0),
        end_point=Point2D(0.0, 0.0),
        height=2.4,
        thickness=0.2
    ))
    # Interior wall
    geometry.walls.append(Wall(
        id="wall_5",
        start_point=Point2D(5.0, 0.0),
        end_point=Point2D(5.0, 8.0),
        height=2.4,
        thickness=0.15
    ))
    
    # Add windows
    geometry.windows.append(Window(
        id="window_1",
        wall_id="wall_1",
        start_point=Point2D(2.0, 0.9),
        end_point=Point2D(4.0, 0.9),
        height=1.2,
        sill_height=0.9
    ))
    geometry.windows.append(Window(
        id="window_2",
        wall_id="wall_1",
        start_point=Point2D(6.0, 0.9),
        end_point=Point2D(8.0, 0.9),
        height=1.2,
        sill_height=0.9
    ))
    geometry.windows.append(Window(
        id="window_3",
        wall_id="wall_2",
        start_point=Point2D(10.0, 0.9),
        end_point=Point2D(10.0, 2.1),
        height=1.2,
        sill_height=0.9
    ))
    geometry.windows.append(Window(
        id="window_4",
        wall_id="wall_3",
        start_point=Point2D(1.0, 0.9),
        end_point=Point2D(3.0, 0.9),
        height=1.2,
        sill_height=0.9
    ))
    
    # Add doors
    geometry.doors.append(Door(
        id="door_1",
        wall_id="wall_5",
        start_point=Point2D(5.0, 0.0),
        end_point=Point2D(5.0, 2.1),
        height=2.1
    ))
    
    # Add thermal zones
    geometry.zones.append(ThermalZone(
        id="zone_1",
        name="Living Room",
        vertices=[
            Point2D(0.0, 0.0),
            Point2D(5.0, 0.0),
            Point2D(5.0, 8.0),
            Point2D(0.0, 8.0)
        ],
        ceiling_height=2.4
    ))
    geometry.zones.append(ThermalZone(
        id="zone_2",
        name="Bedroom",
        vertices=[
            Point2D(5.0, 0.0),
            Point2D(10.0, 0.0),
            Point2D(10.0, 8.0),
            Point2D(5.0, 8.0)
        ],
        ceiling_height=2.4
    ))
    
    # Add metadata
    geometry.metadata = {
        "source": "demo",
        "source_type": "generated",
        "building_type": "residential",
        "year_built": 2020
    }
    
    return geometry


def demo_basic_usage():
    """Demonstrate basic pipeline usage."""
    print("=" * 60)
    print("Geometry Ingestion Pipeline Demo")
    print("=" * 60)
    
    # Create pipeline with mock VLM (for testing without external dependencies)
    pipeline = GeometryIngestionPipeline(
        vlm_provider="mock",
        model_name="llava",
        max_zones=100,
        max_walls=500
    )
    
    # Create sample geometry directly (bypassing VLM for demo)
    geometry = create_sample_geometry()
    
    # Convert to CTA tensors
    tensors = pipeline.converter.to_cta_tensors(geometry)
    
    # Print summary
    print("\n📊 Building Geometry Summary:")
    print("-" * 40)
    summary = geometry.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n📐 CTA Tensor Shapes:")
    print("-" * 40)
    for name, tensor in tensors.items():
        print(f"  {name}: {tensor.shape} (dtype: {tensor.dtype})")
    
    # Validate tensors
    print("\n✅ Tensor Validation:")
    print("-" * 40)
    is_valid, issues = pipeline.converter.validate_tensors(tensors)
    if is_valid:
        print("  All tensors validated successfully!")
    else:
        print(f"  Validation issues: {issues}")
    
    return geometry, tensors


def demo_ingestion_pipeline():
    """Demonstrate the full ingestion pipeline."""
    print("\n" + "=" * 60)
    print("Full Ingestion Pipeline Demo")
    print("=" * 60)
    
    # Create pipeline
    pipeline = GeometryIngestionPipeline(
        vlm_provider="mock",
        max_zones=100,
        max_walls=500
    )
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as output_dir:
        # Run ingestion using mock VLM response (no file needed)
        # This simulates what would happen with a real floor plan image
        geometry, tensors = pipeline.ingest(
            input_path="demo_floor_plan.png",
            input_type="image"
        )
        
        # Save outputs
        pipeline.save_outputs(geometry, tensors, output_dir)
        
        print(f"\n📁 Output files saved to: {output_dir}")
        print("-" * 40)
        
        # List output files
        for f in os.listdir(output_dir):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  {f}: {size} bytes")
    
    return geometry, tensors


def demo_rust_integration():
    """Demonstrate zero-copy integration with Rust core."""
    print("\n" + "=" * 60)
    print("Rust Integration Demo (Zero-Copy)")
    print("=" * 60)
    
    # First, let's demonstrate the tensor format that would be passed to Rust
    geometry, tensors = demo_basic_usage()
    
    # The tensors dictionary contains:
    # - zone_coords: (100, 20) - Zone coordinates and properties
    # - wall_matrix: (500, 6) - Wall geometry
    # - window_matrix: (500, 6) - Window geometry  
    # - adjacency_matrix: (100, 100) - Zone adjacency
    # - zone_properties: (100, 5) - Zone thermal properties
    # - summary: (6,) - Summary statistics
    
    print("\n🔗 Rust Tensor Format:")
    print("-" * 40)
    print("  These tensors can be directly passed to the Rust core via PyO3")
    print("  bindings with zero-copy for maximum performance.")
    print()
    
    # Show sample tensor data
    print("  Zone coords (first zone):")
    print(f"    {tensors['zone_coords'][0][:8]}...")
    print()
    print("  Wall matrix (first wall):")
    print(f"    {tensors['wall_matrix'][0]}")
    print()
    print("  Summary:")
    print(f"    {tensors['summary']}")
    
    # Try to import and use the Rust bindings if available
    try:
        import fluxion
        print("\n✅ Rust fluxion module loaded!")
        
        # Create geometry tensor from numpy arrays
        geo_tensor = fluxion.GeometryTensor.from_numpy(
            tensors['zone_coords'],
            tensors['wall_matrix'],
            tensors['window_matrix'],
            tensors['adjacency_matrix'],
            tensors['zone_properties'],
            tensors['summary']
        )
        
        print(f"  Created Rust GeometryTensor: {geo_tensor}")
        print(f"  Number of zones: {geo_tensor.num_zones()}")
        print(f"  Number of walls: {geo_tensor.num_walls()}")
        print(f"  Total area: {geo_tensor.total_area():.2f} m²")
        print(f"  Total volume: {geo_tensor.total_volume():.2f} m³")
        
        # Validate
        issues = geo_tensor.validate()
        if issues:
            print(f"  Validation issues: {issues}")
        else:
            print("  ✅ Tensor validation passed!")
        
    except ImportError as e:
        print(f"\n⚠️  Rust module not available: {e}")
        print("    Build with: cargo build --features python-bindings")
        print("    Install wheel: pip install -e .")
    except Exception as e:
        print(f"\n⚠️  Error using Rust module: {e}")


def demo_validation():
    """Demonstrate geometry validation against reference."""
    print("\n" + "=" * 60)
    print("Geometry Validation Demo")
    print("=" * 60)
    
    # Create sample geometry
    geometry = create_sample_geometry()
    
    # Calculate metrics
    total_zone_area = sum(z.area for z in geometry.zones)
    total_zone_volume = sum(z.volume for z in geometry.zones)
    total_wall_area = sum(w.area for w in geometry.walls)
    total_window_area = sum(win.area for win in geometry.windows)
    
    print("\n📏 Calculated Metrics:")
    print("-" * 40)
    print(f"  Total zone floor area: {total_zone_area:.2f} m²")
    print(f"  Total zone volume: {total_zone_volume:.2f} m³")
    print(f"  Total wall area: {total_wall_area:.2f} m²")
    print(f"  Total window area: {total_window_area:.2f} m²")
    
    # Calculate window-to-wall ratio
    if total_wall_area > 0:
        wwr = (total_window_area / total_wall_area) * 100
        print(f"  Window-to-Wall Ratio (WWR): {wwr:.1f}%")
        
        # Validate WWR is in reasonable range
        if 15 <= wwr <= 40:
            print("  ✅ WWR is within typical range (15-40%)")
        else:
            print("  ⚠️  WWR is outside typical range")
    
    # Print zone details
    print("\n🏠 Zone Details:")
    print("-" * 40)
    for zone in geometry.zones:
        print(f"  {zone.name}:")
        print(f"    Area: {zone.area:.2f} m²")
        print(f"    Volume: {zone.volume:.2f} m³")
        print(f"    Perimeter: {zone.perimeter:.2f} m")


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for Geometry Ingestion Pipeline (Issue #448)"
    )
    parser.add_argument(
        "--vlm-provider",
        default="mock",
        choices=["ollama", "openai", "mock"],
        help="VLM provider for geometry extraction"
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip Rust integration demo"
    )
    
    args = parser.parse_args()
    
    print("\n🎯 Issue #448: Automated Geometry Ingestion Pipeline")
    print("   PDF/CAD-to-BEM via Vision-Language Models")
    print()
    
    # Run demos
    demo_basic_usage()
    demo_ingestion_pipeline()
    demo_validation()
    
    if not args.skip_rust:
        demo_rust_integration()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("""
The pipeline is ready to use! Here's how:

1. Extract geometry from PDF/CAD:
   python3 tools/geometry_extraction.py input.pdf --output output/

2. Use in Python:
   from tools.geometry_extraction import GeometryIngestionPipeline
   
   pipeline = GeometryIngestionPipeline(vlm_provider='ollama')
   geometry, tensors = pipeline.ingest('floor_plan.png')

3. Pass to Rust core (zero-copy):
   import fluxion
   geo_tensor = fluxion.GeometryTensor.from_numpy(*tensors.values())
""")


if __name__ == "__main__":
    main()
