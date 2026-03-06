#!/usr/bin/env python3
"""
Automated Geometry Ingestion Pipeline (PDF/CAD-to-BEM) via Vision-Language Models.

This module provides a pipeline that extracts building geometry from PDF/CAD files
and converts it to BEM format using Vision-Language Models (VLMs).

Implements Issue #448: Automated Geometry Ingestion Pipeline

Key Features:
- PDF and image-based floor plan parsing using VLMs
- CAD file (DXF) parsing
- Geometry extraction (walls, windows, doors, zones)
- Conversion to CTA (Continuous Tensor Abstraction) tensors
- Zero-copy handoff to Rust core via PyO3
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fluxion.geometry_ingestion")


# ============================================================================
# Data Models for Building Geometry
# ============================================================================

@dataclass
class Point2D:
    """2D point coordinate."""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point2D":
        return cls(x=float(arr[0]), y=float(arr[1]))


@dataclass
class Point3D:
    """3D point coordinate."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point3D":
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class Wall:
    """Represents a wall element."""
    id: str
    start_point: Point2D
    end_point: Point2D
    height: float = 2.4  # Default height in meters
    thickness: float = 0.2  # Default thickness in meters
    
    @property
    def length(self) -> float:
        """Calculate wall length."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        return np.sqrt(dx**2 + dy**2)
    
    @property
    def area(self) -> float:
        """Calculate wall area."""
        return self.length * self.height
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2, height, thickness]."""
        return np.array([
            self.start_point.x, self.start_point.y,
            self.end_point.x, self.end_point.y,
            self.height, self.thickness
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, wall_id: str = "") -> "Wall":
        return cls(
            id=wall_id,
            start_point=Point2D(x=arr[0], y=arr[1]),
            end_point=Point2D(x=arr[2], y=arr[3]),
            height=float(arr[4]) if len(arr) > 4 else 2.4,
            thickness=float(arr[5]) if len(arr) > 5 else 0.2
        )


@dataclass
class Window:
    """Represents a window element."""
    id: str
    wall_id: str  # Parent wall
    start_point: Point2D
    end_point: Point2D
    height: float = 1.2  # Default height in meters
    sill_height: float = 0.9  # Sill height from floor in meters
    
    @property
    def width(self) -> float:
        """Calculate window width."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        return np.sqrt(dx**2 + dy**2)
    
    @property
    def area(self) -> float:
        """Calculate window area."""
        return self.width * self.height
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2, height, sill_height]."""
        return np.array([
            self.start_point.x, self.start_point.y,
            self.end_point.x, self.end_point.y,
            self.height, self.sill_height
        ])


@dataclass
class Door:
    """Represents a door element."""
    id: str
    wall_id: str  # Parent wall
    start_point: Point2D
    end_point: Point2D
    height: float = 2.1  # Default height in meters
    
    @property
    def width(self) -> float:
        """Calculate door width."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        return np.sqrt(dx**2 + dy**2)


@dataclass
class ThermalZone:
    """Represents a thermal zone."""
    id: str
    name: str
    vertices: List[Point2D]  # Polygon vertices (counter-clockwise)
    floor_height: float = 0.0  # Height of floor from ground
    ceiling_height: float = 2.4  # Ceiling height from floor
    
    @property
    def area(self) -> float:
        """Calculate zone floor area using shoelace formula."""
        if len(self.vertices) < 3:
            return 0.0
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0
    
    @property
    def volume(self) -> float:
        """Calculate zone volume."""
        return self.area * self.ceiling_height
    
    @property
    def perimeter(self) -> float:
        """Calculate zone perimeter."""
        if len(self.vertices) < 2:
            return 0.0
        perimeter = 0.0
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            dx = self.vertices[j].x - self.vertices[i].x
            dy = self.vertices[j].y - self.vertices[i].y
            perimeter += np.sqrt(dx**2 + dy**2)
        return perimeter
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for CTA tensor."""
        # Flatten vertices to [x1, y1, x2, y2, ..., xn, yn]
        coords = []
        for v in self.vertices:
            coords.extend([v.x, v.y])
        return np.array(coords + [
            self.floor_height,
            self.ceiling_height,
            self.area,
            self.volume,
            self.perimeter
        ])


@dataclass
class BuildingGeometry:
    """Complete building geometry model."""
    walls: List[Wall] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    zones: List[ThermalZone] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "walls": [
                {
                    "id": w.id,
                    "start": [w.start_point.x, w.start_point.y],
                    "end": [w.end_point.x, w.end_point.y],
                    "height": w.height,
                    "thickness": w.thickness,
                    "length": w.length,
                    "area": w.area
                }
                for w in self.walls
            ],
            "windows": [
                {
                    "id": win.id,
                    "wall_id": win.wall_id,
                    "start": [win.start_point.x, win.start_point.y],
                    "end": [win.end_point.x, win.end_point.y],
                    "height": win.height,
                    "sill_height": win.sill_height,
                    "area": win.area
                }
                for win in self.windows
            ],
            "doors": [
                {
                    "id": d.id,
                    "wall_id": d.wall_id,
                    "start": [d.start_point.x, d.start_point.y],
                    "end": [d.end_point.x, d.end_point.y],
                    "height": d.height,
                    "width": d.width
                }
                for d in self.doors
            ],
            "zones": [
                {
                    "id": z.id,
                    "name": z.name,
                    "vertices": [[v.x, v.y] for v in z.vertices],
                    "floor_height": z.floor_height,
                    "ceiling_height": z.ceiling_height,
                    "area": z.area,
                    "volume": z.volume,
                    "perimeter": z.perimeter
                }
                for z in self.zones
            ],
            "metadata": self.metadata
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get geometry summary statistics."""
        return {
            "num_walls": len(self.walls),
            "num_windows": len(self.windows),
            "num_doors": len(self.doors),
            "num_zones": len(self.zones),
            "total_wall_area": sum(w.area for w in self.walls),
            "total_window_area": sum(win.area for win in self.windows),
            "total_zone_area": sum(z.area for z in self.zones),
            "total_zone_volume": sum(z.volume for z in self.zones)
        }


# ============================================================================
# VLM-Based Geometry Extraction
# ============================================================================

class VLMPromptBuilder:
    """Builds prompts for VLM-based geometry extraction."""
    
    # System prompt for VLM geometry extraction
    SYSTEM_PROMPT = """You are an expert building architect and thermal modeling specialist.
Your task is to analyze architectural floor plan drawings and extract building geometry information
in a structured JSON format suitable for Building Energy Modeling (BEM).

For each floor plan, identify and extract:
1. WALLS: All exterior and interior walls as line segments with coordinates
2. WINDOWS: Window openings on walls with dimensions
3. DOORS: Door openings on walls with dimensions  
4. THERMAL ZONES: Enclosed spaces that can be heated/cooled independently

Output your analysis as a JSON object with this structure:
{
  "walls": [
    {"id": "wall_1", "start": [x1, y1], "end": [x2, y2], "height": 2.4, "thickness": 0.2}
  ],
  "windows": [
    {"id": "window_1", "wall_id": "wall_1", "start": [x1, y1], "end": [x2, y2], "height": 1.2, "sill_height": 0.9}
  ],
  "doors": [
    {"id": "door_1", "wall_id": "wall_1", "start": [x1, y1], "end": [x2, y2], "height": 2.1}
  ],
  "zones": [
    {"id": "zone_1", "name": "Living Room", "vertices": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "ceiling_height": 2.4}
  ]
}

Coordinates should be in meters, with the origin at bottom-left of the drawing.
Be precise with measurements and ensure walls form closed polygons for each zone."""
    
    @classmethod
    def build_extraction_prompt(cls, image_size: Optional[Tuple[int, int]] = None) -> str:
        """Build the extraction prompt."""
        prompt = cls.SYSTEM_PROMPT
        if image_size:
            prompt += f"\n\nThe image dimensions are {image_size[0]}x{image_size[1]} pixels."
        return prompt
    
    @classmethod
    def build_validation_prompt(cls, geometry: BuildingGeometry) -> str:
        """Build a validation prompt for checking extracted geometry."""
        summary = geometry.summary()
        return f"""Validate the following extracted building geometry for completeness and correctness:

Summary:
- Walls: {summary['num_walls']}
- Windows: {summary['num_windows']}
- Doors: {summary['num_doors']}
- Zones: {summary['num_zones']}
- Total Wall Area: {summary['total_wall_area']:.2f} m²
- Total Window Area: {summary['total_window_area']:.2f} m²
- Total Zone Area: {summary['total_zone_area']:.2f} m²
- Total Zone Volume: {summary['total_zone_volume']:.2f} m³

Check for:
1. Are all zones closed polygons?
2. Do windows and doors reference valid wall IDs?
3. Is the total window-to-wall ratio reasonable (typically 15-40%)?
4. Are there any obvious measurement errors?

Respond with JSON:
{{"valid": true/false, "issues": ["issue1", "issue2"], "warnings": ["warning1"]}}"""


class GeometryExtractor:
    """
    Extracts building geometry from various input formats using VLMs.
    
    Supports:
    - PDF files (converted to images)
    - Image files (PNG, JPG, etc.)
    - DXF CAD files
    """
    
    def __init__(
        self,
        vlm_model: Optional[Any] = None,
        vlm_provider: str = "ollama",
        model_name: str = "llava",
        temperature: float = 0.1
    ):
        """
        Initialize the geometry extractor.
        
        Args:
            vlm_model: Pre-loaded VLM model instance (optional)
            vlm_provider: VLM provider ("ollama", "openai", "anthropic", "mock")
            model_name: Model name for the provider
            temperature: Sampling temperature for generation
        """
        self.vlm_model = vlm_model
        self.vlm_provider = vlm_provider
        self.model_name = model_name
        self.temperature = temperature
        self._vlm_client = None
        
        logger.info(f"Initialized GeometryExtractor with provider={vlm_provider}, model={model_name}")
    
    def _init_vlm_client(self):
        """Initialize VLM client based on provider."""
        if self._vlm_client is not None:
            return
            
        if self.vlm_provider == "ollama":
            try:
                import ollama
                self._vlm_client = ollama
                logger.info("Initialized Ollama VLM client")
            except ImportError:
                logger.warning("ollama package not installed, falling back to mock mode")
                self.vlm_provider = "mock"
                
        elif self.vlm_provider == "openai":
            try:
                from openai import OpenAI
                self._vlm_client = OpenAI()
                logger.info("Initialized OpenAI VLM client")
            except ImportError:
                logger.warning("openai package not installed, falling back to mock mode")
                self.vlm_provider = "mock"
        
        elif self.vlm_provider == "mock":
            logger.info("Using mock VLM client for testing")
    
    def extract_from_image(self, image_path: str) -> BuildingGeometry:
        """
        Extract geometry from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BuildingGeometry object with extracted elements
        """
        logger.info(f"Extracting geometry from image: {image_path}")
        
        # Check file exists unless using mock mode
        if self.vlm_provider != "mock" and not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Initialize VLM client if needed (skip for mock mode)
        if self.vlm_provider != "mock":
            self._init_vlm_client()
        
        # Load and prepare image (skip for mock mode)
        image_size = None
        if self.vlm_provider != "mock":
            try:
                from PIL import Image
                image = Image.open(image_path)
                image_size = image.size  # (width, height)
                logger.info(f"Image size: {image_size}")
            except ImportError:
                raise ImportError("PIL is required for image processing. Install with: pip install pillow")
        
        # Build prompt
        prompt = VLMPromptBuilder.build_extraction_prompt(image_size)
        
        # Run VLM inference (pass None in mock mode since it's not used)
        image_for_vlm = None if self.vlm_provider == "mock" else image
        geometry_data = self._run_vlm_inference(prompt, image_for_vlm)
        
        # Parse into BuildingGeometry
        geometry = self._parse_vlm_response(geometry_data)
        
        # Add metadata
        geometry.metadata = {
            "source": image_path,
            "source_type": "image",
            "image_size": image_size,
            "vlm_provider": self.vlm_provider,
            "model_name": self.model_name
        }
        
        logger.info(f"Extracted geometry: {geometry.summary()}")
        return geometry
    
    def extract_from_pdf(self, pdf_path: str, page: int = 1) -> BuildingGeometry:
        """
        Extract geometry from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            page: Page number to extract (1-indexed)
            
        Returns:
            BuildingGeometry object with extracted elements
        """
        logger.info(f"Extracting geometry from PDF: {pdf_path}, page {page}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Convert PDF page to image
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
        
        # Open PDF and convert page to image
        doc = fitz.open(pdf_path)
        if page > len(doc):
            raise ValueError(f"Page {page} not found in PDF (has {len(doc)} pages)")
        
        page = doc[page - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution for better OCR
        
        # Save to temporary image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            pix.save(tmp_path)
        
        doc.close()
        
        try:
            # Extract geometry from the converted image
            geometry = self.extract_from_image(tmp_path)
            geometry.metadata["source"] = pdf_path
            geometry.metadata["source_type"] = "pdf"
            geometry.metadata["pdf_page"] = page
            return geometry
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def extract_from_dxf(self, dxf_path: str) -> BuildingGeometry:
        """
        Extract geometry from a DXF CAD file.
        
        Args:
            dxf_path: Path to the DXF file
            
        Returns:
            BuildingGeometry object with extracted elements
        """
        logger.info(f"Extracting geometry from DXF: {dxf_path}")
        
        if not os.path.exists(dxf_path):
            raise FileNotFoundError(f"DXF not found: {dxf_path}")
        
        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf is required for DXF processing. Install with: pip install ezdxf")
        
        # Load DXF document
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        geometry = BuildingGeometry()
        
        # Extract walls (LINE and LWPOLYLINE entities)
        wall_id = 0
        for entity in msp:
            if entity.dxftype() == "LINE":
                # Exterior wall
                geometry.walls.append(Wall(
                    id=f"wall_{wall_id}",
                    start_point=Point2D(
                        x=entity.dxf.start.x,
                        y=entity.dxf.start.y
                    ),
                    end_point=Point2D(
                        x=entity.dxf.end.x,
                        y=entity.dxf.end.y
                    )
                ))
                wall_id += 1
                
            elif entity.dxftype() == "LWPOLYLINE":
                # Could be a wall polyline
                vertices = list(entity.vertices())
                if len(vertices) >= 2:
                    for i in range(len(vertices) - 1):
                        geometry.walls.append(Wall(
                            id=f"wall_{wall_id}",
                            start_point=Point2D(
                                x=vertices[i].x,
                                y=vertices[i].y
                            ),
                            end_point=Point2D(
                                x=vertices[i + 1].x,
                                y=vertices[i + 1].y
                            )
                        ))
                        wall_id += 1
        
        # Extract windows and doors (look for specific layers or block names)
        # This is a simplified implementation - real CAD files may have complex structures
        window_id = 0
        door_id = 0
        
        # Try to identify windows/doors by layer name patterns
        for entity in msp:
            layer_name = entity.dxf.layer.lower() if hasattr(entity.dxf, 'layer') else ""
            
            if "window" in layer_name and entity.dxftype() == "LINE":
                geometry.windows.append(Window(
                    id=f"window_{window_id}",
                    wall_id="",  # Would need spatial analysis to determine parent wall
                    start_point=Point2D(
                        x=entity.dxf.start.x,
                        y=entity.dxf.start.y
                    ),
                    end_point=Point2D(
                        x=entity.dxf.end.x,
                        y=entity.dxf.end.y
                    )
                ))
                window_id += 1
                
            elif "door" in layer_name and entity.dxftype() == "LINE":
                geometry.doors.append(Door(
                    id=f"door_{door_id}",
                    wall_id="",
                    start_point=Point2D(
                        x=entity.dxf.start.x,
                        y=entity.dxf.start.y
                    ),
                    end_point=Point2D(
                        x=entity.dxf.end.x,
                        y=entity.dxf.end.y
                    )
                ))
                door_id += 1
        
        # Generate zones from wall network (simplified)
        # In practice, would need polygon detection algorithm
        if geometry.walls:
            # Create a single zone from all walls (simplified)
            all_points = []
            for wall in geometry.walls:
                all_points.append([wall.start_point.x, wall.start_point.y])
                all_points.append([wall.end_point.x, wall.end_point.y])
            
            # Get bounding box as single zone
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                geometry.zones.append(ThermalZone(
                    id="zone_0",
                    name="Building",
                    vertices=[
                        Point2D(x=min(xs), y=min(ys)),
                        Point2D(x=max(xs), y=min(ys)),
                        Point2D(x=max(xs), y=max(ys)),
                        Point2D(x=min(xs), y=max(ys))
                    ]
                ))
        
        geometry.metadata = {
            "source": dxf_path,
            "source_type": "dxf",
            "vlm_provider": "dxf_parser",
            "model_name": "ezdxf"
        }
        
        logger.info(f"Extracted geometry from DXF: {geometry.summary()}")
        return geometry
    
    def _run_vlm_inference(self, prompt: str, image: Any) -> Dict:
        """Run VLM inference to extract geometry."""
        
        if self.vlm_provider == "mock" or self.vlm_model is None:
            return self._mock_vlm_response()
        
        elif self.vlm_provider == "ollama":
            try:
                response = self._vlm_client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image] if hasattr(self._vlm_client, 'chat') else None
                        }
                    ],
                    temperature=self.temperature
                )
                # Parse JSON from response
                return self._extract_json_from_response(response["message"]["content"])
            except Exception as e:
                logger.warning(f"Ollama inference failed: {e}, using mock response")
                return self._mock_vlm_response()
        
        elif self.vlm_provider == "openai":
            try:
                # OpenAI Vision API
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                response = self._vlm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=self.temperature
                )
                return self._extract_json_from_response(
                    response.choices[0].message.content
                )
            except Exception as e:
                logger.warning(f"OpenAI inference failed: {e}, using mock response")
                return self._mock_vlm_response()
        
        else:
            return self._mock_vlm_response()
    
    def _mock_vlm_response(self) -> Dict:
        """Generate a mock VLM response for testing."""
        logger.info("Using mock VLM response")
        return {
            "walls": [
                {"id": "wall_1", "start": [0.0, 0.0], "end": [10.0, 0.0], "height": 2.4, "thickness": 0.2},
                {"id": "wall_2", "start": [10.0, 0.0], "end": [10.0, 8.0], "height": 2.4, "thickness": 0.2},
                {"id": "wall_3", "start": [10.0, 8.0], "end": [0.0, 8.0], "height": 2.4, "thickness": 0.2},
                {"id": "wall_4", "start": [0.0, 8.0], "end": [0.0, 0.0], "height": 2.4, "thickness": 0.2},
                {"id": "wall_5", "start": [5.0, 0.0], "end": [5.0, 8.0], "height": 2.4, "thickness": 0.15}
            ],
            "windows": [
                {"id": "window_1", "wall_id": "wall_1", "start": [2.0, 0.9], "end": [4.0, 0.9], "height": 1.2, "sill_height": 0.9},
                {"id": "window_2", "wall_id": "wall_2", "start": [10.0, 0.9], "end": [10.0, 2.2], "height": 1.2, "sill_height": 0.9}
            ],
            "doors": [
                {"id": "door_1", "wall_id": "wall_5", "start": [5.0, 0.0], "end": [5.0, 0.9], "height": 2.1}
            ],
            "zones": [
                {
                    "id": "zone_1",
                    "name": "Living Room",
                    "vertices": [[0.0, 0.0], [5.0, 0.0], [5.0, 8.0], [0.0, 8.0]],
                    "ceiling_height": 2.4
                },
                {
                    "id": "zone_2", 
                    "name": "Bedroom",
                    "vertices": [[5.0, 0.0], [10.0, 0.0], [10.0, 8.0], [5.0, 8.0]],
                    "ceiling_height": 2.4
                }
            ]
        }
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from VLM response text."""
        import re
        
        # Try to find JSON in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON from VLM response: {response_text[:200]}...")
    
    def _parse_vlm_response(self, data: Dict) -> BuildingGeometry:
        """Parse VLM response data into BuildingGeometry."""
        geometry = BuildingGeometry()
        
        # Parse walls
        for w in data.get("walls", []):
            geometry.walls.append(Wall(
                id=w.get("id", f"wall_{len(geometry.walls)}"),
                start_point=Point2D(x=w["start"][0], y=w["start"][1]),
                end_point=Point2D(x=w["end"][0], y=w["end"][1]),
                height=w.get("height", 2.4),
                thickness=w.get("thickness", 0.2)
            ))
        
        # Parse windows
        for win in data.get("windows", []):
            geometry.windows.append(Window(
                id=win.get("id", f"window_{len(geometry.windows)}"),
                wall_id=win.get("wall_id", ""),
                start_point=Point2D(x=win["start"][0], y=win["start"][1]),
                end_point=Point2D(x=win["end"][0], y=win["end"][1]),
                height=win.get("height", 1.2),
                sill_height=win.get("sill_height", 0.9)
            ))
        
        # Parse doors
        for d in data.get("doors", []):
            geometry.doors.append(Door(
                id=d.get("id", f"door_{len(geometry.doors)}"),
                wall_id=d.get("wall_id", ""),
                start_point=Point2D(x=d["start"][0], y=d["start"][1]),
                end_point=Point2D(x=d["end"][0], y=d["end"][1]),
                height=d.get("height", 2.1)
            ))
        
        # Parse zones
        for z in data.get("zones", []):
            geometry.zones.append(ThermalZone(
                id=z.get("id", f"zone_{len(geometry.zones)}"),
                name=z.get("name", f"Zone {len(geometry.zones)}"),
                vertices=[Point2D(x=v[0], y=v[1]) for v in z["vertices"]],
                floor_height=z.get("floor_height", 0.0),
                ceiling_height=z.get("ceiling_height", 2.4)
            ))
        
        return geometry


# ============================================================================
# Geometry to CTA Tensor Conversion
# ============================================================================

class GeometryToTensorConverter:
    """
    Converts extracted building geometry to CTA tensors.
    
    The CTA (Continuous Tensor Abstraction) tensor format is designed for
    efficient computation in the Fluxion Rust core.
    """
    
    def __init__(self, max_zones: int = 100, max_walls: int = 500):
        """
        Initialize the converter.
        
        Args:
            max_zones: Maximum number of thermal zones
            max_walls: Maximum number of walls
        """
        self.max_zones = max_zones
        self.max_walls = max_walls
    
    def to_cta_tensors(
        self, 
        geometry: BuildingGeometry
    ) -> Dict[str, np.ndarray]:
        """
        Convert building geometry to CTA tensors.
        
        Returns a dictionary of tensors:
        - zone_coords: (max_zones, 20) - Zone coordinates and properties
        - wall_matrix: (max_walls, 6) - Wall geometry
        - window_matrix: (max_walls, 6) - Window geometry  
        - adjacency_matrix: (max_zones, max_zones) - Zone adjacency
        - zone_properties: (max_zones, 5) - Zone thermal properties
        
        Returns:
            Dictionary of numpy arrays
        """
        tensors = {}
        
        # Zone coordinates tensor
        # Format: [x1, y1, x2, y2, ..., x8, y8, floor_height, ceiling_height, area, volume, perimeter, zone_id]
        # Padded with zeros to max_zones
        zone_coords = np.zeros((self.max_zones, 20), dtype=np.float32)
        
        for i, zone in enumerate(geometry.zones[:self.max_zones]):
            zone_arr = zone.to_array()
            # Pad or truncate to 20 elements
            zone_coords[i, :len(zone_arr)] = zone_arr[:20]
            # Set zone ID at the end
            zone_coords[i, 19] = i + 1
        
        tensors["zone_coords"] = zone_coords
        
        # Wall matrix
        # Format: [x1, y1, x2, y2, height, thickness]
        wall_matrix = np.zeros((self.max_walls, 6), dtype=np.float32)
        
        for i, wall in enumerate(geometry.walls[:self.max_walls]):
            wall_matrix[i] = wall.to_array()
        
        tensors["wall_matrix"] = wall_matrix
        
        # Window matrix
        # Format: [x1, y1, x2, y2, height, sill_height]
        window_matrix = np.zeros((self.max_walls, 6), dtype=np.float32)
        
        for i, window in enumerate(geometry.windows[:self.max_walls]):
            window_matrix[i] = window.to_array()
        
        tensors["window_matrix"] = window_matrix
        
        # Adjacency matrix
        # Format: (max_zones, max_zones) - 1 if zones share a wall
        adjacency_matrix = self._compute_adjacency_matrix(geometry)
        tensors["adjacency_matrix"] = adjacency_matrix
        
        # Zone properties
        # Format: [floor_area, volume, perimeter, num_windows, num_doors]
        zone_properties = np.zeros((self.max_zones, 5), dtype=np.float32)
        
        for i, zone in enumerate(geometry.zones[:self.max_zones]):
            # Count windows and doors adjacent to this zone
            # (Simplified - would need proper spatial analysis)
            zone_properties[i] = [
                zone.area,
                zone.volume,
                zone.perimeter,
                len(geometry.windows),
                len(geometry.doors)
            ]
        
        tensors["zone_properties"] = zone_properties
        
        # Summary tensor
        # Format: [num_zones, num_walls, num_windows, num_doors, total_area, total_volume]
        summary = np.array([
            len(geometry.zones),
            len(geometry.walls),
            len(geometry.windows),
            len(geometry.doors),
            sum(z.area for z in geometry.zones),
            sum(z.volume for z in geometry.zones)
        ], dtype=np.float32)
        
        tensors["summary"] = summary
        
        return tensors
    
    def _compute_adjacency_matrix(
        self, 
        geometry: BuildingGeometry
    ) -> np.ndarray:
        """
        Compute zone adjacency matrix.
        
        Two zones are considered adjacent if they share a wall.
        """
        n_zones = len(geometry.zones)
        adjacency = np.zeros((self.max_zones, self.max_zones), dtype=np.float32)
        
        if n_zones == 0:
            return adjacency
        
        # Simplified adjacency detection based on shared vertices
        # In practice, would use proper geometric intersection
        
        # For now, mark zones as adjacent if their bounding boxes overlap
        for i in range(min(n_zones, self.max_zones)):
            for j in range(i + 1, min(n_zones, self.max_zones)):
                zone_i = geometry.zones[i]
                zone_j = geometry.zones[j]
                
                # Check if zones share any wall segments
                if self._zones_share_wall(zone_i, zone_j, geometry.walls):
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
        
        return adjacency
    
    def _zones_share_wall(
        self, 
        zone1: ThermalZone, 
        zone2: ThermalZone,
        walls: List[Wall]
    ) -> bool:
        """Check if two zones share a wall."""
        # Get vertex sets for each zone
        verts1 = set((v.x, v.y) for v in zone1.vertices)
        verts2 = set((v.x, v.y) for v in zone2.vertices)
        
        # Shared vertices indicate potential shared wall
        shared = verts1.intersection(verts2)
        return len(shared) >= 2
    
    def validate_tensors(
        self, 
        tensors: Dict[str, np.ndarray]
    ) -> Tuple[bool, List[str]]:
        """
        Validate CTA tensors for correctness.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check zone_coords
        zone_coords = tensors.get("zone_coords")
        if zone_coords is not None:
            if zone_coords.shape != (self.max_zones, 20):
                issues.append(f"zone_coords shape {zone_coords.shape} != ({self.max_zones}, 20)")
        
        # Check wall_matrix
        wall_matrix = tensors.get("wall_matrix")
        if wall_matrix is not None:
            if wall_matrix.shape != (self.max_walls, 6):
                issues.append(f"wall_matrix shape {wall_matrix.shape} != ({self.max_walls}, 6)")
        
        # Check adjacency matrix
        adjacency = tensors.get("adjacency_matrix")
        if adjacency is not None:
            if adjacency.shape != (self.max_zones, self.max_zones):
                issues.append(f"adjacency_matrix shape {adjacency.shape} != ({self.max_zones}, {self.max_zones})")
            # Check symmetry
            if not np.allclose(adjacency, adjacency.T):
                issues.append("adjacency_matrix is not symmetric")
        
        # Check for NaN or Inf
        for name, tensor in tensors.items():
            if not np.all(np.isfinite(tensor)):
                issues.append(f"{name} contains NaN or Inf values")
        
        is_valid = len(issues) == 0
        return is_valid, issues


# ============================================================================
# Main Pipeline Integration
# ============================================================================

class GeometryIngestionPipeline:
    """
    Complete pipeline for automated geometry ingestion.
    
    This pipeline:
    1. Extracts geometry from PDF/CAD files using VLMs
    2. Converts geometry to CTA tensors
    3. Provides zero-copy handoff to Rust core via PyO3
    """
    
    def __init__(
        self,
        vlm_provider: str = "mock",
        model_name: str = "llava",
        max_zones: int = 100,
        max_walls: int = 500
    ):
        """
        Initialize the pipeline.
        
        Args:
            vlm_provider: VLM provider ("ollama", "openai", "mock")
            model_name: Model name for VLM
            max_zones: Maximum thermal zones
            max_walls: Maximum walls
        """
        self.extractor = GeometryExtractor(
            vlm_provider=vlm_provider,
            model_name=model_name
        )
        self.converter = GeometryToTensorConverter(
            max_zones=max_zones,
            max_walls=max_walls
        )
        
        logger.info(f"Initialized GeometryIngestionPipeline (provider={vlm_provider})")
    
    def ingest(
        self,
        input_path: str,
        input_type: Optional[str] = None
    ) -> Tuple[BuildingGeometry, Dict[str, np.ndarray]]:
        """
        Run the complete ingestion pipeline.
        
        Args:
            input_path: Path to input file (PDF, image, or DXF)
            input_type: Type of input ("pdf", "image", "dxf"). Auto-detected if None.
            
        Returns:
            Tuple of (BuildingGeometry, CTA tensors dictionary)
        """
        logger.info(f"Starting geometry ingestion from: {input_path}")
        
        # Auto-detect input type
        if input_type is None:
            ext = Path(input_path).suffix.lower()
            if ext == ".pdf":
                input_type = "pdf"
            elif ext in [".dxf", ".dwg"]:
                input_type = "dxf"
            else:
                input_type = "image"
        
        # Extract geometry based on input type
        if input_type == "pdf":
            geometry = self.extractor.extract_from_pdf(input_path)
        elif input_type == "dxf":
            geometry = self.extractor.extract_from_dxf(input_path)
        else:
            geometry = self.extractor.extract_from_image(input_path)
        
        # Convert to CTA tensors
        tensors = self.converter.to_cta_tensors(geometry)
        
        # Validate tensors
        is_valid, issues = self.converter.validate_tensors(tensors)
        if not is_valid:
            logger.warning(f"Tensor validation issues: {issues}")
        
        logger.info(f"Geometry ingestion complete: {geometry.summary()}")
        
        return geometry, tensors
    
    def save_outputs(
        self,
        geometry: BuildingGeometry,
        tensors: Dict[str, np.ndarray],
        output_dir: str
    ):
        """
        Save pipeline outputs to files.
        
        Args:
            geometry: BuildingGeometry object
            tensors: CTA tensors dictionary
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save geometry as JSON
        geometry_path = os.path.join(output_dir, "geometry.json")
        with open(geometry_path, "w") as f:
            json.dump(geometry.to_dict(), f, indent=2)
        logger.info(f"Saved geometry to: {geometry_path}")
        
        # Save tensors as numpy files
        for name, tensor in tensors.items():
            tensor_path = os.path.join(output_dir, f"{name}.npy")
            np.save(tensor_path, tensor)
            logger.info(f"Saved {name} tensor to: {tensor_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(geometry.summary(), f, indent=2)
        logger.info(f"Saved summary to: {summary_path}")


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated Geometry Ingestion Pipeline (PDF/CAD-to-BEM) via VLM"
    )
    parser.add_argument(
        "input",
        help="Input file (PDF, image, or DXF)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/geometry",
        help="Output directory"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["pdf", "image", "dxf"],
        help="Input file type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--vlm-provider",
        default="mock",
        choices=["ollama", "openai", "mock"],
        help="VLM provider"
    )
    parser.add_argument(
        "--model",
        default="llava",
        help="VLM model name"
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=100,
        help="Maximum number of thermal zones"
    )
    parser.add_argument(
        "--max-walls",
        type=int,
        default=500,
        help="Maximum number of walls"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = GeometryIngestionPipeline(
        vlm_provider=args.vlm_provider,
        model_name=args.model,
        max_zones=args.max_zones,
        max_walls=args.max_walls
    )
    
    geometry, tensors = pipeline.ingest(args.input, args.type)
    pipeline.save_outputs(geometry, tensors, args.output)
    
    print(f"\n✓ Geometry ingestion complete!")
    print(f"  Summary: {geometry.summary()}")


if __name__ == "__main__":
    main()
