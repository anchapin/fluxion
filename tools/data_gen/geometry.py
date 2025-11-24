"""
Geometry generation utilities for OpenStudio.
Handles creation of parametric 'Shoebox' models.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Try importing OpenStudio
try:
    import openstudio
except ImportError:
    openstudio = None

class MockOpenStudio:
    """
    A minimal mock of the OpenStudio API for testing in environments
    without the full library installed.
    """
    class model:
        class Model:
            def __init__(self):
                self.objects = []
            def save(self, path, overwrite):
                logger.info(f"[MOCK] Saving model to {path}")
                return True
            def getThermalZones(self):
                return [MockOpenStudio.model.ThermalZone()]

        class ThermalZone:
            def __init__(self, model=None):
                pass
            def addToNode(self, node):
                pass

        class Space:
            def __init__(self, model):
                pass
            def setThermalZone(self, zone):
                pass

        class Surface:
            def __init__(self, vertices, model):
                self.vertices = vertices
                pass
            def setSpace(self, space):
                pass
            def setSurfaceType(self, type):
                pass
            def setOutsideBoundaryCondition(self, bc):
                pass
            def setSunExposure(self, exp):
                pass
            def setWindExposure(self, exp):
                pass
            def setConstruction(self, c):
                pass

        class SubSurface:
            def __init__(self, vertices, model):
                pass
            def setSurface(self, surface):
                pass
            def setConstruction(self, c):
                pass

        class Material:
            def __init__(self, model):
                pass
            def setThickness(self, v): pass
            def setConductivity(self, v): pass
            def setDensity(self, v): pass
            def setSpecificHeat(self, v): pass

        class Construction:
            def __init__(self, model):
                pass
            def insertLayer(self, idx, mat): pass

        class OutputVariable:
            def __init__(self, var, model): pass
            def setReportingFrequency(self, freq): pass

    class Point3d:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class Point3dVector:
        def __init__(self, points):
            self.points = points

def ensure_openstudio():
    """Returns the openstudio module or a mock if allowed."""
    if openstudio:
        return openstudio
    logger.warning("OpenStudio not found. Using Mock object.")
    return MockOpenStudio()

def create_default_construction(os_api, model, name="Default Construction"):
    """
    Creates a simple concrete construction.
    """
    mat = os_api.model.Material(model)
    mat.setThickness(0.2) # 20cm
    mat.setConductivity(1.8) # Concrete
    mat.setDensity(2400.0)
    mat.setSpecificHeat(880.0)

    construction = os_api.model.Construction(model)
    construction.insertLayer(0, mat)
    return construction

def create_window_construction(os_api, model):
    """
    Creates a simple single pane window construction.
    """
    # For simplicity in API (to avoid SimpleGlazing vs StandardGlazing complexity),
    # we just use a thin material with high conductivity?
    # Or use SimpleGlazing if available in Mock?
    # Let's stick to Material for generic robustness or use standard if known.
    # OpenStudio usually has SimpleGlazing.

    # We'll use a standard Material but with glass-like properties for now to be safe with the mock.
    mat = os_api.model.Material(model)
    mat.setThickness(0.006)
    mat.setConductivity(0.9) # Glass
    mat.setDensity(2500)
    mat.setSpecificHeat(840)

    construction = os_api.model.Construction(model)
    construction.insertLayer(0, mat)
    return construction

def add_outputs(os_api, model):
    """
    Adds standard output variables to the model.
    """
    # Zone Mean Air Temperature
    var1 = os_api.model.OutputVariable("Zone Mean Air Temperature", model)
    var1.setReportingFrequency("Hourly")

    # Zone Ideal Loads Zone Total Heating Energy
    var2 = os_api.model.OutputVariable("Zone Ideal Loads Zone Total Heating Energy", model)
    var2.setReportingFrequency("Hourly")

    # Zone Ideal Loads Zone Total Cooling Energy
    var3 = os_api.model.OutputVariable("Zone Ideal Loads Zone Total Cooling Energy", model)
    var3.setReportingFrequency("Hourly")

    # Surface Outside Face Incident Solar Radiation Amount per Area
    var4 = os_api.model.OutputVariable("Surface Outside Face Incident Solar Radiation Amount per Area", model)
    var4.setReportingFrequency("Hourly")

def create_shoebox_model(
    width: float = 10.0,
    length: float = 10.0,
    height: float = 3.5,
    wwr: float = 0.4,
    orientation_degrees: float = 0.0,
    insulation_r_value: float = None # Placeholder for future material logic
):
    """
    Generates an OpenStudio Model (.osm) for a simple rectangular building.
    """
    os_api = ensure_openstudio()
    model = os_api.model.Model()

    # Create Constructions
    default_const = create_default_construction(os_api, model)
    win_const = create_window_construction(os_api, model)

    # Create Thermal Zone
    thermal_zone = os_api.model.ThermalZone(model)

    # Enable Ideal Air Loads for simple load calculation
    # (Mock doesn't need to support this method, but real OS does)
    try:
        thermal_zone.setUseIdealAirLoads(True)
    except AttributeError:
        pass # Mock object might not have this method yet, or old OS version

    # Create Space
    space = os_api.model.Space(model)
    space.setThermalZone(thermal_zone)

    # Define geometry (Vertices)
    rad = math.radians(orientation_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    def rotate(x, y):
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    p0 = rotate(0, 0)
    p1 = rotate(width, 0)
    p2 = rotate(width, length)
    p3 = rotate(0, length)

    # Floor
    floor_pts = [
        os_api.Point3d(p2[0], p2[1], 0),
        os_api.Point3d(p1[0], p1[1], 0),
        os_api.Point3d(p0[0], p0[1], 0),
        os_api.Point3d(p3[0], p3[1], 0),
    ]
    floor = os_api.model.Surface(os_api.Point3dVector(floor_pts), model)
    floor.setSpace(space)
    floor.setSurfaceType("Floor")
    floor.setOutsideBoundaryCondition("Ground")
    floor.setConstruction(default_const)

    # Roof
    roof_pts = [
        os_api.Point3d(p0[0], p0[1], height),
        os_api.Point3d(p1[0], p1[1], height),
        os_api.Point3d(p2[0], p2[1], height),
        os_api.Point3d(p3[0], p3[1], height),
    ]
    roof = os_api.model.Surface(os_api.Point3dVector(roof_pts), model)
    roof.setSpace(space)
    roof.setSurfaceType("RoofCeiling")
    roof.setOutsideBoundaryCondition("Outdoors")
    roof.setSunExposure("SunExposed")
    roof.setWindExposure("WindExposed")
    roof.setConstruction(default_const)

    # Walls
    make_wall(os_api, model, space, p0, p1, height, wwr, default_const, win_const)
    make_wall(os_api, model, space, p1, p2, height, wwr, default_const, win_const)
    make_wall(os_api, model, space, p2, p3, height, wwr, default_const, win_const)
    make_wall(os_api, model, space, p3, p0, height, wwr, default_const, win_const)

    # Add Output Variables
    add_outputs(os_api, model)

    return model

def make_wall(os_api, model, space, p_start, p_end, height, wwr, wall_const, win_const):
    w_pts = [
        os_api.Point3d(p_end[0], p_end[1], 0),
        os_api.Point3d(p_end[0], p_end[1], height),
        os_api.Point3d(p_start[0], p_start[1], height),
        os_api.Point3d(p_start[0], p_start[1], 0),
    ]

    wall = os_api.model.Surface(os_api.Point3dVector(w_pts), model)
    wall.setSpace(space)
    wall.setSurfaceType("Wall")
    wall.setOutsideBoundaryCondition("Outdoors")
    wall.setSunExposure("SunExposed")
    wall.setWindExposure("WindExposed")
    wall.setConstruction(wall_const)

    if wwr > 0.01:
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        length = math.sqrt(dx*dx + dy*dy)

        ratio = math.sqrt(wwr)
        win_w = length * ratio
        win_h = height * ratio

        offset_x = (length - win_w) / 2
        offset_z = (height - win_h) / 2

        ux = dx / length
        uy = dy / length

        w_p0_x = p_start[0] + offset_x * ux
        w_p0_y = p_start[1] + offset_x * uy
        w_p0_z = offset_z

        w_p1_x = p_start[0] + (length - offset_x) * ux
        w_p1_y = p_start[1] + (length - offset_x) * uy

        w_top_z = offset_z + win_h

        win_pts = [
            os_api.Point3d(w_p1_x, w_p1_y, w_p0_z),
            os_api.Point3d(w_p1_x, w_p1_y, w_top_z),
            os_api.Point3d(w_p0_x, w_p0_y, w_top_z),
            os_api.Point3d(w_p0_x, w_p0_y, w_p0_z),
        ]

        sub = os_api.model.SubSurface(os_api.Point3dVector(win_pts), model)
        sub.setSurface(wall)
        sub.setConstruction(win_const)
