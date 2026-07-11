from pathlib import Path

import pytest

import fluxion

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "osm" / "two_zone.osm"
ROOT = Path(__file__).resolve().parents[2]


def assert_close(left, right, path):
    assert left == pytest.approx(right, abs=1e-6, rel=1e-6), path


def parse_location(schema):
    values = schema["weather"]["location"].split(",")
    return tuple(float(value.strip()) for value in values)


def assert_lossless_fields(left, right):
    assert left["metadata"]["name"] == right["metadata"]["name"]

    left_geometry = left["geometry"]
    right_geometry = right["geometry"]
    assert len(left_geometry["zones"]) == len(right_geometry["zones"])
    for index, (left_zone, right_zone) in enumerate(
        zip(left_geometry["zones"], right_geometry["zones"])
    ):
        assert left_zone["name"] == right_zone["name"]
        assert_close(left_zone["floor_area"], right_zone["floor_area"], f"zone {index} floor_area")
        assert_close(left_zone["volume"], right_zone["volume"], f"zone {index} volume")
        assert_close(left_zone["height"], right_zone["height"], f"zone {index} height")

    assert_close(left_geometry["total_floor_area"], right_geometry["total_floor_area"], "total_floor_area")
    assert_close(left_geometry["total_volume"], right_geometry["total_volume"], "total_volume")
    assert left_geometry["number_of_floors"] == right_geometry["number_of_floors"]
    assert_close(left_geometry["floor_height"], right_geometry["floor_height"], "floor_height")

    for construction_name in ("wall", "roof", "floor"):
        left_layers = left["constructions"][construction_name]["layers"]
        right_layers = right["constructions"][construction_name]["layers"]
        assert len(left_layers) == len(right_layers)
        for index, (left_layer, right_layer) in enumerate(zip(left_layers, right_layers)):
            path = f"{construction_name}.layers[{index}]"
            assert left_layer["name"] == right_layer["name"]
            assert_close(left_layer["thickness"], right_layer["thickness"], f"{path}.thickness")
            assert_close(left_layer["conductivity"], right_layer["conductivity"], f"{path}.conductivity")
            assert_close(left_layer["density"], right_layer["density"], f"{path}.density")
            assert_close(left_layer["specific_heat"], right_layer["specific_heat"], f"{path}.specific_heat")

    left_lat, left_lon = parse_location(left)
    right_lat, right_lon = parse_location(right)
    assert_close(left_lat, right_lat, "weather.latitude")
    assert_close(left_lon, right_lon, "weather.longitude")


def test_osm_reader_writer_roundtrip_lossless_fields(tmp_path):
    original = fluxion.import_osm(str(FIXTURE))
    assert original["metadata"]["name"] == "Two Zone Fixture"
    assert [zone["name"] for zone in original["geometry"]["zones"]] == ["Zone A", "Zone B"]

    output = tmp_path / "roundtrip.osm"
    writer = fluxion.OsmWriter.from_schema_dict(original)
    writer.export(str(output))

    reimported = fluxion.OsmReader(str(output)).to_schema_dict()
    assert_lossless_fields(original, reimported)


def test_osm_free_function_export_roundtrip(tmp_path):
    original = fluxion.OsmReader.from_path(str(FIXTURE)).to_schema_dict()
    output = tmp_path / "free_function.osm"

    fluxion.export_osm(original, str(output))

    assert_lossless_fields(original, fluxion.import_osm(str(output)))


def test_osm_errors_raise_fluxion_error(tmp_path):
    with pytest.raises(fluxion.FluxionError):
        fluxion.import_osm(str(tmp_path / "missing.osm"))


def test_osm_type_stubs_exposed():
    stub = (ROOT / "fluxion.pyi").read_text()
    assert "class OsmReader" in stub
    assert "class OsmWriter" in stub
    assert "def import_osm(path: str) -> SchemaDict" in stub
    assert "def export_osm(schema: SchemaDict, path: str) -> None" in stub
