const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const zlib = require('node:zlib');

function zeros(count) {
  return Array.from({ length: count }, () => 0);
}

function weeklyZeros() {
  return Array.from({ length: 7 }, () => zeros(24));
}

function layer(name, conductivity, density, specificHeat, thickness) {
  return {
    name,
    conductivity,
    density,
    specific_heat: specificHeat,
    thickness,
    emissivity: 0.9,
    absorptance: 0.7,
  };
}

function construction(name, layers) {
  return { name, layers, window: null };
}

function schedule(name) {
  return {
    name,
    schedule_type: 'Weekly',
    values: { Weekly: weeklyZeros() },
  };
}

function constantSchedule(name, value) {
  return {
    name,
    schedule_type: 'Constant',
    values: { Daily: Array.from({ length: 24 }, () => value) },
  };
}

function makeSchema() {
  const wallLayers = [
    layer('Gypsum Board', 0.16, 950, 840, 0.012),
    layer('Insulation', 0.04, 12, 840, 0.066),
  ];
  const roofLayers = [
    layer('Roof Deck', 0.14, 500, 1300, 0.019),
    layer('Roof Insulation', 0.03, 30, 1200, 0.14),
  ];
  const floorLayers = [layer('Concrete Slab', 1.4, 2200, 880, 0.1)];

  return {
    version: 'V1',
    metadata: {
      name: 'NAPI Interop Test Building',
      description: '',
      author: null,
      created_at: null,
      schema_version: 'V1',
    },
    geometry: {
      zones: [
        { name: 'Office Zone', floor_area: 48, volume: 129.6, height: 2.7 },
        { name: 'Lab Zone', floor_area: 24, volume: 64.8, height: 2.7 },
      ],
      total_floor_area: 72,
      total_volume: 194.4,
      number_of_floors: 1,
      floor_height: 2.7,
    },
    constructions: {
      wall: construction('Wall Assembly', wallLayers),
      roof: construction('Roof Assembly', roofLayers),
      floor: construction('Floor Assembly', floorLayers),
      interzone: null,
    },
    schedules: {
      occupancy: schedule('Occupancy'),
      lighting: schedule('Lighting'),
      hvac: {
        heating: constantSchedule('Heating', 20),
        cooling: constantSchedule('Cooling', 24),
      },
      infiltration: null,
    },
    weather: { type: 'tmy', location: '39.739, -104.984' },
    controls: {
      zone_control: {
        heating_setpoint: 20,
        cooling_setpoint: 24,
        deadband_tolerance: 0.5,
        heating_capacity: 100000,
        cooling_capacity: 100000,
      },
      global_control: null,
    },
    output: {
      eui: 0,
      total_energy: 0,
      peak_heating_load: 0,
      peak_cooling_load: 0,
      heating_energy: 0,
      cooling_energy: 0,
      zone_temperatures: null,
      hourly_zone_temperatures: null,
    },
  };
}

function getZipEntry(buffer, name) {
  let eocd = -1;
  for (let i = buffer.length - 22; i >= 0; i -= 1) {
    if (buffer.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  assert.notStrictEqual(eocd, -1);

  const centralDirSize = buffer.readUInt32LE(eocd + 12);
  const centralDirOffset = buffer.readUInt32LE(eocd + 16);
  let offset = centralDirOffset;
  const end = centralDirOffset + centralDirSize;

  while (offset < end) {
    assert.strictEqual(buffer.readUInt32LE(offset), 0x02014b50);
    const method = buffer.readUInt16LE(offset + 10);
    const compressedSize = buffer.readUInt32LE(offset + 20);
    const fileNameLength = buffer.readUInt16LE(offset + 28);
    const extraLength = buffer.readUInt16LE(offset + 30);
    const commentLength = buffer.readUInt16LE(offset + 32);
    const localHeaderOffset = buffer.readUInt32LE(offset + 42);
    const fileName = buffer.toString('utf8', offset + 46, offset + 46 + fileNameLength);

    if (fileName === name) {
      assert.strictEqual(buffer.readUInt32LE(localHeaderOffset), 0x04034b50);
      const localNameLength = buffer.readUInt16LE(localHeaderOffset + 26);
      const localExtraLength = buffer.readUInt16LE(localHeaderOffset + 28);
      const dataStart = localHeaderOffset + 30 + localNameLength + localExtraLength;
      const data = buffer.subarray(dataStart, dataStart + compressedSize);
      return method === 8 ? zlib.inflateRawSync(data).toString('utf8') : data.toString('utf8');
    }

    offset += 46 + fileNameLength + extraLength + commentLength;
  }

  throw new Error(`${name} not found`);
}

describe('interop exporters', () => {
  let OsmExporter;
  let GbXmlExporter;
  let FmiExporter;
  let tmpDir;

  before(() => {
    const fluxion = require('./index.js');
    OsmExporter = fluxion.OsmExporter;
    GbXmlExporter = fluxion.GbXmlExporter;
    FmiExporter = fluxion.FmiExporter;
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fluxion-napi-interop-'));
  });

  after(() => {
    if (tmpDir) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('exports OSM with lossless schema fields', () => {
    const schema = makeSchema();
    const out = path.join(tmpDir, 'building.osm');
    new OsmExporter().exportOsm(JSON.stringify(schema), out);

    const text = fs.readFileSync(out, 'utf8');
    assert.match(text, /OS:Building,/);
    assert.match(text, /OS:ThermalZone,/);
    assert.match(text, /OS:Space,/);
    assert.match(text, /OS:Construction,/);
    assert.match(text, /OS:Material,/);
    assert.match(text, /NAPI Interop Test Building/);
    assert.match(text, /Office Zone/);
    assert.match(text, /Lab Zone/);
    assert.match(text, /72, !- Floor Area/);
    assert.match(text, /1, !- Number of Floors/);
    assert.match(text, /2\.7, !- Floor Height/);
    assert.match(text, /48, !- Floor Area/);
    assert.match(text, /129\.6, !- Volume/);
    assert.match(text, /Gypsum Board/);
    assert.match(text, /Insulation/);
    assert.match(text, /Roof Deck/);
    assert.match(text, /Concrete Slab/);
  });

  it('exports gbXML from schema JSON', () => {
    const schema = makeSchema();
    const out = path.join(tmpDir, 'building.xml');
    new GbXmlExporter().exportGbXml(JSON.stringify(schema), out);

    const text = fs.readFileSync(out, 'utf8');
    assert.match(text, /<gbXML/);
    assert.match(text, /<Campus\b/);
    assert.match(text, /<Building/);
    assert.match(text, /<Space/);
    assert.match(text, /Office Zone/);
    assert.match(text, /Lab Zone/);
    assert.match(text, /<Construction/);
    assert.match(text, /<Layer/);
    assert.match(text, /<Material/);
  });

  it('exports FMI 2.0 FMU with modelDescription.xml', () => {
    const out = path.join(tmpDir, 'building.fmu');
    new FmiExporter().exportFmu(out, '[]', 3600);

    const xml = getZipEntry(fs.readFileSync(out), 'modelDescription.xml');
    assert.match(xml, /<fmiModelDescription/);
    assert.match(xml, /fmiVersion="2\.0"/);
    assert.match(xml, /<CoSimulation/);
    assert.match(xml, /<DefaultExperiment[^>]*stepSize="3600\.0"/);
    assert.match(xml, /name="outdoor_temperature"/);
    assert.match(xml, /name="zone_temperature"/);
  });
});
