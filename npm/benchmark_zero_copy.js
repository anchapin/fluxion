'use strict';

const assert = require('node:assert');
const { performance } = require('node:perf_hooks');
const { transferMatrix } = require('./index.js');

const sizes = [1024, 65536, 1048576];
const results = [];
let sink = 0;

for (const elements of sizes) {
  const matrix = new Float64Array(elements);
  matrix[0] = 1;
  matrix[elements - 1] = 2;

  let transferred = matrix;
  for (let i = 0; i < 1000; i++) {
    transferred = transferMatrix(transferred);
  }

  const transferIterations = 100000;
  const transferStart = performance.now();
  for (let i = 0; i < transferIterations; i++) {
    transferred = transferMatrix(transferred);
  }
  const transferElapsed = performance.now() - transferStart;

  assert.strictEqual(transferred, matrix);
  assert.strictEqual(transferred.buffer, matrix.buffer);
  assert.strictEqual(transferred.byteOffset, matrix.byteOffset);
  assert.strictEqual(transferred.byteLength, matrix.byteLength);

  const copyIterations = Math.max(10, Math.floor((128 * 1024 * 1024) / matrix.byteLength));
  let copied = matrix;
  const copyStart = performance.now();
  for (let i = 0; i < copyIterations; i++) {
    copied = new Float64Array(matrix);
    sink += copied[0];
  }
  const copyElapsed = performance.now() - copyStart;

  assert.notStrictEqual(copied.buffer, matrix.buffer);

  results.push({
    elements,
    bytes: matrix.byteLength,
    zeroCopy: transferred.buffer === matrix.buffer,
    copiedBytesPerTransfer: 0,
    transferNanoseconds: Math.round((transferElapsed * 1e6) / transferIterations),
    explicitCopyNanoseconds: Math.round((copyElapsed * 1e6) / copyIterations),
  });
}

console.table(results);
process.exitCode = sink < 0 ? 1 : 0;
