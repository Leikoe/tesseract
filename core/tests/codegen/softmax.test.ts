/**
 * Softmax kernel test (single-row, fits in one tile).
 *
 * CuTile Python equivalent:
 *   row = ct.load(...)
 *   row_max = ct.max(row, 0)
 *   numerator = ct.exp(row - row_max)
 *   denominator = ct.sum(numerator, 0)
 *   result = numerator / denominator
 *   ct.store(...)
 *
 * Tests: reduce (max, sum), exp, subf, divf, broadcast, reshape.
 */

import { describe, it, expect } from 'vitest';
import { tile, ptr } from '../../src/types.js';
import { emitMLIR } from '../../src/codegen.js';
import { GraphBuilder, makePtrChain } from '../helpers.js';

describe('softmax kernel', () => {
  it('generates valid MLIR for a single-row softmax', () => {
    const b = new GraphBuilder('softmax', 'kernel');

    // Entry args: input and output pointers
    const aPtr = b.entryArg('in_ptr', tile([], ptr('f32')));
    const outPtr = b.entryArg('out_ptr', tile([], ptr('f32')));

    // Build pointer tiles
    const offsets = b.iota([128], 'i32');
    const inPtrs = makePtrChain(b, aPtr, offsets, 128);
    const outPtrs = makePtrChain(b, outPtr, offsets, 128);

    // Load row
    const { data: row } = b.loadPtr(inPtrs);

    // Max reduction: reduce dim=0, identity=-inf
    const rowMax = b.reduce(row, 0, 'max', '-0x1.FFFFFEp+127');

    // Broadcast max back to [128]
    const maxBcast = b.broadcast(b.reshape(rowMax, [1]), [128]);

    // exp(row - max)
    const shifted = b.subf(row, maxBcast);
    const expVals = b.exp(shifted);

    // Sum reduction
    const sum = b.reduce(expVals, 0, 'sum', '0.000000e+0');

    // Broadcast sum and divide
    const sumBcast = b.broadcast(b.reshape(sum, [1]), [128]);
    const result = b.divf(expVals, sumBcast);

    // Store
    b.storePtr(outPtrs, result);

    const mlir = emitMLIR(b.build());

    // -- Structural assertions --
    expect(mlir).toContain('cuda_tile.module @softmax');
    expect(mlir).toContain('entry @kernel(');

    // Reduce max
    expect(mlir).toContain('reduce');
    expect(mlir).toContain('dim=0');
    expect(mlir).toContain('cmpf greater_than ordered');
    expect(mlir).toContain('yield');

    // Exp
    expect(mlir).toContain('exp');

    // Reduce sum
    expect(mlir).toContain('addf %elem, %accum');

    // Division
    expect(mlir).toContain('divf');

    // Store
    expect(mlir).toContain('store_ptr_tko');

    // Return
    expect(mlir).toContain('return');

    // Print for debugging
    // console.log(mlir);
  });
});
