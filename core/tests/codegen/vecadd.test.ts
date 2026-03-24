/**
 * Vector addition kernel — the "hello world" of GPU programming.
 *
 * CuTile Python equivalent:
 *   @ct.kernel
 *   def vector_add(a, b, result):
 *       block_id = ct.bid(0)
 *       a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
 *       b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
 *       result_tile = a_tile + b_tile
 *       ct.store(result, index=(block_id,), tile=result_tile)
 *
 * In Tile IR MLIR (pointer-style, single block):
 *   - Takes three scalar pointers (a, b, out)
 *   - Generates offsets with iota
 *   - Broadcasts pointers and offsets them
 *   - Loads, adds, stores
 */

import { describe, it, expect } from 'vitest';
import { tile, ptr } from '../../src/types.js';
import { emitMLIR } from '../../src/codegen.js';
import { GraphBuilder, makePtrChain } from '../helpers.js';

describe('vecadd kernel', () => {
  it('generates valid MLIR for a vector add kernel', () => {
    const b = new GraphBuilder('vecadd', 'kernel');

    // Entry arguments: three scalar pointers to f32
    const ptrType = tile([], ptr('f32'));
    const aPtr = b.entryArg('a_ptr', ptrType);
    const bPtr = b.entryArg('b_ptr', ptrType);
    const outPtr = b.entryArg('out_ptr', ptrType);

    // Generate offsets [0, 1, 2, ..., 127]
    const offsets = b.iota([128], 'i32');

    // Build pointer tiles: reshape scalar → [1] → broadcast to [128] → offset
    const aPtrs = makePtrChain(b, aPtr, offsets, 128);
    const bPtrs = makePtrChain(b, bPtr, offsets, 128);

    // Load data
    const { data: aData } = b.loadPtr(aPtrs, 'weak');
    const { data: bData } = b.loadPtr(bPtrs, 'weak');

    // Add
    const sum = b.addf(aData, bData, 'nearest_even');

    // Store
    const outPtrs = makePtrChain(b, outPtr, offsets, 128);
    b.storePtr(outPtrs, sum, 'weak');

    const mlir = emitMLIR(b.build());

    // -- Structural assertions --
    expect(mlir).toContain('cuda_tile.module @vecadd');
    expect(mlir).toContain(
      'entry @kernel(%a_ptr : tile<ptr<f32>>, %b_ptr : tile<ptr<f32>>, %out_ptr : tile<ptr<f32>>)',
    );

    // -- Operations present --
    expect(mlir).toContain('iota : tile<128xi32>');
    expect(mlir).toContain('reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>');
    expect(mlir).toContain('broadcast');
    expect(mlir).toContain('offset');
    expect(mlir).toContain('load_ptr_tko weak');
    expect(mlir).toContain('addf');
    expect(mlir).toContain('rounding<nearest_even>');
    expect(mlir).toContain('store_ptr_tko weak');
    expect(mlir).toContain('return');

    // -- Snapshot for regression --
    // Topo sort groups nodes by dependency depth: all reshapes (depth 1) come
    // before broadcasts (depth 2), etc.  This is a valid MLIR ordering.
    expect(mlir).toBe(
      [
        'cuda_tile.module @vecadd {',
        '    entry @kernel(%a_ptr : tile<ptr<f32>>, %b_ptr : tile<ptr<f32>>, %out_ptr : tile<ptr<f32>>) {',
        '        %0 = iota : tile<128xi32>',
        '        %1 = reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>',
        '        %2 = reshape %b_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>',
        '        %3 = reshape %out_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>',
        '        %4 = broadcast %1 : tile<1xptr<f32>> -> tile<128xptr<f32>>',
        '        %5 = broadcast %2 : tile<1xptr<f32>> -> tile<128xptr<f32>>',
        '        %6 = broadcast %3 : tile<1xptr<f32>> -> tile<128xptr<f32>>',
        '        %7 = offset %4, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>',
        '        %8 = offset %5, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>',
        '        %9 = offset %6, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>',
        '        %10, %11 = load_ptr_tko weak %7 : tile<128xptr<f32>> -> tile<128xf32>, token',
        '        %12, %13 = load_ptr_tko weak %8 : tile<128xptr<f32>> -> tile<128xf32>, token',
        '        %14 = addf %10, %12 rounding<nearest_even> : tile<128xf32>',
        '        %15 = store_ptr_tko weak %9, %14 : tile<128xptr<f32>>, tile<128xf32> -> token',
        '        return',
        '    }',
        '}',
      ].join('\n'),
    );
  });
});
