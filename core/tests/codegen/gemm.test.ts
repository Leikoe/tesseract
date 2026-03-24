/**
 * GEMM (General Matrix Multiply) kernel test.
 *
 * CuTile Python equivalent:
 *   @ct.kernel
 *   def matmul(A, B, C, tm, tn, tk):
 *       bidx, bidy = ...
 *       acc = ct.full((tm, tn), 0, dtype=ct.float32)
 *       for k in range(num_tiles_k):
 *           a = ct.load(A, (bidx, k), (tm, tk))
 *           b = ct.load(B, (k, bidy), (tk, tn))
 *           acc = ct.mma(a, b, acc)
 *       ct.store(C, (bidx, bidy), acc)
 *
 * Tests: for loop, mmaf, make_tensor_view, make_partition_view,
 *        load_view_tko, store_view_tko, get_tile_block_id, constants.
 */

import { describe, it, expect } from 'vitest';
import { tile, ptr, tensorView, partitionView } from '../../src/types.js';
import { emitMLIR } from '../../src/codegen.js';
import { GraphBuilder, ForBodyBuilder } from '../helpers.js';

describe('gemm kernel', () => {
  it('generates valid MLIR for a tiled GEMM', () => {
    const b = new GraphBuilder('gemm', 'kernel');

    // Entry args
    const SCALAR_I32 = tile([], 'i32');
    const SCALAR_I64 = tile([], 'i64');
    const aPtr = b.entryArg('A_ptr', tile([], ptr('f16')));
    const bPtr = b.entryArg('B_ptr', tile([], ptr('f16')));
    const cPtr = b.entryArg('C_ptr', tile([], ptr('f32')));
    const M = b.entryArg('M', SCALAR_I64);
    const N = b.entryArg('N', SCALAR_I64);
    const K = b.entryArg('K', SCALAR_I64);
    const numKTiles = b.entryArg('num_k_tiles', SCALAR_I32);

    // Block IDs
    const { bid0, bid1 } = b.getTileBlockId();

    // Constants
    const c0 = b.constant('0', 'i32');
    const c1 = b.constant('1', 'i32');
    const c1_i64 = b.constant('1', 'i64');
    const accInit = b.constant('0.0', 'f32', [128, 128]);

    // Create tensor views (2D, dynamic shape, stride1=1)
    const aView = b.makeTensorView2d(aPtr, M, K, K, c1_i64, 'f16', { staticStrides: '?,1' });
    const bView = b.makeTensorView2d(bPtr, K, N, N, c1_i64, 'f16', { staticStrides: '?,1' });
    const cView = b.makeTensorView2d(cPtr, M, N, N, c1_i64, 'f32', { staticStrides: '?,1' });

    // Partition views
    const aPV = b.makePartitionView(aView, [128, 64]);
    const bPV = b.makePartitionView(bView, [64, 128]);
    const cPV = b.makePartitionView(cView, [128, 128]);

    // Types for carried values
    const accType = tile([128, 128], 'f32');
    const pvAType = partitionView([128, 64], tensorView([null, null], 'f16', [null, 1]));
    const pvBType = partitionView([64, 128], tensorView([null, null], 'f16', [null, 1]));

    // K-loop
    const [accFinal] = b.forLoop(
      c0, numKTiles, c1,
      [{ name: 'acc', init: accInit, type: accType }],
      'k', SCALAR_I32,
      (body) => {
        const k = body.iterVar('k', SCALAR_I32);
        const acc = body.carryIn('acc', 0, accType);

        // Outer refs
        const aPVRef = body.outerRef(aPV, pvAType);
        const bPVRef = body.outerRef(bPV, pvBType);
        const bid0Ref = body.outerRef(bid0, SCALAR_I32);
        const bid1Ref = body.outerRef(bid1, SCALAR_I32);

        // Load tiles
        const { data: aTile } = body.loadView2d(aPVRef, bid0Ref, k);
        const { data: bTile } = body.loadView2d(bPVRef, k, bid1Ref);

        // MMA
        const newAcc = body.mmaf(aTile, bTile, acc);

        return [newAcc];
      },
    );

    // Store result
    b.storeView2d(accFinal, cPV, bid0, bid1);

    const mlir = emitMLIR(b.build());

    // -- Structural assertions --
    expect(mlir).toContain('cuda_tile.module @gemm');
    expect(mlir).toContain('entry @kernel(');
    expect(mlir).toContain('%A_ptr : tile<ptr<f16>>');

    // Grid intrinsics
    expect(mlir).toContain('get_tile_block_id : tile<i32>');

    // Tensor views
    expect(mlir).toContain('make_tensor_view');
    expect(mlir).toContain('make_partition_view');

    // For loop
    expect(mlir).toContain('for %k in (');
    expect(mlir).toContain('iter_values(%acc =');
    expect(mlir).toContain('-> (tile<128x128xf32>)');

    // MMA inside loop
    expect(mlir).toContain('mmaf');
    expect(mlir).toContain('tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>');

    // Continue
    expect(mlir).toContain('continue');
    expect(mlir).toContain(': tile<128x128xf32>');

    // Store
    expect(mlir).toContain('store_view_tko weak');

    // Return
    expect(mlir).toContain('return');

    // Print for debugging
    // console.log(mlir);
  });
});
