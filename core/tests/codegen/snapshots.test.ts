/**
 * MLIR snapshot tests for all template kernels.
 *
 * These verify exact AST equality — if codegen output changes for any reason,
 * these tests will catch it. Update snapshots with `vitest run -u`.
 */

import { describe, it, expect } from 'vitest';
import { GraphBuilder, makePtrChain } from '../helpers.js';
import { tile, ptr, partitionView, tensorView } from '../../src/types.js';
import { emitMLIR } from '../../src/codegen.js';

// ---------------------------------------------------------------------------
// Vector Add
// ---------------------------------------------------------------------------

function buildVecAdd() {
  const b = new GraphBuilder('vecadd', 'kernel');

  const aPtr = b.entryArg('a_ptr', tile([], ptr('f32')));
  const bPtr = b.entryArg('b_ptr', tile([], ptr('f32')));
  const outPtr = b.entryArg('out_ptr', tile([], ptr('f32')));

  const offsets = b.iota([128], 'i32');

  const aPtrs = makePtrChain(b, aPtr, offsets, 128);
  const bPtrs = makePtrChain(b, bPtr, offsets, 128);
  const outPtrs = makePtrChain(b, outPtr, offsets, 128);

  const { data: aData } = b.loadPtr(aPtrs);
  const { data: bData } = b.loadPtr(bPtrs);

  const sum = b.addf(aData, bData);

  b.storePtr(outPtrs, sum);

  return b.build();
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

function buildSoftmax() {
  const b = new GraphBuilder('softmax', 'kernel');

  const inPtr = b.entryArg('in_ptr', tile([], ptr('f32')));
  const outPtr = b.entryArg('out_ptr', tile([], ptr('f32')));

  const offsets = b.iota([128], 'i32');
  const inPtrs = makePtrChain(b, inPtr, offsets, 128);
  const outPtrs = makePtrChain(b, outPtr, offsets, 128);

  const { data: row } = b.loadPtr(inPtrs);

  const rowMax = b.reduce(row, 0, 'max', '-0x1.FFFFFEp+127');
  const maxBcast = b.broadcast(b.reshape(rowMax, [1]), [128]);

  const shifted = b.subf(row, maxBcast);
  const expVals = b.exp(shifted);

  const sum = b.reduce(expVals, 0, 'sum', '0.000000e+0');
  const sumBcast = b.broadcast(b.reshape(sum, [1]), [128]);
  const result = b.divf(expVals, sumBcast);

  b.storePtr(outPtrs, result);

  return b.build();
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

function buildGemm() {
  const b = new GraphBuilder('gemm', 'kernel');

  const SCALAR_I32 = tile([], 'i32');
  const SCALAR_I64 = tile([], 'i64');
  const aPtr = b.entryArg('A_ptr', tile([], ptr('f16')));
  const bPtr = b.entryArg('B_ptr', tile([], ptr('f16')));
  const cPtr = b.entryArg('C_ptr', tile([], ptr('f32')));
  const M = b.entryArg('M', SCALAR_I64);
  const N = b.entryArg('N', SCALAR_I64);
  const K = b.entryArg('K', SCALAR_I64);
  const numKTiles = b.entryArg('num_k_tiles', SCALAR_I32);

  const { bid0, bid1 } = b.getTileBlockId();

  const c0 = b.constant('0', 'i32');
  const c1 = b.constant('1', 'i32');
  const c1_i64 = b.constant('1', 'i64');
  const accInit = b.constant('0.0', 'f32', [128, 128]);

  const aView = b.makeTensorView2d(aPtr, M, K, K, c1_i64, 'f16', { staticStrides: '?,1' });
  const bView = b.makeTensorView2d(bPtr, K, N, N, c1_i64, 'f16', { staticStrides: '?,1' });
  const cView = b.makeTensorView2d(cPtr, M, N, N, c1_i64, 'f32', { staticStrides: '?,1' });

  const aPV = b.makePartitionView(aView, [128, 64]);
  const bPV = b.makePartitionView(bView, [64, 128]);
  const cPV = b.makePartitionView(cView, [128, 128]);

  const accType = tile([128, 128], 'f32');
  const pvAType = partitionView([128, 64], tensorView([null, null], 'f16', [null, 1]));
  const pvBType = partitionView([64, 128], tensorView([null, null], 'f16', [null, 1]));

  const [accFinal] = b.forLoop(
    c0, numKTiles, c1,
    [{ name: 'acc', init: accInit, type: accType }],
    'k', SCALAR_I32,
    (body) => {
      const k = body.iterVar('k', SCALAR_I32);
      const acc = body.carryIn('acc', 0, accType);
      const aPVRef = body.outerRef(aPV, pvAType);
      const bPVRef = body.outerRef(bPV, pvBType);
      const bid0Ref = body.outerRef(bid0, SCALAR_I32);
      const bid1Ref = body.outerRef(bid1, SCALAR_I32);

      const { data: aTile } = body.loadView2d(aPVRef, bid0Ref, k);
      const { data: bTile } = body.loadView2d(bPVRef, k, bid1Ref);
      const newAcc = body.mmaf(aTile, bTile, acc);
      return [newAcc];
    },
  );

  b.storeView2d(accFinal, cPV, bid0, bid1);

  return b.build();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('MLIR snapshot tests', () => {
  it('vecadd', () => {
    expect(emitMLIR(buildVecAdd())).toMatchInlineSnapshot(`
      "cuda_tile.module @vecadd {
          entry @kernel(%a_ptr : tile<ptr<f32>>, %b_ptr : tile<ptr<f32>>, %out_ptr : tile<ptr<f32>>) {
              %0 = iota : tile<128xi32>
              %1 = reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
              %2 = reshape %b_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
              %3 = reshape %out_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
              %4 = broadcast %1 : tile<1xptr<f32>> -> tile<128xptr<f32>>
              %5 = broadcast %2 : tile<1xptr<f32>> -> tile<128xptr<f32>>
              %6 = broadcast %3 : tile<1xptr<f32>> -> tile<128xptr<f32>>
              %7 = offset %4, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
              %8 = offset %5, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
              %9 = offset %6, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
              %10, %11 = load_ptr_tko weak %7 : tile<128xptr<f32>> -> tile<128xf32>, token
              %12, %13 = load_ptr_tko weak %8 : tile<128xptr<f32>> -> tile<128xf32>, token
              %14 = addf %10, %12 rounding<nearest_even> : tile<128xf32>
              %15 = store_ptr_tko weak %9, %14 : tile<128xptr<f32>>, tile<128xf32> -> token
              return
          }
      }"
    `);
  });

  it('softmax', () => {
    expect(emitMLIR(buildSoftmax())).toMatchInlineSnapshot(`
      "cuda_tile.module @softmax {
          entry @kernel(%in_ptr : tile<ptr<f32>>, %out_ptr : tile<ptr<f32>>) {
              %0 = iota : tile<128xi32>
              %1 = reshape %in_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
              %2 = reshape %out_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
              %3 = broadcast %1 : tile<1xptr<f32>> -> tile<128xptr<f32>>
              %4 = broadcast %2 : tile<1xptr<f32>> -> tile<128xptr<f32>>
              %5 = offset %3, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
              %6 = offset %4, %0 : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
              %7, %8 = load_ptr_tko weak %5 : tile<128xptr<f32>> -> tile<128xf32>, token
              %9 = reduce %7 dim=0 identities=[-0x1.FFFFFEp+127 : f32] : tile<128xf32> -> tile<f32>
              (%elem: tile<f32>, %accum: tile<f32>) {
                %cmp = cmpf greater_than ordered %elem, %accum : tile<f32> -> tile<i1>
                %sel = select %cmp, %elem, %accum : tile<i1>, tile<f32>
                yield %sel : tile<f32>
              }
              %10 = reshape %9 : tile<f32> -> tile<1xf32>
              %11 = broadcast %10 : tile<1xf32> -> tile<128xf32>
              %12 = subf %7, %11 rounding<nearest_even> : tile<128xf32>
              %13 = exp %12 : tile<128xf32>
              %14 = reduce %13 dim=0 identities=[0.000000e+0 : f32] : tile<128xf32> -> tile<f32>
              (%elem: tile<f32>, %accum: tile<f32>) {
                %add = addf %elem, %accum : tile<f32>
                yield %add : tile<f32>
              }
              %15 = reshape %14 : tile<f32> -> tile<1xf32>
              %16 = broadcast %15 : tile<1xf32> -> tile<128xf32>
              %17 = divf %13, %16 rounding<nearest_even> : tile<128xf32>
              %18 = store_ptr_tko weak %6, %17 : tile<128xptr<f32>>, tile<128xf32> -> token
              return
          }
      }"
    `);
  });

  it('gemm', () => {
    expect(emitMLIR(buildGemm())).toMatchInlineSnapshot(`
      "cuda_tile.module @gemm {
          entry @kernel(%A_ptr : tile<ptr<f16>>, %B_ptr : tile<ptr<f16>>, %C_ptr : tile<ptr<f32>>, %M : tile<i64>, %N : tile<i64>, %K : tile<i64>, %num_k_tiles : tile<i32>) {
              %0 = constant <i64: 1> : tile<i64>
              %1 = constant <f32: 0.0> : tile<128x128xf32>
              %2, %3, %4 = get_tile_block_id : tile<i32>
              %5 = constant <i32: 0> : tile<i32>
              %6 = constant <i32: 1> : tile<i32>
              %7 = make_tensor_view %C_ptr, shape = [%M, %N], strides = [%N, %0] : tile<i64> -> tensor_view<?x?xf32, strides=[?,1]>
              %8 = make_tensor_view %A_ptr, shape = [%M, %K], strides = [%K, %0] : tile<i64> -> tensor_view<?x?xf16, strides=[?,1]>
              %9 = make_tensor_view %B_ptr, shape = [%K, %N], strides = [%N, %0] : tile<i64> -> tensor_view<?x?xf16, strides=[?,1]>
              %10 = make_partition_view %7 : partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>>
              %11 = make_partition_view %8 : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>>
              %12 = make_partition_view %9 : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>>
              %13 = for %k in (%5 to %num_k_tiles, step %6) : tile<i32> iter_values(%acc = %1) -> (tile<128x128xf32>) {
                  %14, %15 = load_view_tko weak %11[%2, %k] : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>>, tile<i32> -> tile<128x64xf16>, token
                  %16, %17 = load_view_tko weak %12[%k, %3] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>>, tile<i32> -> tile<64x128xf16>, token
                  %18 = mmaf %14, %16, %acc : tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
                  continue %18 : tile<128x128xf32>
              }
              %19 = store_view_tko weak %13, %10[%2, %3] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> token
              return
          }
      }"
    `);
  });
});
