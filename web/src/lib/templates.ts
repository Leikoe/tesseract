/**
 * Pre-built kernel templates for the editor.
 *
 * Each template builds a complete Graph; auto-layout handles positioning.
 */

import type { Graph } from '@core/graph.js';
import { GraphBuilder, makePtrChain } from '@core/builder.js';
import { tile, ptr } from '@core/types.js';
import { autoLayout } from './autolayout.js';

export interface Template {
  id: string;
  name: string;
  description: string;
  build: () => Graph;
}

function withLayout(graph: Graph): Graph {
  autoLayout(graph);
  return graph;
}

// ---------------------------------------------------------------------------
// Vector Add
// ---------------------------------------------------------------------------

function buildVecAdd(): Graph {
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

  return withLayout(b.build());
}

// ---------------------------------------------------------------------------
// Softmax (single-row)
// ---------------------------------------------------------------------------

function buildSoftmax(): Graph {
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

  return withLayout(b.build());
}

// ---------------------------------------------------------------------------
// GEMM (tiled matrix multiply)
// ---------------------------------------------------------------------------

function buildGemm(): Graph {
  const b = new GraphBuilder('gemm', 'kernel');

  const aPtr = b.entryArg('A_ptr', tile([], ptr('f16')));
  const bPtr = b.entryArg('B_ptr', tile([], ptr('f16')));
  const cPtr = b.entryArg('C_ptr', tile([], ptr('f32')));
  const M = b.entryArg('M', tile([], 'i64'));
  const N = b.entryArg('N', tile([], 'i64'));
  const K = b.entryArg('K', tile([], 'i64'));
  const numKTiles = b.entryArg('num_k_tiles', tile([], 'i32'));

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

  // For loop: for k in 0..numKTiles step 1, carrying acc
  const loop = b.forCreate(c0, numKTiles, c1, [{ name: 'acc', init: accInit }], 'k');

  // Loop body: load tiles using k, MMA into accumulator
  const { data: aTile } = b.loadView2d(aPV, bid0, loop.iterVar);
  const { data: bTile } = b.loadView2d(bPV, loop.iterVar, bid1);
  const newAcc = b.mmaf(aTile, bTile, loop.current[0]);

  // Looping edge: MMA result → for.update_0
  b.connectForUpdate(loop.nodeId, 0, newAcc);

  // After loop: store final result
  b.storeView2d(loop.result[0], cPV, bid0, bid1);

  return withLayout(b.build());
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const templates: Template[] = [
  {
    id: 'blank',
    name: 'Blank Kernel',
    description: 'Empty canvas to start from scratch',
    build: () => new GraphBuilder('module', 'kernel').build(),
  },
  {
    id: 'vecadd',
    name: 'Vector Add',
    description: 'Element-wise addition of two f32 vectors (128 elements)',
    build: buildVecAdd,
  },
  {
    id: 'softmax',
    name: 'Softmax',
    description: 'Single-row softmax: max, exp, sum, divide (128 elements)',
    build: buildSoftmax,
  },
  {
    id: 'gemm',
    name: 'GEMM (Matrix Multiply)',
    description: 'Tiled f16 matrix multiply with f32 accumulator (128x128 tiles)',
    build: buildGemm,
  },
];

export function getTemplate(id: string): Template | undefined {
  return templates.find((t) => t.id === id);
}
