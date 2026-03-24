/**
 * Matrix multiply-accumulate nodes: mmaf, mmai, fma.
 *
 * MLIR syntax:
 *   %r = mmaf %a, %b, %acc : tile<MxKxf16>, tile<KxNxf16>, tile<MxNxf32>
 *   %r = mmai %a, %b, %acc signed signed : tile<MxKxi8>, tile<KxNxi8>, tile<MxNxi32>
 *   %r = fma %a, %b, %c : tile<f32>
 */

import { formatType } from '../types.js';
import type { NodeDefinition, EmitContext } from './types.js';

export const mmaf: NodeDefinition = {
  type: 'mmaf',
  category: 'mma',
  label: 'MMA (float)',
  inputs: [
    { id: 'a', name: 'A (M×K)' },
    { id: 'b', name: 'B (K×N)' },
    { id: 'acc', name: 'Accumulator (M×N)' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [],
  resolveOutputTypes(inputTypes) {
    // Result type = same as accumulator
    return [inputTypes[2]];
  },
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = mmaf ${ctx.inputNames[0]}, ${ctx.inputNames[1]}, ${ctx.inputNames[2]} : ${formatType(ctx.inputTypes[0])}, ${formatType(ctx.inputTypes[1])}, ${formatType(ctx.inputTypes[2])}`;
  },
};

export const mmai: NodeDefinition = {
  type: 'mmai',
  category: 'mma',
  label: 'MMA (int)',
  inputs: [
    { id: 'a', name: 'A (M×K)' },
    { id: 'b', name: 'B (K×N)' },
    { id: 'acc', name: 'Accumulator (M×N)' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [
    {
      id: 'aSignedness',
      name: 'A signedness',
      type: 'enum',
      options: ['signed', 'unsigned'],
      default: 'signed',
      required: true,
    },
    {
      id: 'bSignedness',
      name: 'B signedness',
      type: 'enum',
      options: ['signed', 'unsigned'],
      default: 'signed',
      required: true,
    },
  ],
  resolveOutputTypes(inputTypes) {
    return [inputTypes[2]];
  },
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = mmai ${ctx.inputNames[0]}, ${ctx.inputNames[1]}, ${ctx.inputNames[2]} ${ctx.params.aSignedness} ${ctx.params.bSignedness} : ${formatType(ctx.inputTypes[0])}, ${formatType(ctx.inputTypes[1])}, ${formatType(ctx.inputTypes[2])}`;
  },
};

export const fma: NodeDefinition = {
  type: 'fma',
  category: 'mma',
  label: 'FMA (a*b+c)',
  inputs: [
    { id: 'a', name: 'a' },
    { id: 'b', name: 'b' },
    { id: 'c', name: 'c' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [],
  resolveOutputTypes(inputTypes) {
    return [inputTypes[0]];
  },
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = fma ${ctx.inputNames[0]}, ${ctx.inputNames[1]}, ${ctx.inputNames[2]} : ${formatType(ctx.inputTypes[0])}`;
  },
};

export const mmaNodes: NodeDefinition[] = [mmaf, mmai, fma];
