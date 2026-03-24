/**
 * Tensor manipulation nodes: reshape, broadcast, offset, permute, extract,
 * reduce, select, cat, scan.
 *
 * v1 implements: reshape, broadcast, offset  (needed for vector add)
 */

import {
  formatType,
  formatElementType,
  type TileIRType,
  type TileType,
  type ScalarKind,
} from '../types.js';
import type { NodeDefinition, EmitContext } from './types.js';

// -- reshape ---------------------------------------------------------------

export const reshape: NodeDefinition = {
  type: 'reshape',
  category: 'tensor',
  label: 'Reshape',
  inputs: [{ id: 'input', name: 'input' }],
  outputs: [{ id: 'output', name: 'output' }],
  params: [
    { id: 'targetShape', name: 'Target shape', type: 'shape', required: true },
  ],

  resolveOutputTypes(inputTypes, params) {
    const input = inputTypes[0] as TileType | null;
    if (!input) return [null];
    return [
      {
        kind: 'tile',
        shape: params.targetShape as number[],
        element: input.element,
      } as TileType,
    ];
  },

  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = reshape ${ctx.inputNames[0]} : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- broadcast -------------------------------------------------------------

export const broadcast: NodeDefinition = {
  type: 'broadcast',
  category: 'tensor',
  label: 'Broadcast',
  inputs: [{ id: 'input', name: 'input' }],
  outputs: [{ id: 'output', name: 'output' }],
  params: [
    {
      id: 'targetShape',
      name: 'Target shape',
      type: 'shape',
      required: true,
    },
  ],

  resolveOutputTypes(inputTypes, params) {
    const input = inputTypes[0] as TileType | null;
    if (!input) return [null];
    return [
      {
        kind: 'tile',
        shape: params.targetShape as number[],
        element: input.element,
      } as TileType,
    ];
  },

  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = broadcast ${ctx.inputNames[0]} : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- offset ----------------------------------------------------------------

export const offset: NodeDefinition = {
  type: 'offset',
  category: 'tensor',
  label: 'Offset',
  inputs: [
    { id: 'base', name: 'base' },
    { id: 'offsets', name: 'offsets' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [],

  resolveOutputTypes(inputTypes) {
    // Output type = same as the base pointer tile
    return [inputTypes[0]];
  },

  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = offset ${ctx.inputNames[0]}, ${ctx.inputNames[1]} : ${formatType(ctx.inputTypes[0])}, ${formatType(ctx.inputTypes[1])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- permute ---------------------------------------------------------------

export const permute: NodeDefinition = {
  type: 'permute',
  category: 'tensor',
  label: 'Permute',
  inputs: [{ id: 'input', name: 'input' }],
  outputs: [{ id: 'output', name: 'output' }],
  params: [
    { id: 'dims', name: 'Dimension order', type: 'shape', required: true },
  ],

  resolveOutputTypes(inputTypes, params) {
    const input = inputTypes[0] as TileType | null;
    if (!input) return [null];
    const dims = params.dims as number[];
    const newShape = dims.map((d) => input.shape[d]);
    return [{ kind: 'tile', shape: newShape, element: input.element } as TileType];
  },

  emit(ctx: EmitContext) {
    const dims = (ctx.params.dims as number[]).join(', ');
    return `${ctx.outputNames[0]} = permute ${ctx.inputNames[0]} dims=[${dims}] : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- select ----------------------------------------------------------------

export const select: NodeDefinition = {
  type: 'select',
  category: 'tensor',
  label: 'Select (where)',
  inputs: [
    { id: 'cond', name: 'condition (i1)' },
    { id: 'true_val', name: 'true value' },
    { id: 'false_val', name: 'false value' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [],

  resolveOutputTypes(inputTypes) {
    return [inputTypes[1]]; // same type as true_val
  },

  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = select ${ctx.inputNames[0]}, ${ctx.inputNames[1]}, ${ctx.inputNames[2]} : ${formatType(ctx.inputTypes[0])}, ${formatType(ctx.inputTypes[1])}`;
  },
};

// -- reduce ----------------------------------------------------------------
// Pre-defined combiner modes to avoid requiring a sub-graph for the body.

export const reduce: NodeDefinition = {
  type: 'reduce',
  category: 'tensor',
  label: 'Reduce',
  inputs: [{ id: 'input', name: 'input' }],
  outputs: [{ id: 'result', name: 'result' }],
  params: [
    { id: 'dim', name: 'Dimension', type: 'number', required: true },
    {
      id: 'mode',
      name: 'Reduction',
      type: 'enum',
      options: ['sum', 'max', 'min'],
      required: true,
    },
    { id: 'identity', name: 'Identity value', type: 'string', required: true },
  ],

  resolveOutputTypes(inputTypes, params) {
    const input = inputTypes[0] as TileType | null;
    if (!input) return [null];
    const dim = params.dim as number;
    const newShape = input.shape.filter((_, i) => i !== dim);
    // Get the scalar element type (not ptr)
    const elem = typeof input.element === 'string' ? input.element : input.element.pointee;
    return [{ kind: 'tile', shape: newShape, element: elem } as TileType];
  },

  emit(ctx: EmitContext) {
    const dim = ctx.params.dim as number;
    const mode = ctx.params.mode as string;
    const identity = ctx.params.identity as string;
    const inputType = ctx.inputTypes[0] as TileType;
    const elemStr = typeof inputType.element === 'string' ? inputType.element : inputType.element.pointee;
    const scalarType = `tile<${elemStr}>`;

    // Build the body based on combiner mode
    let body: string;
    switch (mode) {
      case 'sum':
        body = [
          `(%elem: ${scalarType}, %accum: ${scalarType}) {`,
          `  %add = addf %elem, %accum : ${scalarType}`,
          `  yield %add : ${scalarType}`,
          `}`,
        ].join('\n');
        break;
      case 'max':
        body = [
          `(%elem: ${scalarType}, %accum: ${scalarType}) {`,
          `  %cmp = cmpf greater_than ordered %elem, %accum : ${scalarType} -> tile<i1>`,
          `  %sel = select %cmp, %elem, %accum : tile<i1>, ${scalarType}`,
          `  yield %sel : ${scalarType}`,
          `}`,
        ].join('\n');
        break;
      case 'min':
        body = [
          `(%elem: ${scalarType}, %accum: ${scalarType}) {`,
          `  %cmp = cmpf less_than ordered %elem, %accum : ${scalarType} -> tile<i1>`,
          `  %sel = select %cmp, %elem, %accum : tile<i1>, ${scalarType}`,
          `  yield %sel : ${scalarType}`,
          `}`,
        ].join('\n');
        break;
      default:
        throw new Error(`Unknown reduce mode: ${mode}`);
    }

    return `${ctx.outputNames[0]} = reduce ${ctx.inputNames[0]} dim=${dim} identities=[${identity} : ${elemStr}] : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}\n${body}`;
  },
};

export const tensorNodes: NodeDefinition[] = [
  reshape, broadcast, offset, permute, select, reduce,
];
