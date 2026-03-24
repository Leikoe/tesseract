/**
 * Constant nodes: constant, iota.
 */

import { formatType, formatElementType, type TileIRType, type TileType, type ScalarKind } from '../types.js';
import type { NodeDefinition, EmitContext } from './types.js';

// -- constant --------------------------------------------------------------

export const constant: NodeDefinition = {
  type: 'constant',
  category: 'constants',
  label: 'Constant',
  inputs: [],
  outputs: [{ id: 'value', name: 'value' }],
  params: [
    { id: 'value', name: 'Value', type: 'string', required: true },
    { id: 'element', name: 'Element type', type: 'element_type', required: true },
    { id: 'shape', name: 'Shape', type: 'shape', required: false, default: [] },
  ],

  resolveOutputTypes(_inputTypes, params) {
    return [
      {
        kind: 'tile',
        shape: (params.shape as number[]) ?? [],
        element: params.element,
      } as TileType,
    ];
  },

  emit(ctx: EmitContext) {
    const tileType = ctx.outputTypes[0] as TileType;
    const elemStr = formatElementType(tileType.element);
    return `${ctx.outputNames[0]} = constant <${elemStr}: ${ctx.params.value}> : ${formatType(tileType)}`;
  },
};

// -- iota ------------------------------------------------------------------

export const iota: NodeDefinition = {
  type: 'iota',
  category: 'constants',
  label: 'Iota',
  inputs: [],
  outputs: [{ id: 'value', name: 'value' }],
  params: [
    { id: 'shape', name: 'Shape', type: 'shape', required: true },
    { id: 'element', name: 'Element type', type: 'element_type', required: true, default: 'i32' },
  ],

  resolveOutputTypes(_inputTypes, params) {
    return [
      {
        kind: 'tile',
        shape: params.shape as number[],
        element: params.element as ScalarKind,
      } as TileType,
    ];
  },

  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = iota : ${formatType(ctx.outputTypes[0])}`;
  },
};

export const constantNodes: NodeDefinition[] = [constant, iota];
