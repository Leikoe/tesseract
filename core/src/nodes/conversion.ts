/**
 * Type conversion nodes: ftof, itof, ftoi, exti, trunci.
 *
 * MLIR syntax:
 *   %r = ftof %x : tile<f16> -> tile<f32>
 *   %r = itof %x : tile<i32> -> tile<f32>
 *   %r = ftoi %x : tile<f32> -> tile<i32>
 */

import { formatType, type TileIRType, type TileType, type ScalarKind } from '../types.js';
import type { NodeDefinition, EmitContext } from './types.js';

function conversionNode(
  opType: string,
  label: string,
  category: string,
): NodeDefinition {
  return {
    type: opType,
    category,
    label,
    inputs: [{ id: 'input', name: 'input' }],
    outputs: [{ id: 'result', name: 'result' }],
    params: [
      {
        id: 'targetElement',
        name: 'Target element type',
        type: 'element_type',
        required: true,
      },
    ],
    resolveOutputTypes(inputTypes, params) {
      const input = inputTypes[0] as TileType | null;
      if (!input) return [null];
      return [
        {
          kind: 'tile',
          shape: input.shape,
          element: params.targetElement as ScalarKind,
        } as TileType,
      ];
    },
    emit(ctx: EmitContext) {
      return `${ctx.outputNames[0]} = ${opType} ${ctx.inputNames[0]} : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}`;
    },
  };
}

export const ftof = conversionNode('ftof', 'Float → Float', 'conversion');
export const itof = conversionNode('itof', 'Int → Float', 'conversion');
export const ftoi = conversionNode('ftoi', 'Float → Int', 'conversion');
export const exti = conversionNode('exti', 'Extend Int', 'conversion');
export const trunci = conversionNode('trunci', 'Truncate Int', 'conversion');

export const conversionNodes: NodeDefinition[] = [ftof, itof, ftoi, exti, trunci];
