/**
 * Math operation nodes: exp, exp2, sqrt, rsqrt, log2, sin, cos, tanh.
 */

import { formatType } from '../types.js';
import type { NodeDefinition, EmitContext, ParamDefinition } from './types.js';

const roundingParam: ParamDefinition = {
  id: 'rounding',
  name: 'Rounding mode',
  type: 'enum',
  options: ['nearest_even', 'zero', 'negative_inf', 'positive_inf', 'approx', 'full'],
  required: false,
};

function mathUnaryOp(
  opType: string,
  label: string,
  hasRounding: boolean,
): NodeDefinition {
  return {
    type: opType,
    category: 'math',
    label,
    inputs: [{ id: 'input', name: 'input' }],
    outputs: [{ id: 'result', name: 'result' }],
    params: hasRounding ? [roundingParam] : [],
    resolveOutputTypes(inputTypes) {
      return [inputTypes[0]];
    },
    emit(ctx: EmitContext) {
      const attrs =
        hasRounding && ctx.params.rounding
          ? ` rounding<${ctx.params.rounding}>`
          : '';
      return `${ctx.outputNames[0]} = ${opType} ${ctx.inputNames[0]}${attrs} : ${formatType(ctx.inputTypes[0])}`;
    },
  };
}

export const exp = mathUnaryOp('exp', 'Exp (e^x)', false);
export const exp2 = mathUnaryOp('exp2', 'Exp2 (2^x)', false);
export const sqrt = mathUnaryOp('sqrt', 'Sqrt', true);
export const rsqrt = mathUnaryOp('rsqrt', 'RSqrt', false);
export const log = mathUnaryOp('log', 'Log (ln)', false);
export const log2 = mathUnaryOp('log2', 'Log2', false);
export const sin = mathUnaryOp('sin', 'Sin', false);
export const cos = mathUnaryOp('cos', 'Cos', false);
export const tanh = mathUnaryOp('tanh', 'Tanh', false);

export const mathNodes: NodeDefinition[] = [
  exp, exp2, sqrt, rsqrt, log, log2, sin, cos, tanh,
];
