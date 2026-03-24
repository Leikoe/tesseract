/**
 * Arithmetic operation nodes.
 *
 * Binary: addf, addi, subf, subi, mulf, muli, divf, divi
 * Unary:  negf, negi, absf, absi
 */

import { formatType } from '../types.js';
import type { NodeDefinition, EmitContext, PortDefinition, ParamDefinition } from './types.js';
import type { TileIRType } from '../types.js';

// -- Helpers to stamp out similar definitions ------------------------------

const roundingParam: ParamDefinition = {
  id: 'rounding',
  name: 'Rounding mode',
  type: 'enum',
  options: ['nearest_even', 'zero', 'negative_inf', 'positive_inf', 'approx', 'full'],
  default: 'nearest_even',
  required: false,
};

const binaryInputs: PortDefinition[] = [
  { id: 'lhs', name: 'lhs' },
  { id: 'rhs', name: 'rhs' },
];

const singleOutput: PortDefinition[] = [{ id: 'result', name: 'result' }];

function binaryPassthrough(
  inputTypes: (TileIRType | null)[],
): (TileIRType | null)[] {
  return [inputTypes[0]];
}

function unaryPassthrough(
  inputTypes: (TileIRType | null)[],
): (TileIRType | null)[] {
  return [inputTypes[0]];
}

function emitBinaryOp(opName: string, float: boolean) {
  return (ctx: EmitContext): string => {
    const attrs =
      float && ctx.params.rounding
        ? ` rounding<${ctx.params.rounding}>`
        : '';
    return `${ctx.outputNames[0]} = ${opName} ${ctx.inputNames[0]}, ${ctx.inputNames[1]}${attrs} : ${formatType(ctx.inputTypes[0])}`;
  };
}

function emitUnaryOp(opName: string, float: boolean) {
  return (ctx: EmitContext): string => {
    const attrs =
      float && ctx.params.rounding
        ? ` rounding<${ctx.params.rounding}>`
        : '';
    return `${ctx.outputNames[0]} = ${opName} ${ctx.inputNames[0]}${attrs} : ${formatType(ctx.inputTypes[0])}`;
  };
}

function binaryFloat(type: string, label: string): NodeDefinition {
  return {
    type,
    category: 'arithmetic',
    label,
    inputs: binaryInputs,
    outputs: singleOutput,
    params: [roundingParam],
    resolveOutputTypes: binaryPassthrough,
    emit: emitBinaryOp(type, true),
  };
}

function binaryInt(type: string, label: string): NodeDefinition {
  return {
    type,
    category: 'arithmetic',
    label,
    inputs: binaryInputs,
    outputs: singleOutput,
    params: [],
    resolveOutputTypes: binaryPassthrough,
    emit: emitBinaryOp(type, false),
  };
}

function unaryFloat(type: string, label: string): NodeDefinition {
  return {
    type,
    category: 'arithmetic',
    label,
    inputs: [{ id: 'input', name: 'input' }],
    outputs: singleOutput,
    params: [roundingParam],
    resolveOutputTypes: unaryPassthrough,
    emit: emitUnaryOp(type, true),
  };
}

function unaryInt(type: string, label: string): NodeDefinition {
  return {
    type,
    category: 'arithmetic',
    label,
    inputs: [{ id: 'input', name: 'input' }],
    outputs: singleOutput,
    params: [],
    resolveOutputTypes: unaryPassthrough,
    emit: emitUnaryOp(type, false),
  };
}

// -- Exported definitions --------------------------------------------------

export const addf = binaryFloat('addf', 'Add (float)');
export const subf = binaryFloat('subf', 'Sub (float)');
export const mulf = binaryFloat('mulf', 'Mul (float)');
export const divf = binaryFloat('divf', 'Div (float)');

export const addi = binaryInt('addi', 'Add (int)');
export const subi = binaryInt('subi', 'Sub (int)');
export const muli = binaryInt('muli', 'Mul (int)');
export const divi = binaryInt('divi', 'Div (int)');

export const negf = unaryFloat('negf', 'Negate (float)');
export const absf = unaryFloat('absf', 'Abs (float)');
export const negi = unaryInt('negi', 'Negate (int)');
export const absi = unaryInt('absi', 'Abs (int)');

export const arithmeticNodes: NodeDefinition[] = [
  addf, subf, mulf, divf,
  addi, subi, muli, divi,
  negf, absf, negi, absi,
];
