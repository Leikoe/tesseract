/**
 * Structural nodes: entry_arg.
 *
 * entry_arg represents a kernel entry-point argument. It has no inputs and
 * one output whose type is set via decomposed parameters (element, shape,
 * isPtr).  These nodes are handled specially by the codegen — they become
 * the entry signature rather than emitting an operation.
 */

import { tile, ptr, type TileIRType, type ScalarKind } from '../types.js';
import type { NodeDefinition } from './types.js';

const ELEMENT_OPTIONS = [
  'f16', 'bf16', 'f32', 'f64',
  'i1', 'i8', 'i16', 'i32', 'i64',
];

export const entryArg: NodeDefinition = {
  type: 'entry_arg',
  category: 'kernel',
  label: 'Entry Argument',
  inputs: [],
  outputs: [{ id: 'value', name: 'value' }],
  params: [
    { id: 'name', name: 'Name', type: 'string', required: true, default: 'arg' },
    { id: 'argIndex', name: 'Index', type: 'number', required: true, default: 0 },
    { id: 'element', name: 'Element type', type: 'enum', required: true, options: ELEMENT_OPTIONS, default: 'f32' },
    { id: 'shape', name: 'Shape', type: 'string', required: false, default: '' },
    { id: 'isPtr', name: 'Pointer?', type: 'boolean', required: false, default: false },
  ],

  resolveOutputTypes(_inputTypes, params) {
    // Support legacy argType (used by builder for complex types)
    if (params.argType) {
      return [params.argType as TileIRType];
    }
    // Build type from decomposed params
    const elem = (params.element as ScalarKind) || 'f32';
    const shapeStr = (params.shape as string) || '';
    const shape = shapeStr
      ? shapeStr.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n))
      : [];
    const isPtr = params.isPtr as boolean;
    return [tile(shape, isPtr ? ptr(elem) : elem)];
  },

  // entry_arg is not emitted as an operation — handled by codegen wrapper
  emit() {
    return '';
  },
};

export const kernelNodes: NodeDefinition[] = [entryArg];
