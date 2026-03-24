/**
 * Control flow nodes.
 *
 * The `for` node operates in two modes:
 *
 * 1. **Flat mode** (visual editor): The `for` node sits in the flat graph.
 *    Its `iter_var` and `current_i` outputs feed into body nodes.  Body nodes
 *    feed back into `update_i` inputs (looping edges).  After the loop, use
 *    `result_i` outputs.  The codegen detects body nodes by tracing forward
 *    from `iter_var`/`current_i` and breaks the cycle by removing `update_i`
 *    edges before topo sort.
 *
 *    Static inputs:  lower, upper, step
 *    Dynamic inputs: init_0..init_N   (initial carry values, from outside)
 *                    update_0..update_N (looping edges, from body)
 *    Static outputs: iter_var          (loop index, used in body)
 *    Dynamic outputs: current_0..current_N (carry value inside body)
 *                     result_0..result_N   (final value after loop)
 *
 * 2. **Compound mode** (builder API): The `for` node has a `body` SubGraph
 *    with structural nodes (for_iter_var, for_carry_in, etc.).  This is the
 *    original model used by the programmatic GraphBuilder.
 *
 * The codegen checks `node.body` to decide which mode to use.
 */

import type { TileIRType } from '../types.js';
import type { NodeDefinition } from './types.js';

// ---------------------------------------------------------------------------
// Compound-mode structural nodes (used inside for-loop body sub-graphs)
// ---------------------------------------------------------------------------

export const forIterVar: NodeDefinition = {
  type: 'for_iter_var',
  category: 'control',
  label: 'Iteration Variable',
  inputs: [],
  outputs: [{ id: 'value', name: 'value' }],
  params: [
    { id: 'name', name: 'Variable name', type: 'string', required: true },
    { id: 'varType', name: 'Type', type: 'string', required: true },
  ],
  resolveOutputTypes(_inputTypes, params) {
    return [params.varType as TileIRType];
  },
  emit() {
    return '';
  },
};

export const forCarryIn: NodeDefinition = {
  type: 'for_carry_in',
  category: 'control',
  label: 'Carry Input',
  inputs: [],
  outputs: [{ id: 'value', name: 'value' }],
  params: [
    { id: 'name', name: 'Variable name', type: 'string', required: true },
    { id: 'index', name: 'Carry index', type: 'number', required: true },
    { id: 'carryType', name: 'Type', type: 'string', required: true },
  ],
  resolveOutputTypes(_inputTypes, params) {
    return [params.carryType as TileIRType];
  },
  emit() {
    return '';
  },
};

export const forCarryOut: NodeDefinition = {
  type: 'for_carry_out',
  category: 'control',
  label: 'Continue (carry out)',
  inputs: [],
  outputs: [],
  params: [
    { id: 'numValues', name: 'Number of values', type: 'number', required: true },
  ],
  resolveOutputTypes() {
    return [];
  },
  emit() {
    return '';
  },
};

export const forOuterRef: NodeDefinition = {
  type: 'for_outer_ref',
  category: 'control',
  label: 'Outer Reference',
  inputs: [],
  outputs: [{ id: 'value', name: 'value' }],
  params: [
    { id: 'outerNodeId', name: 'Outer node ID', type: 'string', required: true },
    { id: 'outerPortId', name: 'Outer port ID', type: 'string', required: true },
    { id: 'refType', name: 'Type', type: 'string', required: true },
  ],
  resolveOutputTypes(_inputTypes, params) {
    return [params.refType as TileIRType];
  },
  emit() {
    return '';
  },
};

// ---------------------------------------------------------------------------
// The `for` node itself
// ---------------------------------------------------------------------------

export const forNode: NodeDefinition = {
  type: 'for',
  category: 'control',
  label: 'For Loop',
  inputs: [
    { id: 'lower', name: 'lower' },
    { id: 'upper', name: 'upper' },
    { id: 'step', name: 'step' },
    // Dynamic: init_0..init_N, update_0..update_N
  ],
  outputs: [
    { id: 'iter_var', name: 'iter_var' },
    // Dynamic: current_0..current_N, result_0..result_N
  ],
  params: [
    { id: 'numCarried', name: 'Number of carried values', type: 'number', required: true },
    { id: 'iterVarName', name: 'Iter variable name', type: 'string', required: false, default: 'i' },
    { id: 'carryNames', name: 'Carry variable names', type: 'string', required: false, default: '' },
  ],
  resolveOutputTypes(inputTypes, params) {
    // iter_var type = same as lower/step
    const ivType = inputTypes[0] ?? { kind: 'tile', shape: [], element: 'i32' };
    const numCarried = (params.numCarried as number) || 0;
    // current_i and result_i types = same as init_i (inputTypes[3 + i])
    const types: TileIRType[] = [ivType as TileIRType];
    for (let i = 0; i < numCarried; i++) {
      const initType = inputTypes[3 + i] ?? null;
      types.push(initType as TileIRType); // current_i
    }
    for (let i = 0; i < numCarried; i++) {
      const initType = inputTypes[3 + i] ?? null;
      types.push(initType as TileIRType); // result_i
    }
    return types;
  },
  emit() {
    // Handled by codegen (both flat and compound modes)
    return '';
  },
};

export const controlNodes: NodeDefinition[] = [
  forIterVar,
  forCarryIn,
  forCarryOut,
  forOuterRef,
  forNode,
];
