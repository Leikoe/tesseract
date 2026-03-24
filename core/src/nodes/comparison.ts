/**
 * Comparison nodes: cmpi, cmpf.
 *
 * cmpi predicates: equal, not_equal, less_than, less_than_or_equal,
 *                  greater_than, greater_than_or_equal
 * cmpi signedness: signed, unsigned
 *
 * cmpf predicates: same as cmpi
 * cmpf ordering:   ordered, unordered
 */

import { formatType, type TileIRType, type TileType } from '../types.js';
import type { NodeDefinition, EmitContext } from './types.js';

const PREDICATES = [
  'equal',
  'not_equal',
  'less_than',
  'less_than_or_equal',
  'greater_than',
  'greater_than_or_equal',
];

function resolveComparisonOutput(
  inputTypes: (TileIRType | null)[],
): (TileIRType | null)[] {
  const input = inputTypes[0] as TileType | null;
  if (!input) return [null];
  return [{ kind: 'tile', shape: input.shape, element: 'i1' } as TileType];
}

export const cmpi: NodeDefinition = {
  type: 'cmpi',
  category: 'comparison',
  label: 'Compare (int)',
  inputs: [
    { id: 'lhs', name: 'lhs' },
    { id: 'rhs', name: 'rhs' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [
    {
      id: 'predicate',
      name: 'Predicate',
      type: 'enum',
      options: PREDICATES,
      required: true,
    },
    {
      id: 'signedness',
      name: 'Signedness',
      type: 'enum',
      options: ['signed', 'unsigned'],
      default: 'signed',
      required: true,
    },
  ],
  resolveOutputTypes: resolveComparisonOutput,
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = cmpi ${ctx.params.predicate} ${ctx.inputNames[0]}, ${ctx.inputNames[1]}, ${ctx.params.signedness} : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

export const cmpf: NodeDefinition = {
  type: 'cmpf',
  category: 'comparison',
  label: 'Compare (float)',
  inputs: [
    { id: 'lhs', name: 'lhs' },
    { id: 'rhs', name: 'rhs' },
  ],
  outputs: [{ id: 'result', name: 'result' }],
  params: [
    {
      id: 'predicate',
      name: 'Predicate',
      type: 'enum',
      options: PREDICATES,
      required: true,
    },
    {
      id: 'ordering',
      name: 'Ordering',
      type: 'enum',
      options: ['ordered', 'unordered'],
      default: 'ordered',
      required: true,
    },
  ],
  resolveOutputTypes: resolveComparisonOutput,
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = cmpf ${ctx.params.predicate} ${ctx.params.ordering} ${ctx.inputNames[0]}, ${ctx.inputNames[1]} : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

export const comparisonNodes: NodeDefinition[] = [cmpi, cmpf];
