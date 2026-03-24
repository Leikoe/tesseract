/**
 * Node registry — central lookup for all node definitions.
 */

import type { NodeDefinition } from './types.js';
import { kernelNodes } from './kernel.js';
import { constantNodes } from './constants.js';
import { arithmeticNodes } from './arithmetic.js';
import { tensorNodes } from './tensor.js';
import { memoryNodes } from './memory.js';
import { mathNodes } from './math.js';
import { comparisonNodes } from './comparison.js';
import { conversionNodes } from './conversion.js';
import { mmaNodes } from './mma.js';
import { intrinsicNodes } from './intrinsics.js';
import { controlNodes } from './control.js';

export type { NodeDefinition, PortDefinition, ParamDefinition, EmitContext } from './types.js';

const allNodes: NodeDefinition[] = [
  ...kernelNodes,
  ...constantNodes,
  ...arithmeticNodes,
  ...tensorNodes,
  ...memoryNodes,
  ...mathNodes,
  ...comparisonNodes,
  ...conversionNodes,
  ...mmaNodes,
  ...intrinsicNodes,
  ...controlNodes,
];

const registry = new Map<string, NodeDefinition>();
for (const node of allNodes) {
  registry.set(node.type, node);
}

export function getNodeDef(type: string): NodeDefinition {
  const def = registry.get(type);
  if (!def) throw new Error(`Unknown node type: ${type}`);
  return def;
}

export function getAllNodeDefs(): NodeDefinition[] {
  return allNodes;
}

export function getNodeDefsByCategory(): Map<string, NodeDefinition[]> {
  const byCategory = new Map<string, NodeDefinition[]>();
  for (const def of allNodes) {
    const list = byCategory.get(def.category) ?? [];
    list.push(def);
    byCategory.set(def.category, list);
  }
  return byCategory;
}
