/**
 * Convert between core Graph and xyflow Node/Edge formats.
 *
 * Also resolves output types for each node so edges can display type labels.
 */

import type { Node as FlowNode, Edge as FlowEdge } from '@xyflow/react';
import type { Graph, GraphNode, Edge } from '@core/graph.js';
import type { TileIRType } from '@core/types.js';
import { topoSortNodes } from '@core/graph.js';
import { getNodeDef } from '@core/nodes/index.js';

export interface TileIRNodeData {
  label: string;
  nodeType: string;
  category: string;
  inputs: { id: string; name: string; typeLabel?: string }[];
  outputs: { id: string; name: string; typeLabel?: string }[];
  params: { id: string; name: string; type: string; required?: boolean; default?: unknown; options?: string[] }[];
  paramValues: Record<string, unknown>;
  numCarried?: number;
}

export function graphToFlow(graph: Graph): {
  nodes: FlowNode<TileIRNodeData>[];
  edges: FlowEdge[];
} {
  const typeMap = resolveAllTypes(graph);

  const nodes = graph.nodes.map((node) => graphNodeToFlowNode(node, typeMap, graph));
  const edges = graph.edges.map((edge) => graphEdgeToFlowEdge(edge, graph, typeMap));
  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// Type resolution — forward pass to label edges
// ---------------------------------------------------------------------------

/**
 * Resolve output types for every node in the graph.
 * Returns nodeId → TileIRType[] (one per output port, including dynamic ports).
 */
function resolveAllTypes(graph: Graph): Map<string, TileIRType[]> {
  const resolved = new Map<string, TileIRType[]>();

  // Topo sort, tolerating cycles from flat for-loop update edges
  let sortEdges = graph.edges;
  const flatForIds = new Set(
    graph.nodes.filter((n) => n.type === 'for' && !n.body).map((n) => n.id),
  );
  if (flatForIds.size > 0) {
    sortEdges = sortEdges.filter(
      (e) => !(flatForIds.has(e.targetNode) && e.targetPort.startsWith('update_')),
    );
  }

  let order: string[];
  try {
    order = topoSortNodes(graph.nodes, sortEdges);
  } catch {
    // If topo sort fails, just return empty — labels won't show
    return resolved;
  }

  for (const nodeId of order) {
    const node = graph.nodes.find((n) => n.id === nodeId)!;
    const def = getNodeDef(node.type);

    // Resolve input types from upstream
    const inputTypes = def.inputs.map((portDef) => {
      const edge = graph.edges.find(
        (e) => e.targetNode === nodeId && e.targetPort === portDef.id,
      );
      if (!edge) return null;
      return getPortType(resolved, graph, edge.sourceNode, edge.sourcePort);
    });

    // For the `for` node, also resolve init_i inputs (dynamic)
    const numCarried = (node.params.numCarried as number) || 0;
    if (node.type === 'for') {
      for (let i = 0; i < numCarried; i++) {
        const edge = graph.edges.find(
          (e) => e.targetNode === nodeId && e.targetPort === `init_${i}`,
        );
        if (edge) {
          inputTypes.push(getPortType(resolved, graph, edge.sourceNode, edge.sourcePort));
        } else {
          inputTypes.push(null);
        }
      }
    }

    const outTypes = def.resolveOutputTypes(inputTypes, node.params) as TileIRType[];
    resolved.set(nodeId, outTypes);
  }

  return resolved;
}

/** Look up the type of a specific output port, handling dynamic ports. */
function getPortType(
  resolved: Map<string, TileIRType[]>,
  graph: Graph,
  nodeId: string,
  portId: string,
): TileIRType | null {
  const types = resolved.get(nodeId);
  if (!types) return null;

  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node) return null;

  const def = getNodeDef(node.type);
  let idx = def.outputs.findIndex((p) => p.id === portId);

  if (idx === -1) {
    // Dynamic ports for `for` node
    const resultMatch = portId.match(/^result_(\d+)$/);
    if (resultMatch) {
      const numCarried = (node.params.numCarried as number) || 0;
      idx = node.body
        ? Number(resultMatch[1]) // compound: types are just carryTypes
        : 1 + numCarried + Number(resultMatch[1]); // flat: [iv, current_0..N, result_0..N]
    }
    const currentMatch = portId.match(/^current_(\d+)$/);
    if (currentMatch) {
      idx = 1 + Number(currentMatch[1]);
    }
  }

  return types[idx] ?? null;
}

// ---------------------------------------------------------------------------
// Node / edge conversion
// ---------------------------------------------------------------------------

function graphNodeToFlowNode(
  node: GraphNode,
  typeMap: Map<string, TileIRType[]>,
  graph: Graph,
): FlowNode<TileIRNodeData> {
  const def = getNodeDef(node.type);
  const numCarried = node.params.numCarried as number | undefined;

  let inputs = [...def.inputs.map((p) => ({ ...p }))];
  let outputs = [...def.outputs.map((p) => ({ ...p }))];
  if (node.type === 'for' && numCarried) {
    const carryNames = ((node.params.carryNames as string) || '').split(',').filter(Boolean);
    for (let i = 0; i < numCarried; i++) {
      const label = carryNames[i] || `carry_${i}`;
      inputs.push({ id: `init_${i}`, name: `${label} init` });
      inputs.push({ id: `update_${i}`, name: `${label} ↩` });
      outputs.push({ id: `current_${i}`, name: label });
      outputs.push({ id: `result_${i}`, name: `${label} ⟶` });
    }
  }

  // Add type labels to output ports
  for (const port of outputs) {
    const type = getPortType(typeMap, graph, node.id, port.id);
    if (type) {
      (port as any).typeLabel = shortType(type);
    }
  }

  return {
    id: node.id,
    type: 'tileIRNode',
    position: node.position ?? { x: 0, y: 0 },
    data: {
      label: def.label,
      nodeType: node.type,
      category: def.category,
      inputs,
      outputs,
      params: def.params,
      paramValues: node.params,
      numCarried,
    },
  };
}

/** Compact type display for port labels. */
function shortType(t: TileIRType): string {
  switch (t.kind) {
    case 'token':
      return 'tok';
    case 'tile': {
      const shape = t.shape.length > 0 ? t.shape.join('×') + '×' : '';
      const elem = typeof t.element === 'string'
        ? t.element
        : `ptr<${t.element.pointee}>`;
      return `${shape}${elem}`;
    }
    case 'tensor_view': {
      const dims = t.shape.map((d) => d ?? '?').join('×');
      return `view<${dims}×${t.element}>`;
    }
    case 'partition_view': {
      const tile = t.tileShape.join('×');
      return `pview<${tile}>`;
    }
  }
}

function graphEdgeToFlowEdge(
  edge: Edge,
  graph: Graph,
  typeMap: Map<string, TileIRType[]>,
): FlowEdge {
  const type = getPortType(typeMap, graph, edge.sourceNode, edge.sourcePort);
  const label = type ? shortType(type) : undefined;

  return {
    id: edge.id,
    source: edge.sourceNode,
    sourceHandle: edge.sourcePort,
    target: edge.targetNode,
    targetHandle: edge.targetPort,
    label,
    labelStyle: {
      fontSize: 9,
      fill: '#585b70',
      fontFamily: 'ui-monospace, monospace',
    },
    labelBgStyle: {
      fill: '#11111b',
      fillOpacity: 0.9,
    },
    labelBgPadding: [4, 2] as [number, number],
  };
}
