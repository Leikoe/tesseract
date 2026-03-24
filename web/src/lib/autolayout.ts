/**
 * Layered auto-layout for graphs.
 *
 * Places nodes left-to-right by topological depth (longest path from a root).
 * Within each column, orders nodes using a barycenter heuristic to reduce
 * edge crossings, then spaces them proportional to their height.
 */

import type { Graph } from '@core/graph.js';
import { getNodeDef } from '@core/nodes/index.js';

const COL_GAP = 380;
const ROW_GAP = 40;
const PADDING_X = 60;
const PADDING_Y = 60;

/** Estimate pixel height of a node based on port count + params. */
function estimateNodeHeight(nodeId: string, graph: Graph): number {
  const node = graph.nodes.find((n) => n.id === nodeId)!;
  const def = getNodeDef(node.type);
  const numCarried = (node.params.numCarried as number) || 0;

  let inputs = def.inputs.length;
  let outputs = def.outputs.length;
  if (node.type === 'for' && numCarried) {
    inputs += numCarried * 2;  // init + update per carry
    outputs += numCarried * 2; // current + result per carry
  }

  const ports = Math.max(inputs, outputs, 1);
  const header = 28;
  const portArea = ports * 22 + 12;
  const params = 20; // rough estimate for param line
  return header + portArea + params;
}

/**
 * Reorder a single layer by a scoring function, keeping pinned nodes in place.
 */
function reorderLayer(
  d: number,
  layers: Map<number, string[]>,
  graph: Graph,
  pinnedIds: Set<string>,
  yCenter: Map<string, number>,
  rowGap: number,
  scoreFn: (id: string) => number | null,
): void {
  const ids = layers.get(d)!;

  // Split into pinned (keep position) and free (reorder)
  const pinned: { id: string; idx: number }[] = [];
  const free: { id: string; score: number }[] = [];

  for (let i = 0; i < ids.length; i++) {
    if (pinnedIds.has(ids[i])) {
      pinned.push({ id: ids[i], idx: i });
    } else {
      const score = scoreFn(ids[i]);
      free.push({ id: ids[i], score: score ?? yCenter.get(ids[i])! });
    }
  }

  free.sort((a, b) => a.score - b.score);

  // Merge: pinned stay at their indices, free fill the gaps
  const newOrder: string[] = new Array(ids.length);
  for (const p of pinned) newOrder[p.idx] = p.id;
  let fi = 0;
  for (let i = 0; i < newOrder.length; i++) {
    if (!newOrder[i]) newOrder[i] = free[fi++].id;
  }

  // Recompute y-centers
  let y = 0;
  for (const id of newOrder) {
    const h = estimateNodeHeight(id, graph);
    yCenter.set(id, y + h / 2);
    y += h + rowGap;
  }

  layers.set(d, newOrder);
}

export function autoLayout(graph: Graph): void {
  const nodeIds = new Set(graph.nodes.map((n) => n.id));

  // Filter out update edges from flat for loops (cycles)
  const flatForIds = new Set(
    graph.nodes.filter((n) => n.type === 'for' && !n.body).map((n) => n.id),
  );
  const layoutEdges = graph.edges.filter((e) => {
    if (flatForIds.has(e.targetNode) && e.targetPort.startsWith('update_')) return false;
    if (!nodeIds.has(e.sourceNode) || !nodeIds.has(e.targetNode)) return false;
    return true;
  });

  // Build adjacency
  const dependents = new Map<string, Set<string>>();
  const inDegree = new Map<string, number>();
  for (const n of graph.nodes) {
    dependents.set(n.id, new Set());
    inDegree.set(n.id, 0);
  }
  for (const e of layoutEdges) {
    dependents.get(e.sourceNode)!.add(e.targetNode);
    inDegree.set(e.targetNode, (inDegree.get(e.targetNode) ?? 0) + 1);
  }

  // Compute depth (longest path from any root)
  const depth = new Map<string, number>();
  const queue: string[] = [];

  for (const [id, deg] of inDegree) {
    if (deg === 0) {
      depth.set(id, 0);
      queue.push(id);
    }
  }

  while (queue.length > 0) {
    const current = queue.shift()!;
    const d = depth.get(current)!;
    for (const dep of dependents.get(current)!) {
      const newDepth = Math.max(depth.get(dep) ?? 0, d + 1);
      depth.set(dep, newDepth);
      const newIn = inDegree.get(dep)! - 1;
      inDegree.set(dep, newIn);
      if (newIn === 0) queue.push(dep);
    }
  }

  for (const n of graph.nodes) {
    if (!depth.has(n.id)) depth.set(n.id, 0);
  }

  // Group nodes by depth
  const layers = new Map<number, string[]>();
  for (const n of graph.nodes) {
    const d = depth.get(n.id)!;
    if (!layers.has(d)) layers.set(d, []);
    layers.get(d)!.push(n.id);
  }

  // Pin entry_arg nodes at the top of layer 0 in argIndex order
  if (layers.has(0)) {
    const layer0 = layers.get(0)!;
    const entryArgs: string[] = [];
    const others: string[] = [];
    for (const id of layer0) {
      const node = graph.nodes.find((n) => n.id === id)!;
      if (node.type === 'entry_arg') {
        entryArgs.push(id);
      } else {
        others.push(id);
      }
    }
    entryArgs.sort((a, b) => {
      const na = graph.nodes.find((n) => n.id === a)!;
      const nb = graph.nodes.find((n) => n.id === b)!;
      return ((na.params.argIndex as number) ?? 0) - ((nb.params.argIndex as number) ?? 0);
    });
    layers.set(0, [...entryArgs, ...others]);
  }

  // Set of entry_arg IDs — these are pinned and shouldn't be reordered
  const pinnedIds = new Set(
    graph.nodes.filter((n) => n.type === 'entry_arg').map((n) => n.id),
  );

  // Build reverse adjacency for barycenter
  const parents = new Map<string, string[]>();
  for (const n of graph.nodes) parents.set(n.id, []);
  for (const e of layoutEdges) {
    parents.get(e.targetNode)!.push(e.sourceNode);
  }

  const sortedDepths = [...layers.keys()].sort((a, b) => a - b);

  // Assign initial y index
  const yIndex = new Map<string, number>();
  for (const d of sortedDepths) {
    const ids = layers.get(d)!;
    for (let i = 0; i < ids.length; i++) {
      yIndex.set(ids[i], i);
    }
  }

  // Barycenter: reorder by average parent y-center position
  const yCenter = new Map<string, number>(); // actual pixel center y

  // First pass: assign initial pixel centers
  for (const d of sortedDepths) {
    const ids = layers.get(d)!;
    let y = 0;
    for (const id of ids) {
      const h = estimateNodeHeight(id, graph);
      yCenter.set(id, y + h / 2);
      y += h + ROW_GAP;
    }
  }

  // Iterate barycenter (reorder layers to reduce edge crossings)
  for (let iter = 0; iter < 6; iter++) {
    // Forward pass: order by average parent y
    for (const d of sortedDepths) {
      reorderLayer(d, layers, graph, pinnedIds, yCenter, ROW_GAP, (id) => {
        const pars = parents.get(id)!;
        if (pars.length === 0) return null;
        return pars.reduce((s, p) => s + (yCenter.get(p) ?? 0), 0) / pars.length;
      });
    }

    // Backward pass: order by average child y
    for (let di = sortedDepths.length - 1; di >= 0; di--) {
      const d = sortedDepths[di];
      reorderLayer(d, layers, graph, pinnedIds, yCenter, ROW_GAP, (id) => {
        const children = [...(dependents.get(id) ?? [])];
        if (children.length === 0) return null;
        return children.reduce((s, c) => s + (yCenter.get(c) ?? 0), 0) / children.length;
      });
    }
  }

  // Vertically center all columns relative to the tallest one
  let maxColHeight = 0;
  const colHeights = new Map<number, number>();
  for (const d of sortedDepths) {
    const ids = layers.get(d)!;
    let h = 0;
    for (const id of ids) h += estimateNodeHeight(id, graph) + ROW_GAP;
    h -= ROW_GAP; // no gap after last
    colHeights.set(d, h);
    if (h > maxColHeight) maxColHeight = h;
  }

  // Assign final positions
  for (const d of sortedDepths) {
    const ids = layers.get(d)!;
    const colH = colHeights.get(d)!;
    const offsetY = (maxColHeight - colH) / 2;

    let y = 0;
    for (const id of ids) {
      const node = graph.nodes.find((n) => n.id === id)!;
      const h = estimateNodeHeight(id, graph);
      node.position = {
        x: PADDING_X + d * COL_GAP,
        y: PADDING_Y + offsetY + y,
      };
      y += h + ROW_GAP;
    }
  }
}
