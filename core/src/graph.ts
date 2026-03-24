/**
 * Graph data structure for the node editor.
 *
 * Plain JSON-serializable objects + pure functions.
 */

// -- Data types ------------------------------------------------------------

export interface SubGraph {
  nodes: GraphNode[];
  edges: Edge[];
}

export interface GraphNode {
  id: string;
  type: string; // references a NodeDefinition.type
  params: Record<string, unknown>;
  position?: { x: number; y: number }; // UI only, ignored by core
  body?: SubGraph; // For compound nodes (for, if)
}

export interface Edge {
  id: string;
  sourceNode: string;
  sourcePort: string;
  targetNode: string;
  targetPort: string;
}

export interface Graph {
  moduleName: string;
  entryName: string;
  nodes: GraphNode[];
  edges: Edge[];
}

// -- Construction ----------------------------------------------------------

let _idCounter = 0;

/** Reset the ID counter (useful for deterministic tests). */
export function resetIdCounter(start = 0): void {
  _idCounter = start;
}

function nextId(prefix: string): string {
  return `${prefix}_${_idCounter++}`;
}

export function createGraph(moduleName: string, entryName: string): Graph {
  return { moduleName, entryName, nodes: [], edges: [] };
}

export function addNode(
  graph: Graph | SubGraph,
  type: string,
  params: Record<string, unknown> = {},
  body?: SubGraph,
): string {
  const id = nextId('n');
  const node: GraphNode = { id, type, params };
  if (body) node.body = body;
  graph.nodes.push(node);
  return id;
}

export function addEdgeToSubGraph(
  sg: SubGraph,
  sourceNode: string,
  sourcePort: string,
  targetNode: string,
  targetPort: string,
): string {
  const id = nextId('e');
  sg.edges.push({ id, sourceNode, sourcePort, targetNode, targetPort });
  return id;
}

export function addEdge(
  graph: Graph,
  sourceNode: string,
  sourcePort: string,
  targetNode: string,
  targetPort: string,
): string {
  const id = nextId('e');
  graph.edges.push({ id, sourceNode, sourcePort, targetNode, targetPort });
  return id;
}

// -- Queries ---------------------------------------------------------------

export function getNode(graph: Graph, nodeId: string): GraphNode {
  const node = graph.nodes.find((n) => n.id === nodeId);
  if (!node) throw new Error(`Node not found: ${nodeId}`);
  return node;
}

export function getIncomingEdges(graph: Graph, nodeId: string): Edge[] {
  return graph.edges.filter((e) => e.targetNode === nodeId);
}

export function getOutgoingEdges(graph: Graph, nodeId: string): Edge[] {
  return graph.edges.filter((e) => e.sourceNode === nodeId);
}

/**
 * Find the edge connected to a specific input port.
 * Returns null if the port is unconnected.
 */
export function getEdgeToPort(
  graph: Graph,
  targetNode: string,
  targetPort: string,
): Edge | null {
  return (
    graph.edges.find(
      (e) => e.targetNode === targetNode && e.targetPort === targetPort,
    ) ?? null
  );
}

// -- Topological sort ------------------------------------------------------

/**
 * Kahn's algorithm.  Returns node IDs in dependency order.
 * Throws if the graph contains a cycle.
 */
export function topoSortNodes(
  nodes: GraphNode[],
  edges: Edge[],
): string[] {
  // Build adjacency & in-degree maps
  const inDegree = new Map<string, number>();
  // nodeId -> Map<targetNodeId, edgeCount> (tracks multi-edges correctly)
  const dependents = new Map<string, Map<string, number>>();

  for (const node of nodes) {
    inDegree.set(node.id, 0);
    dependents.set(node.id, new Map());
  }

  for (const edge of edges) {
    inDegree.set(edge.targetNode, (inDegree.get(edge.targetNode) ?? 0) + 1);
    const depMap = dependents.get(edge.sourceNode)!;
    depMap.set(edge.targetNode, (depMap.get(edge.targetNode) ?? 0) + 1);
  }

  // Seed with nodes that have no incoming edges
  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  // Stable: sort the initial queue by node id for determinism
  queue.sort();

  const sorted: string[] = [];
  while (queue.length > 0) {
    const current = queue.shift()!;
    sorted.push(current);
    for (const [dep, count] of [...dependents.get(current)!].sort((a, b) => a[0].localeCompare(b[0]))) {
      const newDeg = inDegree.get(dep)! - count;
      inDegree.set(dep, newDeg);
      if (newDeg === 0) queue.push(dep);
    }
  }

  if (sorted.length !== nodes.length) {
    throw new Error('Graph contains a cycle');
  }

  return sorted;
}

/** Convenience: topo-sort a full Graph. */
export function topologicalSort(graph: Graph): string[] {
  return topoSortNodes(graph.nodes, graph.edges);
}
