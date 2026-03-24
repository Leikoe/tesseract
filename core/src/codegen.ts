/**
 * MLIR code generation.
 *
 * Walks the graph in topological order and emits valid CUDA Tile IR MLIR text.
 *
 * The `for` node supports two modes:
 *   - **Compound mode**: node.body is a SubGraph with structural nodes
 *   - **Flat mode**: looping edges (update_i → for → current_i) in the main graph;
 *     body nodes are detected by tracing from iter_var / current_i outputs.
 */

import { formatType, type TileIRType, type TileType } from './types.js';
import {
  type Graph,
  type GraphNode,
  type SubGraph,
  type Edge,
  topoSortNodes,
  getNode,
} from './graph.js';
import { getNodeDef, type EmitContext } from './nodes/index.js';

const INDENT = '    '; // 4 spaces

// -- Flat-mode for-loop detection ------------------------------------------

interface FlatLoopInfo {
  forId: string;
  bodyNodeIds: Set<string>;
}

/**
 * Detect flat-mode for loops: `for` nodes without a body sub-graph that have
 * looping edges (update_i inputs coming from downstream body nodes).
 */
function detectFlatLoops(nodes: GraphNode[], edges: Edge[]): FlatLoopInfo[] {
  const flatForNodes = nodes.filter((n) => n.type === 'for' && !n.body);
  if (flatForNodes.length === 0) return [];

  // Build node-level forward adjacency (excluding update_i edges back to for)
  const nodeForward = new Map<string, Set<string>>();
  const forIds = new Set(flatForNodes.map((n) => n.id));
  for (const e of edges) {
    // Skip update edges (looping back to the for node)
    if (forIds.has(e.targetNode) && e.targetPort.startsWith('update_')) continue;
    if (!nodeForward.has(e.sourceNode)) nodeForward.set(e.sourceNode, new Set());
    nodeForward.get(e.sourceNode)!.add(e.targetNode);
  }

  const loops: FlatLoopInfo[] = [];

  for (const forNode of flatForNodes) {
    const numCarried = (forNode.params.numCarried as number) || 0;

    // Collect "seed" ports: iter_var, current_0..N
    const seedPorts = ['iter_var'];
    for (let i = 0; i < numCarried; i++) seedPorts.push(`current_${i}`);

    // Find direct consumers of these seed ports
    const seedConsumers = new Set<string>();
    for (const e of edges) {
      if (e.sourceNode === forNode.id && seedPorts.includes(e.sourcePort)) {
        seedConsumers.add(e.targetNode);
      }
    }

    // BFS forward from seed consumers to find all body nodes
    const bodyNodes = new Set<string>();
    const queue = [...seedConsumers];
    while (queue.length > 0) {
      const nid = queue.shift()!;
      if (bodyNodes.has(nid)) continue;
      // Don't include the for node itself
      if (nid === forNode.id) continue;
      bodyNodes.add(nid);
      const deps = nodeForward.get(nid);
      if (deps) {
        for (const dep of deps) {
          if (!bodyNodes.has(dep) && dep !== forNode.id) {
            queue.push(dep);
          }
        }
      }
    }

    loops.push({ forId: forNode.id, bodyNodeIds: bodyNodes });
  }

  return loops;
}

// -- Emitter ---------------------------------------------------------------

class MLIREmitter {
  /** "nodeId:portId" → SSA name */
  private ssaNames = new Map<string, string>();
  /** nodeId → resolved output types */
  private resolvedTypes = new Map<string, TileIRType[]>();
  /** Sequential counter for SSA names */
  private counter = 0;

  emit(graph: Graph): string {
    // Detect flat-mode for loops
    const flatLoops = detectFlatLoops(graph.nodes, graph.edges);

    // For topo sort: remove update edges (looping) and add synthetic
    // dependency edges so that outer nodes feeding body nodes are processed
    // before the for node.
    let sortEdges = graph.edges;
    if (flatLoops.length > 0) {
      const forIds = new Set(flatLoops.map((l) => l.forId));

      // Remove update_i edges (they create cycles)
      sortEdges = sortEdges.filter(
        (e) => !(forIds.has(e.targetNode) && e.targetPort.startsWith('update_')),
      );

      // Add synthetic edges: outer nodes → for node
      const syntheticEdges: Edge[] = [];
      let synId = 0;
      for (const loop of flatLoops) {
        const seenDeps = new Set<string>();
        for (const e of sortEdges) {
          if (
            loop.bodyNodeIds.has(e.targetNode) &&
            !loop.bodyNodeIds.has(e.sourceNode) &&
            e.sourceNode !== loop.forId
          ) {
            if (!seenDeps.has(e.sourceNode)) {
              seenDeps.add(e.sourceNode);
              syntheticEdges.push({
                id: `_syn_${synId++}`,
                sourceNode: e.sourceNode,
                sourcePort: '_dep',
                targetNode: loop.forId,
                targetPort: `_dep_${e.sourceNode}`,
              });
            }
          }
        }
      }
      sortEdges = [...sortEdges, ...syntheticEdges];
    }

    const order = topoSortNodes(graph.nodes, sortEdges);

    // Separate entry_arg nodes from operation nodes
    const entryArgs: GraphNode[] = [];
    const opNodeIds: string[] = [];
    for (const nodeId of order) {
      const node = getNode(graph, nodeId);
      if (node.type === 'entry_arg') {
        entryArgs.push(node);
      } else {
        opNodeIds.push(nodeId);
      }
    }

    // Sort entry args by argIndex
    entryArgs.sort(
      (a, b) => (a.params.argIndex as number) - (b.params.argIndex as number),
    );

    // Register entry arg SSA names and types
    for (const arg of entryArgs) {
      const name = arg.params.name as string;
      this.ssaNames.set(`${arg.id}:value`, `%${name}`);
      const def = getNodeDef(arg.type);
      const outTypes = def.resolveOutputTypes([], arg.params) as TileIRType[];
      this.resolvedTypes.set(arg.id, outTypes);
    }

    // Emit operations
    const opLines = this.emitBlock(graph.nodes, graph.edges, opNodeIds, 2, flatLoops);

    // Build output
    const lines: string[] = [];
    lines.push(`cuda_tile.module @${graph.moduleName} {`);

    const argParts = entryArgs.map((arg) => {
      const name = arg.params.name as string;
      const outTypes = this.resolvedTypes.get(arg.id)!;
      return `%${name} : ${formatType(outTypes[0])}`;
    });
    lines.push(`${INDENT}entry @${graph.entryName}(${argParts.join(', ')}) {`);

    lines.push(...opLines);

    lines.push(`${INDENT}${INDENT}return`);
    lines.push(`${INDENT}}`);
    lines.push(`}`);

    return lines.join('\n');
  }

  /**
   * Emit a block of operations.
   */
  private emitBlock(
    allNodes: GraphNode[],
    edges: Edge[],
    nodeIds: string[],
    indentLevel: number,
    flatLoops: FlatLoopInfo[] = [],
  ): string[] {
    const lines: string[] = [];
    const prefix = INDENT.repeat(indentLevel);

    // Track which nodes belong to loop bodies (skip in main emission)
    const loopBodyNodes = new Set<string>();
    for (const loop of flatLoops) {
      for (const bodyId of loop.bodyNodeIds) {
        loopBodyNodes.add(bodyId);
      }
    }

    for (const nodeId of nodeIds) {
      if (loopBodyNodes.has(nodeId)) continue;

      const node = allNodes.find((n) => n.id === nodeId)!;
      const def = getNodeDef(node.type);

      // Skip structural nodes
      if (
        ['entry_arg', 'for_iter_var', 'for_carry_in', 'for_carry_out', 'for_outer_ref'].includes(
          node.type,
        )
      ) {
        continue;
      }

      // --- Resolve input types ---
      const inputTypes = this.resolveInputTypes(def, node, edges);

      // --- Resolve output types ---
      const outTypes = def.resolveOutputTypes(inputTypes, node.params) as TileIRType[];
      this.resolvedTypes.set(nodeId, outTypes);

      // --- Handle compound for (sub-graph) ---
      if (node.type === 'for' && node.body) {
        lines.push(...this.emitCompoundForLoop(node, edges, indentLevel));
        continue;
      }

      // --- Handle flat for (looping edges) ---
      if (node.type === 'for' && !node.body) {
        const loop = flatLoops.find((l) => l.forId === nodeId);
        if (loop) {
          lines.push(...this.emitFlatForLoop(node, loop, allNodes, edges, indentLevel));
          continue;
        }
      }

      // --- Assign SSA names for outputs ---
      const outputNames: string[] = [];
      for (const portDef of def.outputs) {
        const name = `%${this.counter++}`;
        this.ssaNames.set(`${nodeId}:${portDef.id}`, name);
        outputNames.push(name);
      }

      // --- Build emit context ---
      const inputNames = this.resolveInputNames(def, node, edges);

      const ctx: EmitContext = {
        inputNames,
        outputNames,
        inputTypes,
        outputTypes: outTypes,
        params: node.params,
      };

      const mlir = def.emit(ctx);
      if (mlir) {
        for (const line of mlir.split('\n')) {
          lines.push(prefix + line);
        }
      }
    }

    return lines;
  }

  /**
   * Emit a flat-mode for loop (looping edges in the main graph).
   */
  private emitFlatForLoop(
    forNode: GraphNode,
    loop: FlatLoopInfo,
    allNodes: GraphNode[],
    edges: Edge[],
    indentLevel: number,
  ): string[] {
    const prefix = INDENT.repeat(indentLevel);
    const numCarried = (forNode.params.numCarried as number) || 0;

    // --- Resolve for-node inputs (lower, upper, step, init_i) ---
    const findInput = (portId: string): string => {
      const edge = edges.find(
        (e) => e.targetNode === forNode.id && e.targetPort === portId,
      );
      if (!edge) return '';
      return this.ssaNames.get(`${edge.sourceNode}:${edge.sourcePort}`) ?? '';
    };

    const findInputType = (portId: string): TileIRType | null => {
      const edge = edges.find(
        (e) => e.targetNode === forNode.id && e.targetPort === portId,
      );
      if (!edge) return null;
      const srcTypes = this.resolvedTypes.get(edge.sourceNode);
      if (!srcTypes) return null;
      const srcNode = this.findNodeAnywhere(edge.sourceNode);
      if (!srcNode) return null;
      const srcDef = getNodeDef(srcNode.type);
      const portIdx = srcDef.outputs.findIndex((p) => p.id === edge.sourcePort);
      return srcTypes[portIdx] ?? null;
    };

    const lowerName = findInput('lower');
    const upperName = findInput('upper');
    const stepName = findInput('step');

    // Iter var
    const ivName = (forNode.params.iterVarName as string) || 'i';
    const ivType = findInputType('lower') ?? { kind: 'tile', shape: [], element: 'i32' } as TileIRType;

    // Assign SSA for iter_var
    this.ssaNames.set(`${forNode.id}:iter_var`, `%${ivName}`);

    // --- Process carried values ---
    const carryNamesList = ((forNode.params.carryNames as string) || '').split(',').filter(Boolean);
    const carryNames: string[] = [];
    const carryTypes: TileIRType[] = [];
    const initNames: string[] = [];

    for (let i = 0; i < numCarried; i++) {
      const name = carryNamesList[i] || `carry_${i}`;
      const initType = findInputType(`init_${i}`);
      const type = initType ?? ({ kind: 'tile', shape: [], element: 'f32' } as TileIRType);

      // Assign SSA for current_i (used inside body)
      this.ssaNames.set(`${forNode.id}:current_${i}`, `%${name}`);

      carryNames.push(name);
      carryTypes.push(type);
      initNames.push(findInput(`init_${i}`));
    }

    // Update resolvedTypes for the for node so downstream nodes can resolve
    // outTypes layout: [ivType, current_0..N, result_0..N]
    const forOutTypes: TileIRType[] = [ivType];
    for (const t of carryTypes) forOutTypes.push(t); // current_i
    for (const t of carryTypes) forOutTypes.push(t); // result_i
    this.resolvedTypes.set(forNode.id, forOutTypes);

    // --- Assign SSA for result_i (used after loop) ---
    const resultNames: string[] = [];
    for (let i = 0; i < numCarried; i++) {
      const name = `%${this.counter++}`;
      this.ssaNames.set(`${forNode.id}:result_${i}`, name);
      resultNames.push(name);
    }

    // --- Topo-sort body nodes ---
    const bodyNodes = allNodes.filter((n) => loop.bodyNodeIds.has(n.id));
    const bodyNodeIdSet = new Set(bodyNodes.map((n) => n.id));
    const bodyEdges = edges.filter(
      (e) => bodyNodeIdSet.has(e.sourceNode) && bodyNodeIdSet.has(e.targetNode),
    );
    const bodyOrder = topoSortNodes(bodyNodes, bodyEdges);

    // --- Emit body operations ---
    const bodyLines = this.emitBlock(allNodes, edges, bodyOrder, indentLevel + 1);

    // --- Find continue values (update_i sources) ---
    const continueNames: string[] = [];
    for (let i = 0; i < numCarried; i++) {
      const updateEdge = edges.find(
        (e) => e.targetNode === forNode.id && e.targetPort === `update_${i}`,
      );
      if (updateEdge) {
        continueNames.push(
          this.ssaNames.get(`${updateEdge.sourceNode}:${updateEdge.sourcePort}`) ?? '',
        );
      }
    }

    // --- Build MLIR ---
    const lines: string[] = [];
    const resultStr = resultNames.join(', ');

    let header = `${resultStr} = for %${ivName} in (${lowerName} to ${upperName}, step ${stepName}) : ${formatType(ivType)}`;
    if (carryNames.length > 0) {
      const iterValuesStr = carryNames
        .map((name, i) => `%${name} = ${initNames[i]}`)
        .join(', ');
      const resultTypeStr = carryTypes.map(formatType).join(', ');
      header += ` iter_values(${iterValuesStr}) -> (${resultTypeStr})`;
    }
    header += ' {';
    lines.push(prefix + header);

    lines.push(...bodyLines);

    if (continueNames.length > 0) {
      const continueTypeStr = carryTypes.map(formatType).join(', ');
      lines.push(
        `${prefix}${INDENT}continue ${continueNames.join(', ')} : ${continueTypeStr}`,
      );
    } else {
      lines.push(`${prefix}${INDENT}continue`);
    }

    lines.push(prefix + '}');
    return lines;
  }

  /**
   * Emit a compound for loop (sub-graph based).
   */
  private emitCompoundForLoop(
    node: GraphNode,
    outerEdges: Edge[],
    indentLevel: number,
  ): string[] {
    const prefix = INDENT.repeat(indentLevel);
    const body = node.body!;

    const iterVarNode = body.nodes.find((n) => n.type === 'for_iter_var')!;
    const carryInNodes = body.nodes
      .filter((n) => n.type === 'for_carry_in')
      .sort((a, b) => (a.params.index as number) - (b.params.index as number));
    const carryOutNode = body.nodes.find((n) => n.type === 'for_carry_out')!;

    const findOuterInput = (portId: string): string => {
      const edge = outerEdges.find(
        (e) => e.targetNode === node.id && e.targetPort === portId,
      );
      if (!edge) return '';
      return this.ssaNames.get(`${edge.sourceNode}:${edge.sourcePort}`) ?? '';
    };

    const lowerName = findOuterInput('lower');
    const upperName = findOuterInput('upper');
    const stepName = findOuterInput('step');

    const ivName = iterVarNode.params.name as string;
    this.ssaNames.set(`${iterVarNode.id}:value`, `%${ivName}`);
    const ivType = iterVarNode.params.varType as TileIRType;
    this.resolvedTypes.set(iterVarNode.id, [ivType]);

    const carryNames: string[] = [];
    const carryTypes: TileIRType[] = [];
    const initNames: string[] = [];
    for (let i = 0; i < carryInNodes.length; i++) {
      const cin = carryInNodes[i];
      const name = cin.params.name as string;
      const type = cin.params.carryType as TileIRType;
      this.ssaNames.set(`${cin.id}:value`, `%${name}`);
      this.resolvedTypes.set(cin.id, [type]);
      carryNames.push(name);
      carryTypes.push(type);
      initNames.push(findOuterInput(`init_${i}`));
    }

    for (const refNode of body.nodes.filter((n) => n.type === 'for_outer_ref')) {
      const outerNodeId = refNode.params.outerNodeId as string;
      const outerPortId = refNode.params.outerPortId as string;
      const refType = refNode.params.refType as TileIRType;
      const outerSSA = this.ssaNames.get(`${outerNodeId}:${outerPortId}`) ?? '';
      this.ssaNames.set(`${refNode.id}:value`, outerSSA);
      this.resolvedTypes.set(refNode.id, [refType]);
    }

    const resultNames: string[] = [];
    for (let i = 0; i < carryTypes.length; i++) {
      const name = `%${this.counter++}`;
      this.ssaNames.set(`${node.id}:result_${i}`, name);
      resultNames.push(name);
    }
    this.resolvedTypes.set(node.id, carryTypes);

    const bodyOrder = topoSortNodes(body.nodes, body.edges);
    const bodyOpIds = bodyOrder.filter((id) => {
      const n = body.nodes.find((bn) => bn.id === id)!;
      return !['for_iter_var', 'for_carry_in', 'for_carry_out', 'for_outer_ref'].includes(n.type);
    });

    const bodyLines = this.emitBlock(body.nodes, body.edges, bodyOpIds, indentLevel + 1);

    const numCarry = carryInNodes.length;
    const continueNames: string[] = [];
    for (let i = 0; i < numCarry; i++) {
      const edge = body.edges.find(
        (e) => e.targetNode === carryOutNode.id && e.targetPort === `value_${i}`,
      );
      if (edge) {
        continueNames.push(
          this.ssaNames.get(`${edge.sourceNode}:${edge.sourcePort}`) ?? '',
        );
      }
    }

    const lines: string[] = [];
    const resultStr = resultNames.join(', ');
    const iterValuesStr = carryNames
      .map((name, i) => `%${name} = ${initNames[i]}`)
      .join(', ');
    const resultTypeStr = carryTypes.map(formatType).join(', ');

    let header = `${resultStr} = for %${ivName} in (${lowerName} to ${upperName}, step ${stepName}) : ${formatType(ivType)}`;
    if (carryNames.length > 0) {
      header += ` iter_values(${iterValuesStr}) -> (${resultTypeStr})`;
    }
    header += ' {';
    lines.push(prefix + header);

    lines.push(...bodyLines);

    if (continueNames.length > 0) {
      const continueTypeStr = carryTypes.map(formatType).join(', ');
      lines.push(
        `${prefix}${INDENT}continue ${continueNames.join(', ')} : ${continueTypeStr}`,
      );
    } else {
      lines.push(`${prefix}${INDENT}continue`);
    }

    lines.push(prefix + '}');
    return lines;
  }

  // --- Helpers ---

  private resolveInputTypes(
    def: { inputs: { id: string }[] },
    node: GraphNode,
    edges: Edge[],
  ): (TileIRType | null)[] {
    return def.inputs.map((portDef) => {
      const edge = edges.find(
        (e) => e.targetNode === node.id && e.targetPort === portDef.id,
      );
      if (!edge) return null;
      const srcTypes = this.resolvedTypes.get(edge.sourceNode);
      if (!srcTypes) return null;
      const srcNode = this.findNodeAnywhere(edge.sourceNode);
      if (!srcNode) return null;
      const srcDef = getNodeDef(srcNode.type);
      let portIdx = srcDef.outputs.findIndex((p) => p.id === edge.sourcePort);
      // Handle dynamic outputs (e.g., for node's result_0, current_0)
      if (portIdx === -1) {
        const resultMatch = edge.sourcePort.match(/^result_(\d+)$/);
        if (resultMatch) {
          if (srcNode.type === 'for' && !srcNode.body) {
            // Flat for: outTypes = [ivType, current_0..N, result_0..N]
            const numCarried = (srcNode.params.numCarried as number) || 0;
            portIdx = 1 + numCarried + Number(resultMatch[1]);
          } else {
            // Compound for: outTypes = carryTypes (just the carry types)
            portIdx = Number(resultMatch[1]);
          }
        }
        const currentMatch = edge.sourcePort.match(/^current_(\d+)$/);
        if (currentMatch) {
          portIdx = 1 + Number(currentMatch[1]);
        }
      }
      return srcTypes[portIdx] ?? null;
    });
  }

  private resolveInputNames(
    def: { inputs: { id: string }[] },
    node: GraphNode,
    edges: Edge[],
  ): string[] {
    return def.inputs.map((portDef) => {
      const edge = edges.find(
        (e) => e.targetNode === node.id && e.targetPort === portDef.id,
      );
      if (!edge) return '';
      return this.ssaNames.get(`${edge.sourceNode}:${edge.sourcePort}`) ?? '';
    });
  }

  private findNodeAnywhere(nodeId: string): GraphNode | null {
    return this.nodeCache.get(nodeId) ?? null;
  }

  private nodeCache = new Map<string, GraphNode>();

  registerNodes(nodes: GraphNode[]): void {
    for (const n of nodes) {
      this.nodeCache.set(n.id, n);
      if (n.body) {
        this.registerNodes(n.body.nodes);
      }
    }
  }
}

/**
 * Generate MLIR text from a graph.
 */
export function emitMLIR(graph: Graph): string {
  const emitter = new MLIREmitter();
  emitter.registerNodes(graph.nodes);
  return emitter.emit(graph);
}
