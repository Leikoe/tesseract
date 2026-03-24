/**
 * GraphBuilder — fluent API for constructing graphs programmatically.
 *
 * Methods return PortRef handles that auto-wire connections when passed
 * as arguments to subsequent method calls.
 */

import {
  type Graph,
  type SubGraph,
  type TileIRType,
  type ElementType,
  type ScalarKind,
  createGraph,
  addNode,
  addEdge,
  addEdgeToSubGraph,
  resetIdCounter,
} from './index.js';

export interface PortRef {
  nodeId: string;
  portId: string;
}

// ---------------------------------------------------------------------------
// Base builder (shared ops for outer graph and for-loop body)
// ---------------------------------------------------------------------------

export class BaseBuilder {
  constructor(
    protected nodes: { nodes: any[]; edges: any[] },
    protected addEdgeFn: (
      sg: any,
      src: string,
      srcPort: string,
      dst: string,
      dstPort: string,
    ) => string,
  ) {}

  protected _addNode(type: string, params: Record<string, unknown> = {}, body?: SubGraph): string {
    return addNode(this.nodes as any, type, params, body);
  }

  protected _connect(src: PortRef, targetNode: string, targetPort: string): void {
    this.addEdgeFn(this.nodes, src.nodeId, src.portId, targetNode, targetPort);
  }

  // -- Constants -----------------------------------------------------------

  constant(value: string | number | boolean, element: ElementType, shape: number[] = []): PortRef {
    const id = this._addNode('constant', { value: String(value), element, shape });
    return { nodeId: id, portId: 'value' };
  }

  iota(shape: number[], element: ScalarKind = 'i32'): PortRef {
    const id = this._addNode('iota', { shape, element });
    return { nodeId: id, portId: 'value' };
  }

  // -- Arithmetic ----------------------------------------------------------

  private binaryOp(type: string, lhs: PortRef, rhs: PortRef, params: Record<string, unknown> = {}): PortRef {
    const id = this._addNode(type, params);
    this._connect(lhs, id, 'lhs');
    this._connect(rhs, id, 'rhs');
    return { nodeId: id, portId: 'result' };
  }

  addf(lhs: PortRef, rhs: PortRef, rounding = 'nearest_even'): PortRef {
    return this.binaryOp('addf', lhs, rhs, { rounding });
  }
  addi(lhs: PortRef, rhs: PortRef): PortRef {
    return this.binaryOp('addi', lhs, rhs);
  }
  subf(lhs: PortRef, rhs: PortRef, rounding = 'nearest_even'): PortRef {
    return this.binaryOp('subf', lhs, rhs, { rounding });
  }
  mulf(lhs: PortRef, rhs: PortRef, rounding = 'nearest_even'): PortRef {
    return this.binaryOp('mulf', lhs, rhs, { rounding });
  }
  divf(lhs: PortRef, rhs: PortRef, rounding = 'nearest_even'): PortRef {
    return this.binaryOp('divf', lhs, rhs, { rounding });
  }

  // -- Math ----------------------------------------------------------------

  private unaryOp(type: string, input: PortRef, params: Record<string, unknown> = {}): PortRef {
    const id = this._addNode(type, params);
    this._connect(input, id, 'input');
    return { nodeId: id, portId: 'result' };
  }

  exp(input: PortRef): PortRef { return this.unaryOp('exp', input); }
  exp2(input: PortRef): PortRef { return this.unaryOp('exp2', input); }
  sqrt(input: PortRef, rounding = 'nearest_even'): PortRef { return this.unaryOp('sqrt', input, { rounding }); }
  rsqrt(input: PortRef): PortRef { return this.unaryOp('rsqrt', input); }
  negf(input: PortRef): PortRef { return this.unaryOp('negf', input); }

  // -- Tensor manipulation -------------------------------------------------

  reshape(input: PortRef, targetShape: number[]): PortRef {
    const id = this._addNode('reshape', { targetShape });
    this._connect(input, id, 'input');
    return { nodeId: id, portId: 'output' };
  }

  broadcast(input: PortRef, targetShape: number[]): PortRef {
    const id = this._addNode('broadcast', { targetShape });
    this._connect(input, id, 'input');
    return { nodeId: id, portId: 'output' };
  }

  offset(base: PortRef, offsets: PortRef): PortRef {
    const id = this._addNode('offset', {});
    this._connect(base, id, 'base');
    this._connect(offsets, id, 'offsets');
    return { nodeId: id, portId: 'result' };
  }

  select(cond: PortRef, trueVal: PortRef, falseVal: PortRef): PortRef {
    const id = this._addNode('select', {});
    this._connect(cond, id, 'cond');
    this._connect(trueVal, id, 'true_val');
    this._connect(falseVal, id, 'false_val');
    return { nodeId: id, portId: 'result' };
  }

  reduce(input: PortRef, dim: number, mode: 'sum' | 'max' | 'min', identity: string): PortRef {
    const id = this._addNode('reduce', { dim, mode, identity });
    this._connect(input, id, 'input');
    return { nodeId: id, portId: 'result' };
  }

  // -- MMA -----------------------------------------------------------------

  mmaf(a: PortRef, b: PortRef, acc: PortRef): PortRef {
    const id = this._addNode('mmaf', {});
    this._connect(a, id, 'a');
    this._connect(b, id, 'b');
    this._connect(acc, id, 'acc');
    return { nodeId: id, portId: 'result' };
  }

  // -- Conversion ----------------------------------------------------------

  ftof(input: PortRef, targetElement: ScalarKind): PortRef {
    return this.unaryOp('ftof', input, { targetElement });
  }

  // -- Memory (ptr-based) --------------------------------------------------

  loadPtr(ptrs: PortRef, ordering = 'weak', tokenIn?: PortRef): { data: PortRef; token: PortRef } {
    const id = this._addNode('load_ptr_tko', { ordering });
    this._connect(ptrs, id, 'ptrs');
    if (tokenIn) this._connect(tokenIn, id, 'token');
    return {
      data: { nodeId: id, portId: 'data' },
      token: { nodeId: id, portId: 'token' },
    };
  }

  storePtr(ptrs: PortRef, data: PortRef, ordering = 'weak', tokenIn?: PortRef): PortRef {
    const id = this._addNode('store_ptr_tko', { ordering });
    this._connect(ptrs, id, 'ptrs');
    this._connect(data, id, 'data');
    if (tokenIn) this._connect(tokenIn, id, 'token');
    return { nodeId: id, portId: 'token' };
  }

  // -- Memory (view-based) -------------------------------------------------

  makeTensorView2d(
    base: PortRef, dim0: PortRef, dim1: PortRef,
    stride0: PortRef, stride1: PortRef,
    element: ScalarKind,
    opts?: { indexType?: 'i32' | 'i64'; staticShape?: string; staticStrides?: string },
  ): PortRef {
    const indexType = opts?.indexType ?? 'i64';
    const params: Record<string, unknown> = { element, indexType };
    if (opts?.staticShape) params.staticShape = opts.staticShape;
    if (opts?.staticStrides) params.staticStrides = opts.staticStrides;
    const id = this._addNode('make_tensor_view_2d', params);
    this._connect(base, id, 'base');
    this._connect(dim0, id, 'dim0');
    this._connect(dim1, id, 'dim1');
    this._connect(stride0, id, 'stride0');
    this._connect(stride1, id, 'stride1');
    return { nodeId: id, portId: 'view' };
  }

  makePartitionView(view: PortRef, tileShape: number[]): PortRef {
    const id = this._addNode('make_partition_view', { tileShape });
    this._connect(view, id, 'view');
    return { nodeId: id, portId: 'pview' };
  }

  loadView2d(
    pview: PortRef, idx0: PortRef, idx1: PortRef,
    ordering = 'weak', tokenIn?: PortRef,
  ): { data: PortRef; token: PortRef } {
    const id = this._addNode('load_view_tko_2d', { ordering });
    this._connect(pview, id, 'pview');
    this._connect(idx0, id, 'idx0');
    this._connect(idx1, id, 'idx1');
    if (tokenIn) this._connect(tokenIn, id, 'token');
    return {
      data: { nodeId: id, portId: 'data' },
      token: { nodeId: id, portId: 'token' },
    };
  }

  storeView2d(
    data: PortRef, pview: PortRef, idx0: PortRef, idx1: PortRef,
    ordering = 'weak', tokenIn?: PortRef,
  ): PortRef {
    const id = this._addNode('store_view_tko_2d', { ordering });
    this._connect(data, id, 'data');
    this._connect(pview, id, 'pview');
    this._connect(idx0, id, 'idx0');
    this._connect(idx1, id, 'idx1');
    if (tokenIn) this._connect(tokenIn, id, 'token');
    return { nodeId: id, portId: 'token' };
  }
}

// ---------------------------------------------------------------------------
// ForBodyBuilder — builds a for-loop body sub-graph
// ---------------------------------------------------------------------------

export class ForBodyBuilder extends BaseBuilder {
  private subGraph: SubGraph = { nodes: [], edges: [] };

  constructor() {
    const sg: SubGraph = { nodes: [], edges: [] };
    super(sg, (s, src, srcP, dst, dstP) => addEdgeToSubGraph(s as SubGraph, src, srcP, dst, dstP));
    this.subGraph = sg;
  }

  iterVar(name: string, varType: TileIRType): PortRef {
    const id = this._addNode('for_iter_var', { name, varType });
    return { nodeId: id, portId: 'value' };
  }

  carryIn(name: string, index: number, carryType: TileIRType): PortRef {
    const id = this._addNode('for_carry_in', { name, index, carryType });
    return { nodeId: id, portId: 'value' };
  }

  outerRef(ref: PortRef, refType: TileIRType): PortRef {
    const id = this._addNode('for_outer_ref', {
      outerNodeId: ref.nodeId,
      outerPortId: ref.portId,
      refType,
    });
    return { nodeId: id, portId: 'value' };
  }

  /** Finalize the body. carryOutValues must match the carry-in order. */
  build(carryOutValues: PortRef[]): SubGraph {
    const id = this._addNode('for_carry_out', { numValues: carryOutValues.length });
    for (let i = 0; i < carryOutValues.length; i++) {
      this._connect(carryOutValues[i], id, `value_${i}`);
    }
    return this.subGraph;
  }
}

// ---------------------------------------------------------------------------
// GraphBuilder — top-level graph builder
// ---------------------------------------------------------------------------

export class GraphBuilder extends BaseBuilder {
  private graph: Graph;
  private argCount = 0;

  constructor(moduleName: string, entryName: string) {
    resetIdCounter(0);
    const g = createGraph(moduleName, entryName);
    super(g, (s, src, srcP, dst, dstP) => addEdge(s as Graph, src, srcP, dst, dstP));
    this.graph = g;
  }

  // -- Structural ----------------------------------------------------------

  entryArg(name: string, argType: TileIRType): PortRef {
    const id = this._addNode('entry_arg', {
      name,
      argType,
      argIndex: this.argCount++,
    });
    return { nodeId: id, portId: 'value' };
  }

  getTileBlockId(): { bid0: PortRef; bid1: PortRef; bid2: PortRef } {
    const id = this._addNode('get_tile_block_id', {});
    return {
      bid0: { nodeId: id, portId: 'bid0' },
      bid1: { nodeId: id, portId: 'bid1' },
      bid2: { nodeId: id, portId: 'bid2' },
    };
  }

  // -- For loop (flat graph mode) -------------------------------------------

  /**
   * Create a flat `for` node. Returns handles to iter_var and per-carry
   * current/result ports. Connect looping edges later via connectForUpdate().
   */
  forCreate(
    lower: PortRef,
    upper: PortRef,
    step: PortRef,
    carries: { name: string; init: PortRef }[],
    iterVarName = 'i',
  ): {
    nodeId: string;
    iterVar: PortRef;
    current: PortRef[];
    result: PortRef[];
  } {
    const carryNames = carries.map((c) => c.name).join(',');
    const id = this._addNode('for', {
      numCarried: carries.length,
      iterVarName,
      carryNames,
    });
    this._connect(lower, id, 'lower');
    this._connect(upper, id, 'upper');
    this._connect(step, id, 'step');
    for (let i = 0; i < carries.length; i++) {
      this._connect(carries[i].init, id, `init_${i}`);
    }
    return {
      nodeId: id,
      iterVar: { nodeId: id, portId: 'iter_var' },
      current: carries.map((_, i) => ({ nodeId: id, portId: `current_${i}` })),
      result: carries.map((_, i) => ({ nodeId: id, portId: `result_${i}` })),
    };
  }

  /** Connect the looping edge for carry index i. */
  connectForUpdate(forNodeId: string, index: number, update: PortRef): void {
    this._connect(update, forNodeId, `update_${index}`);
  }

  // -- For loop (compound/sub-graph mode) ----------------------------------

  forLoop(
    lower: PortRef,
    upper: PortRef,
    step: PortRef,
    carries: { name: string; init: PortRef; type: TileIRType }[],
    iterVarName: string,
    iterVarType: TileIRType,
    buildBody: (body: ForBodyBuilder) => PortRef[],
  ): PortRef[] {
    const bodyBuilder = new ForBodyBuilder();
    const carryOutValues = buildBody(bodyBuilder);
    const body = bodyBuilder.build(carryOutValues);

    const id = this._addNode('for', { numCarried: carries.length }, body);

    // Connect outer inputs
    this._connect(lower, id, 'lower');
    this._connect(upper, id, 'upper');
    this._connect(step, id, 'step');
    for (let i = 0; i < carries.length; i++) {
      this._connect(carries[i].init, id, `init_${i}`);
    }

    // Add implicit dependency edges from outer-referenced nodes to the for node
    const outerRefs = body.nodes.filter((n) => n.type === 'for_outer_ref');
    const seenOuter = new Set<string>();
    for (const ref of outerRefs) {
      const outerNodeId = ref.params.outerNodeId as string;
      const outerPortId = ref.params.outerPortId as string;
      const key = `${outerNodeId}:${outerPortId}`;
      if (!seenOuter.has(key)) {
        seenOuter.add(key);
        this._connect({ nodeId: outerNodeId, portId: outerPortId }, id, `_dep_${key}`);
      }
    }

    return carries.map((_, i) => ({ nodeId: id, portId: `result_${i}` }));
  }

  // -- Build ---------------------------------------------------------------

  build(): Graph {
    return this.graph;
  }
}

/**
 * Helper to build a pointer-offset chain:
 *   reshape(scalar_ptr, [1]) -> broadcast([1], [N]) -> offset(broadcasted, offsets)
 */
export function makePtrChain(
  b: BaseBuilder,
  scalarPtr: PortRef,
  offsets: PortRef,
  size: number,
): PortRef {
  const reshaped = b.reshape(scalarPtr, [1]);
  const broadcasted = b.broadcast(reshaped, [size]);
  return b.offset(broadcasted, offsets);
}
