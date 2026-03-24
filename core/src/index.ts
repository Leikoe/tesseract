/**
 * Public API for the spatial-gpu core library.
 */

// Types
export {
  type ScalarKind,
  type IntKind,
  type FloatKind,
  type PtrType,
  type ElementType,
  type TileType,
  type TokenType,
  type TensorViewType,
  type PartitionViewType,
  type TileIRType,
  tile,
  ptr,
  token,
  tensorView,
  partitionView,
  formatType,
  formatElementType,
  typesEqual,
  typesCompatible,
  isIntKind,
  isFloatKind,
  isPtr,
} from './types.js';

// Graph
export {
  type Graph,
  type GraphNode,
  type Edge,
  type SubGraph,
  createGraph,
  addNode,
  addEdge,
  addEdgeToSubGraph,
  getNode,
  getIncomingEdges,
  getOutgoingEdges,
  getEdgeToPort,
  topologicalSort,
  topoSortNodes,
  resetIdCounter,
} from './graph.js';

// Nodes
export {
  type NodeDefinition,
  type PortDefinition,
  type ParamDefinition,
  type EmitContext,
} from './nodes/types.js';
export { getNodeDef, getAllNodeDefs, getNodeDefsByCategory } from './nodes/index.js';

// Codegen
export { emitMLIR } from './codegen.js';

// Builder
export {
  type PortRef,
  BaseBuilder,
  ForBodyBuilder,
  GraphBuilder,
  makePtrChain,
} from './builder.js';
