/**
 * Memory operation nodes: load_ptr_tko, store_ptr_tko.
 *
 * These use Token-Keyed Ordering (TKO) — memory ops produce tokens for
 * explicit dependency tracking.  Token inputs are optional.
 */

import {
  formatType,
  token as tokenType,
  type TileIRType,
  type TileType,
  type TensorViewType,
  type PartitionViewType,
  type PtrType,
  type ScalarKind,
} from '../types.js';
import type { NodeDefinition, EmitContext, ParamDefinition } from './types.js';

const orderingParam: ParamDefinition = {
  id: 'ordering',
  name: 'Memory ordering',
  type: 'enum',
  options: ['weak', 'acquire', 'release', 'acq_rel', 'seq_cst'],
  default: 'weak',
  required: true,
};

// -- load_ptr_tko ----------------------------------------------------------

export const loadPtrTko: NodeDefinition = {
  type: 'load_ptr_tko',
  category: 'memory',
  label: 'Load (ptr)',
  inputs: [
    { id: 'ptrs', name: 'pointers' },
    { id: 'token', name: 'token' },
  ],
  outputs: [
    { id: 'data', name: 'data' },
    { id: 'token', name: 'token' },
  ],
  params: [orderingParam],

  resolveOutputTypes(inputTypes) {
    const ptrTile = inputTypes[0] as TileType | null;
    if (!ptrTile) return [null, tokenType()];
    const ptrElem = ptrTile.element as PtrType;
    return [
      { kind: 'tile', shape: ptrTile.shape, element: ptrElem.pointee } as TileType,
      tokenType(),
    ];
  },

  emit(ctx: EmitContext) {
    const ordering = ctx.params.ordering as string;
    const tokenIn =
      ctx.inputNames[1] ? ` token=${ctx.inputNames[1]}` : '';
    const outNames = ctx.outputNames.join(', ');
    return `${outNames} = load_ptr_tko ${ordering} ${ctx.inputNames[0]}${tokenIn} : ${formatType(ctx.inputTypes[0])} -> ${formatType(ctx.outputTypes[0])}, ${formatType(ctx.outputTypes[1])}`;
  },
};

// -- store_ptr_tko ---------------------------------------------------------

export const storePtrTko: NodeDefinition = {
  type: 'store_ptr_tko',
  category: 'memory',
  label: 'Store (ptr)',
  inputs: [
    { id: 'ptrs', name: 'pointers' },
    { id: 'data', name: 'data' },
    { id: 'token', name: 'token' },
  ],
  outputs: [{ id: 'token', name: 'token' }],
  params: [orderingParam],

  resolveOutputTypes() {
    return [tokenType()];
  },

  emit(ctx: EmitContext) {
    const ordering = ctx.params.ordering as string;
    const tokenIn =
      ctx.inputNames[2] ? ` token=${ctx.inputNames[2]}` : '';
    return `${ctx.outputNames[0]} = store_ptr_tko ${ordering} ${ctx.inputNames[0]}, ${ctx.inputNames[1]}${tokenIn} : ${formatType(ctx.inputTypes[0])}, ${formatType(ctx.inputTypes[1])} -> ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- make_tensor_view (2D, all-dynamic) ------------------------------------
// MLIR: %v = make_tensor_view %ptr, shape = [%d0, %d1], strides = [%s0, %s1]
//         : tile<i64> -> tensor_view<?x?xf16, strides=[?,?]>

export const makeTensorView2d: NodeDefinition = {
  type: 'make_tensor_view_2d',
  category: 'memory',
  label: 'Make Tensor View (2D)',
  inputs: [
    { id: 'base', name: 'base pointer' },
    { id: 'dim0', name: 'dim 0' },
    { id: 'dim1', name: 'dim 1' },
    { id: 'stride0', name: 'stride 0' },
    { id: 'stride1', name: 'stride 1' },
  ],
  outputs: [{ id: 'view', name: 'tensor view' }],
  params: [
    { id: 'element', name: 'Element type', type: 'element_type', required: true },
    { id: 'indexType', name: 'Index type', type: 'enum', options: ['i32', 'i64'], default: 'i64', required: true },
    { id: 'staticShape', name: 'Static shape', type: 'string', required: false },
    { id: 'staticStrides', name: 'Static strides', type: 'string', required: false },
  ],

  resolveOutputTypes(_inputTypes, params) {
    const elem = params.element as ScalarKind;
    // Parse optional static shape/strides (e.g. "?,128" or "128,1")
    const parseNullable = (s: string | undefined): (number | null)[] => {
      if (!s) return [null, null];
      return s.split(',').map((v) => {
        const trimmed = v.trim();
        if (trimmed === '?' || trimmed === '') return null;
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : null;
      });
    };
    return [
      {
        kind: 'tensor_view',
        shape: parseNullable(params.staticShape as string | undefined),
        element: elem,
        strides: parseNullable(params.staticStrides as string | undefined),
      } as TensorViewType,
    ];
  },

  emit(ctx: EmitContext) {
    const idxType = ctx.params.indexType as string;
    return `${ctx.outputNames[0]} = make_tensor_view ${ctx.inputNames[0]}, shape = [${ctx.inputNames[1]}, ${ctx.inputNames[2]}], strides = [${ctx.inputNames[3]}, ${ctx.inputNames[4]}] : tile<${idxType}> -> ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- make_partition_view ---------------------------------------------------
// MLIR: %pv = make_partition_view %tv : partition_view<tile=(MxN), tensor_view<...>>

export const makePartitionView: NodeDefinition = {
  type: 'make_partition_view',
  category: 'memory',
  label: 'Make Partition View',
  inputs: [{ id: 'view', name: 'tensor view' }],
  outputs: [{ id: 'pview', name: 'partition view' }],
  params: [
    { id: 'tileShape', name: 'Tile shape', type: 'shape', required: true },
  ],

  resolveOutputTypes(inputTypes, params) {
    const tv = inputTypes[0] as TensorViewType | null;
    if (!tv) return [null];
    return [
      {
        kind: 'partition_view',
        tileShape: params.tileShape as number[],
        tensorView: tv,
      } as PartitionViewType,
    ];
  },

  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]} = make_partition_view ${ctx.inputNames[0]} : ${formatType(ctx.outputTypes[0])}`;
  },
};

// -- load_view_tko (2D) ----------------------------------------------------
// MLIR: %d, %t = load_view_tko weak %pv[%i0, %i1] : pv_type, tile<i32> -> tile<...>, token

export const loadViewTko2d: NodeDefinition = {
  type: 'load_view_tko_2d',
  category: 'memory',
  label: 'Load (view, 2D)',
  inputs: [
    { id: 'pview', name: 'partition view' },
    { id: 'idx0', name: 'index 0' },
    { id: 'idx1', name: 'index 1' },
    { id: 'token', name: 'token' },
  ],
  outputs: [
    { id: 'data', name: 'data' },
    { id: 'token', name: 'token' },
  ],
  params: [orderingParam],

  resolveOutputTypes(inputTypes) {
    const pv = inputTypes[0] as PartitionViewType | null;
    if (!pv) return [null, tokenType()];
    return [
      {
        kind: 'tile',
        shape: pv.tileShape,
        element: pv.tensorView.element,
      } as TileType,
      tokenType(),
    ];
  },

  emit(ctx: EmitContext) {
    const ordering = ctx.params.ordering as string;
    const tokenIn = ctx.inputNames[3] ? ` token = ${ctx.inputNames[3]}` : '';
    // Index type from the index input — assume i32 for now
    const idxType = 'tile<i32>';
    return `${ctx.outputNames[0]}, ${ctx.outputNames[1]} = load_view_tko ${ordering} ${ctx.inputNames[0]}[${ctx.inputNames[1]}, ${ctx.inputNames[2]}]${tokenIn} : ${formatType(ctx.inputTypes[0])}, ${idxType} -> ${formatType(ctx.outputTypes[0])}, token`;
  },
};

// -- store_view_tko (2D) ---------------------------------------------------
// MLIR: %t = store_view_tko weak %data, %pv[%i0, %i1] : tile<...>, pv_type, tile<i32> -> token

export const storeViewTko2d: NodeDefinition = {
  type: 'store_view_tko_2d',
  category: 'memory',
  label: 'Store (view, 2D)',
  inputs: [
    { id: 'data', name: 'data' },
    { id: 'pview', name: 'partition view' },
    { id: 'idx0', name: 'index 0' },
    { id: 'idx1', name: 'index 1' },
    { id: 'token', name: 'token' },
  ],
  outputs: [{ id: 'token', name: 'token' }],
  params: [orderingParam],

  resolveOutputTypes() {
    return [tokenType()];
  },

  emit(ctx: EmitContext) {
    const ordering = ctx.params.ordering as string;
    const tokenIn = ctx.inputNames[4] ? ` token = ${ctx.inputNames[4]}` : '';
    const idxType = 'tile<i32>';
    return `${ctx.outputNames[0]} = store_view_tko ${ordering} ${ctx.inputNames[0]}, ${ctx.inputNames[1]}[${ctx.inputNames[2]}, ${ctx.inputNames[3]}]${tokenIn} : ${formatType(ctx.inputTypes[0])}, ${formatType(ctx.inputTypes[1])}, ${idxType} -> token`;
  },
};

export const memoryNodes: NodeDefinition[] = [
  loadPtrTko, storePtrTko,
  makeTensorView2d, makePartitionView, loadViewTko2d, storeViewTko2d,
];
