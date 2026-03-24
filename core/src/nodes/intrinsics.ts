/**
 * GPU intrinsic nodes: get_tile_block_id, get_num_tile_blocks.
 *
 * MLIR syntax:
 *   %bid0, %bid1, %bid2 = get_tile_block_id : tile<i32>
 *   %nb0, %nb1, %nb2 = get_num_tile_blocks : tile<i32>
 *
 * Both return a 3-tuple (x, y, z).
 */

import type { TileIRType, TileType } from '../types.js';
import type { NodeDefinition, EmitContext } from './types.js';

const SCALAR_I32: TileType = { kind: 'tile', shape: [], element: 'i32' };

export const getTileBlockId: NodeDefinition = {
  type: 'get_tile_block_id',
  category: 'intrinsics',
  label: 'Block ID',
  inputs: [],
  outputs: [
    { id: 'bid0', name: 'bid_x' },
    { id: 'bid1', name: 'bid_y' },
    { id: 'bid2', name: 'bid_z' },
  ],
  params: [],
  resolveOutputTypes() {
    return [SCALAR_I32, SCALAR_I32, SCALAR_I32];
  },
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]}, ${ctx.outputNames[1]}, ${ctx.outputNames[2]} = get_tile_block_id : tile<i32>`;
  },
};

export const getNumTileBlocks: NodeDefinition = {
  type: 'get_num_tile_blocks',
  category: 'intrinsics',
  label: 'Num Blocks',
  inputs: [],
  outputs: [
    { id: 'nb0', name: 'grid_x' },
    { id: 'nb1', name: 'grid_y' },
    { id: 'nb2', name: 'grid_z' },
  ],
  params: [],
  resolveOutputTypes() {
    return [SCALAR_I32, SCALAR_I32, SCALAR_I32];
  },
  emit(ctx: EmitContext) {
    return `${ctx.outputNames[0]}, ${ctx.outputNames[1]}, ${ctx.outputNames[2]} = get_num_tile_blocks : tile<i32>`;
  },
};

export const intrinsicNodes: NodeDefinition[] = [
  getTileBlockId,
  getNumTileBlocks,
];
