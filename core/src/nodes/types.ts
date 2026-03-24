/**
 * Node definition interfaces.
 *
 * Every Tile IR operation is described by a NodeDefinition that specifies its
 * ports, parameters, type-resolution rules, and MLIR emission logic.
 */

import type { TileIRType } from '../types.js';

// -- Port & param definitions ----------------------------------------------

export interface PortDefinition {
  id: string;
  name: string;
}

export interface ParamDefinition {
  id: string;
  name: string;
  type: 'number' | 'string' | 'boolean' | 'shape' | 'element_type' | 'enum';
  options?: string[]; // for enum
  default?: unknown;
  required: boolean;
}

// -- Emit context ----------------------------------------------------------

/** Passed to NodeDefinition.emit() during code generation. */
export interface EmitContext {
  /** SSA names for each input port (in definition order). */
  inputNames: string[];
  /** SSA names for each output port (in definition order). */
  outputNames: string[];
  /** Resolved types for each input port. */
  inputTypes: TileIRType[];
  /** Resolved types for each output port. */
  outputTypes: TileIRType[];
  /** Node parameters. */
  params: Record<string, unknown>;
}

// -- Node definition -------------------------------------------------------

export interface NodeDefinition {
  /** Unique identifier — matches the Tile IR operation name. */
  type: string;
  /** UI category for the palette. */
  category: string;
  /** Human-readable label. */
  label: string;

  inputs: PortDefinition[];
  outputs: PortDefinition[];
  params: ParamDefinition[];

  /**
   * Given the resolved types flowing into each input port and the node's
   * parameter values, compute the type of each output port.
   */
  resolveOutputTypes(
    inputTypes: (TileIRType | null)[],
    params: Record<string, unknown>,
  ): (TileIRType | null)[];

  /**
   * Emit one or more lines of MLIR for this operation.
   * Should NOT include leading indentation — the emitter adds that.
   */
  emit(ctx: EmitContext): string;
}
