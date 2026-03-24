/**
 * CUDA Tile IR type system.
 *
 * Models the types from the Tile IR dialect:
 *   - Scalar element types (i1, i32, f16, f32, ...)
 *   - Pointer types: ptr<elementType>
 *   - Tile types: tile<shape x elementType>  (shape [] = scalar tile)
 *   - Token type: ordering constraint, not runtime data
 */

// -- Scalar element kinds --------------------------------------------------

export type IntKind = 'i1' | 'i8' | 'i16' | 'i32' | 'i64';
export type FloatKind =
  | 'f16'
  | 'bf16'
  | 'f32'
  | 'tf32'
  | 'f64'
  | 'f8E4M3FN'
  | 'f8E5M2';

export type ScalarKind = IntKind | FloatKind;

export function isIntKind(k: ScalarKind): k is IntKind {
  return k.startsWith('i');
}

export function isFloatKind(k: ScalarKind): k is FloatKind {
  return !k.startsWith('i');
}

// -- Composite element types -----------------------------------------------

export interface PtrType {
  kind: 'ptr';
  pointee: ScalarKind;
}

export type ElementType = ScalarKind | PtrType;

export function isPtr(e: ElementType): e is PtrType {
  return typeof e !== 'string' && e.kind === 'ptr';
}

// -- IR-level types --------------------------------------------------------

export interface TileType {
  kind: 'tile';
  shape: number[]; // [] = scalar tile, [128] = 1-D, [64,32] = 2-D, …
  element: ElementType;
}

export interface TokenType {
  kind: 'token';
}

export interface TensorViewType {
  kind: 'tensor_view';
  shape: (number | null)[]; // null = dynamic
  element: ScalarKind;
  strides: (number | null)[];
}

export interface PartitionViewType {
  kind: 'partition_view';
  tileShape: number[];
  tensorView: TensorViewType;
}

export type TileIRType = TileType | TokenType | TensorViewType | PartitionViewType;

// -- Constructors ----------------------------------------------------------

export function tile(shape: number[], element: ElementType): TileType {
  return { kind: 'tile', shape, element };
}

export function ptr(pointee: ScalarKind): PtrType {
  return { kind: 'ptr', pointee };
}

export function token(): TokenType {
  return { kind: 'token' };
}

export function tensorView(
  shape: (number | null)[],
  element: ScalarKind,
  strides: (number | null)[],
): TensorViewType {
  return { kind: 'tensor_view', shape, element, strides };
}

export function partitionView(
  tileShape: number[],
  tv: TensorViewType,
): PartitionViewType {
  return { kind: 'partition_view', tileShape, tensorView: tv };
}

// -- Formatting (for MLIR text output) -------------------------------------

export function formatElementType(e: ElementType): string {
  if (typeof e === 'string') return e;
  return `ptr<${e.pointee}>`;
}

export function formatType(t: TileIRType): string {
  switch (t.kind) {
    case 'token':
      return 'token';
    case 'tile': {
      const shapePart =
        t.shape.length > 0 ? t.shape.join('x') + 'x' : '';
      return `tile<${shapePart}${formatElementType(t.element)}>`;
    }
    case 'tensor_view': {
      const dims = t.shape.map((d) => (d === null ? '?' : String(d)));
      const strides = t.strides.map((s) => (s === null ? '?' : String(s)));
      return `tensor_view<${dims.join('x')}x${t.element}, strides=[${strides.join(',')}]>`;
    }
    case 'partition_view': {
      const tilePart = t.tileShape.join('x');
      return `partition_view<tile=(${tilePart}), ${formatType(t.tensorView)}>`;
    }
  }
}

// -- Equality / compatibility ----------------------------------------------

export function elementTypesEqual(a: ElementType, b: ElementType): boolean {
  if (typeof a === 'string' && typeof b === 'string') return a === b;
  if (typeof a === 'string' || typeof b === 'string') return false;
  return a.kind === b.kind && a.pointee === b.pointee;
}

export function typesEqual(a: TileIRType, b: TileIRType): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'token') return true;
  if (a.kind === 'tile' && b.kind === 'tile') {
    if (a.shape.length !== b.shape.length) return false;
    if (!a.shape.every((d, i) => d === b.shape[i])) return false;
    return elementTypesEqual(a.element, b.element);
  }
  return false; // tensor_view equality not needed yet
}

/**
 * Check whether an output port of type `from` can connect to an input port
 * expecting type `to`.  For now this is strict equality.
 */
export function typesCompatible(from: TileIRType, to: TileIRType): boolean {
  return typesEqual(from, to);
}
