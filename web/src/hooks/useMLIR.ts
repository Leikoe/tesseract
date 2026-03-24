import { useMemo } from 'react';
import type { Graph } from '@core/graph.js';
import { emitMLIR } from '@core/codegen.js';

export function useMLIR(graph: Graph): { mlir: string | null; error: string | null } {
  return useMemo(() => {
    if (graph.nodes.length === 0) {
      return { mlir: null, error: null };
    }
    try {
      return { mlir: emitMLIR(graph), error: null };
    } catch (e) {
      return { mlir: null, error: (e as Error).message };
    }
  }, [graph]);
}
