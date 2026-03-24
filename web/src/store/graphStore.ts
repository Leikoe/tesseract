/**
 * Graph state management via React context + useReducer.
 */

import { createContext, useContext } from 'react';
import type { Graph } from '@core/graph.js';
import { createGraph, addNode, addEdge, resetIdCounter } from '@core/graph.js';
import { getNodeDef } from '@core/nodes/index.js';

// -- Actions --

export type GraphAction =
  | { type: 'ADD_NODE'; nodeType: string; position: { x: number; y: number } }
  | { type: 'REMOVE_NODE'; nodeId: string }
  | { type: 'CONNECT'; sourceNode: string; sourcePort: string; targetNode: string; targetPort: string }
  | { type: 'DISCONNECT'; edgeId: string }
  | { type: 'UPDATE_PARAMS'; nodeId: string; params: Record<string, unknown> }
  | { type: 'SET_POSITION'; nodeId: string; position: { x: number; y: number } }
  | { type: 'LOAD_GRAPH'; graph: Graph };

// -- Reducer --

export function graphReducer(state: Graph, action: GraphAction): Graph {
  switch (action.type) {
    case 'ADD_NODE': {
      const def = getNodeDef(action.nodeType);
      // Build default params
      const params: Record<string, unknown> = {};
      for (const p of def.params) {
        if (p.default !== undefined) params[p.id] = p.default;
      }
      const newGraph = {
        ...state,
        nodes: [...state.nodes],
        edges: [...state.edges],
      };
      const id = addNode(newGraph, action.nodeType, params);
      // Set position on the newly added node
      const node = newGraph.nodes.find((n) => n.id === id)!;
      node.position = action.position;
      return newGraph;
    }

    case 'REMOVE_NODE': {
      return {
        ...state,
        nodes: state.nodes.filter((n) => n.id !== action.nodeId),
        edges: state.edges.filter(
          (e) => e.sourceNode !== action.nodeId && e.targetNode !== action.nodeId,
        ),
      };
    }

    case 'CONNECT': {
      // Don't allow duplicate connections to the same input port
      const existing = state.edges.find(
        (e) => e.targetNode === action.targetNode && e.targetPort === action.targetPort,
      );
      if (existing) {
        // Replace existing connection
        const filtered = state.edges.filter((e) => e.id !== existing.id);
        const newGraph = { ...state, nodes: [...state.nodes], edges: filtered };
        addEdge(newGraph, action.sourceNode, action.sourcePort, action.targetNode, action.targetPort);
        return newGraph;
      }
      const newGraph = { ...state, nodes: [...state.nodes], edges: [...state.edges] };
      addEdge(newGraph, action.sourceNode, action.sourcePort, action.targetNode, action.targetPort);
      return newGraph;
    }

    case 'DISCONNECT': {
      return {
        ...state,
        edges: state.edges.filter((e) => e.id !== action.edgeId),
      };
    }

    case 'UPDATE_PARAMS': {
      return {
        ...state,
        nodes: state.nodes.map((n) =>
          n.id === action.nodeId
            ? { ...n, params: { ...n.params, ...action.params } }
            : n,
        ),
      };
    }

    case 'SET_POSITION': {
      return {
        ...state,
        nodes: state.nodes.map((n) =>
          n.id === action.nodeId
            ? { ...n, position: action.position }
            : n,
        ),
      };
    }

    case 'LOAD_GRAPH': {
      return action.graph;
    }

    default:
      return state;
  }
}

// -- Context --

export interface GraphContextValue {
  graph: Graph;
  dispatch: React.Dispatch<GraphAction>;
}

export const GraphContext = createContext<GraphContextValue | null>(null);

export function useGraph(): GraphContextValue {
  const ctx = useContext(GraphContext);
  if (!ctx) throw new Error('useGraph must be used within GraphContext.Provider');
  return ctx;
}

// -- Initial state --

export function createInitialGraph(): Graph {
  resetIdCounter(0);
  return createGraph('module', 'kernel');
}
