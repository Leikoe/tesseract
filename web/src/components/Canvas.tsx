import { useCallback, useState, useEffect, useRef, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type Node as FlowNode,
  type Edge as FlowEdge,
  useReactFlow,
} from '@xyflow/react';
import { useGraph } from '../store/graphStore.js';
import { graphToFlow } from '../lib/convert.js';
import { TileIRNode } from './customNodes/TileIRNode.js';

const nodeTypes = { tileIRNode: TileIRNode };

export function Canvas() {
  const { graph, dispatch } = useGraph();
  const { screenToFlowPosition, fitView } = useReactFlow();
  const ref = useRef<HTMLDivElement>(null);

  // Local xyflow state — owns positions for smooth dragging
  const [flowNodes, setFlowNodes] = useState<FlowNode[]>([]);
  const [flowEdges, setFlowEdges] = useState<FlowEdge[]>([]);

  // Track graph identity to detect structural changes
  const graphRef = useRef(graph);

  // Sync from graph → local flow state when graph structure changes
  useEffect(() => {
    const { nodes, edges } = graphToFlow(graph);
    setFlowNodes(nodes);
    setFlowEdges(edges);
    graphRef.current = graph;

    // Fit view after template load (slight delay for React to render)
    requestAnimationFrame(() => {
      fitView({ padding: 0.15, duration: 200 });
    });
  }, [graph, fitView]);

  // Handle node changes: apply locally for smooth dragging, sync back on drag end
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      setFlowNodes((nds) => applyNodeChanges(changes, nds));

      for (const change of changes) {
        if (change.type === 'position' && change.position && change.dragging === false) {
          dispatch({
            type: 'SET_POSITION',
            nodeId: change.id,
            position: change.position,
          });
        }
        if (change.type === 'remove') {
          dispatch({ type: 'REMOVE_NODE', nodeId: change.id });
        }
      }
    },
    [dispatch],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      setFlowEdges((eds) => applyEdgeChanges(changes, eds));

      for (const change of changes) {
        if (change.type === 'remove') {
          dispatch({ type: 'DISCONNECT', edgeId: change.id });
        }
      }
    },
    [dispatch],
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      if (connection.source && connection.target && connection.sourceHandle && connection.targetHandle) {
        dispatch({
          type: 'CONNECT',
          sourceNode: connection.source,
          sourcePort: connection.sourceHandle,
          targetNode: connection.target,
          targetPort: connection.targetHandle,
        });
      }
    },
    [dispatch],
  );

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      const nodeType = e.dataTransfer.getData('application/tile-ir-node');
      if (!nodeType) return;

      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      dispatch({ type: 'ADD_NODE', nodeType, position });
    },
    [dispatch, screenToFlowPosition],
  );

  return (
    <div ref={ref} style={{ flex: 1, height: '100%' }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onConnect={onConnect}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onDragOver={onDragOver}
        onDrop={onDrop}
        fitView
        deleteKeyCode="Backspace"
        style={{ background: '#11111b' }}
        defaultEdgeOptions={{
          style: { stroke: '#585b70', strokeWidth: 1.5 },
          type: 'default',
        }}
      >
        <Background color="#313244" gap={20} />
        <Controls
          style={{ background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6 }}
        />
      </ReactFlow>
    </div>
  );
}
