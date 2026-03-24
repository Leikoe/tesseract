import { useReducer, useState, useMemo, useCallback } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import {
  GraphContext,
  graphReducer,
  createInitialGraph,
} from './store/graphStore.js';
import { Canvas } from './components/Canvas.js';
import { NodePalette } from './components/NodePalette.js';
import { ParamEditor } from './components/ParamEditor.js';
import { MLIROutput } from './components/MLIROutput.js';
import { TemplateSelector } from './components/TemplateSelector.js';

export function App() {
  const [graph, dispatch] = useReducer(graphReducer, null, createInitialGraph);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const selectedNode = useMemo(
    () => graph.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [graph.nodes, selectedNodeId],
  );

  const onLoadTemplate = useCallback(
    (newGraph: import('@core/graph.js').Graph) => {
      setSelectedNodeId(null);
      dispatch({ type: 'LOAD_GRAPH', graph: newGraph });
    },
    [dispatch],
  );

  return (
    <GraphContext.Provider value={{ graph, dispatch }}>
      <ReactFlowProvider>
        <div
          style={{
            display: 'flex',
            height: '100vh',
            width: '100vw',
            overflow: 'hidden',
            background: '#11111b',
            color: '#cdd6f4',
          }}
          onClick={(e) => {
            if ((e.target as HTMLElement).classList.contains('react-flow__pane')) {
              setSelectedNodeId(null);
            }
          }}
          onClickCapture={(e) => {
            const nodeEl = (e.target as HTMLElement).closest('[data-id]');
            if (nodeEl) {
              setSelectedNodeId(nodeEl.getAttribute('data-id'));
            }
          }}
        >
          {/* Left: Node Palette */}
          <NodePalette />

          {/* Center: Toolbar + Canvas + Output */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* Toolbar */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '6px 12px',
                background: '#181825',
                borderBottom: '1px solid #313244',
                fontFamily: 'ui-monospace, monospace',
                fontSize: 12,
              }}
            >
              <span style={{ fontWeight: 700, color: '#cba6f7', fontSize: 14 }}>
                Spatial GPU
              </span>
              <span style={{ color: '#585b70' }}>|</span>
              <TemplateSelector onLoad={onLoadTemplate} />
            </div>
            <Canvas />
            <MLIROutput />
          </div>

          {/* Right: Param Editor */}
          <div
            style={{
              width: 220,
              background: '#181825',
              borderLeft: '1px solid #313244',
              overflowY: 'auto',
              fontFamily: 'ui-monospace, monospace',
              fontSize: 11,
            }}
          >
            <div style={{ padding: '10px 12px', fontWeight: 700, fontSize: 13, color: '#cba6f7' }}>
              Inspector
            </div>
            <ParamEditor node={selectedNode} />
          </div>
        </div>
      </ReactFlowProvider>
    </GraphContext.Provider>
  );
}
