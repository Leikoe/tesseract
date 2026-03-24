import { useCallback } from 'react';
import { useGraph } from '../store/graphStore.js';
import { useMLIR } from '../hooks/useMLIR.js';

export function MLIROutput() {
  const { graph } = useGraph();
  const { mlir, error } = useMLIR(graph);

  const copy = useCallback(() => {
    if (mlir) navigator.clipboard.writeText(mlir);
  }, [mlir]);

  return (
    <div
      style={{
        background: '#11111b',
        borderTop: '1px solid #313244',
        height: 200,
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 11,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '6px 12px',
          background: '#181825',
          borderBottom: '1px solid #313244',
        }}
      >
        <span style={{ fontWeight: 700, color: '#cba6f7', fontSize: 12 }}>MLIR Output</span>
        {mlir && (
          <button
            onClick={copy}
            style={{
              background: '#313244',
              border: 'none',
              borderRadius: 3,
              color: '#cdd6f4',
              padding: '2px 8px',
              fontSize: 10,
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            Copy
          </button>
        )}
      </div>
      <pre
        style={{
          flex: 1,
          margin: 0,
          padding: 12,
          overflowY: 'auto',
          color: error ? '#f38ba8' : '#a6e3a1',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
        }}
      >
        {error ?? mlir ?? 'Add nodes and connect them to generate MLIR'}
      </pre>
    </div>
  );
}
