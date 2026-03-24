import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { TileIRNodeData } from '../../lib/convert.js';

// Internal params to never show
const HIDDEN_PARAMS = new Set([
  'argType', 'argIndex', 'numCarried', 'numValues',
  'outerNodeId', 'outerPortId', 'refType', 'varType', 'carryType',
  'index', 'rounding', 'ordering', 'indexType',
]);

function formatInlineParams(nodeType: string, params: Record<string, unknown>): JSX.Element[] {
  // Special display for entry_arg
  if (nodeType === 'entry_arg') {
    const name = params.name as string || 'arg';
    const elem = params.element as string || 'f32';
    const shape = params.shape as string || '';
    const isPtr = params.isPtr as boolean;
    const shapeStr = shape ? `${shape}x` : '';
    const typeStr = isPtr ? `tile<${shapeStr}ptr<${elem}>>` : `tile<${shapeStr}${elem}>`;
    return [<div key="type">{name}: {typeStr}</div>];
  }

  // Special display for constant
  if (nodeType === 'constant') {
    const val = params.value as string || '0';
    const elem = params.element as string || 'f32';
    const shape = Array.isArray(params.shape) ? (params.shape as number[]).join('x') : '';
    return [<div key="val">{shape ? `${shape}x` : ''}{elem}: {val}</div>];
  }

  return Object.entries(params)
    .filter(([k, v]) => !HIDDEN_PARAMS.has(k) && v !== undefined && v !== '')
    .slice(0, 3)
    .map(([k, v]) => (
      <div key={k}>
        {k}: {Array.isArray(v) ? (v as unknown[]).join(',') : String(v)}
      </div>
    ));
}

const CATEGORY_COLORS: Record<string, string> = {
  kernel: '#6366f1',      // indigo
  constants: '#f59e0b',   // amber
  arithmetic: '#10b981',  // emerald
  math: '#14b8a6',        // teal
  tensor: '#8b5cf6',      // violet
  memory: '#ef4444',      // red
  comparison: '#f97316',  // orange
  conversion: '#06b6d4',  // cyan
  mma: '#ec4899',         // pink
  intrinsics: '#84cc16',  // lime
  control: '#64748b',     // slate
};

const PORT_HEIGHT = 22;
const HEADER_HEIGHT = 28;
const PADDING_TOP = 6;

export function TileIRNode({ data, selected }: NodeProps) {
  const d = data as unknown as TileIRNodeData;
  const color = CATEGORY_COLORS[d.category] ?? '#6b7280';
  const maxPorts = Math.max(d.inputs.length, d.outputs.length, 1);
  const bodyHeight = maxPorts * PORT_HEIGHT + PADDING_TOP * 2;

  return (
    <div
      style={{
        minWidth: 160,
        background: '#1e1e2e',
        borderRadius: 6,
        border: selected ? `2px solid ${color}` : '2px solid #313244',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 11,
        color: '#cdd6f4',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: color,
          color: '#fff',
          padding: '4px 10px',
          fontSize: 11,
          fontWeight: 600,
          height: HEADER_HEIGHT,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span style={{ opacity: 0.7, fontSize: 9 }}>{d.category}</span>
        <span>{d.label}</span>
      </div>

      {/* Ports */}
      <div style={{ position: 'relative', height: bodyHeight }}>
        {/* Input ports (left) */}
        {d.inputs.map((port, i) => (
          <div
            key={`in-${port.id}`}
            style={{
              position: 'absolute',
              left: 8,
              top: PADDING_TOP + i * PORT_HEIGHT,
              height: PORT_HEIGHT,
              display: 'flex',
              alignItems: 'center',
              fontSize: 10,
              color: '#a6adc8',
            }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={port.id}
              style={{
                position: 'absolute',
                left: -12,
                top: PORT_HEIGHT / 2,
                width: 8,
                height: 8,
                background: '#45475a',
                border: '2px solid #585b70',
                borderRadius: '50%',
              }}
            />
            {port.name}
          </div>
        ))}

        {/* Output ports (right) */}
        {d.outputs.map((port, i) => (
          <div
            key={`out-${port.id}`}
            style={{
              position: 'absolute',
              right: 8,
              top: PADDING_TOP + i * PORT_HEIGHT,
              height: PORT_HEIGHT,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              gap: 4,
              fontSize: 10,
              color: '#a6adc8',
            }}
          >
            {port.name}
            <Handle
              type="source"
              position={Position.Right}
              id={port.id}
              style={{
                position: 'absolute',
                right: -12,
                top: PORT_HEIGHT / 2,
                width: 8,
                height: 8,
                background: '#45475a',
                border: `2px solid ${color}`,
                borderRadius: '50%',
              }}
            />
          </div>
        ))}
      </div>

      {/* Show key param values inline */}
      {d.paramValues && Object.keys(d.paramValues).length > 0 && (
        <div
          style={{
            padding: '2px 10px 6px',
            fontSize: 9,
            color: '#7f849c',
            borderTop: '1px solid #313244',
          }}
        >
          {formatInlineParams(d.nodeType, d.paramValues)}
        </div>
      )}
    </div>
  );
}
