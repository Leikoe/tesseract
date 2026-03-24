import type { GraphNode } from '@core/graph.js';
import { getNodeDef } from '@core/nodes/index.js';
import { useGraph } from '../store/graphStore.js';

interface Props {
  node: GraphNode | null;
}

const ELEMENT_OPTIONS = [
  'f16', 'bf16', 'f32', 'f64',
  'i1', 'i8', 'i16', 'i32', 'i64',
];

export function ParamEditor({ node }: Props) {
  const { dispatch } = useGraph();

  if (!node) {
    return (
      <div style={{ padding: 12, color: '#585b70', fontSize: 11, fontStyle: 'italic' }}>
        Select a node to edit parameters
      </div>
    );
  }

  const def = getNodeDef(node.type);

  const update = (paramId: string, value: unknown) => {
    dispatch({ type: 'UPDATE_PARAMS', nodeId: node.id, params: { [paramId]: value } });
  };

  // Filter out internal params not meant for UI editing
  const visibleParams = def.params.filter(
    (p) => !['argType'].includes(p.id),
  );

  return (
    <div style={{ padding: 12 }}>
      <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 8, color: '#cba6f7' }}>
        {def.label}
      </div>
      <div style={{ fontSize: 9, color: '#585b70', marginBottom: 10 }}>{node.id}</div>

      {visibleParams.map((p) => {
        const value = node.params[p.id];
        return (
          <div key={p.id} style={{ marginBottom: 8 }}>
            <label style={{ display: 'block', fontSize: 10, color: '#a6adc8', marginBottom: 2 }}>
              {p.name}
            </label>
            {renderParam(p, value, (v) => update(p.id, v))}
          </div>
        );
      })}
    </div>
  );
}

function renderParam(
  p: { id: string; type: string; options?: string[]; default?: unknown },
  value: unknown,
  onChange: (v: unknown) => void,
) {
  // Enum with options
  if (p.options) {
    return (
      <select
        value={String(value ?? p.default ?? '')}
        onChange={(e) => onChange(e.target.value)}
        style={selectStyle}
      >
        {!value && !p.default && <option value="">—</option>}
        {p.options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    );
  }

  // Element type selector (render as dropdown even without options)
  if (p.type === 'element_type') {
    return (
      <select
        value={String(value ?? p.default ?? 'f32')}
        onChange={(e) => onChange(e.target.value)}
        style={selectStyle}
      >
        {ELEMENT_OPTIONS.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    );
  }

  // Shape input (comma-separated integers)
  if (p.type === 'shape') {
    const display = Array.isArray(value)
      ? (value as number[]).join(', ')
      : String(value ?? '');
    return (
      <>
        <input
          type="text"
          value={display}
          placeholder="e.g. 128,64"
          onChange={(e) => {
            const str = e.target.value;
            // Store as string; node defs parse it
            const nums = str
              .split(',')
              .map((s) => parseInt(s.trim(), 10))
              .filter((n) => !isNaN(n));
            onChange(nums.length > 0 ? nums : []);
          }}
          style={inputStyle}
        />
        <div style={{ fontSize: 9, color: '#585b70', marginTop: 1 }}>
          comma-separated: 128 or 128,64
        </div>
      </>
    );
  }

  // Boolean
  if (p.type === 'boolean') {
    return (
      <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
        <input
          type="checkbox"
          checked={Boolean(value ?? p.default)}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span style={{ fontSize: 10, color: '#a6adc8' }}>
          {value ? 'yes' : 'no'}
        </span>
      </label>
    );
  }

  // Number
  if (p.type === 'number') {
    return (
      <input
        type="number"
        value={value as number ?? p.default ?? ''}
        onChange={(e) => onChange(Number(e.target.value))}
        style={inputStyle}
      />
    );
  }

  // Default: string
  return (
    <input
      type="text"
      value={String(value ?? p.default ?? '')}
      onChange={(e) => onChange(e.target.value)}
      style={inputStyle}
    />
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '3px 6px',
  fontSize: 11,
  background: '#1e1e2e',
  border: '1px solid #313244',
  borderRadius: 3,
  color: '#cdd6f4',
  outline: 'none',
  fontFamily: 'ui-monospace, monospace',
  boxSizing: 'border-box',
};

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  appearance: 'auto',
  cursor: 'pointer',
};
