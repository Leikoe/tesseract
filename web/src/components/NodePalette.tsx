import { useState, useMemo, type DragEvent } from 'react';
import { getNodeDefsByCategory } from '@core/nodes/index.js';

// Nodes that are structural / not user-addable
const HIDDEN_TYPES = new Set([
  'for_iter_var', 'for_carry_in', 'for_carry_out', 'for_outer_ref',
]);

const CATEGORY_ORDER = [
  'kernel', 'constants', 'arithmetic', 'math', 'comparison',
  'conversion', 'tensor', 'memory', 'mma', 'intrinsics', 'control',
];

export function NodePalette() {
  const categories = useMemo(() => {
    const map = getNodeDefsByCategory();
    // Filter hidden nodes and sort categories
    const result: { name: string; nodes: { type: string; label: string }[] }[] = [];
    for (const cat of CATEGORY_ORDER) {
      const defs = map.get(cat);
      if (!defs) continue;
      const visible = defs.filter((d) => !HIDDEN_TYPES.has(d.type));
      if (visible.length > 0) {
        result.push({
          name: cat,
          nodes: visible.map((d) => ({ type: d.type, label: d.label })),
        });
      }
    }
    return result;
  }, []);

  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggle = (cat: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  const onDragStart = (e: DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/tile-ir-node', nodeType);
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      style={{
        width: 200,
        background: '#181825',
        borderRight: '1px solid #313244',
        overflowY: 'auto',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 11,
        color: '#cdd6f4',
        userSelect: 'none',
      }}
    >
      <div style={{ padding: '10px 12px', fontWeight: 700, fontSize: 13, color: '#cba6f7' }}>
        Nodes
      </div>
      {categories.map((cat) => (
        <div key={cat.name}>
          <div
            onClick={() => toggle(cat.name)}
            style={{
              padding: '5px 12px',
              cursor: 'pointer',
              background: '#1e1e2e',
              fontWeight: 600,
              fontSize: 10,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: '#a6adc8',
              borderBottom: '1px solid #313244',
            }}
          >
            {collapsed.has(cat.name) ? '▸' : '▾'} {cat.name}
          </div>
          {!collapsed.has(cat.name) &&
            cat.nodes.map((node) => (
              <div
                key={node.type}
                draggable
                onDragStart={(e) => onDragStart(e, node.type)}
                style={{
                  padding: '4px 12px 4px 20px',
                  cursor: 'grab',
                  borderBottom: '1px solid #11111b',
                }}
                onMouseOver={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = '#313244';
                }}
                onMouseOut={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = 'transparent';
                }}
              >
                {node.label}
              </div>
            ))}
        </div>
      ))}
    </div>
  );
}
