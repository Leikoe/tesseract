import { useState, useRef, useEffect } from 'react';
import type { Graph } from '@core/graph.js';
import { templates } from '../lib/templates.js';

interface Props {
  onLoad: (graph: Graph) => void;
}

export function TemplateSelector({ onLoad }: Props) {
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState('blank');
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const select = (id: string) => {
    const tmpl = templates.find((t) => t.id === id);
    if (!tmpl) return;
    setCurrent(id);
    setOpen(false);
    onLoad(tmpl.build());
  };

  const currentName = templates.find((t) => t.id === current)?.name ?? 'Blank';

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: '#313244',
          border: '1px solid #45475a',
          borderRadius: 4,
          color: '#cdd6f4',
          padding: '4px 12px',
          fontSize: 11,
          cursor: 'pointer',
          fontFamily: 'ui-monospace, monospace',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span style={{ color: '#a6adc8' }}>Template:</span>
        <span style={{ fontWeight: 600 }}>{currentName}</span>
        <span style={{ fontSize: 8, color: '#585b70' }}>{open ? '▴' : '▾'}</span>
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: 4,
            background: '#1e1e2e',
            border: '1px solid #45475a',
            borderRadius: 6,
            minWidth: 280,
            zIndex: 1000,
            overflow: 'hidden',
            boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          }}
        >
          {templates.map((tmpl) => (
            <div
              key={tmpl.id}
              onClick={() => select(tmpl.id)}
              style={{
                padding: '8px 14px',
                cursor: 'pointer',
                borderBottom: '1px solid #313244',
                background: tmpl.id === current ? '#313244' : 'transparent',
              }}
              onMouseOver={(e) => {
                if (tmpl.id !== current) {
                  (e.currentTarget as HTMLDivElement).style.background = '#292940';
                }
              }}
              onMouseOut={(e) => {
                if (tmpl.id !== current) {
                  (e.currentTarget as HTMLDivElement).style.background = 'transparent';
                }
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 12, color: '#cdd6f4' }}>
                {tmpl.name}
                {tmpl.id === current && (
                  <span style={{ marginLeft: 6, fontSize: 10, color: '#a6e3a1' }}>current</span>
                )}
              </div>
              <div style={{ fontSize: 10, color: '#7f849c', marginTop: 2 }}>
                {tmpl.description}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
