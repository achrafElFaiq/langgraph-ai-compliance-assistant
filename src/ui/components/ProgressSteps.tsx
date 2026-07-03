'use client'

import { useState, useEffect } from 'react'

export interface Step {
  node: string
  label: string
  status: 'active' | 'done'
}

function Dots() {
  const [dots, setDots] = useState('.')
  useEffect(() => {
    const id = setInterval(() => setDots(d => d.length >= 3 ? '.' : d + '.'), 400)
    return () => clearInterval(id)
  }, [])
  return <span>{dots}</span>
}

export default function ProgressSteps({ steps }: { steps: Step[] }) {
  if (steps.length === 0) {
    return (
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '12px 16px',
        maxWidth: '340px',
      }}>
        <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
          [.] starting...
        </p>
      </div>
    )
  }

  return (
    <div style={{
      backgroundColor: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      borderRadius: '6px',
      padding: '12px 16px',
      maxWidth: '340px',
    }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '8px' }}>
        [.] processing
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {steps.map((step, i) => (
          <div key={i} style={{ fontSize: '13px', display: 'flex', gap: '8px', alignItems: 'baseline' }}>
            {step.status === 'done' ? (
              <span style={{ color: 'var(--primary)', flexShrink: 0 }}>[ok]</span>
            ) : (
              <span className="step-bracket-active" style={{ color: 'var(--primary)', flexShrink: 0 }}>[&gt;]</span>
            )}
            <span
              className={step.status === 'active' ? 'step-label-active' : undefined}
              style={{ color: step.status === 'done' ? 'var(--text-muted)' : undefined }}
            >
              {step.label}{step.status === 'active' && <Dots />}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
