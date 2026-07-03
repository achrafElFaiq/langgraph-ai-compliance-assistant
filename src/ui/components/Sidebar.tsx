'use client'

import { useState, useEffect, useCallback } from 'react'

interface RegStat {
  name: string
  articles: number
}

interface Stats {
  regulations: RegStat[]
  total_articles: number
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

const S = {
  aside: {
    width: '240px',
    flexShrink: 0,
    borderRight: '1px solid var(--border)',
    padding: '16px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '20px',
    overflowY: 'auto' as const,
    backgroundColor: 'var(--bg)',
  },
  heading: {
    color: 'var(--primary)',
    fontWeight: 700,
    fontSize: '13px',
    marginBottom: '10px',
  },
  panel: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    padding: '12px 14px',
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    fontSize: '13px',
    padding: '3px 0',
  },
  regName: {
    color: 'var(--text)',
  },
  regCount: {
    color: 'var(--text-muted)',
    fontVariantNumeric: 'tabular-nums' as const,
  },
  divider: {
    border: 'none',
    borderTop: '1px solid var(--border)',
    margin: '8px 0',
  },
  totalRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    fontSize: '13px',
    color: 'var(--primary)',
    fontWeight: 700,
  },
  status: {
    color: 'var(--text-muted)',
    fontSize: '12px',
    marginTop: '4px',
  },
}

export default function Sidebar() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/ingestion/stats`)
      if (!res.ok) throw new Error(`stats ${res.status}`)
      setStats(await res.json())
      setError(null)
    } catch {
      setError('stats indisponibles')
    }
  }, [])

  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  return (
    <aside style={S.aside}>
      <div>
        <p style={S.heading}>&gt; base de données</p>
        <div style={S.panel}>
          {stats ? (
            <>
              {stats.regulations.map(r => (
                <div key={r.name} style={S.row}>
                  <span style={S.regName}>{r.name}</span>
                  <span style={S.regCount}>{r.articles} articles</span>
                </div>
              ))}
              <hr style={S.divider} />
              <div style={S.totalRow}>
                <span>total</span>
                <span style={{ fontVariantNumeric: 'tabular-nums' }}>{stats.total_articles} articles</span>
              </div>
            </>
          ) : (
            <p style={S.status}>[.] chargement...</p>
          )}
        </div>
        {error && <p style={{ ...S.status, color: 'var(--primary)' }}>[!] {error}</p>}
      </div>
    </aside>
  )
}
