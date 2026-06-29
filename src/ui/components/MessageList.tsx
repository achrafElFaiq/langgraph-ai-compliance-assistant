'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

export interface Citation {
  breadcrumb: string
  relevant: boolean
  excerpts: string[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  regulations?: string[]
  route?: string
  retryCount?: number
  fallbackAttempted?: boolean
  citations?: Citation[]
  finalReport?: string
}

const REGULATION_LABELS: Record<string, string> = {
  mica: 'MiCA',
  dora: 'DORA',
  ai_act: 'AI Act',
  gdpr: 'GDPR',
}

function CitationItem({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div style={{ borderBottom: '1px solid var(--border-light)', paddingBottom: '4px' }}>
      <button
        onClick={() => setExpanded(v => !v)}
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: '8px',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '4px 0',
          width: '100%',
          textAlign: 'left',
          fontFamily: 'inherit',
          fontSize: '12px',
        }}
      >
        <span style={{ color: 'var(--primary)', flexShrink: 0 }}>
          {expanded ? '[v]' : '[>]'}
        </span>
        <span style={{
          color: 'var(--text-muted)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          flex: 1,
        }}>
          {citation.breadcrumb}
        </span>
      </button>
      {expanded && citation.excerpts.length > 0 && (
        <div style={{ paddingLeft: '28px', paddingBottom: '6px' }}>
          {citation.excerpts.map((excerpt, i) => (
            <p key={i} style={{
              fontSize: '12px',
              color: 'var(--text-muted)',
              fontStyle: 'italic',
              lineHeight: '1.6',
              margin: '3px 0',
            }}>
              &ldquo;{excerpt}&rdquo;
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

function AssistantMessage({ message }: { message: Message }) {
  const [showCitations, setShowCitations] = useState(false)
  const relevantCitations = message.citations?.filter(c => c.relevant) ?? []

  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
      <div style={{ maxWidth: '680px', width: '100%' }}>
        {/* Message card */}
        <div style={{
          backgroundColor: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: '6px',
          padding: '16px',
        }}>
          <div className="md" style={{ fontSize: '13px', lineHeight: '1.7', color: 'var(--text)' }}>
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>

        {/* Meta row */}
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '12px',
          marginTop: '6px',
          paddingLeft: '4px',
          fontSize: '12px',
        }}>
          {message.regulations?.map(reg => (
            <span key={reg} style={{ color: 'var(--primary)', fontWeight: 700 }}>
              [{REGULATION_LABELS[reg] ?? reg.toUpperCase()}]
            </span>
          ))}
          {message.fallbackAttempted && (
            <span style={{ color: 'var(--text-muted)' }}>[~] search broadened</span>
          )}
          {(message.retryCount ?? 0) > 0 && (
            <span style={{ color: 'var(--text-muted)' }}>
              [!] {message.retryCount} verification{(message.retryCount ?? 0) > 1 ? 's' : ''}
            </span>
          )}
          {relevantCitations.length > 0 && (
            <button
              onClick={() => setShowCitations(v => !v)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                fontSize: '12px',
                fontFamily: 'inherit',
                padding: 0,
              }}
            >
              {showCitations
                ? '[v] hide sources'
                : `[>] ${relevantCitations.length} source${relevantCitations.length > 1 ? 's' : ''}`}
            </button>
          )}
        </div>

        {/* Citations panel */}
        {showCitations && relevantCitations.length > 0 && (
          <div style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '10px 16px',
            marginTop: '6px',
          }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '8px' }}>
              [&gt;] sources
            </p>
            {relevantCitations.map((citation, i) => (
              <CitationItem key={i} citation={citation} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function MessageList({ messages }: { messages: Message[] }) {
  return (
    <>
      {messages.map(message =>
        message.role === 'user' ? (
          <div key={message.id} style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <div style={{
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '10px 14px',
              maxWidth: '520px',
              fontSize: '13px',
              color: 'var(--text)',
            }}>
              {message.content}
            </div>
          </div>
        ) : (
          <AssistantMessage key={message.id} message={message} />
        )
      )}
    </>
  )
}
