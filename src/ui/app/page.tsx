'use client'

import { useState, useEffect, useRef } from 'react'
import MessageList, { Message, Citation } from '@/components/MessageList'
import ProgressSteps, { Step } from '@/components/ProgressSteps'
import Sidebar from '@/components/Sidebar'

const S = {
  shell: {
    display: 'flex',
    flexDirection: 'row' as const,
    height: '100vh',
    backgroundColor: 'var(--bg)',
    color: 'var(--text)',
  },
  root: {
    display: 'flex',
    flexDirection: 'column' as const,
    flex: 1,
    minWidth: 0,
    height: '100vh',
    backgroundColor: 'var(--bg)',
    color: 'var(--text)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 20px',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  h1: {
    color: 'var(--primary)',
    fontSize: '18px',
    fontWeight: 700,
  },
  subtitle: {
    color: 'var(--text-muted)',
    fontSize: '12px',
    marginTop: '2px',
  },
  btnPrimary: (disabled: boolean) => ({
    fontSize: '13px',
    padding: '6px 12px',
    backgroundColor: 'var(--primary)',
    color: 'var(--bg)',
    border: 'none',
    borderRadius: '6px',
    cursor: disabled ? 'default' : 'pointer',
    opacity: disabled ? 0.4 : 1,
    fontFamily: 'inherit',
  }),
  btnSecondary: (disabled: boolean) => ({
    fontSize: '13px',
    padding: '6px 12px',
    backgroundColor: 'transparent',
    color: 'var(--primary)',
    border: '1px solid var(--primary)',
    borderRadius: '6px',
    cursor: disabled ? 'default' : 'pointer',
    opacity: disabled ? 0.4 : 1,
    fontFamily: 'inherit',
  }),
  messages: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '24px 20px',
  },
  empty: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
  },
  emptyTitle: {
    color: 'var(--primary)',
    fontWeight: 700,
    marginBottom: '12px',
  },
  emptyHint: {
    color: 'var(--text-muted)',
    fontSize: '12px',
    lineHeight: '2.2',
    cursor: 'pointer',
  },
  footer: {
    borderTop: '1px solid var(--border)',
    padding: '12px 20px',
    flexShrink: 0,
  },
  inputRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    maxWidth: '760px',
    margin: '0 auto',
  },
  prompt: {
    color: 'var(--primary)',
    fontWeight: 700,
    flexShrink: 0,
    fontSize: '15px',
    lineHeight: 1,
  },
  input: {
    flex: 1,
    border: '1px dashed var(--border)',
    borderRadius: '4px',
    padding: '8px 12px',
    backgroundColor: 'var(--bg)',
    color: 'var(--text)',
    fontSize: '13px',
    fontFamily: 'inherit',
    outline: 'none',
  },
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [steps, setSteps] = useState<Step[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, steps])

  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return

    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content: text }])
    setInput('')
    setIsLoading(true)
    setSteps([])

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_text: text, thread_id: threadId }),
      })

      if (!response.body) throw new Error('no response body')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'node_start') {
              setSteps(prev => [...prev, { node: event.node, label: event.label, status: 'active' as const }])
            } else if (event.type === 'node_end') {
              setSteps(prev => {
                const result = [...prev]
                for (let i = result.length - 1; i >= 0; i--) {
                  if (result[i].node === event.node && result[i].status === 'active') {
                    result[i] = { ...result[i], label: event.label, status: 'done' as const }
                    break
                  }
                }
                return result
              })
            } else if (event.type === 'done') {
              if (event.thread_id) {
                setThreadId(event.thread_id)
              }
              setMessages(prev => [...prev, {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: event.final_report || event.answer || '',
                regulations: event.regulations ?? [],
                route: event.route ?? '',
                retryCount: event.retry_count ?? 0,
                fallbackAttempted: event.fallback_attempted ?? false,
                citations: (event.citations as Citation[]) ?? [],
                finalReport: event.final_report ?? '',
              }])
              setIsLoading(false)
              setSteps([])
              setTimeout(() => inputRef.current?.focus(), 50)
            }
          } catch { /* skip malformed lines */ }
        }
      }
    } catch (err) {
      console.error(err)
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '[error] An error occurred. Please try again.',
      }])
      setIsLoading(false)
      setSteps([])
    }
  }

  const resetConversation = () => {
    setMessages([])
    setThreadId(null)
    setSteps([])
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  return (
    <div style={S.shell}>
      <Sidebar />
      <div style={S.root}>
      {/* Header */}
      <header style={S.header}>
        <div>
          <h1 style={S.h1}>&gt; Assistant de conformité réglementaire </h1>
          <p style={S.subtitle}>MiCA · DORA · AI Act · GDPR</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {messages.length > 0 && (
            <button
              onClick={() => sendMessage('Generate synthesis')}
              disabled={isLoading}
              style={S.btnPrimary(isLoading)}
            >
              [+] synthesis
            </button>
          )}
          {messages.length > 0 && (
            <button
              onClick={resetConversation}
              disabled={isLoading}
              style={S.btnSecondary(isLoading)}
            >
              [x] new
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div style={S.messages}>
        {messages.length === 0 && !isLoading && (
          <div style={S.empty}>
            <div>
              <p style={S.emptyTitle}>[.] Posez votre question de conformité réglementaire...</p>
              <div>
                {[
                  'Nous lançons un token utilitaire en France, quelles sont nos obligations avant l\'offre au public ?',
                  'Notre banque fait appel à un prestataire cloud tiers, quelles sont nos obligations contractuelles sous DORA ?',
                  'Notre modèle de scoring crédit est-il considéré comme un système IA à haut risque sous l\'AI Act ?',
                ].map(q => (
                  <p key={q} style={S.emptyHint} onClick={() => setInput(q)}>
                    [&gt;] {q}
                  </p>
                ))}
            </div>
            </div>
          </div>
        )}

        <div style={{ maxWidth: '760px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <MessageList messages={messages} />
          {isLoading && <ProgressSteps steps={steps} />}
        </div>
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <footer style={S.footer}>
        <div style={S.inputRow}>
          <span style={S.prompt}>&gt;</span>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                sendMessage(input)
              }
            }}
            placeholder="Posez votre question de conformité réglementaire..."
            disabled={isLoading}
            style={S.input}
            onFocus={e => {
              e.target.style.borderColor = 'var(--primary)'
              e.target.style.borderStyle = 'solid'
            }}
            onBlur={e => {
              e.target.style.borderColor = 'var(--border)'
              e.target.style.borderStyle = 'dashed'
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={isLoading || !input.trim()}
            style={S.btnPrimary(isLoading || !input.trim())}
          >
            [send]
          </button>
        </div>
      </footer>
      </div>
    </div>
  )
}
