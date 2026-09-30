import { useRef } from 'react'

import { KIND_COLOR } from '../lib/derive'
import { duration } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import { followRoute, ROUTE_PATH, type Route } from '../lib/route'
import type { EngineHandle } from '../lib/use-engine'
import { Legend } from './charts'
import { Pill, StatusDot, type Tone } from './ui'

const CONNECTION_TONE: Record<string, Tone> = {
  live: 'accent',
  replay: 'warning',
  connecting: 'warning',
  offline: 'danger',
}

const VIEWS: Route[] = ['dashboard', 'playground']

export function TopBar({
  engine,
  route,
  systemOne,
  binding,
}: {
  engine: EngineHandle
  route: Route
  /** Whether this server answers System One, or a replayed log recorded decisions. */
  systemOne: boolean
  /** The decision adapter the SDK alias resolves to; empty when none is bound. */
  binding: string | undefined
}) {
  const { state, loadFile, resumeLive } = engine
  const filePicker = useRef<HTMLInputElement>(null)
  const serverEngine = state.serverStart?.engine
  const server = state.serverStart?.server
  const replay = state.source === 'file'

  return (
    <header className="topbar">
      <div className="topbar__identity">
        <StatusDot tone={CONNECTION_TONE[state.connection]} />
        <strong>{state.telemetry?.model_id ?? server?.public_model_id ?? 'ninfer'}</strong>
        {serverEngine ? (
          <span className="topbar__config">
            {serverEngine.kv_cache} · {serverEngine.max_concurrency} lanes ·{' '}
            {serverEngine.max_context.toLocaleString('en-US')} ctx
            {serverEngine.speculative_backend !== 'none'
              ? ` · ${serverEngine.speculative_backend}`
              : ''}
            {serverEngine.cuda_graph ? ' · graphs' : ''}
          </span>
        ) : null}
        <span className="topbar__surfaces">
          <Legend
            items={[
              {
                label: 'chat /v1',
                color: KIND_COLOR.generative,
                hint: 'OpenAI and Anthropic generation, on the base weights or a chat adapter.',
              },
              ...(systemOne
                ? [
                    {
                      label: `System One /typesafe${binding ? ` → ${binding}` : ''}`,
                      color: KIND_COLOR.decision,
                      hint: GLOSSARY.systemOne.body,
                    },
                  ]
                : []),
            ]}
          />
        </span>
      </div>

      <nav className="topbar__nav" aria-label="Views">
        {VIEWS.map((view) => (
          <a
            key={view}
            href={ROUTE_PATH[view]}
            aria-current={route === view ? 'page' : undefined}
            onClick={followRoute(view)}
          >
            {view}
          </a>
        ))}
      </nav>

      <div className="topbar__actions">
        {replay ? (
          <Pill tone="warning">replay · {state.fileName}</Pill>
        ) : state.telemetry ? (
          <span className="topbar__config">up {duration(state.telemetry.uptime_seconds)}</span>
        ) : null}
        {state.error && !replay ? <Pill tone="danger">{state.error}</Pill> : null}
        {route === 'dashboard' ? (
          <>
            <input
              ref={filePicker}
              type="file"
              accept=".jsonl,.json,application/jsonl,text/plain"
              className="sr-only"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void loadFile(file)
                event.target.value = ''
              }}
            />
            <button className="button" type="button" onClick={() => filePicker.current?.click()}>
              load jsonl
            </button>
          </>
        ) : null}
        {replay ? (
          <button className="button" type="button" onClick={resumeLive}>
            go live
          </button>
        ) : null}
      </div>
    </header>
  )
}
