import { useMemo, useRef } from 'react'

import { Legend } from './components/charts'
import { Pill, StatusDot, type Tone } from './components/ui'
import { adapterPool, KIND_COLOR, summarizeDecisions, summarizeRequests } from './lib/derive'
import { duration } from './lib/format'
import { GLOSSARY } from './lib/glossary'
import { latest } from './lib/series'
import { cacheFromRecords } from './lib/telemetry'
import { useEngine } from './lib/use-engine'
import { AdaptersPanel } from './panels/AdaptersPanel'
import { CachePanel } from './panels/CachePanel'
import { ChurnPanel } from './panels/ChurnPanel'
import { EnergyPanel } from './panels/EnergyPanel'
import { GpuPanel } from './panels/GpuPanel'
import { Headline } from './panels/Headline'
import { LatencyPanel } from './panels/LatencyPanel'
import { MemoryPanel } from './panels/MemoryPanel'
import { RequestsPanel } from './panels/RequestsPanel'
import { SchedulerPanel } from './panels/SchedulerPanel'
import { SlotsPanel } from './panels/SlotsPanel'
import { SystemOnePanel } from './panels/SystemOnePanel'
import { ThroughputPanel } from './panels/ThroughputPanel'

const CONNECTION_TONE: Record<string, Tone> = {
  live: 'accent',
  replay: 'warning',
  connecting: 'warning',
  offline: 'danger',
}

export function App() {
  const { state, loadFile, resumeLive } = useEngine()
  const filePicker = useRef<HTMLInputElement>(null)

  // Summaries are pure over the retained record window; recomputing them on every 1 Hz telemetry
  // poll would be wasted work on a list that only changes when a request completes.
  const summary = useMemo(() => summarizeRequests(state.requests), [state.requests])
  const decisionSummary = useMemo(() => summarizeDecisions(state.decisions), [state.decisions])

  const engine = state.serverStart?.engine
  const server = state.serverStart?.server
  const replay = state.source === 'file'

  // Live inventory wins; a replayed log carries the same block on its own server_start, so a
  // registered-but-unused adapter is still named offline.
  const adapters = state.telemetry?.adapters ?? state.serverStart?.adapters
  const pool = useMemo(() => adapterPool(adapters), [adapters])
  const surface = state.telemetry?.systemone ?? state.serverStart?.systemone
  const lanes = state.telemetry?.scheduler.max_concurrency ?? engine?.max_concurrency
  // System One earns its readings when this server can answer a decision or a log recorded one.
  const systemOne = pool.some((entry) => entry.kind === 'decision') || state.decisions.length > 0

  // Live telemetry wins; a replayed log reconstructs the same cache view from its own records.
  const cache = useMemo(
    () => state.telemetry?.cache ?? cacheFromRecords(latest(state.throughput), state.serverStart),
    [state.telemetry, state.throughput, state.serverStart],
  )

  return (
    <div className="console">
      <header className="topbar">
        <div className="topbar__identity">
          <StatusDot tone={CONNECTION_TONE[state.connection]} />
          <strong>{state.telemetry?.model_id ?? server?.public_model_id ?? 'ninfer'}</strong>
          {engine ? (
            <span className="topbar__config">
              {engine.kv_cache} · {engine.max_concurrency} lanes ·{' '}
              {engine.max_context.toLocaleString('en-US')} ctx
              {engine.speculative_backend !== 'none' ? ` · ${engine.speculative_backend}` : ''}
              {engine.cuda_graph ? ' · graphs' : ''}
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
                ...(systemOne || surface?.supported
                  ? [
                      {
                        label: `System One /systemone${
                          surface?.binding ? ` → ${surface.binding}` : ''
                        }`,
                        color: KIND_COLOR.decision,
                        hint: GLOSSARY.systemOne.body,
                      },
                    ]
                  : []),
              ]}
            />
          </span>
        </div>

        <div className="topbar__actions">
          {replay ? (
            <Pill tone="warning">replay · {state.fileName}</Pill>
          ) : state.telemetry ? (
            <span className="topbar__config">up {duration(state.telemetry.uptime_seconds)}</span>
          ) : null}
          {state.error && !replay ? <Pill tone="danger">{state.error}</Pill> : null}
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
          {replay ? (
            <button className="button" type="button" onClick={resumeLive}>
              go live
            </button>
          ) : null}
        </div>
      </header>

      <Headline
        telemetry={state.telemetry}
        records={state.throughput}
        summary={summary}
        decisions={decisionSummary}
        systemOne={systemOne}
        engine={engine}
      />

      <main className="grid">
        <ThroughputPanel records={state.throughput} lanes={lanes} />
        <SchedulerPanel telemetry={state.telemetry} records={state.throughput} engine={engine} />
        <GpuPanel gpu={state.telemetry?.gpu} history={state.gpu} replay={replay} />
        <EnergyPanel records={state.throughput} gpu={state.telemetry?.gpu} replay={replay} />
        <MemoryPanel memory={state.telemetry?.memory} gpu={state.telemetry?.gpu} replay={replay} />
        <CachePanel cache={cache} summary={summary} replay={replay} />
        <ChurnPanel records={state.throughput} summary={summary} replay={replay} />
        <LatencyPanel summary={summary} />
        <SlotsPanel
          slots={state.telemetry?.slots}
          maxContext={state.telemetry?.memory.max_context ?? 0}
          replay={replay}
        />
        <RequestsPanel requests={state.requests} active={state.active} />
        <SystemOnePanel
          decisions={state.decisions}
          summary={decisionSummary}
          active={state.activeDecisions}
          errors={state.decisionErrors}
          pool={pool}
          surface={surface}
          replay={replay}
        />
        <AdaptersPanel
          inventory={adapters}
          bank={replay ? undefined : state.telemetry?.adapters}
          pool={pool}
          binding={surface?.binding}
          requests={state.requests}
          decisions={state.decisions}
          records={state.throughput}
        />
      </main>
    </div>
  )
}
