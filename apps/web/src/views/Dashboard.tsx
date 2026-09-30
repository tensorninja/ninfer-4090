import { useMemo } from 'react'

import { summarizeDecisions, summarizeRequests, type PoolEntry } from '../lib/derive'
import type { EngineState } from '../lib/engine-client'
import type { AdapterInventory, SystemOneSurface } from '../lib/records'
import { latest } from '../lib/series'
import { cacheFromRecords } from '../lib/telemetry'
import { AdaptersPanel } from '../panels/AdaptersPanel'
import { CachePanel } from '../panels/CachePanel'
import { ChurnPanel } from '../panels/ChurnPanel'
import { EnergyPanel } from '../panels/EnergyPanel'
import { GpuPanel } from '../panels/GpuPanel'
import { Headline } from '../panels/Headline'
import { LatencyPanel } from '../panels/LatencyPanel'
import { MemoryPanel } from '../panels/MemoryPanel'
import { RequestsPanel } from '../panels/RequestsPanel'
import { SchedulerPanel } from '../panels/SchedulerPanel'
import { SlotsPanel } from '../panels/SlotsPanel'
import { SystemOnePanel } from '../panels/SystemOnePanel'
import { ThroughputPanel } from '../panels/ThroughputPanel'

export function Dashboard({
  state,
  adapters,
  pool,
  surface,
  systemOne,
}: {
  state: EngineState
  adapters: AdapterInventory | undefined
  pool: PoolEntry[]
  surface: SystemOneSurface | undefined
  systemOne: boolean
}) {
  // Summaries are pure over the retained record window; recomputing them on every 1 Hz telemetry
  // poll would be wasted work on a list that only changes when a request completes.
  const summary = useMemo(() => summarizeRequests(state.requests), [state.requests])
  const decisionSummary = useMemo(() => summarizeDecisions(state.decisions), [state.decisions])

  const engine = state.serverStart?.engine
  const replay = state.source === 'file'
  const lanes = state.telemetry?.scheduler.max_concurrency ?? engine?.max_concurrency

  // Live telemetry wins; a replayed log reconstructs the same cache view from its own records.
  const cache = useMemo(
    () => state.telemetry?.cache ?? cacheFromRecords(latest(state.throughput), state.serverStart),
    [state.telemetry, state.throughput, state.serverStart],
  )

  return (
    <>
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
    </>
  )
}
