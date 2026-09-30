// Transport to a running ninfer-serve, plus the equivalent offline path for a JSONL file.
//
// Two channels with different jobs. GET /events carries the schema-22 record stream, which is
// append-only history: throughput samples, completed chat requests and completed System One
// decisions. GET /telemetry is polled for instantaneous state - board sensors, scheduler
// occupancy, lane work, the adapter bank, VRAM, cache fill - because those are levels rather than
// events and a snapshot cannot be reconstructed by replaying deltas.
//
// Because the stream is lossy under backpressure, nothing derived from it is treated as
// authoritative for current state; the poll always wins for levels.

import {
  parseRecordLine,
  RECORD_EVENTS,
  type DecisionDoneRecord,
  type DecisionErrorRecord,
  type DecisionStartRecord,
  type EngineRecord,
  type RequestDoneRecord,
  type RequestStartRecord,
  type ServerStartRecord,
  type ThroughputRecord,
} from './records'
import { activeThrottles, type Telemetry } from './telemetry'

/** One board sample, retained client-side because /telemetry reports levels without history. */
export interface GpuSample {
  at: number
  utilization: number
  memoryUtilization: number
  temperature: number
  power: number
  smClock: number
  /** Active non-benign throttle reasons, joined for display. Empty when the board is unlimited. */
  reasons: string
}

export type ConnectionState = 'connecting' | 'live' | 'offline' | 'replay'

export interface EngineState {
  connection: ConnectionState
  source: 'live' | 'file'
  fileName: string | null
  error: string | null
  telemetry: Telemetry | null
  serverStart: ServerStartRecord | null
  throughput: ThroughputRecord[]
  /** Completed chat requests. */
  requests: RequestDoneRecord[]
  /** Chat requests started and not yet finished. */
  active: RequestStartRecord[]
  /** Completed System One decisions. */
  decisions: DecisionDoneRecord[]
  /** Decisions started and not yet answered. */
  activeDecisions: DecisionStartRecord[]
  /**
   * Decisions that passed validation and then failed: refused at ingress, timed out in the queue,
   * cancelled by a disconnect, or faulted. A body kev would reject never gets this far.
   */
  decisionErrors: DecisionErrorRecord[]
  gpu: GpuSample[]
  droppedRecords: number
}

export const initialEngineState: EngineState = {
  connection: 'connecting',
  source: 'live',
  fileName: null,
  error: null,
  telemetry: null,
  serverStart: null,
  throughput: [],
  requests: [],
  active: [],
  decisions: [],
  activeDecisions: [],
  decisionErrors: [],
  gpu: [],
  droppedRecords: 0,
}

// Retention. Throughput covers an hour at the default 5s reporting interval; the GPU ring covers
// ten minutes at the 1s poll. Both are bounded so an overnight session cannot grow without limit.
const MAX_THROUGHPUT = 720
const MAX_REQUESTS = 500
const MAX_DECISIONS = 500
const MAX_DECISION_ERRORS = 100
const MAX_GPU = 600
const TELEMETRY_INTERVAL_MS = 1000

function ring<T>(items: readonly T[], next: T, limit: number): T[] {
  const out = items.length >= limit ? items.slice(items.length - limit + 1) : items.slice()
  out.push(next)
  return out
}

export class EngineClient {
  private state: EngineState = initialEngineState
  private readonly notify: (state: EngineState) => void
  private events: EventSource | null = null
  private poll: ReturnType<typeof setInterval> | null = null
  private stopped = false

  constructor(notify: (state: EngineState) => void) {
    this.notify = notify
  }

  start(): void {
    this.stopped = false
    this.openStream()
    this.startPolling()
  }

  stop(): void {
    this.stopped = true
    this.events?.close()
    this.events = null
    if (this.poll !== null) clearInterval(this.poll)
    this.poll = null
  }

  private update(patch: Partial<EngineState>): void {
    this.state = { ...this.state, ...patch }
    this.notify(this.state)
  }

  /**
   * Whether a live poll should be discarded. Read through a method rather than inline so the
   * check is re-evaluated after an await: narrowing `this.state.source` at the top of a function
   * otherwise persists across the suspension point, which is precisely the state that can change
   * while a request is in flight.
   */
  private discardPoll(): boolean {
    return this.stopped || this.state.source === 'file'
  }

  private startPolling(): void {
    if (this.poll !== null) clearInterval(this.poll)
    void this.readTelemetry()
    this.poll = setInterval(() => void this.readTelemetry(), TELEMETRY_INTERVAL_MS)
  }

  private async readTelemetry(): Promise<void> {
    if (this.discardPoll()) return
    try {
      const response = await fetch('/telemetry', { headers: { accept: 'application/json' } })
      if (!response.ok) throw new Error(`/telemetry responded ${response.status}`)
      const telemetry = (await response.json()) as Telemetry
      // Re-check after awaiting: a file may have been loaded while this request was in flight,
      // and merging a live snapshot into replay state would show the running engine's board and
      // slots beside a historical log.
      if (this.discardPoll()) return
      const gpu = telemetry.gpu
      const patch: Partial<EngineState> = { telemetry, error: null, connection: 'live' }
      if (gpu.available) {
        patch.gpu = ring(
          this.state.gpu,
          {
            at: telemetry.timestamp_unix_ms,
            utilization: gpu.utilization_gpu_percent ?? 0,
            memoryUtilization: gpu.utilization_memory_percent ?? 0,
            temperature: gpu.temperature_c ?? 0,
            power: gpu.power_watts ?? 0,
            smClock: gpu.sm_clock_mhz ?? 0,
            reasons: activeThrottles(gpu).join(' · '),
          },
          MAX_GPU,
        )
      }
      this.update(patch)
    } catch (error) {
      if (this.discardPoll()) return
      this.update({
        connection: 'offline',
        error: error instanceof Error ? error.message : 'telemetry unavailable',
      })
    }
  }

  private openStream(): void {
    this.events?.close()
    const source = new EventSource('/events')
    this.events = source
    // The server names each frame after the record's own `event` field, so one listener per
    // record type replaces a discriminating switch on the client.
    for (const name of RECORD_EVENTS) {
      source.addEventListener(name, (message) => {
        const record = parseRecordLine((message as MessageEvent<string>).data)
        if (record !== null) this.ingest(record)
      })
    }
    // EventSource reconnects on its own using the server's `retry` hint; the poll is what decides
    // whether the dashboard reads as live, so an error here only records the transition.
    source.onerror = () => {
      if (this.stopped) return
      if (this.state.connection === 'live') this.update({ connection: 'connecting' })
    }
  }

  private ingest(record: EngineRecord): void {
    switch (record.event) {
      case 'server_start':
        // A new instance id means the engine restarted underneath us: history from the previous
        // process describes a different configuration and must not share an axis with this one.
        if (
          this.state.serverStart !== null &&
          this.state.serverStart.server_instance_id !== record.server_instance_id
        ) {
          this.update({
            serverStart: record,
            throughput: [],
            requests: [],
            active: [],
            decisions: [],
            activeDecisions: [],
            decisionErrors: [],
            gpu: [],
          })
          return
        }
        this.update({ serverStart: record })
        return
      case 'throughput':
        this.update({ throughput: ring(this.state.throughput, record, MAX_THROUGHPUT) })
        return
      case 'request_start':
        this.update({ active: [...this.state.active, record] })
        return
      case 'request_done':
        this.update({
          requests: ring(this.state.requests, record, MAX_REQUESTS),
          active: this.state.active.filter(
            (started) => started.request.request_id !== record.request.request_id,
          ),
        })
        return
      case 'request_error':
        this.update({
          active: this.state.active.filter(
            (started) => started.request.request_id !== record.request.request_id,
          ),
        })
        return
      case 'decision_start':
        this.update({ activeDecisions: [...this.state.activeDecisions, record] })
        return
      case 'decision_done':
        this.update({
          decisions: ring(this.state.decisions, record, MAX_DECISIONS),
          activeDecisions: this.state.activeDecisions.filter(
            (started) => started.request.request_id !== record.request.request_id,
          ),
        })
        return
      case 'decision_error':
        this.update({
          decisionErrors: ring(this.state.decisionErrors, record, MAX_DECISION_ERRORS),
          activeDecisions: this.state.activeDecisions.filter(
            (started) => started.request.request_id !== record.request.request_id,
          ),
        })
        return
    }
  }

  /**
   * Replaces live state with a parsed JSONL file. Records from more than one server instance can
   * share a file (it is opened in append mode), so only the last instance is kept: mixing two
   * configurations on one axis would misattribute every derived figure.
   */
  loadFile(name: string, text: string): void {
    this.stop()
    const records: EngineRecord[] = []
    for (const line of text.split('\n')) {
      const record = parseRecordLine(line)
      if (record !== null) records.push(record)
    }
    const instances = records.map((record) => record.server_instance_id)
    const lastInstance = instances.length === 0 ? null : instances[instances.length - 1]!
    const scoped = records.filter((record) => record.server_instance_id === lastInstance)

    const droppedInstances = new Set(instances).size - 1
    this.state = {
      ...initialEngineState,
      connection: 'replay',
      source: 'file',
      fileName: name,
      serverStart:
        (scoped.find((record) => record.event === 'server_start') as
          ServerStartRecord | undefined) ?? null,
      throughput: scoped.filter((r): r is ThroughputRecord => r.event === 'throughput'),
      requests: scoped.filter((r): r is RequestDoneRecord => r.event === 'request_done'),
      decisions: scoped.filter((r): r is DecisionDoneRecord => r.event === 'decision_done'),
      decisionErrors: scoped.filter((r): r is DecisionErrorRecord => r.event === 'decision_error'),
      droppedRecords: droppedInstances,
      // `lastInstance` is taken from a record that exists, so an empty scope means nothing parsed
      // at all rather than a run that was filtered out.
      error: scoped.length === 0 ? `${name} contains no engine records` : null,
    }
    this.notify(this.state)
  }

  /** Leaves replay and reconnects to the engine. */
  resumeLive(): void {
    this.state = { ...initialEngineState }
    this.notify(this.state)
    this.start()
  }
}
