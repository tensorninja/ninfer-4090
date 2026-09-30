// GET /telemetry payload, as built by HttpServer::handle_telemetry.
//
// This is a complete snapshot rather than a delta, which is what lets the dashboard resynchronize
// after dropped stream records without replaying anything.

import type {
  AdapterInventory,
  PooledAdapter,
  ServerStartRecord,
  SystemOneSurface,
  ThroughputRecord,
} from './records'

export interface GpuTelemetry {
  available: boolean
  error?: string
  name?: string
  uuid?: string
  driver_version?: string
  temperature_c?: number
  fan_percent?: number
  /** Driver-averaged over a trailing 1 s window, which is the right statistic at a 1 Hz poll. */
  power_watts?: number
  power_limit_watts?: number
  /** False on boards with no cumulative energy counter; every energy field is then null. */
  energy_available?: boolean
  /** Board energy since driver load. Only meaningful as a difference within one driver session. */
  energy_joules_total?: number | null
  /** Board energy since this server started, accumulated by the interval reporter. */
  server_energy_joules?: number | null
  /** Board draw measured during intervals that ran no execution unit at all. */
  idle_watts?: number | null
  utilization_gpu_percent?: number
  utilization_memory_percent?: number
  sm_clock_mhz?: number
  sm_clock_max_mhz?: number
  memory_clock_mhz?: number
  memory_used_bytes?: number
  memory_total_bytes?: number
  pcie_rx_bytes_per_second?: number
  pcie_tx_bytes_per_second?: number
  throttle_reasons?: string[]
}

// Boards report throttle reasons that are not faults. `gpu_idle` means there is nothing to run,
// and `applications_clocks_setting` is a deliberate cap, so neither indicates a limit on work the
// engine wanted to do.
const BENIGN_THROTTLES = new Set(['gpu_idle', 'applications_clocks_setting'])

/** Throttle reasons that actually constrain achievable throughput. */
export function activeThrottles(gpu: GpuTelemetry | undefined): string[] {
  return (gpu?.throttle_reasons ?? []).filter((reason) => !BENIGN_THROTTLES.has(reason))
}

export interface SchedulerTelemetry {
  running: number
  prefilling: number
  decode_ready: number
  waiting: number
  max_concurrency: number
  max_pending_requests: number
  admitted_requests: number
  queue_seconds_total: number
  rejected_overloaded: number
  rejected_queue_timeout: number
  decode_rounds: number
  decode_row_rounds: number
  /** Wall clock of the single execution thread, split by unit. These sum to its busy time. */
  worker_seconds: {
    decode: number
    prefill: number
    admission: number
    publish: number
    upkeep: number
  }
  /** Decomposition of `worker_seconds.admission`, over the calls that entered the attempt. */
  worker_admission: {
    calls: number
    plan_seconds: number
    restore_seconds: number
    commit_seconds: number
  }
  worker_decode_rounds: number
  worker_prefill_steps: number
}

export interface CacheTierTelemetry {
  entries: number
  bytes: number
  capacity_bytes: number
}

/** Capacity-driven evictions from a tier. A TTL expiry is deliberately not counted as one. */
export interface CacheTierEvictions {
  evictions: number
  evicted_bytes: number
}

export interface CacheTelemetry {
  l1: CacheTierTelemetry & { evictions: number; demotions: number }
  l2: CacheTierTelemetry & CacheTierEvictions
  l3: CacheTierTelemetry & CacheTierEvictions
  kv_growth: { attempts: number; forced_spills: number; curtailed: number }
  restore_successes: number
  restore_failures: number
  restore_deferrals: number
  lookup_hits: number
  lookup_misses: number
  publication_successes: number
  publication_failures: number
  persistence_successes: number
  persistence_failures: number
}

export interface ArenaTelemetry {
  capacity_bytes: number
  used_bytes: number
  peak_used_bytes: number
}

export interface MemoryTelemetry {
  device: number
  max_context: number
  kv_capacity: number
  kv_capacity_page_groups: number
  kv_capacity_max_page_groups: number
  weights: ArenaTelemetry
  sequence: ArenaTelemetry
  workspace: ArenaTelemetry
  request_transient: ArenaTelemetry
  kv_payload_bytes: number
  text_kv_bytes: number
  mtp_kv_bytes: number
  gdn_state_bytes: number
  dflash_kv_bytes: number
  replay_records_bytes: number
  cuda_graph_observed_bytes: number
  cuda_graph_allowance_bytes: number
  available_after_startup_bytes: number
  available_after_weights_bytes: number
  /** Resident adapter bank, held outside the weights arena. */
  lora_bank_bytes: number
}

/** What a lane is running, or the kind of state it retains once idle. */
export type LaneWork = 'none' | 'generation' | 'decision'

export interface SlotTelemetry {
  processing: boolean
  retained: boolean
  prompt_tokens: number
  cached_tokens: number
  session_digest: string
  checkpoints: number
  /** `generation` is chat; `decision` is System One. `none` for an empty lane. */
  work: LaneWork
  /** The adapter the lane runs or retains state under; empty for the base weights. */
  adapter: string
}

/** One device slot of the adapter bank. */
export interface BankSlot {
  /** Pool name of the occupant; empty for a slot never staged. */
  adapter: string
  /** A running lane executes against the occupant, so admission cannot swap it out. */
  pinned: boolean
}

/** The live bank: the whole pool, what each device slot holds, and what swapping has cost. */
export interface AdapterTelemetry extends AdapterInventory {
  pool: PooledAdapter[]
  resident: BankSlot[]
  /** Adapters staged into a slot since startup, and the execution-thread time they took. */
  stages: number
  stage_seconds: number
  /** Requests whose admission waited at least once because every slot was pinned. */
  slot_waits: number
}

export interface Telemetry {
  timestamp_unix_ms: number
  server_instance_id: string
  uptime_seconds: number
  attached: boolean
  model_id: string
  gpu: GpuTelemetry
  scheduler: SchedulerTelemetry
  cache: CacheTelemetry
  memory: MemoryTelemetry
  slots: SlotTelemetry[]
  events: { jsonl_enabled: boolean; subscribers: number }
  adapters: AdapterTelemetry
  systemone: SystemOneSurface
}

/**
 * Reconstructs the live cache view from a replayed log.
 *
 * A throughput record carries the same occupancy and cumulative counters the live snapshot
 * reports, and the capacities they are measured against come from the log's own `server_start`.
 * Together those are sufficient, so cache health is fully readable offline - unlike board
 * telemetry and lane occupancy, which are never recorded.
 */
export function cacheFromRecords(
  sample: ThroughputRecord | null,
  start: ServerStartRecord | null,
): CacheTelemetry | undefined {
  if (sample === null || start === null) return undefined
  const occupancy = sample.continuation_cache.occupancy
  const configured = start.engine.continuation_cache
  const mib = 1024 * 1024
  return {
    l1: {
      entries: occupancy.l1_entries,
      bytes: occupancy.l1_bytes,
      capacity_bytes: configured.l1_capacity_mib * mib,
      evictions: occupancy.l1_evictions,
      demotions: occupancy.l1_demotions,
    },
    l2: {
      entries: occupancy.l2_entries,
      bytes: occupancy.l2_bytes,
      capacity_bytes: configured.l2_capacity_mib * mib,
      evictions: occupancy.l2_evictions ?? 0,
      evicted_bytes: occupancy.l2_evicted_bytes ?? 0,
    },
    l3: {
      entries: occupancy.l3_entries,
      bytes: occupancy.l3_bytes,
      capacity_bytes: configured.l3_capacity_mib * mib,
      evictions: occupancy.l3_evictions ?? 0,
      evicted_bytes: occupancy.l3_evicted_bytes ?? 0,
    },
    kv_growth: {
      attempts: occupancy.kv_growth_attempts,
      forced_spills: occupancy.kv_growth_forced_spills,
      curtailed: occupancy.kv_growth_curtailed,
    },
    restore_successes: sample.continuation_cache.restore_successes,
    restore_failures: sample.continuation_cache.restore_failures,
    restore_deferrals: sample.continuation_cache.restore_deferrals ?? 0,
    lookup_hits: sample.continuation_cache.lookup_hits,
    lookup_misses: sample.continuation_cache.lookup_misses,
    publication_successes: sample.continuation_cache.publication_successes,
    publication_failures: sample.continuation_cache.publication_failures,
    persistence_successes: sample.continuation_cache.persistence_total.successes,
    persistence_failures: sample.continuation_cache.persistence_total.failures,
  }
}

/** Prometheus text body from GET /metrics, flattened to a name→value map. */
export function parsePrometheus(body: string): Map<string, number> {
  const out = new Map<string, number>()
  for (const line of body.split('\n')) {
    if (line.length === 0 || line.startsWith('#')) continue
    const split = line.lastIndexOf(' ')
    if (split <= 0) continue
    const value = Number(line.slice(split + 1))
    if (Number.isFinite(value)) out.set(line.slice(0, split).trim(), value)
  }
  return out
}
