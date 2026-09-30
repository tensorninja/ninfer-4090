// Record shapes, as emitted by src/serve/request_log.cpp (schema 22).
//
// GET /events streams these live and `--request-log-jsonl` appends the identical lines, so one
// set of types serves both the live dashboard and file replay. Only the fields the dashboard
// reads are declared; a record carries more, and unread fields are deliberately not mirrored
// here because every declaration is a second place to keep in step with the C++ formatter.
//
// Records are discriminated on `event`, never on `schema_version`, so an older log still
// replays: fields added by a later schema are declared optional and read as absent.
//
// One process serves two systems. Chat (OpenAI/Anthropic, `request_*`) generates text; System
// One (`/typesafe`, `decision_*`, schema 21 on) answers multiple-choice questions over a state
// with a prefill-only pass. Both run on the same weights, lanes, KV pool and adapter bank.

export interface RequestContext {
  request_id: number
  x_request_id: string
  protocol: string
  model: string
  adapter: string
  stream: boolean
  message_count: number
  tool_count: number
  has_tool_history: boolean
  enable_thinking: boolean
  requested_output_tokens: number
  prompt_cache_key_digest: string
}

/** Where a prompt's resident state came from. `none` means it was prefilled from zero. */
export type ContinuationSource = 'none' | 'l1' | 'l2' | 'l3'

export interface ContinuationDiagnostics {
  source: ContinuationSource
  alias_kind: string
  final_miss_reason: string
  restore_failure: string
  lookup_microseconds: number
  preflight_microseconds: number
  restore_microseconds: number
  restored_tokens: number
  restored_bytes: number
  destructive_rollback: boolean
  completion_publication_queued: boolean
  // Deepest prefix any preflighted candidate agreed with. `null` when nothing was preflighted,
  // and 0 only for genuine zero agreement. Against `prompt_tokens` on a request that still
  // prefilled from zero, this is how much recomputed state the cache actually had in hand.
  deepest_candidate_agreement?: number | null
}

/**
 * Restores broken down by the tier that served them, as cumulative totals paired with the
 * interval delta. The tier mix is the churn signal: a working set that no longer fits in L1
 * keeps its hit rate but pays a host or disk import on every turn instead of nothing.
 */
export interface ContinuationTierTotals {
  l1_restore_successes: number
  l2_restore_successes: number
  l3_restore_successes: number
  delta_l1_restore_successes: number
  delta_l2_restore_successes: number
  delta_l3_restore_successes: number
  l1_restored_tokens: number
  l2_restored_tokens: number
  l3_restored_tokens: number
  delta_l1_restored_tokens: number
  delta_l2_restored_tokens: number
  delta_l3_restored_tokens: number
  l1_restored_bytes: number
  l2_restored_bytes: number
  l3_restored_bytes: number
  delta_l1_restored_bytes: number
  delta_l2_restored_bytes: number
  delta_l3_restored_bytes: number
}

/** Per-request phase durations. `ttft` is the quantity the queue/restore/prefill split explains. */
export interface RequestTimings {
  prepare: number
  queue: number
  restore: number
  prefill: number
  decode: number
  publish: number
  vision: number
  ttft: number
  total: number
}

export interface SpeculativeStats {
  backend: string
  draft_window: number
  rounds: number
  drafted_tokens: number
  accepted_tokens: number
  fallback_steps: number
  /** Acceptance count per draft position; its decay is the shape of the MTP accept curve. */
  accepted_per_position: number[]
}

/**
 * A chat adapter corrects generation and is selected on `/v1` and the Anthropic route; a decision
 * adapter adds a System One pointer head and is selected on `/typesafe`. Both kinds share one
 * pool and one device bank.
 */
export type AdapterKind = 'generative' | 'decision'

/** One discovered adapter, schema 22. */
export interface PooledAdapter {
  name: string
  kind: AdapterKind
  rank: number
  /**
   * The id a client selects it by: `<model>-<name>` for a chat adapter, the bare name for a
   * decision adapter, which is a System One model in its own right.
   */
  model_id: string
  /** Decision adapters only: the head's calibration temperature and model-card fields. */
  temperature?: number
  pointer_dim?: number
  description?: string
  release_date?: string
}

/**
 * The adapter pool and its device bank, as reported by `server_start` and `/telemetry`.
 *
 * Every registered adapter shares one rank by construction, so a single rank and one bank byte
 * count describe the whole set. `device_bytes` is the bank's own arena, which sits outside the
 * weights arena and is otherwise visible only as a reduction in free memory.
 */
export interface AdapterInventory {
  /** Discovered pool size. Every entry is selectable regardless of what is resident. */
  count: number
  rank: number
  /** Device-resident slots the pool is swapped through. Absent in logs written before slots. */
  slots?: number
  device_bytes: number
  file_bytes: number
  /** Present from schema 22: every pooled adapter with its kind and served id. */
  pool?: PooledAdapter[]
  /** Schema 15–21: pooled names of every kind, without kind or served id. */
  names?: string[]
  /** Schema 21: the subset of `names` that are decision adapters. */
  decision_names?: string[]
}

/** The System One surface, schema 22. */
export interface SystemOneSurface {
  /** Whether this target runs decisions at all. */
  supported: boolean
  /** The SDK's default model name, served as an alias. */
  alias: string
  /** The decision adapter the alias resolves to; empty when none is bound. */
  binding: string
}

interface RecordEnvelope {
  schema_version: number
  server_instance_id: string
  timestamp_unix_ms: number
  artifact_type: string
}

export interface ServerStartRecord extends RecordEnvelope {
  event: 'server_start'
  argv: string[]
  artifact: {
    path: string
    target: string
    weights_id: string
    size_bytes: number
    tensor_count: number
    load_seconds: number
    upload_seconds: number
  }
  engine: {
    device: number
    kv_cache: string
    kv_capacity: number
    kv_capacity_mode: string
    max_concurrency: number
    max_context: number
    max_pending_requests: number
    pending_timeout_ms: number
    prefill_chunk: number
    prefix_reuse: boolean
    prefix_checkpoint_policy: string
    speculative_backend: string
    speculative_draft_window: number
    cuda_graph: boolean
    vision: boolean
    log_stats_interval_ms: number
    continuation_cache: {
      tiers: string
      policy: string
      directory: string
      namespace: string
      l1_capacity_mib: number
      l2_capacity_mib: number
      l3_capacity_mib: number
      persist_min_tokens: number
    }
  }
  /** Discovered LoRA pool, present from schema 15. Names appear even with no adapter traffic. */
  adapters?: AdapterInventory
  /** Present from schema 22. */
  systemone?: SystemOneSurface
  environment: {
    device: number
    gpu_name: string
    gpu_uuid: string
    total_device_memory_bytes: number
    compute_capability_major: number
    compute_capability_minor: number
    cuda_runtime_version: string
    cuda_driver_version: string
  }
  memory: Record<string, unknown> & { lora_bank_bytes?: number }
  server: {
    host: string
    port: number
    public_model_id: string
    cors_enabled: boolean
    api_key_configured: boolean
  }
}

export interface RequestStartRecord extends RecordEnvelope {
  event: 'request_start'
  request: RequestContext
}

export interface RequestDoneRecord extends RecordEnvelope {
  event: 'request_done'
  request: RequestContext
  result: {
    prompt_tokens: number
    completion_tokens: number
    computed_prefill_tokens: number
    prefix_cache_hit_tokens: number
    prefix_reuse_path: string
    finish_reason: string
    tool_call_count: number
    /** Present from schema 20: calls the output cut short before they closed. */
    tool_calls_incomplete?: number
    /** Present from schema 20: completed calls whose arguments are not a JSON object. */
    tool_calls_invalid_arguments?: number
    /** Present from schema 20: bytes of non-call output after the first call, dropped. */
    tool_call_discarded_bytes?: number
  }
  timings_seconds: RequestTimings
  continuation_cache: ContinuationDiagnostics
  speculative?: SpeculativeStats
}

export interface RequestErrorRecord extends RecordEnvelope {
  event: 'request_error'
  request: RequestContext
  error: { message: string }
}

/** A System One decision as submitted: its layout, before anything ran. */
export interface DecisionContext {
  request_id: number
  /** The `x-typesafe-request-id` the response carries. */
  x_request_id: string
  protocol: 'systemone'
  /** The requested model name, which may be the SDK alias. */
  model: string
  /** The decision adapter it resolved to. */
  adapter: string
  allow_prefix_reuse: boolean
  questions: number
  /** Options summed over every question. */
  options: number
  state_tokens: number
  state_truncated: boolean
  /** Question and option tokens summed over every branch. */
  branch_tokens: number
  longest_branch: number
  input_tokens: number
}

/**
 * Decision phases, in seconds.
 *
 * `queue` runs from submission to admission and includes `restore`, the import of a cached state
 * just before admission. `execution` runs from admission to the result and decomposes into
 * `state` (state prefill) and `branch` (the branch passes, readout and head). System One's own
 * `latency_ms` is `restore + execution`: the engine's time without the wait for a lane. `total`
 * is receipt to response, including `prepare` on the HTTP side.
 */
export interface DecisionTimings {
  prepare: number
  queue: number
  restore: number
  state: number
  branch: number
  execution: number
  total: number
}

export interface DecisionStartRecord extends RecordEnvelope {
  event: 'decision_start'
  request: DecisionContext
}

export interface DecisionDoneRecord extends RecordEnvelope {
  event: 'decision_done'
  request: DecisionContext
  result: {
    output_tokens: number
    /** State tokens resident in the lane or restored from a tier, rather than computed. */
    reused_state_tokens: number
    computed_state_tokens: number
    state_source: ContinuationSource
    branch_passes: number
    long_branch_chunks: number
    /** The lane that ran it. */
    slot: number
  }
  timings_seconds: DecisionTimings
  /** Null where the phase computed no tokens, or none measurably. */
  rates: { state_tok_s: number | null; branch_tok_s: number | null }
}

export interface DecisionErrorRecord extends RecordEnvelope {
  event: 'decision_error'
  request: DecisionContext
  /** `status` is the HTTP status System One answered with. */
  error: { status: number; message: string }
}

/**
 * One reporting interval. The reporter only emits when the interval had activity, so samples are
 * irregularly spaced and `interval_seconds` — not the gap between timestamps — is the authority
 * on the window a sample covers.
 */
export interface ThroughputRecord extends RecordEnvelope {
  event: 'throughput'
  interval_seconds: number
  throughput_tokens_per_second: { prefill: number; decode: number }
  /**
   * Board energy for the interval, or null where the board exposes no cumulative energy counter.
   *
   * `board_joules` is measured by the board itself. The prefill and decode splits are integrated
   * from instantaneous power at execution-unit boundaries and are estimates; `residual_fraction`
   * is how much of the measured total those splits plus the idle baseline fail to explain, so a
   * reader can tell a trustworthy split from a noisy one instead of being handed a bare number.
   */
  energy: {
    board_joules: number
    prefill_joules: number
    decode_joules: number
    idle_joules: number
    idle_watts: number
    accounted_seconds: number
    residual_joules: number
    residual_fraction: number
    joules_per_token: {
      served: number | null
      active: number | null
      prefill: number | null
      decode: number | null
    }
  } | null
  tokens: {
    /** Every prefilled token, chat and System One together. */
    computed_prefill: number
    /** Present from schema 22: the System One part of `computed_prefill`. */
    decision_prefill?: number
    committed_decode: number
  }
  /**
   * Present from schema 22. Adapters staged into a device slot, and requests whose admission
   * waited because every slot held an adapter that a running lane needed.
   */
  adapters?: {
    stages: number
    delta_stages: number
    slot_waits: number
    delta_slot_waits: number
  }
  decode_batch: { rounds: number; row_rounds: number; average_size: number | null }
  scheduler: {
    running: number
    prefilling: number
    decode_ready: number
    waiting: number
    admitted_requests: number
    delta_admitted_requests: number
    queue_seconds_total: number
    delta_queue_seconds: number
    rejected_overloaded: number
    rejected_queue_timeout: number
  }
  continuation_cache: {
    occupancy: {
      l1_entries: number
      l1_bytes: number
      l2_entries: number
      l2_bytes: number
      l3_entries: number
      l3_bytes: number
      l1_evictions: number
      l1_demotions: number
      kv_growth_attempts: number
      kv_growth_forced_spills: number
      kv_growth_curtailed: number
      delta_l1_evictions: number
      delta_l1_demotions: number
      // Capacity-driven evictions from the host and disk tiers. A TTL expiry is not counted here.
      l2_evictions?: number
      l2_evicted_bytes?: number
      l3_evictions?: number
      l3_evicted_bytes?: number
      delta_l2_evictions?: number
      delta_l2_evicted_bytes?: number
      delta_l3_evictions?: number
      delta_l3_evicted_bytes?: number
    }
    delta_lookup_hits: number
    delta_lookup_misses: number
    delta_restore_successes: number
    delta_restore_failures: number
    delta_restored_tokens: number
    delta_restored_bytes: number
    // A restore refused for shared-KV capacity with the candidate left live, so it is restore
    // pressure rather than a lost restore. Reported apart from `restore_failures`.
    restore_deferrals?: number
    delta_restore_deferrals?: number
    // Cumulative totals alongside the deltas. These are what let a replayed log reconstruct the
    // same cache view the live /telemetry snapshot reports.
    lookup_hits: number
    lookup_misses: number
    restore_successes: number
    restore_failures: number
    publication_successes: number
    publication_failures: number
    // Publication work that completed and was then discarded because the alias had moved on.
    publication_superseded?: number
    delta_publication_superseded?: number
    persistence_total: {
      queued: number
      coalesced: number
      successes: number
      failures: number
    }
    tiers: ContinuationTierTotals
    miss_reasons: Record<string, number>
  }
}

export type EngineRecord =
  | ServerStartRecord
  | RequestStartRecord
  | RequestDoneRecord
  | RequestErrorRecord
  | DecisionStartRecord
  | DecisionDoneRecord
  | DecisionErrorRecord
  | ThroughputRecord

export type RecordEvent = EngineRecord['event']

/** Every record event, which is also the SSE frame name `/events` sends it under. */
export const RECORD_EVENTS: readonly RecordEvent[] = [
  'server_start',
  'request_start',
  'request_done',
  'request_error',
  'decision_start',
  'decision_done',
  'decision_error',
  'throughput',
]

export function isEngineRecord(value: unknown): value is EngineRecord {
  if (typeof value !== 'object' || value === null) return false
  const event = (value as { event?: unknown }).event
  return RECORD_EVENTS.includes(event as RecordEvent)
}

/**
 * Parses one JSONL line. Returns null for blank lines and for a torn trailing record, which a
 * reader tailing a file the server is still appending to will encounter.
 */
export function parseRecordLine(line: string): EngineRecord | null {
  const trimmed = line.trim()
  if (trimmed.length === 0) return null
  try {
    const parsed: unknown = JSON.parse(trimmed)
    return isEngineRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}
