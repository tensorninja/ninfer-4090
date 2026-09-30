import { expect, test } from 'bun:test'

import {
  adapterPool,
  percentile,
  summarizeByAdapter,
  summarizeChurn,
  summarizeDecisions,
  summarizeDecisionsByAdapter,
  summarizeEnergy,
  summarizeRequests,
} from './derive'
import { energyPerMillionTokens, WH_PER_MILLION_TOKENS_PER_JOULE } from './format'
import {
  parseRecordLine,
  type AdapterInventory,
  type ContinuationSource,
  type DecisionDoneRecord,
  type RequestDoneRecord,
  type ThroughputRecord,
} from './records'

// Fixture shaped like real schema-19 records: a mix of tier sources, one cold miss, one restore
// failure. Expected values below were produced by the reference implementation
// (`cache_health.py`) over this same input, so a change here that drifts from the script fails.
function done(
  id: number,
  values: {
    prompt: number
    computed: number
    generated: number
    source: 'none' | 'l1' | 'l2' | 'l3'
    missReason?: string
    restoreFailure?: string
    reusePath?: string
    queue: number
    restore: number
    prefill: number
    decode: number
    ttft: number
    restoredTokens?: number
    restoredBytes?: number
    restoreMicroseconds?: number
    preflightMicroseconds?: number
    drafted?: number
    accepted?: number
    perPosition?: number[]
    adapter?: string
    agreement?: number | null
  },
): RequestDoneRecord {
  return {
    event: 'request_done',
    schema_version: 19,
    server_instance_id: 'serve-test-1',
    timestamp_unix_ms: 1_700_000_000_000 + id * 1000,
    artifact_type: 'ninfer_serve_request_log',
    request: {
      request_id: id,
      x_request_id: `req_${id}`,
      protocol: 'openai_responses',
      model: values.adapter ? `qwen3.8-27b-${values.adapter}` : 'qwen3.8-27b',
      adapter: values.adapter ?? '',
      stream: true,
      message_count: 4,
      tool_count: 0,
      has_tool_history: false,
      enable_thinking: true,
      requested_output_tokens: 4096,
      prompt_cache_key_digest: '',
    },
    result: {
      prompt_tokens: values.prompt,
      completion_tokens: values.generated,
      computed_prefill_tokens: values.computed,
      prefix_cache_hit_tokens: values.prompt - values.computed,
      prefix_reuse_path: values.reusePath ?? 'restore_turn_checkpoint',
      finish_reason: 'stop_token',
      tool_call_count: 0,
    },
    timings_seconds: {
      prepare: 0,
      queue: values.queue,
      restore: values.restore,
      prefill: values.prefill,
      decode: values.decode,
      publish: 0,
      vision: 0,
      ttft: values.ttft,
      total: values.ttft + values.decode,
    },
    continuation_cache: {
      source: values.source,
      alias_kind: 'routed_session',
      final_miss_reason: values.missReason ?? 'none',
      restore_failure: values.restoreFailure ?? 'none',
      lookup_microseconds: 0,
      preflight_microseconds: values.preflightMicroseconds ?? 0,
      restore_microseconds: values.restoreMicroseconds ?? 0,
      restored_tokens: values.restoredTokens ?? 0,
      restored_bytes: values.restoredBytes ?? 0,
      destructive_rollback: false,
      completion_publication_queued: true,
      deepest_candidate_agreement: values.agreement ?? null,
    },
    speculative:
      values.drafted === undefined
        ? undefined
        : {
            backend: 'mtp',
            draft_window: 3,
            rounds: 100,
            drafted_tokens: values.drafted,
            accepted_tokens: values.accepted ?? 0,
            fallback_steps: 0,
            accepted_per_position: values.perPosition ?? [],
          },
  }
}

// Only the churn and energy inputs vary; everything else is a plausible constant, because
// `summarizeChurn` reads nothing but the eviction, restore-tier and publication deltas, and
// `summarizeEnergy` reads nothing but the energy block and the interval token counts.
function throughput(
  id: number,
  values: {
    evictions?: number
    demotions?: number
    l1?: number
    l2?: number
    l3?: number
    l2Evictions?: number
    l3Evictions?: number
    deferrals?: number
    superseded?: number
    prefillTokens?: number
    decodeTokens?: number
    energy?: ThroughputRecord['energy']
  },
): ThroughputRecord {
  const zeroTier = {
    l1_restore_successes: 0,
    l2_restore_successes: 0,
    l3_restore_successes: 0,
    l1_restored_tokens: 0,
    l2_restored_tokens: 0,
    l3_restored_tokens: 0,
    delta_l1_restored_tokens: 0,
    delta_l2_restored_tokens: 0,
    delta_l3_restored_tokens: 0,
    l1_restored_bytes: 0,
    l2_restored_bytes: 0,
    l3_restored_bytes: 0,
    delta_l1_restored_bytes: 0,
    delta_l2_restored_bytes: 0,
    delta_l3_restored_bytes: 0,
  }
  return {
    event: 'throughput',
    schema_version: 19,
    server_instance_id: 'serve-test-1',
    timestamp_unix_ms: 1_700_000_000_000 + id * 10_000,
    artifact_type: 'ninfer_serve_request_log',
    interval_seconds: 10,
    throughput_tokens_per_second: { prefill: 0, decode: 0 },
    energy: values.energy ?? null,
    tokens: {
      computed_prefill: values.prefillTokens ?? 0,
      committed_decode: values.decodeTokens ?? 0,
    },
    decode_batch: { rounds: 0, row_rounds: 0, average_size: null },
    scheduler: {
      running: 0,
      prefilling: 0,
      decode_ready: 0,
      waiting: 0,
      admitted_requests: 0,
      delta_admitted_requests: 0,
      queue_seconds_total: 0,
      delta_queue_seconds: 0,
      rejected_overloaded: 0,
      rejected_queue_timeout: 0,
    },
    continuation_cache: {
      occupancy: {
        l1_entries: 0,
        l1_bytes: 0,
        l2_entries: 0,
        l2_bytes: 0,
        l3_entries: 0,
        l3_bytes: 0,
        l1_evictions: 0,
        l1_demotions: 0,
        kv_growth_attempts: 0,
        kv_growth_forced_spills: 0,
        kv_growth_curtailed: 0,
        delta_l1_evictions: values.evictions ?? 0,
        delta_l1_demotions: values.demotions ?? 0,
        l2_evictions: 0,
        l2_evicted_bytes: 0,
        l3_evictions: 0,
        l3_evicted_bytes: 0,
        delta_l2_evictions: values.l2Evictions ?? 0,
        delta_l2_evicted_bytes: 0,
        delta_l3_evictions: values.l3Evictions ?? 0,
        delta_l3_evicted_bytes: 0,
      },
      delta_lookup_hits: 0,
      delta_lookup_misses: 0,
      delta_restore_successes: 0,
      delta_restore_failures: 0,
      delta_restored_tokens: 0,
      delta_restored_bytes: 0,
      restore_deferrals: 0,
      delta_restore_deferrals: values.deferrals ?? 0,
      lookup_hits: 0,
      lookup_misses: 0,
      restore_successes: 0,
      restore_failures: 0,
      publication_successes: 0,
      publication_failures: 0,
      publication_superseded: 0,
      delta_publication_superseded: values.superseded ?? 0,
      persistence_total: { queued: 0, coalesced: 0, successes: 0, failures: 0 },
      tiers: {
        ...zeroTier,
        delta_l1_restore_successes: values.l1 ?? 0,
        delta_l2_restore_successes: values.l2 ?? 0,
        delta_l3_restore_successes: values.l3 ?? 0,
      },
      miss_reasons: {},
    },
  }
}

const FIXTURE: RequestDoneRecord[] = [
  done(1, {
    prompt: 1000,
    computed: 100,
    generated: 200,
    source: 'l2',
    queue: 1,
    restore: 0.5,
    prefill: 0.25,
    decode: 2,
    ttft: 1.75,
    restoredTokens: 900,
    restoredBytes: 9_000_000,
    restoreMicroseconds: 500_000,
    preflightMicroseconds: 100_000,
    drafted: 300,
    accepted: 200,
    perPosition: [120, 50, 30],
  }),
  done(2, {
    prompt: 2000,
    computed: 2000,
    generated: 400,
    source: 'none',
    missReason: 'no_lane',
    reusePath: 'full_reset',
    // Preflight agreed with 1800 of the 2000 prompt tokens and the request still prefilled all
    // 2000, so 1800 tokens of the cache's own state were recomputed.
    agreement: 1800,
    queue: 4,
    restore: 0,
    prefill: 1,
    decode: 4,
    ttft: 5,
    drafted: 600,
    accepted: 300,
    perPosition: [180, 80, 40],
  }),
  done(3, {
    adapter: 'caveman',
    prompt: 4000,
    computed: 400,
    generated: 5000,
    source: 'l3',
    queue: 2,
    restore: 1.5,
    prefill: 0.5,
    decode: 50,
    ttft: 4,
    restoredTokens: 3600,
    restoredBytes: 36_000_000,
    restoreMicroseconds: 1_500_000,
    preflightMicroseconds: 300_000,
    drafted: 900,
    accepted: 600,
    perPosition: [400, 150, 50],
  }),
  done(4, {
    prompt: 500,
    computed: 0,
    generated: 50,
    source: 'l1',
    queue: 0.5,
    restore: 0,
    prefill: 0,
    decode: 0.5,
    ttft: 0.5,
    restoredTokens: 500,
    restoredBytes: 5_000_000,
    restoreMicroseconds: 10_000,
    preflightMicroseconds: 5_000,
  }),
  done(5, {
    adapter: 'caveman',
    prompt: 3000,
    computed: 3000,
    generated: 100,
    source: 'none',
    missReason: 'not_attempted',
    restoreFailure: 'kv_reservation',
    reusePath: 'full_reset',
    // Nothing was preflighted, so there is no evidence the cache held this prefix at all.
    agreement: null,
    queue: 3,
    restore: 0,
    prefill: 1.5,
    decode: 1,
    ttft: 4.5,
  }),
]

test('percentile matches the reference truncating nearest-rank rule', () => {
  const values = [1, 2, 3, 4, 5]
  // int(q * len) indexes into the sorted array: 0.5 * 5 -> 2, 0.9 * 5 -> 4.
  expect(percentile(values, 0.5)).toBe(3)
  expect(percentile(values, 0.9)).toBe(5)
  expect(percentile(values, 0)).toBe(1)
  expect(percentile(values, 1)).toBe(5)
  expect(percentile([], 0.5)).toBe(0)
})

test('request summary reproduces the reference aggregations', () => {
  const summary = summarizeRequests(FIXTURE)

  expect(summary.count).toBe(5)

  // Sorted prompts [500, 1000, 2000, 3000, 4000]: p50 -> index 2, p90 -> index 4.
  expect(summary.promptTokens.p50).toBe(2000)
  expect(summary.promptTokens.p90).toBe(4000)
  expect(summary.promptTokens.max).toBe(4000)

  expect(summary.totalPromptTokens).toBe(10500)
  expect(summary.totalComputedPrefill).toBe(5500)
  expect(summary.prefillAvoided).toBeCloseTo(1 - 5500 / 10500, 12)

  expect(summary.generated.max).toBe(5000)
  expect(summary.overLongGenerations).toBe(1)

  expect(summary.bySource).toEqual({ l2: 1, none: 2, l3: 1, l1: 1 })
  expect(summary.byMissReason).toEqual({ no_lane: 1, not_attempted: 1 })
  expect(summary.byRestoreFailure).toEqual({ kv_reservation: 1 })
  expect(summary.byReusePath).toEqual({ restore_turn_checkpoint: 3, full_reset: 2 })
  expect(summary.byFinishReason).toEqual({ stop_token: 5 })

  // Sorted TTFT [0.5, 1.75, 4, 4.5, 5].
  expect(summary.ttft.p50).toBe(4)
  expect(summary.ttft.p90).toBe(5)

  // Shares are over summed TTFT (15.75), not per-request means.
  expect(summary.ttftShare.queue).toBeCloseTo(10.5 / 15.75, 12)
  expect(summary.ttftShare.restore).toBeCloseTo(2 / 15.75, 12)
  expect(summary.ttftShare.prefill).toBeCloseTo(3.25 / 15.75, 12)

  expect(summary.restores.count).toBe(3)
  expect(summary.restores.totalTokens).toBe(5000)
  expect(summary.restores.totalBytes).toBe(50_000_000)
  expect(summary.restores.meanSeconds).toBeCloseTo(0.67, 10)
  expect(summary.restores.meanPreflightSeconds).toBeCloseTo(0.135, 10)

  expect(summary.speculative.drafted).toBe(1800)
  expect(summary.speculative.accepted).toBe(1100)
  expect(summary.speculative.acceptRate).toBeCloseTo(1100 / 1800, 12)
  expect(summary.speculative.perPosition).toEqual([700, 280, 120])
})

test('an empty window summarizes without dividing by zero', () => {
  const summary = summarizeRequests([])
  expect(summary.count).toBe(0)
  expect(summary.prefillAvoided).toBe(0)
  expect(summary.ttftShare).toEqual({ queue: 0, restore: 0, prefill: 0 })
  expect(summary.speculative.acceptRate).toBe(0)
  expect(summary.coverageWaste).toEqual({
    requests: 0,
    tokens: 0,
    requestShare: 0,
    prefillShare: 0,
  })
})

test('coverage waste counts prefill spent on state the cache had agreed with', () => {
  const summary = summarizeRequests(FIXTURE)

  // Only request 2 qualifies: it prefilled from zero with 1800 tokens of agreement and no lane
  // reuse. Request 5 also prefilled from zero but preflighted nothing, so it is not evidence.
  expect(summary.coverageWaste.requests).toBe(1)
  expect(summary.coverageWaste.tokens).toBe(1800)
  expect(summary.coverageWaste.requestShare).toBeCloseTo(1 / 5, 12)
  expect(summary.coverageWaste.prefillShare).toBeCloseTo(1800 / 5500, 12)
})

test('coverage waste excludes agreement the lane had already reused', () => {
  // `not_deeper`: preflight agreed with 800 tokens, but the resident frontier already covered
  // those same 800, so nothing was recomputed and this is not waste.
  const covered = done(10, {
    prompt: 1000,
    computed: 200,
    generated: 10,
    source: 'none',
    missReason: 'not_deeper',
    agreement: 800,
    queue: 0,
    restore: 0,
    prefill: 0.1,
    decode: 0.1,
    ttft: 0.1,
  })
  // True zero agreement is a real measurement, and it is not waste either.
  const noAgreement = { ...covered, continuation_cache: { ...covered.continuation_cache } }
  noAgreement.continuation_cache.deepest_candidate_agreement = 0
  // A restore that happened is never waste, whatever it agreed with.
  const restored = { ...covered, continuation_cache: { ...covered.continuation_cache } }
  restored.continuation_cache.source = 'l2'
  restored.continuation_cache.deepest_candidate_agreement = 900

  const summary = summarizeRequests([covered, noAgreement, restored])
  expect(summary.coverageWaste.requests).toBe(0)
  expect(summary.coverageWaste.tokens).toBe(0)
})

test('churn sums interval deltas and splits surviving from lost evictions', () => {
  const window = [
    throughput(1, { evictions: 4, demotions: 3, l1: 2, l2: 1, l3: 0, deferrals: 1 }),
    throughput(2, {
      evictions: 6,
      demotions: 2,
      l1: 0,
      l2: 3,
      l3: 2,
      l2Evictions: 5,
      l3Evictions: 1,
      superseded: 2,
    }),
  ]
  const churn = summarizeChurn(window)

  expect(churn.evictions).toBe(10)
  expect(churn.demotions).toBe(5)
  // Five sessions were evicted with no publication ticket, so they survive nowhere.
  expect(churn.lost).toBe(5)
  expect(churn.restores).toEqual({ l1: 2, l2: 4, l3: 2 })
  // Six of eight restores had to be imported rather than found resident.
  expect(churn.importShare).toBeCloseTo(6 / 8, 12)
  expect(churn.l2Evictions).toBe(5)
  expect(churn.l3Evictions).toBe(1)
  expect(churn.deferrals).toBe(1)
  expect(churn.superseded).toBe(2)
})

test('an empty churn window reports no import pressure rather than NaN', () => {
  const churn = summarizeChurn([])
  expect(churn.evictions).toBe(0)
  expect(churn.lost).toBe(0)
  expect(churn.importShare).toBe(0)
})

test('a torn trailing line is skipped rather than throwing', () => {
  // A reader tailing a file the server is still appending to sees a partial final record.
  expect(parseRecordLine('{"event":"throughput","schema_ver')).toBeNull()
  expect(parseRecordLine('')).toBeNull()
  expect(parseRecordLine('   ')).toBeNull()
  expect(parseRecordLine('{"not":"a record"}')).toBeNull()
  expect(parseRecordLine('{"event":"request_start","request":{}}')).not.toBeNull()
})

test('adapter grouping reproduces the reference per-adapter aggregations', () => {
  const usage = summarizeByAdapter(FIXTURE)

  // Base first, then adapters by descending request count.
  expect(usage.map((entry) => entry.name)).toEqual(['', 'caveman'])

  const base = usage[0]!
  expect(base.summary.count).toBe(3)
  expect(base.generatedTokens).toBe(650)
  expect(base.summary.ttft.p50).toBe(1.75)
  expect(base.summary.decodeTokensPerSecond.p50).toBeCloseTo(100, 12)
  expect(base.summary.prefillAvoided).toBeCloseTo(0.4, 12)
  expect(base.summary.speculative.acceptRate).toBeCloseTo(500 / 900, 12)

  const caveman = usage[1]!
  expect(caveman.summary.count).toBe(2)
  expect(caveman.generatedTokens).toBe(5100)
  expect(caveman.summary.ttft.p50).toBe(4.5)
  expect(caveman.summary.decodeTokensPerSecond.p50).toBeCloseTo(100, 12)
  expect(caveman.summary.prefillAvoided).toBeCloseTo(1 - 3400 / 7000, 12)
  expect(caveman.summary.speculative.acceptRate).toBeCloseTo(600 / 900, 12)

  // Groups partition the input: no request is counted twice or dropped.
  expect(usage.reduce((total, entry) => total + entry.summary.count, 0)).toBe(FIXTURE.length)
})

test('adapter grouping keys on the resolved adapter, not the requested model', () => {
  // The Anthropic route passes the client's own model string through and silently falls back to
  // the base weights when it does not resolve, so `model` cannot identify what actually ran.
  const record = structuredClone(FIXTURE[0]!)
  record.request.model = 'claude-sonnet-4'
  record.request.adapter = 'caveman'
  const usage = summarizeByAdapter([record])
  expect(usage.map((entry) => entry.name)).toEqual(['caveman'])
})

// --- energy ----------------------------------------------------------------------------------

function energyBlock(values: {
  board: number
  prefill: number
  decode: number
  idle: number
  idleWatts: number
}): ThroughputRecord['energy'] {
  const residual = values.board - values.prefill - values.decode - values.idle
  return {
    board_joules: values.board,
    prefill_joules: values.prefill,
    decode_joules: values.decode,
    idle_joules: values.idle,
    idle_watts: values.idleWatts,
    accounted_seconds: 0,
    residual_joules: residual,
    residual_fraction: values.board === 0 ? 0 : residual / values.board,
    joules_per_token: { served: null, active: null, prefill: null, decode: null },
  }
}

test('energy sums measured intervals and divides by the matching token counts', () => {
  const records = [
    throughput(1, {
      prefillTokens: 1000,
      decodeTokens: 100,
      energy: energyBlock({ board: 1000, prefill: 200, decode: 400, idle: 380, idleWatts: 76 }),
    }),
    throughput(2, {
      prefillTokens: 0,
      decodeTokens: 100,
      energy: energyBlock({ board: 600, prefill: 0, decode: 400, idle: 190, idleWatts: 76 }),
    }),
  ]
  const energy = summarizeEnergy(records)

  expect(energy.available).toBe(true)
  expect(energy.boardJoules).toBe(1600)
  expect(energy.prefillJoules).toBe(200)
  expect(energy.decodeJoules).toBe(800)
  expect(energy.idleJoules).toBe(570)

  // Served prices every joule against every token; active removes only the measured idle draw.
  expect(energy.servedJoulesPerToken).toBeCloseTo(1600 / 1200, 12)
  expect(energy.activeJoulesPerToken).toBeCloseTo((1600 - 570) / 1200, 12)

  // Each phase divides by its own denominator, never by the combined token count.
  expect(energy.prefillJoulesPerToken).toBeCloseTo(200 / 1000, 12)
  expect(energy.decodeJoulesPerToken).toBeCloseTo(800 / 200, 12)

  // Residual is carried through as a share of the measured total, not recomputed from the parts.
  expect(energy.residualFraction).toBeCloseTo((1600 - 200 - 800 - 570) / 1600, 12)
})

test('a phase that produced no tokens reports no energy per token rather than zero', () => {
  // A decode-only window has no prefill denominator. Zero J/tok would read as "free", which is the
  // opposite of "not measured", so the distinction has to survive into the summary.
  const energy = summarizeEnergy([
    throughput(1, {
      prefillTokens: 0,
      decodeTokens: 50,
      energy: energyBlock({ board: 500, prefill: 0, decode: 300, idle: 200, idleWatts: 76 }),
    }),
  ])
  expect(energy.prefillJoulesPerToken).toBeNull()
  expect(energy.decodeJoulesPerToken).toBeCloseTo(6, 12)
})

test('a board with no energy counter reports unavailable rather than zero joules per token', () => {
  const energy = summarizeEnergy([throughput(1, { prefillTokens: 1000, decodeTokens: 100 })])
  expect(energy.available).toBe(false)
  expect(energy.boardJoules).toBe(0)
  expect(energy.servedJoulesPerToken).toBeNull()
  expect(energy.residualFraction).toBe(0)
})

test('an empty energy window does not divide by zero', () => {
  const energy = summarizeEnergy([])
  expect(energy.available).toBe(false)
  expect(energy.servedJoulesPerToken).toBeNull()
  expect(energy.activeJoulesPerToken).toBeNull()
  expect(Number.isFinite(energy.residualFraction)).toBe(true)
})

test('energy per million tokens is an exact restatement of joules per token', () => {
  // 1 Wh is exactly 3600 J, so a million tokens at 3.6 J each is exactly 1 kWh. The whole point of
  // the alternate denominator is that it is a pure rescale; if this drifts, two numbers on the same
  // panel would describe the same measurement differently.
  expect(3.6 * WH_PER_MILLION_TOKENS_PER_JOULE).toBeCloseTo(1000, 9)
  expect(energyPerMillionTokens(3.6)).toBe('1.00 kWh')

  // Measured on this target: prefill ~0.176 J/tok, decode ~3.80 J/tok, served ~0.504 J/tok.
  // All three must land on a readable scale, which is why the formatter switches Wh/kWh.
  expect(energyPerMillionTokens(0.1763)).toBe('49 Wh')
  expect(energyPerMillionTokens(3.7993)).toBe('1.06 kWh')
  expect(energyPerMillionTokens(0.5042)).toBe('140 Wh')
})

test('energy per million tokens reports absence rather than a zero cost', () => {
  expect(energyPerMillionTokens(null)).toBe('—')
  expect(energyPerMillionTokens(Number.NaN)).toBe('—')
  expect(energyPerMillionTokens(-1)).toBe('—')
  expect(energyPerMillionTokens(0)).toBe('0 Wh')
})

// --- System One ------------------------------------------------------------------------------

// A decision shaped like a real schema-22 `decision_done` record. Only the layout, reuse and
// timing inputs vary.
function decided(
  id: number,
  values: {
    adapter: string
    model?: string
    questions: number
    options: number
    stateTokens: number
    truncated?: boolean
    reused: number
    source: ContinuationSource
    branchTokens: number
    passes: number
    prepare: number
    queue: number
    restore: number
    state: number
    branch: number
    execution: number
    total: number
  },
): DecisionDoneRecord {
  return {
    event: 'decision_done',
    schema_version: 22,
    server_instance_id: 'serve-test-1',
    timestamp_unix_ms: 1_700_000_000_000 + id * 1000,
    artifact_type: 'ninfer_serve_request_log',
    request: {
      request_id: id,
      x_request_id: `ts_${id}`,
      protocol: 'systemone',
      model: values.model ?? values.adapter,
      adapter: values.adapter,
      allow_prefix_reuse: true,
      questions: values.questions,
      options: values.options,
      state_tokens: values.stateTokens,
      state_truncated: values.truncated ?? false,
      branch_tokens: values.branchTokens,
      longest_branch: 64,
      input_tokens: values.stateTokens + values.branchTokens,
    },
    result: {
      output_tokens: 40,
      reused_state_tokens: values.reused,
      computed_state_tokens: values.stateTokens - values.reused,
      state_source: values.source,
      branch_passes: values.passes,
      long_branch_chunks: 0,
      slot: 0,
    },
    timings_seconds: {
      prepare: values.prepare,
      queue: values.queue,
      restore: values.restore,
      state: values.state,
      branch: values.branch,
      execution: values.execution,
      total: values.total,
    },
    rates: { state_tok_s: null, branch_tok_s: null },
  }
}

// A cold state cut to kev's 65,536-token limit, a repeat of another state still resident in its
// lane, and a smaller state restored from host memory. The restore sits inside `queue`.
const DECISIONS: DecisionDoneRecord[] = [
  decided(1, {
    adapter: 'decision-v7',
    questions: 6,
    options: 18,
    stateTokens: 65_536,
    truncated: true,
    reused: 0,
    source: 'none',
    branchTokens: 300,
    passes: 1,
    prepare: 0.05,
    queue: 0.1,
    restore: 0,
    state: 16.384,
    branch: 0.3,
    execution: 16.7,
    total: 16.85,
  }),
  decided(2, {
    adapter: 'decision-v7',
    model: 'jev-latest',
    questions: 6,
    options: 18,
    stateTokens: 8000,
    reused: 8000,
    source: 'l1',
    branchTokens: 300,
    passes: 1,
    prepare: 0.05,
    queue: 0.02,
    restore: 0,
    state: 0,
    branch: 0.3,
    execution: 0.32,
    total: 0.39,
  }),
  decided(3, {
    adapter: 'pilot',
    questions: 2,
    options: 4,
    stateTokens: 1000,
    reused: 1000,
    source: 'l2',
    branchTokens: 100,
    passes: 1,
    prepare: 0.03,
    queue: 0.25,
    restore: 0.2,
    state: 0,
    branch: 0.1,
    execution: 0.12,
    total: 0.4,
  }),
]

test('decision summary reports System One latency, phase shares and state reuse', () => {
  const summary = summarizeDecisions(DECISIONS)
  expect(summary.count).toBe(3)
  expect(summary.questions).toBe(14)
  expect(summary.options).toBe(40)
  expect(summary.truncatedStates).toBe(1)

  // latency_ms is restore + execution: 16.7, 0.32 and 0.2 + 0.12. The queue wait is excluded.
  expect(summary.latency.p50).toBeCloseTo(0.32, 12)
  expect(summary.latency.p90).toBeCloseTo(16.7, 12)
  expect(summary.latency.max).toBeCloseTo(16.7, 12)
  expect(summary.latency.mean).toBeCloseTo(17.34 / 3, 12)
  expect(summary.total.p50).toBeCloseTo(0.4, 12)
  expect(summary.total.p90).toBeCloseTo(16.85, 12)

  // Engine time is queue + execution = 16.8 + 0.34 + 0.37. The wait is the queue without the
  // restore it contains: 0.1 + 0.02 + 0.05.
  const engine = 17.51
  expect(summary.phaseShare.wait).toBeCloseTo(0.17 / engine, 12)
  expect(summary.phaseShare.restore).toBeCloseTo(0.2 / engine, 12)
  expect(summary.phaseShare.state).toBeCloseTo(16.384 / engine, 12)
  expect(summary.phaseShare.branch).toBeCloseTo(0.7 / engine, 12)

  expect(summary.stateTokens).toBe(74_536)
  expect(summary.reusedStateTokens).toBe(9000)
  expect(summary.stateReuse).toBeCloseTo(9000 / 74_536, 12)
  expect(summary.bySource).toEqual({ none: 1, l1: 1, l2: 1 })
  expect(summary.branchTokens).toBe(700)
  expect(summary.branchPasses).toBe(3)
  // Pooled over the phases that ran: only the cold state was prefilled.
  expect(summary.stateTokensPerSecond).toBeCloseTo(4000, 9)
  expect(summary.branchTokensPerSecond).toBeCloseTo(1000, 9)
})

test('an empty decision window has no rates rather than zero ones', () => {
  const summary = summarizeDecisions([])
  expect(summary.count).toBe(0)
  expect(summary.latency.p50).toBe(0)
  expect(summary.stateReuse).toBe(0)
  expect(summary.phaseShare).toEqual({ wait: 0, restore: 0, state: 0, branch: 0 })
  expect(summary.stateTokensPerSecond).toBeNull()
  expect(summary.branchTokensPerSecond).toBeNull()
})

test('decisions group by the resolved adapter, not the requested model', () => {
  // Decision 2 asked for the SDK alias; it ran on the adapter the alias is bound to.
  const usage = summarizeDecisionsByAdapter(DECISIONS)
  expect(usage.map((entry) => entry.name)).toEqual(['decision-v7', 'pilot'])
  expect(usage[0]!.summary.count).toBe(2)
  expect(usage[0]!.summary.questions).toBe(12)
  expect(usage[0]!.summary.latency.p50).toBeCloseTo(16.7, 12)
  expect(usage[1]!.summary.stateReuse).toBe(1)
  expect(usage.reduce((total, entry) => total + entry.summary.count, 0)).toBe(DECISIONS.length)
})

test('the adapter pool reads schema-22 kinds and older name lists alike', () => {
  const bank = { count: 2, rank: 16, slots: 2, device_bytes: 168_800_000, file_bytes: 80_000_000 }
  const current: AdapterInventory = {
    ...bank,
    pool: [
      { name: 'math7', kind: 'generative', rank: 16, model_id: 'qwen3.8-27b-math7' },
      {
        name: 'decision-v7',
        kind: 'decision',
        rank: 16,
        model_id: 'decision-v7',
        temperature: 1.25,
        pointer_dim: 256,
        description: 'triage',
        release_date: '2026-09-29',
      },
    ],
  }
  expect(adapterPool(current)).toEqual([
    { name: 'math7', kind: 'generative', modelId: 'qwen3.8-27b-math7', temperature: null },
    { name: 'decision-v7', kind: 'decision', modelId: 'decision-v7', temperature: 1.25 },
  ])

  // Schema 21 named the decision adapters separately and carried no served ids.
  const schema21: AdapterInventory = {
    ...bank,
    names: ['decision-v7', 'math7'],
    decision_names: ['decision-v7'],
  }
  expect(adapterPool(schema21).map((entry) => [entry.name, entry.kind, entry.modelId])).toEqual([
    ['decision-v7', 'decision', null],
    ['math7', 'generative', null],
  ])

  // Before schema 21 every adapter was a chat adapter.
  const schema19: AdapterInventory = { ...bank, count: 1, names: ['math7'] }
  expect(adapterPool(schema19).map((entry) => entry.kind)).toEqual(['generative'])
  expect(adapterPool(undefined)).toEqual([])
})

test('decision records pass through the line reader as the server writes them', () => {
  const done = parseRecordLine(
    '{"artifact_type":"ninfer_serve_request_log","event":"decision_done","rates":' +
      '{"branch_tok_s":1234.5,"state_tok_s":null},"request":{"adapter":"decision-v7",' +
      '"allow_prefix_reuse":true,"branch_tokens":310,"input_tokens":8310,"longest_branch":58,' +
      '"model":"jev-latest","options":17,"protocol":"systemone","questions":6,"request_id":42,' +
      '"state_tokens":8000,"state_truncated":false,"x_request_id":"0f5c"},"result":' +
      '{"branch_passes":1,"computed_state_tokens":0,"long_branch_chunks":0,"output_tokens":61,' +
      '"reused_state_tokens":8000,"slot":2,"state_source":"l1"},"schema_version":22,' +
      '"server_instance_id":"serve-1-2","timestamp_unix_ms":1700000000000,"timings_seconds":' +
      '{"branch":0.25,"execution":0.26,"prepare":0.004,"queue":0.001,"restore":0.0,"state":0.0,' +
      '"total":0.265}}',
  )
  expect(done?.event).toBe('decision_done')
  const summary = summarizeDecisions([done as DecisionDoneRecord])
  expect(summary.stateReuse).toBe(1)
  expect(summary.latency.p50).toBeCloseTo(0.26, 12)

  expect(parseRecordLine('{"event":"decision_start","request":{}}')?.event).toBe('decision_start')
  expect(parseRecordLine('{"event":"decision_error","error":{"status":503}}')?.event).toBe(
    'decision_error',
  )
  expect(parseRecordLine('{"event":"decision_progress"}')).toBeNull()
})
