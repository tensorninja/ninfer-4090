// Derived request analytics.
//
// The chat aggregations are a port of the maintainer script `cache_health.py`, which is the
// checked reference for their definitions: same percentile rule, same denominators, same
// partitioning of misses and restore failures. `derive.test.ts` pins the agreement, so the
// dashboard and the script cannot drift into reporting different numbers for the same log. The
// System One summaries have no reference script; they use the same percentile rule and are
// pinned against hand-computed values instead.

import { CHART } from './palette'
import type {
  AdapterInventory,
  AdapterKind,
  ContinuationSource,
  DecisionDoneRecord,
  RequestDoneRecord,
  ThroughputRecord,
} from './records'

/**
 * Nearest-rank percentile with truncating index selection. Deliberately identical to the
 * reference script's `s[min(len(s) - 1, int(q * len(s)))]` rather than an interpolating
 * definition, because the two disagree on small samples and the script is the oracle.
 */
export function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.trunc(quantile * sorted.length))
  return sorted[index]!
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0
  return values.reduce((total, value) => total + value, 0) / values.length
}

export function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0)
}

/** A counter a replayed log predates reads as zero, never as `NaN` in someone's total. */
function counter(value: number | undefined): number {
  return value ?? 0
}

function tally<T extends string>(values: readonly T[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1
  return counts
}

export interface Spread {
  p50: number
  p90: number
  p99: number
  max: number
  mean: number
}

function spread(values: readonly number[]): Spread {
  return {
    p50: percentile(values, 0.5),
    p90: percentile(values, 0.9),
    p99: percentile(values, 0.99),
    max: values.length === 0 ? 0 : Math.max(...values),
    mean: mean(values),
  }
}

/**
 * Prefill spent on state the cache demonstrably held.
 *
 * `deepest_candidate_agreement` is the deepest prefix any preflighted candidate agreed with, so a
 * request that still recomputed from zero proves the engine had that much of the prompt in hand
 * and prefilled anyway. Waste is measured beyond what the lane already reused
 * (`prefix_cache_hit_tokens`), which is what makes `not_deeper` - where the resident frontier
 * already covered the agreement - correctly score as no waste at all.
 *
 * `null` agreement means nothing was preflighted, so there is no evidence either way and the
 * request is excluded rather than counted as clean.
 */
export interface CoverageWaste {
  /** Requests that recomputed a prefix the cache had agreed with. */
  requests: number
  /** Tokens recomputed despite that agreement. */
  tokens: number
  /** Share of completed requests affected. */
  requestShare: number
  /** Share of all computed prefill that this waste accounts for. */
  prefillShare: number
}

export interface RestoreSummary {
  count: number
  meanTokens: number
  totalTokens: number
  meanBytes: number
  totalBytes: number
  meanSeconds: number
  p90Seconds: number
  meanPreflightSeconds: number
}

export interface SpeculativeSummary {
  drafted: number
  accepted: number
  /** Accepted fraction of drafted tokens, the headline MTP figure. */
  acceptRate: number
  /** Acceptance count per draft position, summed across requests. */
  perPosition: number[]
  fallbackSteps: number
}

export interface RequestSummary {
  count: number
  promptTokens: Spread
  computedPrefill: Spread
  generated: Spread
  totalPromptTokens: number
  totalComputedPrefill: number
  /** Fraction of prompt tokens that prefix reuse kept out of prefill entirely. */
  prefillAvoided: number
  overLongGenerations: number
  bySource: Record<string, number>
  byMissReason: Record<string, number>
  byRestoreFailure: Record<string, number>
  byReusePath: Record<string, number>
  byFinishReason: Record<string, number>
  ttft: Spread
  /** Share of summed TTFT attributable to each phase. These are the actionable latency terms. */
  ttftShare: { queue: number; restore: number; prefill: number }
  coverageWaste: CoverageWaste
  restores: RestoreSummary
  speculative: SpeculativeSummary
  decodeTokensPerSecond: Spread
}

export function summarizeRequests(records: readonly RequestDoneRecord[]): RequestSummary {
  const prompt = records.map((record) => record.result.prompt_tokens)
  const computed = records.map((record) => record.result.computed_prefill_tokens)
  const generated = records.map((record) => record.result.completion_tokens)
  const ttft = records.map((record) => record.timings_seconds.ttft)

  const totalPrompt = sum(prompt)
  const totalComputed = sum(computed)
  const totalTtft = sum(ttft)

  const restored = records.filter((record) => record.continuation_cache.source !== 'none')
  const restoreSeconds = restored.map(
    (record) => record.continuation_cache.restore_microseconds / 1e6,
  )

  const perPosition: number[] = []
  let drafted = 0
  let accepted = 0
  let fallbackSteps = 0
  for (const record of records) {
    const speculative = record.speculative
    if (!speculative) continue
    drafted += speculative.drafted_tokens
    accepted += speculative.accepted_tokens
    fallbackSteps += speculative.fallback_steps
    speculative.accepted_per_position.forEach((value, index) => {
      perPosition[index] = (perPosition[index] ?? 0) + value
    })
  }

  // Tokens the cache agreed with and the engine recomputed anyway, counted only beyond what the
  // lane had already reused. A restore that happened is not waste, and an agreement the resident
  // frontier already covered is not waste either.
  let wasteTokens = 0
  let wasteRequests = 0
  for (const record of records) {
    if (record.continuation_cache.source !== 'none') continue
    const agreement = record.continuation_cache.deepest_candidate_agreement
    if (agreement === null || agreement === undefined) continue
    const recomputed = agreement - record.result.prefix_cache_hit_tokens
    if (recomputed <= 0) continue
    wasteTokens += recomputed
    wasteRequests += 1
  }

  // Only requests that actually decoded contribute a rate; a fully restored prompt that emitted
  // one token in near-zero time would otherwise dominate the spread with a meaningless value.
  const decodeRates = records
    .filter((record) => record.timings_seconds.decode > 0.01)
    .map((record) => record.result.completion_tokens / record.timings_seconds.decode)

  return {
    count: records.length,
    promptTokens: spread(prompt),
    computedPrefill: spread(computed),
    generated: spread(generated),
    totalPromptTokens: totalPrompt,
    totalComputedPrefill: totalComputed,
    prefillAvoided: totalPrompt === 0 ? 0 : 1 - totalComputed / totalPrompt,
    overLongGenerations: generated.filter((value) => value > 4096).length,
    bySource: tally(records.map((record) => record.continuation_cache.source)),
    byMissReason: tally(
      records
        .filter((record) => record.continuation_cache.source === 'none')
        .map((record) => record.continuation_cache.final_miss_reason),
    ),
    byRestoreFailure: tally(
      records
        .map((record) => record.continuation_cache.restore_failure)
        .filter((value) => value !== 'none'),
    ),
    byReusePath: tally(records.map((record) => record.result.prefix_reuse_path)),
    byFinishReason: tally(records.map((record) => record.result.finish_reason)),
    ttft: spread(ttft),
    ttftShare: {
      queue: totalTtft === 0 ? 0 : sum(records.map((r) => r.timings_seconds.queue)) / totalTtft,
      restore: totalTtft === 0 ? 0 : sum(records.map((r) => r.timings_seconds.restore)) / totalTtft,
      prefill: totalTtft === 0 ? 0 : sum(records.map((r) => r.timings_seconds.prefill)) / totalTtft,
    },
    coverageWaste: {
      requests: wasteRequests,
      tokens: wasteTokens,
      requestShare: records.length === 0 ? 0 : wasteRequests / records.length,
      prefillShare: totalComputed === 0 ? 0 : wasteTokens / totalComputed,
    },
    restores: {
      count: restored.length,
      meanTokens: mean(restored.map((r) => r.continuation_cache.restored_tokens)),
      totalTokens: sum(restored.map((r) => r.continuation_cache.restored_tokens)),
      meanBytes: mean(restored.map((r) => r.continuation_cache.restored_bytes)),
      totalBytes: sum(restored.map((r) => r.continuation_cache.restored_bytes)),
      meanSeconds: mean(restoreSeconds),
      p90Seconds: percentile(restoreSeconds, 0.9),
      meanPreflightSeconds: mean(
        restored.map((r) => r.continuation_cache.preflight_microseconds / 1e6),
      ),
    },
    speculative: {
      drafted,
      accepted,
      acceptRate: drafted === 0 ? 0 : accepted / drafted,
      perPosition,
      fallbackSteps,
    },
    decodeTokensPerSecond: spread(decodeRates),
  }
}

/** Per-adapter usage. `name` is `""` for the base model, which is not an adapter. */
export interface AdapterUsage {
  name: string
  summary: RequestSummary
  generatedTokens: number
}

/**
 * Groups completed requests by the adapter that served them.
 *
 * `request.adapter` is the authoritative field, not `request.model`: on the Anthropic route the
 * model string is whatever the client sent and an unknown value silently falls back to the base
 * model rather than 404ing, so only the resolved adapter name identifies what actually ran.
 *
 * Each group is summarized with the same `summarizeRequests` used for the global figures, so a
 * per-adapter number and its global counterpart are computed identically.
 */
export function summarizeByAdapter(records: readonly RequestDoneRecord[]): AdapterUsage[] {
  const groups = new Map<string, RequestDoneRecord[]>()
  for (const record of records) {
    const name = record.request.adapter ?? ''
    const bucket = groups.get(name)
    if (bucket === undefined) groups.set(name, [record])
    else bucket.push(record)
  }
  return (
    [...groups.entries()]
      .map(([name, group]) => ({
        name,
        summary: summarizeRequests(group),
        generatedTokens: sum(group.map((record) => record.result.completion_tokens)),
      }))
      // Base first, then adapters by traffic. The base row is the comparison every other row is
      // read against, so it does not compete for position on request count.
      .sort((a, b) => (a.name === '' ? -1 : b.name === '' ? 1 : b.summary.count - a.summary.count))
  )
}

/**
 * One pooled adapter, whichever schema reported the pool.
 *
 * Schema 22 lists every adapter with its kind and served id. Schema 21 lists names and,
 * separately, which of them are decision adapters; schemas 15–20 list names alone, and those are
 * all chat adapters because decision adapters did not exist yet. `modelId` is null where the
 * record did not carry it.
 */
export interface PoolEntry {
  name: string
  kind: AdapterKind
  modelId: string | null
  /** A decision adapter's calibration temperature, where reported. */
  temperature: number | null
}

export function adapterPool(inventory: AdapterInventory | undefined): PoolEntry[] {
  if (inventory === undefined) return []
  if (inventory.pool !== undefined) {
    return inventory.pool.map((entry) => ({
      name: entry.name,
      kind: entry.kind,
      modelId: entry.model_id,
      temperature: entry.temperature ?? null,
    }))
  }
  const decisions = new Set(inventory.decision_names ?? [])
  return (inventory.names ?? []).map((name) => ({
    name,
    kind: decisions.has(name) ? 'decision' : 'generative',
    modelId: null,
    temperature: null,
  }))
}

/**
 * System One decisions over the retained window.
 *
 * `latency` is what System One itself reports as `latency_ms`: restore plus execution, the
 * engine's time on a decision without its wait for a lane. `total` is what the client saw, from
 * receipt to response. The phase shares divide summed engine time (queue plus execution) into
 * waiting for a lane, importing a cached state, prefilling the state, and the branch passes; the
 * record's `queue` includes the restore, so the wait is the difference.
 */
export interface DecisionSummary {
  images: number
  visionTokens: number
  count: number
  questions: number
  options: number
  latency: Spread
  total: Spread
  phaseShare: { wait: number; restore: number; state: number; vision: number; branch: number }
  stateTokens: number
  reusedStateTokens: number
  /** Share of state tokens resident in a lane or restored from a tier instead of prefilled. */
  stateReuse: number
  bySource: Record<string, number>
  branchTokens: number
  branchPasses: number
  /** Computed state tokens over summed state seconds; null when no state was prefilled. */
  stateTokensPerSecond: number | null
  /** Branch tokens over summed branch seconds; null when no branch ran measurably. */
  branchTokensPerSecond: number | null
  /** States cut to their first 65,536 tokens, as kev cuts them. */
  truncatedStates: number
}

/** Pooled tokens per second: summed tokens over summed seconds, or null with nothing to divide. */
function pooledRate(tokens: number, seconds: number): number | null {
  return tokens === 0 || !(seconds > 0) ? null : tokens / seconds
}

/** One decision's engine time by phase. The record's `queue` contains the restore. */
export function decisionPhases(record: DecisionDoneRecord): {
  wait: number
  restore: number
  state: number
  vision: number
  branch: number
} {
  const timings = record.timings_seconds
  const vision = timings.vision ?? 0
  return {
    wait: Math.max(0, timings.queue - timings.restore),
    restore: timings.restore,
    state: Math.max(0, timings.state - vision),
    vision,
    branch: timings.branch,
  }
}

/** System One's `latency_ms`, in seconds: the restore plus execution, without the lane wait. */
export function decisionLatency(record: DecisionDoneRecord): number {
  return record.timings_seconds.restore + record.timings_seconds.execution
}

export function summarizeDecisions(records: readonly DecisionDoneRecord[]): DecisionSummary {
  let images = 0
  let visionTokens = 0
  let vision = 0
  let questions = 0
  let options = 0
  let wait = 0
  let restore = 0
  let state = 0
  let branch = 0
  let engine = 0
  let stateTokens = 0
  let reusedStateTokens = 0
  let computedStateTokens = 0
  let branchTokens = 0
  let branchPasses = 0
  let truncatedStates = 0
  for (const record of records) {
    const phases = decisionPhases(record)
    images += record.request.images ?? 0
    visionTokens += record.request.vision_tokens ?? 0
    vision += phases.vision
    questions += record.request.questions
    options += record.request.options
    wait += phases.wait
    restore += phases.restore
    state += phases.state
    branch += phases.branch
    engine += record.timings_seconds.queue + record.timings_seconds.execution
    stateTokens += record.request.state_tokens
    reusedStateTokens += record.result.reused_state_tokens
    computedStateTokens += record.result.computed_state_tokens
    branchTokens += record.request.branch_tokens
    branchPasses += record.result.branch_passes
    if (record.request.state_truncated) truncatedStates += 1
  }
  return {
    images,
    visionTokens,
    count: records.length,
    questions,
    options,
    latency: spread(records.map(decisionLatency)),
    total: spread(records.map((record) => record.timings_seconds.total)),
    phaseShare: {
      wait: engine === 0 ? 0 : wait / engine,
      restore: engine === 0 ? 0 : restore / engine,
      state: engine === 0 ? 0 : state / engine,
      vision: engine === 0 ? 0 : vision / engine,
      branch: engine === 0 ? 0 : branch / engine,
    },
    stateTokens,
    reusedStateTokens,
    stateReuse: stateTokens === 0 ? 0 : reusedStateTokens / stateTokens,
    bySource: tally(records.map((record) => record.result.state_source)),
    branchTokens,
    branchPasses,
    stateTokensPerSecond: pooledRate(computedStateTokens, state + vision),
    branchTokensPerSecond: pooledRate(branchTokens, branch),
    truncatedStates,
  }
}

/** Decisions served by one decision adapter. */
export interface DecisionAdapterUsage {
  name: string
  summary: DecisionSummary
}

/**
 * Groups decisions by the adapter that answered them, busiest first.
 *
 * `request.adapter` is the resolved decision adapter; `request.model` may be the SDK alias, which
 * names whichever adapter it is bound to, so only the adapter says what actually ran.
 */
export function summarizeDecisionsByAdapter(
  records: readonly DecisionDoneRecord[],
): DecisionAdapterUsage[] {
  const groups = new Map<string, DecisionDoneRecord[]>()
  for (const record of records) {
    const bucket = groups.get(record.request.adapter)
    if (bucket === undefined) groups.set(record.request.adapter, [record])
    else bucket.push(record)
  }
  return [...groups.entries()]
    .map(([name, group]) => ({ name, summary: summarizeDecisions(group) }))
    .sort((a, b) => b.summary.count - a.summary.count || a.name.localeCompare(b.name))
}

/**
 * Cache churn over the retained throughput window.
 *
 * Eviction on its own is a cache doing its job, so none of these are pathological in isolation.
 * The two readings that matter are `lost`, which is state discarded with no handoff and therefore
 * guaranteed to be recomputed if it is wanted again, and `importShare`, which is the fraction of
 * reuse that had to come from host or disk instead of resident VRAM. A working set that outgrows
 * L1 keeps its hit rate and quietly starts paying an import on every turn; that shift shows up
 * here and nowhere else.
 *
 * Summed from interval deltas rather than differenced endpoints, so a server restart inside the
 * window cannot turn a counter reset into a negative or absurd reading. A counter a replayed log
 * predates reads as zero rather than poisoning the whole summary with `NaN`.
 */
export interface ChurnSummary {
  evictions: number
  demotions: number
  /** Evicted with no publication ticket, so the session did not survive anywhere. */
  lost: number
  l2Evictions: number
  l3Evictions: number
  l2EvictedBytes: number
  l3EvictedBytes: number
  restores: { l1: number; l2: number; l3: number }
  /** Share of restores served from L2 or L3 rather than resident L1. */
  importShare: number
  /** Restores refused for shared-KV capacity with the candidate left live for a retry. */
  deferrals: number
  /** Publications that completed and were discarded because the alias had already moved on. */
  superseded: number
}

export function summarizeChurn(records: readonly ThroughputRecord[]): ChurnSummary {
  let evictions = 0
  let demotions = 0
  let l2Evictions = 0
  let l3Evictions = 0
  let l2EvictedBytes = 0
  let l3EvictedBytes = 0
  let l1 = 0
  let l2 = 0
  let l3 = 0
  let deferrals = 0
  let superseded = 0
  for (const record of records) {
    const cache = record.continuation_cache
    const occupancy = cache.occupancy
    evictions += occupancy.delta_l1_evictions
    demotions += occupancy.delta_l1_demotions
    l2Evictions += counter(occupancy.delta_l2_evictions)
    l3Evictions += counter(occupancy.delta_l3_evictions)
    l2EvictedBytes += counter(occupancy.delta_l2_evicted_bytes)
    l3EvictedBytes += counter(occupancy.delta_l3_evicted_bytes)
    l1 += cache.tiers.delta_l1_restore_successes
    l2 += cache.tiers.delta_l2_restore_successes
    l3 += cache.tiers.delta_l3_restore_successes
    deferrals += counter(cache.delta_restore_deferrals)
    superseded += counter(cache.delta_publication_superseded)
  }
  const restores = l1 + l2 + l3
  return {
    evictions,
    demotions,
    // A demotion is one kind of eviction, never an extra one, so this cannot go negative.
    lost: evictions - demotions,
    l2Evictions,
    l3Evictions,
    l2EvictedBytes,
    l3EvictedBytes,
    restores: { l1, l2, l3 },
    importShare: restores === 0 ? 0 : (l2 + l3) / restores,
    deferrals,
    superseded,
  }
}

export interface EnergySummary {
  /** False when no record in the window carried energy, i.e. the board has no counter. */
  available: boolean
  boardJoules: number
  prefillJoules: number
  decodeJoules: number
  idleJoules: number
  /** Latest measured idle draw, in watts. */
  idleWatts: number
  prefillTokens: number
  decodeTokens: number
  /**
   * Joules per token, or null where no tokens of that kind were produced.
   *
   * `served` prices every joule the board drew across the window, including the idle draw between
   * requests; it is what the work actually costs and it degrades when the server is mostly idle.
   * `active` removes the measured idle baseline and tracks the schedule rather than the duty
   * cycle. They are both reported because neither answers the other's question.
   */
  servedJoulesPerToken: number | null
  activeJoulesPerToken: number | null
  prefillJoulesPerToken: number | null
  decodeJoulesPerToken: number | null
  /**
   * Share of measured energy the phase splits and idle baseline together do not explain.
   *
   * The board refreshes power at roughly 50 Hz while a decode round is shorter than that, so the
   * split is an estimate over a measured total. A large residual means the split should not be
   * read closely; the total remains exact either way.
   */
  residualFraction: number
}

/** Energy per token, or null when the denominator is zero: no tokens is not zero joules each. */
function perToken(joules: number, tokens: number): number | null {
  return tokens === 0 ? null : joules / tokens
}

export function summarizeEnergy(records: readonly ThroughputRecord[]): EnergySummary {
  let boardJoules = 0
  let prefillJoules = 0
  let decodeJoules = 0
  let idleJoules = 0
  let residualJoules = 0
  let prefillTokens = 0
  let decodeTokens = 0
  let idleWatts = 0
  let available = false
  for (const record of records) {
    const energy = record.energy
    if (!energy) continue
    available = true
    boardJoules += energy.board_joules
    prefillJoules += energy.prefill_joules
    decodeJoules += energy.decode_joules
    idleJoules += energy.idle_joules
    residualJoules += energy.residual_joules
    prefillTokens += record.tokens.computed_prefill
    decodeTokens += record.tokens.committed_decode
    // The baseline is a running calibration, so the most recent reading is the current one.
    idleWatts = energy.idle_watts
  }
  const tokens = prefillTokens + decodeTokens
  return {
    available,
    boardJoules,
    prefillJoules,
    decodeJoules,
    idleJoules,
    idleWatts,
    prefillTokens,
    decodeTokens,
    servedJoulesPerToken: perToken(boardJoules, tokens),
    activeJoulesPerToken: perToken(Math.max(0, boardJoules - idleJoules), tokens),
    prefillJoulesPerToken: perToken(prefillJoules, prefillTokens),
    decodeJoulesPerToken: perToken(decodeJoules, decodeTokens),
    residualFraction: boardJoules === 0 ? 0 : residualJoules / boardJoules,
  }
}

/** The two systems, named and coloured the same way on every panel that tells them apart. */
export const KIND_LABEL: Record<AdapterKind, string> = {
  generative: 'chat',
  decision: 'System One',
}

export const KIND_COLOR: Record<AdapterKind, string> = {
  generative: CHART.accent,
  decision: CHART.systemOne,
}

export const SOURCE_ORDER: readonly ContinuationSource[] = ['l1', 'l2', 'l3', 'none']

/**
 * Tier colors, shared by the source breakdown and the cache occupancy panel.
 *
 * Literal values rather than `var()` because the same palette feeds ECharts, which renders to
 * canvas and cannot resolve custom properties.
 */
export const SOURCE_COLOR: Record<string, string> = {
  l1: CHART.accent,
  l2: CHART.blue,
  l3: CHART.violet,
  none: CHART.dim,
}
