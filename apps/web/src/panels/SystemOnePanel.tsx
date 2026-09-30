import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term, Tooltip } from '../components/tooltip'
import { Empty, Panel, Pill, Stat } from '../components/ui'
import {
  decisionLatency,
  decisionPhases,
  SOURCE_COLOR,
  type DecisionSummary,
  type PoolEntry,
} from '../lib/derive'
import { clock, count, percent, rate, seconds } from '../lib/format'
import type {
  DecisionDoneRecord,
  DecisionErrorRecord,
  DecisionStartRecord,
  SystemOneSurface,
} from '../lib/records'

const PHASES = [
  {
    key: 'wait',
    label: 'wait',
    color: CHART.warning,
    hint: 'Waiting in the bounded FIFO for a lane. Chat and System One share the lanes, so this is where the two systems contend.',
  },
  {
    key: 'restore',
    label: 'restore',
    color: CHART.blue,
    hint: 'Importing a cached state from host memory (L2) or disk (L3).',
  },
  {
    key: 'state',
    label: 'state',
    color: CHART.violet,
    hint: 'Prefilling the state tokens that no lane or tier already held.',
  },
  {
    key: 'branch',
    label: 'branches',
    color: CHART.systemOne,
    hint: 'The branch passes, one per packed group of questions, with their readout and the pointer head.',
  },
] as const

/** Per-decision phase split, on the same colours as the summary bar. */
function Waterfall({ record }: { record: DecisionDoneRecord }) {
  const phases = decisionPhases(record)
  const total = phases.wait + phases.restore + phases.state + phases.branch
  return (
    <StackedBar
      height={5}
      segments={PHASES.map((phase) => ({
        label: phase.label,
        value: phases[phase.key],
        color: phase.color,
        hint: phase.hint,
        display: `${seconds(phases[phase.key])} of ${seconds(total)}`,
      }))}
    />
  )
}

/** Everything the record knows about one decision, shown on its timestamp. */
function detail(record: DecisionDoneRecord): string {
  const request = record.request
  const result = record.result
  const timings = record.timings_seconds
  const lines = [
    request.model === request.adapter
      ? `${request.adapter} · request ${request.request_id}`
      : `${request.model} → ${request.adapter} · request ${request.request_id}`,
    `x-typesafe-request-id ${request.x_request_id}`,
    `${request.questions} questions · ${request.options} options · longest branch ${count(
      request.longest_branch,
    )} tokens`,
    `state ${count(request.state_tokens)} tokens${
      request.state_truncated ? ' (cut to 65,536)' : ''
    }: ${count(result.reused_state_tokens)} reused from ${result.state_source}, ${count(
      result.computed_state_tokens,
    )} prefilled`,
    `branches ${count(request.branch_tokens)} tokens in ${result.branch_passes} pass${
      result.branch_passes === 1 ? '' : 'es'
    }${result.long_branch_chunks > 0 ? ` + ${result.long_branch_chunks} long-branch chunks` : ''}`,
    `lane ${result.slot} · ${count(request.input_tokens)} input · ${count(
      result.output_tokens,
    )} output tokens`,
    `latency_ms ${(decisionLatency(record) * 1000).toFixed(1)} · total ${seconds(
      timings.total,
    )} with ${seconds(timings.prepare)} preparing`,
  ]
  return lines.join('\n')
}

/**
 * System One: decisions answered by the same process, weights and lanes that serve chat.
 *
 * Everything here comes from `decision_*` records, so it survives replay. What the panel adds over
 * the chat panels is the phase split of a prefill-only request - lane wait, state restore, state
 * prefill, branch passes - and the state reuse that makes a repeated state cost only its branches.
 */
export function SystemOnePanel({
  decisions,
  summary,
  active,
  errors,
  pool,
  surface,
  replay,
}: {
  decisions: DecisionDoneRecord[]
  summary: DecisionSummary
  active: DecisionStartRecord[]
  errors: DecisionErrorRecord[]
  pool: PoolEntry[]
  surface: SystemOneSurface | undefined
  replay: boolean
}) {
  const models = pool.filter((entry) => entry.kind === 'decision')
  const lastError = errors.length === 0 ? null : errors[errors.length - 1]!

  const note =
    surface?.supported === false ? (
      'not served by this target'
    ) : (
      <>
        {models.length} model{models.length === 1 ? '' : 's'} ·{' '}
        <Term k="sdkAlias">
          {surface?.alias ?? 'jev-latest'} → {surface?.binding || 'unbound'}
        </Term>
      </>
    )

  if (surface?.supported === false || (models.length === 0 && decisions.length === 0)) {
    return (
      <Panel title="System One" hint="systemOne" className="panel--wide" note={note}>
        <Empty>
          {surface?.supported === false
            ? 'this target does not answer System One decisions'
            : 'no decision adapter in the pool — convert one with --decision-head into --lora-dir'}
        </Empty>
      </Panel>
    )
  }

  const recent = [...decisions].reverse().slice(0, 12)
  const share = summary.phaseShare

  return (
    <Panel
      title="System One"
      hint="systemOne"
      className="panel--wide"
      note={
        <>
          {active.length > 0 ? <Pill tone="accent">{active.length} deciding</Pill> : null} {note}
        </>
      }
    >
      <div className="stat-row">
        <Stat value={count(summary.count)} label="decisions" hint="systemOne" />
        <Stat value={count(summary.questions)} label="questions" />
        <Stat
          value={seconds(summary.latency.p50)}
          label="latency p50"
          hint="decisionLatency"
          tone="accent"
        />
        <Stat value={seconds(summary.latency.p90)} label="latency p90" hint="decisionLatency" />
        <Stat
          value={percent(summary.stateReuse)}
          label="state reused"
          hint="stateReuse"
          tone={summary.stateReuse > 0.5 ? 'accent' : 'neutral'}
        />
        <Stat
          value={replay ? '—' : count(active.length)}
          label="in flight"
          hint="laneWork"
          tone={active.length > 0 ? 'accent' : 'neutral'}
        />
        <Stat
          value={count(errors.length)}
          label="errors"
          tone={errors.length > 0 ? 'danger' : 'neutral'}
        />
      </div>

      {summary.count === 0 ? (
        <Empty>no decisions yet — POST /systemone/v1/systemone</Empty>
      ) : (
        <>
          <div className="latency__split">
            <div className="eyebrow">
              <Term k="decisionPhases">engine time, decomposed</Term>
            </div>
            <StackedBar
              height={10}
              segments={PHASES.map((phase) => ({
                label: phase.label,
                value: share[phase.key],
                color: phase.color,
                hint: phase.hint,
                display: `${percent(share[phase.key], 1)} of summed engine time`,
              }))}
            />
            <Legend
              items={PHASES.map((phase) => ({
                label: phase.label,
                color: phase.color,
                hint: phase.hint,
              }))}
            />
            <p className="panel__footnote">
              State prefill{' '}
              {summary.stateTokensPerSecond === null
                ? '— (every state reused)'
                : `${rate(summary.stateTokensPerSecond)} tok/s`}{' '}
              · <Term k="branchPasses">branches</Term>{' '}
              {summary.branchTokensPerSecond === null
                ? '—'
                : `${rate(summary.branchTokensPerSecond)} tok/s`}{' '}
              over {count(summary.branchPasses)} passes
              {summary.truncatedStates > 0
                ? ` · ${summary.truncatedStates} state(s) cut to 65,536 tokens`
                : ''}
              .
            </p>
          </div>

          <table className="table">
            <thead>
              <tr>
                <th>at</th>
                <th>model</th>
                <th className="numeric">q · opt</th>
                <th className="numeric">state</th>
                <th className="numeric">
                  <Term k="stateReuse">reused</Term>
                </th>
                <th>
                  <Term k="promptSource">src</Term>
                </th>
                <th className="numeric">
                  <Term k="branchPasses">branch</Term>
                </th>
                <th className="numeric">
                  <Term k="decisionLatency">latency</Term>
                </th>
                <th style={{ width: '22%' }}>
                  <Term k="decisionPhases">phases</Term>
                </th>
              </tr>
            </thead>
            <tbody>
              {recent.map((record) => {
                const request = record.request
                const source = record.result.state_source
                return (
                  <tr key={`${request.request_id}-${record.timestamp_unix_ms}`}>
                    <td>
                      <Tooltip
                        title={clock(record.timestamp_unix_ms)}
                        body={<span className="tipwrap">{detail(record)}</span>}
                        className="tip--term"
                      >
                        {clock(record.timestamp_unix_ms)}
                      </Tooltip>
                    </td>
                    <td className="emphasis">{request.adapter}</td>
                    <td className="numeric">
                      {request.questions} · {request.options}
                    </td>
                    <td className="numeric emphasis">{count(request.state_tokens)}</td>
                    <td className="numeric">
                      {request.state_tokens === 0
                        ? '—'
                        : percent(record.result.reused_state_tokens / request.state_tokens)}
                    </td>
                    <td style={{ color: SOURCE_COLOR[source] }}>{source}</td>
                    <td className="numeric">{count(request.branch_tokens)}</td>
                    <td className="numeric emphasis">{seconds(decisionLatency(record))}</td>
                    <td>
                      <Waterfall record={record} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </>
      )}

      <p className="panel__footnote">
        Latency is System One&apos;s own <code>latency_ms</code>: restore plus execution, without
        the wait for a lane. Decisions share the weights, lanes, KV pool and adapter bank with chat
        and never decode. Hover a timestamp for the full decision.
        {lastError ? (
          <>
            {' '}
            Last error: <strong>{lastError.error.status}</strong> {lastError.error.message}
          </>
        ) : null}
      </p>
    </Panel>
  )
}
