import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term } from '../components/tooltip'
import { Empty, Panel, Stat } from '../components/ui'
import type { RequestSummary } from '../lib/derive'
import { bytes, count, percent, seconds } from '../lib/format'

const PHASES = [
  {
    key: 'queue',
    label: 'queue',
    color: CHART.warning,
    hint: 'Waiting in the bounded FIFO for a lane to free up. Responds to lane count and generation length, not to kernel speed.',
  },
  {
    key: 'restore',
    label: 'restore',
    color: CHART.blue,
    hint: 'Importing cached continuation state into the lane, from L2 host memory or L3 disk.',
  },
  {
    key: 'prefill',
    label: 'prefill',
    color: CHART.violet,
    hint: 'Evaluating the prompt tokens that no reused prefix covered.',
  },
] as const

export function LatencyPanel({ summary }: { summary: RequestSummary }) {
  if (summary.count === 0) {
    return (
      <Panel title="Chat latency" hint="ttft">
        <Empty>no completed chat requests in the retained window</Empty>
      </Panel>
    )
  }

  const share = summary.ttftShare
  const dominant = PHASES.reduce((best, phase) =>
    share[phase.key] > share[best.key] ? phase : best,
  )

  return (
    <Panel title="Chat latency" hint="ttft" note={`${summary.count} completed requests`}>
      <div className="stat-row">
        {/* No fixed threshold makes a TTFT "bad" here - acceptable latency depends on prompt size
            and lane count - so these report without an alarm colour. */}
        <Stat value={seconds(summary.ttft.p50)} label="ttft p50" hint="ttft" />
        <Stat value={seconds(summary.ttft.p90)} label="ttft p90" hint="percentiles" />
        <Stat value={seconds(summary.ttft.p99)} label="ttft p99" hint="percentiles" />
        <Stat
          value={`${summary.decodeTokensPerSecond.p50.toFixed(1)}`}
          unit="tok/s"
          label="decode p50"
          hint="decodePerRequest"
          tone="accent"
        />
        <Stat
          value={percent(summary.speculative.acceptRate)}
          label="mtp accept"
          hint="mtpAccept"
          tone={summary.speculative.acceptRate > 0.5 ? 'accent' : 'neutral'}
        />
      </div>

      <div className="latency__split">
        <div className="eyebrow">
          <Term k="ttftSplit">time to first token, decomposed</Term>
        </div>
        <StackedBar
          segments={PHASES.map((phase) => ({
            label: phase.label,
            value: share[phase.key],
            color: phase.color,
            hint: phase.hint,
            display: `${percent(share[phase.key], 1)} of summed TTFT`,
          }))}
          height={10}
        />
        <Legend
          items={PHASES.map((phase) => ({
            label: phase.label,
            color: phase.color,
            hint: phase.hint,
          }))}
        />
        <p className="panel__footnote">
          {PHASES.map((phase) => `${phase.label} ${percent(share[phase.key], 1)}`).join(' · ')}.
          TTFT is dominated by <strong>{dominant.label}</strong>
          {dominant.key === 'queue'
            ? ' — requests are waiting for a lane, not being computed, so more lanes or shorter generations move this before any kernel work does.'
            : dominant.key === 'restore'
              ? ' — continuation import is on the critical path; check L2/L3 tier latency.'
              : ' — prompts are genuinely being recomputed; check the cache source split.'}
        </p>
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>tokens</th>
            <th className="numeric">
              <Term k="percentiles">p50</Term>
            </th>
            <th className="numeric">
              <Term k="percentiles">p90</Term>
            </th>
            <th className="numeric">max</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>prompt</td>
            <td className="numeric emphasis">{count(summary.promptTokens.p50)}</td>
            <td className="numeric emphasis">{count(summary.promptTokens.p90)}</td>
            <td className="numeric emphasis">{count(summary.promptTokens.max)}</td>
          </tr>
          <tr>
            <td>
              <Term k="prefillAvoided">recomputed prefill</Term>
            </td>
            <td className="numeric emphasis">{count(summary.computedPrefill.p50)}</td>
            <td className="numeric emphasis">{count(summary.computedPrefill.p90)}</td>
            <td className="numeric emphasis">{count(summary.computedPrefill.max)}</td>
          </tr>
          <tr>
            <td>generated</td>
            <td className="numeric emphasis">{count(summary.generated.p50)}</td>
            <td className="numeric emphasis">{count(summary.generated.p90)}</td>
            <td className="numeric emphasis">{count(summary.generated.max)}</td>
          </tr>
        </tbody>
      </table>

      {summary.restores.count > 0 ? (
        <p className="panel__footnote">
          <Term k="restores">{summary.restores.count} restores</Term> moved{' '}
          {count(summary.restores.totalTokens)} tokens ({bytes(summary.restores.totalBytes)}) at{' '}
          {seconds(summary.restores.meanSeconds)} mean / {seconds(summary.restores.p90Seconds)} p90,
          after {seconds(summary.restores.meanPreflightSeconds)} mean preflight.
        </p>
      ) : null}
    </Panel>
  )
}
