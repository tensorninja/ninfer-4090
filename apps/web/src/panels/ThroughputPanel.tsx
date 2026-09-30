import { BandChart, Legend } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Panel } from '../components/ui'
import { rate, seconds } from '../lib/format'
import type { ThroughputRecord } from '../lib/records'
import { bands, span, weightedRate } from '../lib/series'

/** System One's share of an interval's prefill rate; zero in a log that predates the split. */
function decisionPrefillRate(record: ThroughputRecord): number {
  const tokens = record.tokens.decision_prefill ?? 0
  return record.interval_seconds > 0 ? tokens / record.interval_seconds : 0
}

export function ThroughputPanel({
  records,
  lanes,
}: {
  records: ThroughputRecord[]
  /** Configured lane count, used as the ceiling for the batch chart. */
  lanes: number | undefined
}) {
  // Prefill is one rate over both systems. A schema-22 record splits out the System One part, so
  // the chart stacks chat under System One and their sum is the rate the engine reported.
  const split = records.some((record) => record.tokens.decision_prefill !== undefined)
  const decisionPrefill = bands(records, decisionPrefillRate)
  const chatPrefill = bands(
    records,
    (record) => record.throughput_tokens_per_second.prefill - decisionPrefillRate(record),
  )
  const prefill = bands(records, (record) => record.throughput_tokens_per_second.prefill)
  const decode = bands(records, (record) => record.throughput_tokens_per_second.decode)
  const window = span(records)

  // Prefill runs an order of magnitude faster than decode on this target, so a shared linear axis
  // renders decode as a flat sliver. Each series gets its own scale and the time axis is pinned to
  // a common domain, which keeps them comparable in time without one flattening the other. Both
  // are interval averages, so both stay bands.
  const domain: [number, number] | undefined =
    records.length === 0
      ? undefined
      : [
          Math.min(...prefill.map((band) => band.start)),
          Math.max(...prefill.map((band) => band.end)),
        ]

  // Mean sequences advanced per decode round. An interval mean like the rates, so it is a band
  // too, and it is what explains a change in aggregate decode: the same per-sequence speed at a
  // larger batch is more total tokens.
  const batch = bands(records, (record) => record.decode_batch.average_size ?? 0)

  const report = [
    ...(split
      ? [
          { name: 'chat prefill', bands: chatPrefill, color: CHART.violet },
          { name: 'System One prefill', bands: decisionPrefill, color: CHART.systemOne },
        ]
      : [{ name: 'prefill', bands: prefill, color: CHART.violet }]),
    { name: 'decode (all lanes)', bands: decode, color: CHART.accent },
    { name: 'decode batch', bands: batch, color: CHART.blue },
  ]
  const meanPrefill = weightedRate(records, (r) => r.throughput_tokens_per_second.prefill)
  const meanDecisionPrefill = weightedRate(records, decisionPrefillRate)

  // Both decode figures are taken over the intervals that actually ran a decode round. A mean
  // across the whole window would divide by idle time and report a rate the engine never
  // sustained, and pairing that with a per-sequence figure computed only over decoding time
  // would make the two look contradictory when they are simply different denominators.
  const decoding = records.filter((record) => record.decode_batch.rounds > 0)
  const meanDecode = weightedRate(decoding, (r) => r.throughput_tokens_per_second.decode)
  const meanBatch = weightedRate(decoding, (r) => r.decode_batch.average_size ?? 0)
  // Aggregate ÷ mean batch is the rate one sequence sees: a sequence in a batch of b receives
  // 1/b of what those rounds committed.
  const perSequence = meanBatch > 0 ? meanDecode / meanBatch : 0

  return (
    <Panel
      title="Throughput"
      hint="intervalBands"
      note={
        records.length === 0 ? undefined : `${seconds(window)} window · ${records.length} samples`
      }
    >
      {records.length === 0 ? (
        <Empty>waiting for a reporting interval with activity</Empty>
      ) : (
        <>
          <div className="throughput__row">
            <span className="eyebrow" style={{ color: CHART.violet }}>
              <Term k="prefillRate">prefill</Term>
            </span>
            <span className="chart__caption">
              mean {rate(meanPrefill)} tok/s
              {split ? (
                <>
                  {' '}
                  · <Term k="decisionPrefill">{rate(meanDecisionPrefill)} System One</Term>
                </>
              ) : null}
            </span>
          </div>
          <BandChart
            label="Prefill tokens per second"
            unit="tok/s"
            height={86}
            domain={domain}
            report={report}
            stack={split}
            series={
              split
                ? [
                    {
                      name: 'chat prefill',
                      bands: chatPrefill,
                      color: CHART.violet,
                      opacity: 0.85,
                    },
                    {
                      name: 'System One prefill',
                      bands: decisionPrefill,
                      color: CHART.systemOne,
                      opacity: 0.85,
                    },
                  ]
                : [{ name: 'prefill', bands: prefill, color: CHART.violet, opacity: 0.85 }]
            }
            legend={
              split ? (
                <Legend
                  items={[
                    { label: 'chat', color: CHART.violet },
                    {
                      label: 'System One',
                      color: CHART.systemOne,
                      hint: 'Decision state and branch prefill, stacked on chat prefill.',
                    },
                  ]}
                />
              ) : undefined
            }
          />

          <div className="throughput__row">
            <span className="eyebrow" style={{ color: CHART.accent }}>
              <Term k="decodeRate">decode · all lanes</Term>
            </span>
            <span className="chart__caption">
              {decoding.length === 0 ? (
                'no decode rounds yet'
              ) : (
                <>
                  {rate(meanDecode)} tok/s ·{' '}
                  <Term k="perSequenceRate">{rate(perSequence)} per seq</Term> · while decoding
                </>
              )}
            </span>
          </div>
          <BandChart
            label="Decode tokens per second, summed across lanes"
            unit="tok/s"
            height={86}
            domain={domain}
            report={report}
            series={[
              { name: 'decode (all lanes)', bands: decode, color: CHART.accent, opacity: 0.9 },
            ]}
          />

          <div className="throughput__row">
            <span className="eyebrow" style={{ color: CHART.blue }}>
              <Term k="decodeBatch">decode batch</Term>
            </span>
            <span className="chart__caption">
              {decoding.length === 0
                ? 'no decode rounds yet'
                : `${meanBatch.toFixed(2)} seq/round while decoding`}
            </span>
          </div>
          <BandChart
            label="Mean sequences advanced per decode round"
            unit="seq"
            height={64}
            domain={domain}
            report={report}
            ceiling={lanes}
            series={[{ name: 'decode batch', bands: batch, color: CHART.blue, opacity: 0.8 }]}
            caption={
              <>
                shared time axis, independent scales <Info k="intervalBands" />
              </>
            }
          />
          <p className="panel__footnote">
            Decode is the sum across every active lane, so it rises with concurrency even when each
            request gets no faster; divide by the batch for the rate one sequence sees. Prefill is
            scaled separately because it runs roughly an order of magnitude faster — compare each
            series against its own axis, not against the others. Bands cover each sample&apos;s own
            reporting interval, and the reporter skips intervals with no activity, so a wide band is
            idle time folded forward rather than a long sustained rate. The decode means are taken
            over intervals that ran a decode round, not over the whole window, so idle time does not
            deflate them.
          </p>
        </>
      )}
    </Panel>
  )
}
