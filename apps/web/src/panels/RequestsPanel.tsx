import { StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term, Tooltip } from '../components/tooltip'
import { Empty, Panel, Pill } from '../components/ui'
import { SOURCE_COLOR } from '../lib/derive'
import { bytes, clock, count, percent, seconds } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { RequestDoneRecord, RequestStartRecord } from '../lib/records'

const PHASES = [
  {
    key: 'queue',
    color: CHART.warning,
    hint: 'Waiting in the FIFO before a lane was available.',
  },
  {
    key: 'restore',
    color: CHART.blue,
    hint: 'Importing cached continuation state into the lane.',
  },
  { key: 'prefill', color: CHART.violet, hint: 'Evaluating prompt tokens no prefix covered.' },
  {
    key: 'decode',
    color: CHART.accent,
    hint: 'Generating tokens once the first one had been emitted.',
  },
] as const

const SOURCE_HINT: Record<string, string> = {
  l1: GLOSSARY.tierL1.body,
  l2: GLOSSARY.tierL2.body,
  l3: GLOSSARY.tierL3.body,
  none: 'Nothing was reused: the whole prompt was prefilled from zero.',
}

/** Per-request TTFT split, drawn on the same phase colors the latency panel uses. */
function Waterfall({ record }: { record: RequestDoneRecord }) {
  const t = record.timings_seconds
  return (
    <StackedBar
      height={5}
      segments={PHASES.map((phase) => ({
        label: phase.key,
        value: t[phase.key],
        color: phase.color,
        hint: phase.hint,
        display: `${seconds(t[phase.key])} of ${seconds(t.queue + t.restore + t.prefill + t.decode)}`,
      }))}
    />
  )
}

/** Everything the record knows about one request, shown on its timestamp. */
function detail(record: RequestDoneRecord) {
  const cache = record.continuation_cache
  const spec = record.speculative
  const lines: string[] = [
    `${record.request.protocol} · ${record.request.request_id}`,
    `finish: ${record.result.finish_reason}`,
    `reuse path: ${record.result.prefix_reuse_path}`,
    `reused ${count(record.result.prefix_cache_hit_tokens)} of ${count(
      record.result.prompt_tokens,
    )} prompt tokens`,
  ]
  if (cache.source !== 'none') {
    lines.push(
      `restored ${count(cache.restored_tokens)} tokens (${bytes(cache.restored_bytes)}) from ${
        cache.source
      } in ${seconds(cache.restore_microseconds / 1e6)}`,
    )
  }
  if (cache.final_miss_reason && cache.final_miss_reason !== 'none') {
    lines.push(`miss: ${cache.final_miss_reason}`)
  }
  if (cache.restore_failure && cache.restore_failure !== 'none') {
    lines.push(`restore failure: ${cache.restore_failure}`)
  }
  if (cache.destructive_rollback) lines.push('a destructive rollback was required')
  if (spec && spec.rounds > 0) {
    lines.push(
      `mtp: ${spec.accepted_tokens}/${spec.drafted_tokens} accepted (${percent(
        spec.drafted_tokens === 0 ? 0 : spec.accepted_tokens / spec.drafted_tokens,
      )})`,
    )
  }
  return lines.join('\n')
}

export function RequestsPanel({
  requests,
  active,
}: {
  requests: RequestDoneRecord[]
  active: RequestStartRecord[]
}) {
  const recent = [...requests].reverse().slice(0, 14)

  return (
    <Panel
      title="Chat requests"
      hint="ttftSplit"
      className="panel--wide"
      note={
        active.length > 0 ? (
          <Pill tone="accent">{active.length} in flight</Pill>
        ) : (
          `${requests.length} retained`
        )
      }
    >
      {recent.length === 0 ? (
        <Empty>no completed chat requests yet</Empty>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>at</th>
              <th>proto</th>
              <th className="numeric">prompt</th>
              <th className="numeric">gen</th>
              <th>
                <Term k="promptSource">src</Term>
              </th>
              <th className="numeric">
                <Term k="ttft">ttft</Term>
              </th>
              <th className="numeric">
                <Term k="decodePerRequest">tok/s</Term>
              </th>
              <th style={{ width: '22%' }}>
                <Term k="ttftSplit">phases</Term>
              </th>
            </tr>
          </thead>
          <tbody>
            {recent.map((record) => {
              const source = record.continuation_cache.source
              const decodeRate =
                record.timings_seconds.decode > 0.01
                  ? record.result.completion_tokens / record.timings_seconds.decode
                  : 0
              return (
                <tr key={`${record.request.request_id}-${record.timestamp_unix_ms}`}>
                  <td>
                    <Tooltip
                      title={clock(record.timestamp_unix_ms)}
                      body={<span className="tipwrap">{detail(record)}</span>}
                      className="tip--term"
                    >
                      {clock(record.timestamp_unix_ms)}
                    </Tooltip>
                  </td>
                  <td>{record.request.protocol.replace('openai_', '')}</td>
                  <td className="numeric emphasis">{count(record.result.prompt_tokens)}</td>
                  <td className="numeric emphasis">{count(record.result.completion_tokens)}</td>
                  <td style={{ color: SOURCE_COLOR[source] }}>
                    <Tooltip title={source} body={SOURCE_HINT[source] ?? ''} className="tip--term">
                      {source}
                    </Tooltip>
                  </td>
                  <td className="numeric emphasis">{seconds(record.timings_seconds.ttft)}</td>
                  <td className="numeric">{decodeRate > 0 ? decodeRate.toFixed(0) : '—'}</td>
                  <td>
                    <Waterfall record={record} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      <p className="panel__footnote">
        Phase bars are proportional to each request&apos;s own total, coloured queue / restore /
        prefill / decode. Hover a timestamp for that request&apos;s cache decision and MTP result.
      </p>
    </Panel>
  )
}
