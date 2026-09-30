import { Term, Tooltip } from '../components/tooltip'
import { Empty, Panel, Pill } from '../components/ui'
import {
  KIND_COLOR,
  KIND_LABEL,
  summarizeByAdapter,
  summarizeDecisionsByAdapter,
  type PoolEntry,
} from '../lib/derive'
import { bytes, count, percent, seconds } from '../lib/format'
import type {
  AdapterInventory,
  DecisionDoneRecord,
  RequestDoneRecord,
  ThroughputRecord,
} from '../lib/records'
import { latest } from '../lib/series'
import type { AdapterTelemetry } from '../lib/telemetry'

/** Pooled names first, then names only the records know, each once. */
function union(pooled: string[], used: string[]): string[] {
  const names = [...pooled, ...used.filter((name) => !pooled.includes(name))]
  return names.filter((name, index) => names.indexOf(name) === index)
}

function NotInPool() {
  return (
    <span className="adapters__flag">
      <Pill tone="warning">not in pool</Pill>
    </span>
  )
}

/** Which adapter each device slot holds right now, and whether a running lane pins it. */
function SlotStrip({ bank, pool }: { bank: AdapterTelemetry; pool: PoolEntry[] }) {
  return (
    <div className="bank">
      {bank.resident.map((slot, index) => {
        const entry = pool.find((candidate) => candidate.name === slot.adapter)
        const color = entry ? KIND_COLOR[entry.kind] : 'var(--line)'
        return (
          <Tooltip
            key={index}
            title={`Slot ${index}`}
            body={
              slot.adapter === ''
                ? 'Never staged. The next adapter a request selects goes here first.'
                : `${slot.adapter}, a ${entry ? KIND_LABEL[entry.kind] : 'pooled'} adapter. ${
                    slot.pinned
                      ? 'Pinned: a running lane executes against it, so it cannot be swapped out.'
                      : 'Unpinned: the next swap may replace it, least recently used first.'
                  }`
            }
            className="bank__slot"
          >
            <span className="bank__swatch" style={{ background: color }} />
            <span className={slot.adapter === '' ? 'bank__name bank__name--empty' : 'bank__name'}>
              {slot.adapter === '' ? 'empty' : slot.adapter}
            </span>
            {slot.pinned ? <Pill tone="accent">pinned</Pill> : null}
          </Tooltip>
        )
      })}
    </div>
  )
}

/**
 * The adapter bank: one pool and one set of device slots behind both systems.
 *
 * Chat adapters and System One decision adapters are discovered from the same `--lora-dir` and
 * staged through the same slots, so the panel shows them side by side over one residency strip.
 * The inventory comes from the engine, so an adapter that has taken no traffic still appears;
 * usage is derived from the chat and decision records, so it survives replay. Which adapter holds
 * which slot is a live reading and is never recorded; the swap and wait counters are, on every
 * throughput record.
 */
export function AdaptersPanel({
  inventory,
  bank,
  pool,
  binding,
  requests,
  decisions,
  records,
}: {
  inventory: AdapterInventory | undefined
  /** The live bank, or undefined in replay. */
  bank: AdapterTelemetry | undefined
  pool: PoolEntry[]
  /** The decision adapter the SDK alias resolves to. */
  binding: string | undefined
  requests: RequestDoneRecord[]
  decisions: DecisionDoneRecord[]
  records: ThroughputRecord[]
}) {
  const chatUsage = summarizeByAdapter(requests)
  const chatUsed = new Map(chatUsage.map((entry) => [entry.name, entry]))
  const decisionUsage = summarizeDecisionsByAdapter(decisions)
  const decisionUsed = new Map(decisionUsage.map((entry) => [entry.name, entry]))
  const pooled = new Map(pool.map((entry) => [entry.name, entry]))
  // Membership can only be asserted when the engine actually reported its pool. Without an
  // inventory the correct statement is "unknown", not "not in pool".
  const known = inventory !== undefined

  const chatRows = union(
    ['', ...pool.filter((entry) => entry.kind === 'generative').map((entry) => entry.name)],
    chatUsage.map((entry) => entry.name),
  )
  const decisionRows = union(
    pool.filter((entry) => entry.kind === 'decision').map((entry) => entry.name),
    decisionUsage.map((entry) => entry.name),
  )

  if (pool.length === 0 && chatRows.length === 1 && decisionRows.length === 0) {
    return (
      <Panel title="Adapter bank" hint="adapterBank">
        <Empty>no adapters discovered — start with --lora-dir</Empty>
      </Panel>
    )
  }

  const slots = inventory?.slots ?? inventory?.count ?? 0
  const perSlot = inventory && slots > 0 ? inventory.device_bytes / slots : 0

  // Swap counters are cumulative: live from the poll, offline from the last throughput record.
  const recorded = latest(records)?.adapters
  const stages = bank?.stages ?? recorded?.stages
  const slotWaits = bank?.slot_waits ?? recorded?.slot_waits

  return (
    <Panel
      title="Adapter bank"
      hint="adapterBank"
      className="panel--wide"
      note={
        inventory && inventory.count > 0 ? (
          <>
            {inventory.count} pooled · rank {inventory.rank} ·{' '}
            <Tooltip
              title="Adapter bank"
              body={`${inventory.count} adapter(s) of both kinds are selectable; ${slots} slot(s) at ${bytes(
                perSlot,
              )} each are device-resident. The bank is committed at startup in its own device arena, before KV capacity is resolved, and the engine stages an adapter into an empty or least recently used unpinned slot when a request selects one that is not resident.`}
              className="tip--term"
            >
              {slots} slots · {bytes(inventory.device_bytes)} vram
            </Tooltip>
          </>
        ) : known ? (
          'none discovered'
        ) : (
          'usage only · no inventory reported'
        )
      }
    >
      {bank && bank.resident.length > 0 ? <SlotStrip bank={bank} pool={pool} /> : null}
      {stages !== undefined ? (
        <div className="bank__counters">
          <Term k="swaps">
            {count(stages)} swap{stages === 1 ? '' : 's'}
            {bank && bank.stages > 0 ? ` · ${seconds(bank.stage_seconds / bank.stages)} each` : ''}
          </Term>
          {' · '}
          <Term k="slotWaits">
            {count(slotWaits ?? 0)} slot wait{slotWaits === 1 ? '' : 's'}
          </Term>
          {bank ? null : ' · residency is live only'}
        </div>
      ) : null}

      <div className="eyebrow" style={{ color: KIND_COLOR.generative }}>
        <Term k="chatAdapter">chat</Term>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>adapter</th>
            <th className="numeric">reqs</th>
            <th className="numeric">gen</th>
            <th className="numeric">
              <Term k="ttft">ttft p50</Term>
            </th>
            <th className="numeric">
              <Term k="decodePerRequest">tok/s</Term>
            </th>
            <th className="numeric">
              <Term k="prefillAvoided">reused</Term>
            </th>
            <th className="numeric">
              <Term k="mtpAccept">mtp</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {chatRows.map((name) => {
            const entry = chatUsed.get(name)
            const inPool = name === '' || pooled.has(name)
            return (
              <tr key={name || '<base>'}>
                <td className="emphasis">
                  <Tooltip
                    title={name === '' ? 'Base model' : name}
                    body={
                      name === ''
                        ? 'Requests served by the base weights, with no adapter applied.'
                        : inPool
                          ? `Served as model id "${
                              pooled.get(name)?.modelId ?? name
                            }". In the pool and always selectable; the engine stages it into a device slot on demand.`
                          : known
                            ? 'Served requests in this window but is not in the pool of the engine now reporting.'
                            : 'Served requests in this window. This source reports no adapter inventory, so pool membership is unknown.'
                    }
                    className="tip--term"
                  >
                    {name === '' ? 'base' : name}
                  </Tooltip>
                  {known && !inPool ? <NotInPool /> : null}
                </td>
                <td className="numeric emphasis">{entry ? count(entry.summary.count) : '—'}</td>
                <td className="numeric">{entry ? count(entry.generatedTokens) : '—'}</td>
                <td className="numeric">{entry ? seconds(entry.summary.ttft.p50) : '—'}</td>
                <td className="numeric">
                  {entry && entry.summary.decodeTokensPerSecond.p50 > 0
                    ? entry.summary.decodeTokensPerSecond.p50.toFixed(0)
                    : '—'}
                </td>
                <td className="numeric">{entry ? percent(entry.summary.prefillAvoided) : '—'}</td>
                <td className="numeric">
                  {entry && entry.summary.speculative.drafted > 0
                    ? percent(entry.summary.speculative.acceptRate)
                    : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {decisionRows.length > 0 ? (
        <>
          <div className="eyebrow" style={{ color: KIND_COLOR.decision }}>
            <Term k="decisionAdapter">System One</Term>
          </div>
          <table className="table">
            <thead>
              <tr>
                <th>adapter</th>
                <th className="numeric">decisions</th>
                <th className="numeric">questions</th>
                <th className="numeric">
                  <Term k="decisionLatency">latency p50</Term>
                </th>
                <th className="numeric">
                  <Term k="stateReuse">reused</Term>
                </th>
                <th className="numeric">temp</th>
              </tr>
            </thead>
            <tbody>
              {decisionRows.map((name) => {
                const entry = decisionUsed.get(name)
                const pooledEntry = pooled.get(name)
                return (
                  <tr key={name}>
                    <td className="emphasis">
                      <Tooltip
                        title={name}
                        body={
                          pooledEntry
                            ? `System One model "${name}": a LoRA adapter plus a pointer head, in the same pool and slots as the chat adapters.`
                            : known
                              ? 'Answered decisions in this window but is not in the pool of the engine now reporting.'
                              : 'Answered decisions in this window. This source reports no adapter inventory, so pool membership is unknown.'
                        }
                        className="tip--term"
                      >
                        {name}
                      </Tooltip>
                      {binding === name ? (
                        <span className="adapters__flag">
                          <Pill>sdk default</Pill>
                        </span>
                      ) : null}
                      {known && !pooledEntry ? <NotInPool /> : null}
                    </td>
                    <td className="numeric emphasis">{entry ? count(entry.summary.count) : '—'}</td>
                    <td className="numeric">{entry ? count(entry.summary.questions) : '—'}</td>
                    <td className="numeric">{entry ? seconds(entry.summary.latency.p50) : '—'}</td>
                    <td className="numeric">{entry ? percent(entry.summary.stateReuse) : '—'}</td>
                    <td className="numeric">
                      {pooledEntry && pooledEntry.temperature !== null
                        ? pooledEntry.temperature.toFixed(2)
                        : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </>
      ) : null}

      <p className="panel__footnote">
        One pool, one bank: chat and decision adapters come from the same --lora-dir and are swapped
        through the same slots, each refused on the other&apos;s route. A pooled adapter costs disk,
        not VRAM — only the slots are committed. Rows group by the resolved adapter, not the
        requested model id: the Anthropic route falls back to the base weights on an unknown model,
        and the SDK alias names whichever decision adapter it is bound to.
      </p>
    </Panel>
  )
}
