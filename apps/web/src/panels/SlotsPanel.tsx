import { Term, Tooltip } from '../components/tooltip'
import { Empty, Meter, Panel, StatusDot } from '../components/ui'
import { KIND_COLOR, KIND_LABEL } from '../lib/derive'
import { count, percent } from '../lib/format'
import type { AdapterKind } from '../lib/records'
import type { SlotTelemetry } from '../lib/telemetry'

const STATE_HINT = {
  busy: 'Executing a request right now.',
  retained: 'Idle but still holding a resident session in VRAM, reusable with no import at all.',
  idle: 'Empty and immediately available to the next admitted request.',
} as const

/** Chat or System One, and under which adapter; a retained lane keeps the kind of its state. */
function Work({ slot }: { slot: SlotTelemetry }) {
  if (slot.work === 'none') return <>—</>
  const kind: AdapterKind = slot.work === 'decision' ? 'decision' : 'generative'
  const adapter = slot.adapter === '' ? 'base' : slot.adapter
  const body =
    kind === 'decision'
      ? slot.processing
        ? `Answering a System One decision with ${adapter}.`
        : `Holds a decision state computed with ${adapter}. A decision on the same adapter whose state extends it prefills only the new tokens.`
      : slot.processing
        ? `Running a chat generation on ${adapter === 'base' ? 'the base weights' : adapter}.`
        : `Holds a chat session produced on ${adapter === 'base' ? 'the base weights' : adapter}. Only a chat request on the same adapter can reuse it.`
  return (
    <Tooltip title={KIND_LABEL[kind]} body={body} className="tip--term slots__work">
      <span style={{ color: KIND_COLOR[kind] }}>
        {KIND_LABEL[kind]} · {adapter}
      </span>
    </Tooltip>
  )
}

export function SlotsPanel({
  slots,
  maxContext,
  replay,
}: {
  slots: SlotTelemetry[] | undefined
  maxContext: number
  replay: boolean
}) {
  if (!slots || slots.length === 0) {
    return (
      <Panel title="Slots" hint={replay ? 'replayMode' : 'lanes'} className="panel--wide">
        <Empty>
          {replay ? 'live only — lane occupancy is not recorded' : 'engine not attached'}
        </Empty>
      </Panel>
    )
  }

  const busy = slots.filter((slot) => slot.processing)
  const deciding = busy.filter((slot) => slot.work === 'decision').length

  return (
    <Panel
      title="Slots"
      hint="lanes"
      className="panel--wide"
      note={`${busy.length} busy of ${slots.length}${
        busy.length > 0 ? ` · ${busy.length - deciding} chat · ${deciding} System One` : ''
      }`}
    >
      <table className="table">
        <thead>
          <tr>
            <th />
            <th>slot</th>
            <th>
              <Term k="laneWork">work</Term>
            </th>
            <th className="numeric">prompt</th>
            <th className="numeric">
              <Term k="slotReused">reused</Term>
            </th>
            <th className="numeric">
              <Tooltip
                title="Context fill"
                body="Resident prompt tokens against the model's maximum context."
                className="tip--term"
              >
                ctx
              </Tooltip>
            </th>
            <th>
              <Term k="sessionDigest">session</Term>
            </th>
            <th className="numeric">
              <Term k="checkpoints">ckpt</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {slots.map((slot, index) => {
            const fill = maxContext === 0 ? 0 : slot.prompt_tokens / maxContext
            const state = slot.processing ? 'busy' : slot.retained ? 'retained' : 'idle'
            return (
              <tr key={index}>
                <td>
                  <StatusDot
                    tone={slot.processing ? 'accent' : slot.retained ? 'warning' : 'neutral'}
                  />
                </td>
                <td className="emphasis">
                  {index}
                  <Tooltip
                    title={state}
                    body={STATE_HINT[state]}
                    className="slots__state tip--term"
                  >
                    {state}
                  </Tooltip>
                </td>
                <td>
                  <Work slot={slot} />
                </td>
                <td className="numeric emphasis">{count(slot.prompt_tokens)}</td>
                <td className="numeric">
                  {slot.prompt_tokens === 0
                    ? '—'
                    : percent(slot.cached_tokens / slot.prompt_tokens)}
                </td>
                <td className="numeric" style={{ width: 72 }}>
                  <Meter fraction={fill} color={fill > 0.9 ? 'var(--warning)' : 'var(--blue)'} />
                </td>
                <td className="slots__digest">{slot.session_digest || '—'}</td>
                <td className="numeric">{slot.checkpoints || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="panel__footnote">
        Chat and System One share these lanes. A <Term k="retainedLane">retained lane</Term> holds a
        resident chat session or decision state in VRAM (L1) that a matching request of the same
        kind and adapter can reuse without any import. <Term k="slotReused">reused</Term> is the
        share of the prompt served from resident prefix.
      </p>
    </Panel>
  )
}
