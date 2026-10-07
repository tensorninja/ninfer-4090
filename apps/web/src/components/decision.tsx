// The phases of one System One decision, shared by the dashboard's System One panel and the
// playground's per-run engine facts so both draw the same split in the same colours.

import type { DecisionDoneRecord } from '../lib/records'
import { decisionPhases } from '../lib/derive'
import { seconds } from '../lib/format'
import { StackedBar } from './charts'
import { CHART } from './echart'

export const DECISION_PHASES = [
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
    key: 'vision',
    label: 'vision',
    color: CHART.accent,
    hint: 'Image encoding within state execution. Shown separately from the remaining state prefill, never counted twice.',
  },
  {
    key: 'state',
    label: 'state',
    color: CHART.violet,
    hint: 'Prefilling state tokens, excluding image encoding shown as vision.',
  },
  {
    key: 'branch',
    label: 'branches',
    color: CHART.systemOne,
    hint: 'The branch passes, one per packed group of questions, with their readout and the pointer head.',
  },
] as const

/** Per-decision phase split, on the same colours as the summary bar. */
export function DecisionWaterfall({
  record,
  height = 5,
}: {
  record: DecisionDoneRecord
  height?: number
}) {
  const phases = decisionPhases(record)
  const total = phases.wait + phases.restore + phases.vision + phases.state + phases.branch
  return (
    <StackedBar
      height={height}
      segments={DECISION_PHASES.map((phase) => ({
        label: phase.label,
        value: phases[phase.key],
        color: phase.color,
        hint: phase.hint,
        display: `${seconds(phases[phase.key])} of ${seconds(total)}`,
      }))}
    />
  )
}
