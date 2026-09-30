// What the engine did for the decision a run made: state reuse, branch passes, the lane, and the
// phase split, from the run's own records. See engine.ts for how a run finds them.

import { DECISION_PHASES, DecisionWaterfall } from '../components/decision'
import { Info } from '../components/tooltip'
import { Pill } from '../components/ui'
import { decisionLatency, decisionPhases, SOURCE_COLOR } from '../lib/derive'
import { count, rate, seconds } from '../lib/format'
import type { DecisionContext } from '../lib/records'
import type { Facts } from './engine'

const plural = (n: number, word: string) => `${count(n)} ${word}${n === 1 ? '' : 's'}`

function Layout({ request }: { request: DecisionContext }) {
  return (
    <>
      <dt>request</dt>
      <dd>
        {plural(request.questions, 'question')} · {plural(request.options, 'option')} ·{' '}
        {count(request.input_tokens)} input tokens
      </dd>
    </>
  )
}

function Phases({ record }: { record: Extract<Facts, { kind: 'done' }>['record'] }) {
  const phases = decisionPhases(record)
  return (
    <div className="pg-facts__phases">
      <DecisionWaterfall record={record} height={6} />
      <div className="pg-facts__legend">
        {DECISION_PHASES.map((phase) => (
          <span key={phase.key} title={phase.hint}>
            <i style={{ background: phase.color }} />
            {phase.label} {seconds(phases[phase.key])}
          </span>
        ))}
      </div>
    </div>
  )
}

export function EngineFacts({ facts, onGoLive }: { facts: Facts; onGoLive: () => void }) {
  let body: React.ReactNode
  let tone: 'neutral' | 'accent' | 'warning' | 'danger' = 'neutral'
  let badge = ''

  if (facts.kind === 'done') {
    const { request, result, timings_seconds: timings, rates } = facts.record
    tone = 'accent'
    badge = `lane ${result.slot}`
    const reused = result.reused_state_tokens
    body = (
      <>
        <Phases record={facts.record} />
        <dl className="pg-facts__list">
          <dt>state</dt>
          <dd>
            {plural(request.state_tokens, 'token')}
            {request.state_truncated ? ' (cut to 65,536)' : ''}:{' '}
            {reused > 0 ? (
              <>
                <b style={{ color: SOURCE_COLOR[result.state_source] }}>{count(reused)} reused</b>{' '}
                from {result.state_source}
                {result.computed_state_tokens > 0
                  ? `, ${count(result.computed_state_tokens)} computed`
                  : ''}
              </>
            ) : (
              <>{count(result.computed_state_tokens)} computed</>
            )}
            {request.allow_prefix_reuse ? '' : ' · reuse off (--no-prefix-reuse)'}
          </dd>
          <dt>branches</dt>
          <dd>
            {plural(request.branch_tokens, 'token')} in {plural(result.branch_passes, 'pass')}
            {result.long_branch_chunks > 0
              ? ` + ${plural(result.long_branch_chunks, 'long-branch chunk')}`
              : ''}{' '}
            · longest {count(request.longest_branch)}
          </dd>
          <dt>rates</dt>
          <dd>
            state {rates.state_tok_s === null ? '\u2014' : `${rate(rates.state_tok_s)} tok/s`} ·
            branches {rates.branch_tok_s === null ? '\u2014' : `${rate(rates.branch_tok_s)} tok/s`}
          </dd>
          <dt>time</dt>
          <dd>
            latency_ms {(decisionLatency(facts.record) * 1000).toFixed(1)} · total{' '}
            {seconds(timings.total)} with {seconds(timings.prepare)} preparing
          </dd>
          <dt>adapter</dt>
          <dd>
            {request.model === request.adapter
              ? request.adapter
              : `${request.model} \u2192 ${request.adapter}`}
          </dd>
          <Layout request={request} />
        </dl>
      </>
    )
  } else if (facts.kind === 'error') {
    const { request, error } = facts.record
    tone = 'danger'
    badge = `HTTP ${error.status}`
    body = (
      <dl className="pg-facts__list">
        <dt>error</dt>
        <dd>{error.message}</dd>
        <dt>state</dt>
        <dd>
          {plural(request.state_tokens, 'token')}
          {request.state_truncated ? ' (cut to 65,536)' : ''} · longest branch{' '}
          {count(request.longest_branch)}
        </dd>
        <Layout request={request} />
      </dl>
    )
  } else if (facts.kind === 'running') {
    tone = 'warning'
    badge = facts.start ? 'admitted' : 'submitted'
    body = facts.start ? (
      <dl className="pg-facts__list">
        <dt>state</dt>
        <dd>
          {plural(facts.start.request.state_tokens, 'token')} ·{' '}
          {plural(facts.start.request.branch_tokens, 'branch token')}
        </dd>
        <Layout request={facts.start.request} />
      </dl>
    ) : (
      <p>Waiting for the engine to take the decision.</p>
    )
  } else if (facts.kind === 'waiting') {
    badge = 'waiting'
    body = <p>The response arrived; waiting for the decision&rsquo;s record on the event stream…</p>
  } else if (facts.kind === 'rejected') {
    tone = 'danger'
    badge = 'no record'
    body = (
      <p>
        Rejected before submission: the server refused the request while preparing it, before the
        engine ran anything, and such a request is not logged.
      </p>
    )
  } else if (facts.kind === 'missing') {
    tone = 'warning'
    badge = 'no record'
    body = (
      <p>
        The decision was answered, but its record was not received: the event stream drops records
        under backpressure and while it reconnects.
      </p>
    )
  } else if (facts.kind === 'offline') {
    tone = 'warning'
    badge = facts.replay ? 'replay' : 'offline'
    body = facts.replay ? (
      <p>
        The dashboard is replaying a log, so this run&rsquo;s records cannot arrive.{' '}
        <button type="button" className="button" onClick={onGoLive}>
          go live
        </button>
      </p>
    ) : (
      <p>
        The live event stream is not connected, so this run&rsquo;s records cannot arrive. It
        reconnects on its own.
      </p>
    )
  } else {
    tone = 'danger'
    badge = 'no response'
    body = <p>The server did not answer, so there is no decision to describe.</p>
  }

  return (
    <section className="pg-facts" aria-label="Engine facts">
      <header className="pg-facts__head">
        <h3>
          engine <Info k="engineFacts" />
        </h3>
        <Pill tone={tone}>{badge}</Pill>
      </header>
      {body}
    </section>
  )
}
