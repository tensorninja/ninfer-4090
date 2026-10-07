// Answers as distributions, and the questions before anything has run.
// Adapted from laya's playground (examples/server.py: answerRow, renderPreview; Apache-2.0).

import type { ReactNode } from 'react'

import { cx } from '../components/ui'
import {
  codeSpans,
  describeNode,
  pct,
  positionalLabel,
  previewOf,
  UNSURE,
  type AnswerView,
} from './answers'
import { dedupe, lastEntry, type JsonNode } from './json'
import type { Protocol } from './request'

/** Instructions with `backtick` spans as code. */
export function Instructions({ text }: { text: string }) {
  return (
    <>
      {codeSpans(text).map((part, j) =>
        part.code ? <code key={j}>{part.text}</code> : <span key={j}>{part.text}</span>,
      )}
    </>
  )
}

function IdCell({
  answerKey,
  type,
  instructions,
}: {
  answerKey: string
  type: string
  instructions: string
}) {
  return (
    <span className="pg-id">
      <span className="pg-id__key">{answerKey || '\u2205'}</span>
      <span className="pg-id__type">{type || '?'}</span>
      <span className="pg-id__ins">
        <Instructions text={instructions} />
      </span>
    </span>
  )
}

function ListHead({ right }: { right: string }) {
  return (
    <div className="pg-alist__head" aria-hidden="true">
      <span>question & instructions</span>
      <span>{right}</span>
    </div>
  )
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))

/** The expected score marked on the 0..top scale. */
function Scale({ score, top }: { score: number; top: number }) {
  const at = `${clamp01(score / top) * 100}%`
  return (
    <div
      className="pg-scale"
      role="img"
      aria-label={`Expected score ${score.toFixed(2)} on a 0 to ${top} scale`}
    >
      <span>0</span>
      <span className="pg-scale__track">
        {Array.from({ length: top + 1 }, (_, j) => (
          <span key={j} className="pg-scale__tick" style={{ left: `${(j / top) * 100}%` }} />
        ))}
        <span className="pg-scale__fill" style={{ width: at }} />
        <span className="pg-scale__mark" style={{ left: at }} />
      </span>
      <span>{top}</span>
    </div>
  )
}

function AnswerRow({
  view,
  open,
  onToggle,
}: {
  view: AnswerView
  open: boolean
  onToggle: (open: boolean) => void
}) {
  const head = <IdCell answerKey={view.label} type={view.type} instructions={view.instructions} />
  if (!view.rows.length) {
    return (
      <details
        className="pg-ans"
        open={open}
        onToggle={(event) => onToggle(event.currentTarget.open)}
      >
        <summary>
          <span className="pg-ans__chev" aria-hidden="true" />
          {head}
          <span className="pg-ans__verdict">
            <span className="pg-ans__sub">raw answer</span>
          </span>
        </summary>
        <pre className="pg-ans__raw">{JSON.stringify(view.raw ?? null, null, 2)}</pre>
      </details>
    )
  }
  const top = view.rows[view.win]!
  const sub: ReactNode[] = []
  if (view.score !== null) {
    sub.push(
      <span key="score">
        score <b>{view.score.toFixed(2)}</b> on 0{'\u2013'}
        {view.maxLevel}
      </span>,
    )
  }
  if (view.unsure) {
    sub.push(
      <span key="unsure">
        <span
          className="pg-ans__unsure"
          title={`The answer's probability is below ${UNSURE.toFixed(2)}`}
        >
          uncertain
        </span>
        {view.runnerUp ? (
          <>
            , runner-up <b>{view.runnerUp.label}</b> {pct(view.runnerUp.p)}
          </>
        ) : null}
      </span>,
    )
  }
  return (
    <details
      className="pg-ans"
      open={open}
      onToggle={(event) => onToggle(event.currentTarget.open)}
    >
      <summary>
        <span className="pg-ans__chev" aria-hidden="true" />
        {head}
        <span className="pg-ans__verdict">
          <span className="pg-ans__top">
            <span className={cx('pg-ans__label', view.type === 'score' && 'pg-ans__label--prose')}>
              {top.label}
            </span>
            <span className={cx('pg-ans__pct', view.unsure && 'pg-ans__pct--unsure')}>
              {pct(view.headline)}
            </span>
          </span>
          {sub.length ? (
            <span className="pg-ans__sub">
              {sub.map((part, j) => (
                <span key={j}>
                  {j ? ' \u00b7 ' : ''}
                  {part}
                </span>
              ))}
            </span>
          ) : null}
        </span>
      </summary>
      <div className="pg-ans__body">
        {view.score !== null && view.maxLevel > 0 ? (
          <Scale score={view.score} top={view.maxLevel} />
        ) : null}
        <ol className="pg-dist">
          {view.rows.map((row, j) => (
            <li key={j} className={cx('pg-dist__row', j === view.win && 'pg-dist__row--win')}>
              <span
                className={cx('pg-dist__label', view.type === 'score' && 'pg-dist__label--prose')}
              >
                {row.level !== undefined ? (
                  <span className="pg-dist__level">{row.level}</span>
                ) : null}
                {row.label}
              </span>
              <span className="pg-dist__bar" aria-hidden="true">
                <span style={{ width: `${clamp01(row.p) * 100}%` }} />
              </span>
              <span className="pg-dist__p">{pct(row.p)}</span>
              {row.desc ? <span className="pg-dist__desc">{row.desc}</span> : null}
            </li>
          ))}
        </ol>
        {view.confidence !== null ? (
          <div className="pg-ans__meta">
            <code>confidence</code> <b>{view.confidence.toFixed(4)}</b>{' '}
            {view.type === 'choice'
              ? '(p\u2098\u2090\u2093 \u2212 1/K) / (1 \u2212 1/K): 0 at uniform, 1 at certainty'
              : '1 \u2212 E|level \u2212 mode| / D: 1 when all mass is on one level'}
          </div>
        ) : null}
      </div>
    </details>
  )
}

export function AnswerList({
  views,
  collapsed,
  onToggle,
}: {
  views: readonly AnswerView[]
  collapsed: ReadonlySet<string>
  onToggle: (key: string, open: boolean) => void
}) {
  return (
    <div className="pg-alist">
      <ListHead right="answer & probability" />
      {views.map((view) => (
        <AnswerRow
          key={view.key}
          view={view}
          open={!collapsed.has(view.key)}
          onToggle={(open) => onToggle(view.key, open)}
        />
      ))}
    </div>
  )
}

/** The questions as they would be asked, before the first run and under an error. */
export function PreviewList({
  questions,
  protocol = 'typesafe',
}: {
  questions: JsonNode | null
  protocol?: Protocol
}) {
  // As the server reads them: a repeated key keeps its first position and its last value.
  const entries =
    protocol === 'openai'
      ? questions?.t === 'arr'
        ? questions.items.map((v, index) => {
            const name = lastEntry(v, 'name')?.v
            return {
              k: String(index),
              label: positionalLabel(name?.t === 'str' ? name.v : null, index),
              v,
            }
          })
        : []
      : questions?.t === 'obj'
        ? (dedupe(questions) as typeof questions).entries.map((entry) => ({
            ...entry,
            label: entry.k,
          }))
        : []
  if (!entries.length) return null
  return (
    <div className="pg-alist">
      <ListHead right="possible answers" />
      {entries.map((entry) => {
        const type = lastEntry(entry.v, 'type')?.v
        return (
          <div key={entry.k} className="pg-pv">
            <span className="pg-pv__dot" aria-hidden="true" />
            <IdCell
              answerKey={entry.label}
              type={type?.t === 'str' ? type.v : ''}
              instructions={describeNode(lastEntry(entry.v, 'instructions')?.v)}
            />
            <span className="pg-ans__verdict pg-pv__what">{previewOf(entry.v, protocol)}</span>
          </div>
        )
      })}
    </div>
  )
}
