// Pieces the state and questions editors share: the section frame with its mode switch, notes and
// problem count; the problems list; auto-growing text areas; and focus that follows an edit.

import {
  useLayoutEffect,
  useRef,
  type CSSProperties,
  type ReactNode,
  type RefObject,
  type TextareaHTMLAttributes,
} from 'react'

import { cx } from '../components/ui'
import type { EditorMarks } from './CodeEditor'
import { lineOf } from './json'
import { Popover, usePopover } from './Popover'
import { plural } from './request'
import type { Note } from './store'
import type { Problem, Severity } from './validate'

/** The worst severity reported for each form control. */
export function severities(problems: readonly Problem[]): Map<string, Severity> {
  const out = new Map<string, Severity>()
  for (const p of problems) {
    if (!p.loc || !('el' in p.loc)) continue
    if (out.get(p.loc.el) !== 'error') out.set(p.loc.el, p.severity)
  }
  return out
}

/** Line marks for a JSON editor from its section's problems. */
export function marksFor(problems: readonly Problem[], text: string): EditorMarks {
  const bad = new Set<number>()
  const warn = new Set<number>()
  let errAt: number | null = null
  for (const p of problems) {
    if (p.parse) errAt = p.parse.at
    else if (p.loc && 'at' in p.loc)
      (p.severity === 'error' ? bad : warn).add(lineOf(text, p.loc.at[0]))
  }
  return { errAt, bad, warn }
}

/**
 * Focus for a control that the next render creates or moves, named by its data-loc. Set it
 * before the store edit that causes the render.
 */
export function useFocusAfterRender(root: RefObject<HTMLElement | null>) {
  const pending = useRef<{ loc: string; select: boolean } | null>(null)
  useLayoutEffect(() => {
    const next = pending.current
    if (!next) return
    const el = root.current?.querySelector<HTMLElement>(`[data-loc="${CSS.escape(next.loc)}"]`)
    if (!el) return
    pending.current = null
    el.focus()
    el.scrollIntoView({ block: 'nearest' })
    if (next.select && el instanceof HTMLInputElement) el.select()
  })
  return (loc: string, select = false) => {
    pending.current = { loc, select }
  }
}

const FIELD_SIZING = typeof CSS !== 'undefined' && CSS.supports?.('field-sizing', 'content')

/** A textarea as tall as its text: CSS field-sizing where supported, measured elsewhere. */
export function GrowArea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (FIELD_SIZING || !el) return
    const grow = () => {
      el.style.height = 'auto'
      el.style.height = `${el.scrollHeight}px`
    }
    grow()
    const observer = new ResizeObserver(grow)
    observer.observe(el)
    return () => observer.disconnect()
  }, [props.value])
  return <textarea rows={1} {...props} ref={ref} className={cx('pg-grow', props.className)} />
}

export function ProblemList({
  problems,
  scope,
  onJump,
}: {
  problems: readonly Problem[]
  /** The section, for the empty message; the whole request when absent. */
  scope?: string
  onJump: (problem: Problem) => void
}) {
  const errors = problems.filter((p) => p.severity === 'error').length
  const title = problems.length
    ? [
        errors && plural(errors, 'error'),
        problems.length - errors && plural(problems.length - errors, 'warning'),
      ]
        .filter(Boolean)
        .join(', ') + ` in ${scope ?? 'the request'}`
    : 'No problems'
  return (
    <>
      <h3>{title}</h3>
      {problems.length ? (
        <ul className="pg-problems">
          {problems.map((p, j) => (
            <li key={j}>
              <button
                type="button"
                className={cx('pg-problem', `pg-problem--${p.severity}`)}
                onClick={() => onJump(p)}
              >
                <span className="pg-problem__mark" aria-hidden="true">
                  {p.severity === 'error' ? '\u2715' : '!'}
                </span>
                <span>
                  <span className="sr-only">{p.severity}: </span>
                  {p.msg}
                </span>
                <small>{p.where}</small>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="pg-pop__empty">
          Nothing to fix here; the {scope ?? 'request'} {scope === 'questions' ? 'are' : 'is'} ready
          to run.
        </p>
      )}
    </>
  )
}

/** A section's problem count, opening the list. Errors block Run; warnings do not. */
export function ProblemsButton({
  problems,
  scope,
  onJump,
}: {
  problems: readonly Problem[]
  scope: string
  onJump: (problem: Problem) => void
}) {
  const pop = usePopover()
  if (!problems.length) return null
  const error = problems.some((p) => p.severity === 'error')
  const label = `${scope}: ${plural(problems.length, error ? 'problem' : 'warning')}`
  return (
    <>
      <button
        type="button"
        className={cx('pg-count', error ? 'pg-count--error' : 'pg-count--warning')}
        aria-haspopup="dialog"
        aria-expanded="false"
        aria-label={label}
        title={`${label} (F8 jumps to the next one)`}
        onClick={(event) => pop.toggle(event.currentTarget)}
      >
        {error ? '\u2715' : '!'} {problems.length}
      </button>
      {pop.anchor ? (
        <Popover anchor={pop.anchor} label="Problems" onClose={pop.close}>
          <ProblemList
            problems={problems}
            scope={scope}
            onJump={(p) => {
              pop.close()
              onJump(p)
            }}
          />
        </Popover>
      ) : null}
    </>
  )
}

export function EditorSection<M extends string>({
  id,
  title,
  sub,
  modes,
  mode,
  onMode,
  onFormat,
  problems,
  note,
  onShowNote,
  onJump,
  children,
  className,
  style,
}: {
  id: string
  title: string
  sub: string
  modes: ReadonlyArray<{ value: M; label: string }>
  mode: M
  onMode: (mode: M) => void
  /** Present in JSON mode. */
  onFormat?: () => void
  problems: readonly Problem[]
  note?: Note
  onShowNote?: (at: number) => void
  onJump: (problem: Problem) => void
  children: ReactNode
  className?: string
  style?: CSSProperties
}) {
  return (
    <section className={cx('pg-sec', className)} aria-labelledby={`${id}-title`} style={style}>
      <header className="pg-sec__head">
        <h2 id={`${id}-title`}>{title}</h2>
        <span className="pg-sec__sub">{sub}</span>
        <span className="pg-sec__tools">
          <ProblemsButton problems={problems} scope={title.toLowerCase()} onJump={onJump} />
          {onFormat ? (
            <button
              type="button"
              className="button"
              onClick={onFormat}
              title="Format (Shift+Alt+F)"
            >
              format
            </button>
          ) : null}
          <span className="pg-seg" role="group" aria-label={`${title} editor`}>
            {modes.map((m) => (
              <button
                key={m.value}
                type="button"
                aria-pressed={mode === m.value}
                onClick={() => onMode(m.value)}
              >
                {m.label}
              </button>
            ))}
          </span>
        </span>
      </header>
      {note ? (
        <p
          className={cx('pg-note', note.kind === 'warn' && 'pg-note--warn')}
          role={note.kind === 'warn' ? 'status' : undefined}
        >
          <span aria-hidden="true">{note.kind === 'warn' ? '!' : 'i'}</span>
          <span>
            {note.text}
            {note.at !== undefined && onShowNote ? (
              <>
                {' '}
                <button type="button" onClick={() => onShowNote(note.at!)}>
                  Show me
                </button>
              </>
            ) : null}
          </span>
        </p>
      ) : null}
      <div className="pg-sec__body">{children}</div>
    </section>
  )
}
