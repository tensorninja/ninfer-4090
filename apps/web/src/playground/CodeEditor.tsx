// A JSON editor: a transparent textarea over a highlighted copy of its text.
//
// The textarea does the editing, so selection, IME, spellcheck-off typing and the browser's own
// undo all behave natively; the painted `pre` under it shows syntax colours, line numbers, the
// error character and lines with problems. Edits the editor makes itself (indent, auto-indent,
// Format) go through execCommand('insertText'), which keeps them on the browser's undo stack.
// Ported from laya's playground (examples/server.py: paint, CodeEditor; Apache-2.0).

import {
  memo,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
  type Ref,
} from 'react'

import { cx } from '../components/ui'
import { formatJson, lineCol, lineOf, tokenize, tryParse, type JsonSyntaxError } from './json'

export interface CodeEditorHandle {
  /** Focuses the editor, selects `start..end` and flashes its line. */
  select: (start: number, end: number) => void
  /** Reformats valid JSON as one undoable edit; jumps to the error when it does not parse. */
  format: () => void
  focus: () => void
}

export interface EditorMarks {
  /** Offset of the syntax error, if any. */
  errAt: number | null
  bad: ReadonlySet<number>
  warn: ReadonlySet<number>
}

const NO_MARKS: EditorMarks = { errAt: null, bad: new Set(), warn: new Set() }

export const plainEnter = (event: KeyboardEvent) =>
  event.key === 'Enter' &&
  !event.nativeEvent.isComposing &&
  event.keyCode !== 229 &&
  !event.shiftKey &&
  !event.ctrlKey &&
  !event.metaKey &&
  !event.altKey

export const smoothScroll = (): ScrollBehavior =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'

/** The key shortcuts name with Enter: Command on Apple platforms, Control elsewhere. */
export const MOD =
  typeof navigator !== 'undefined' &&
  /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent)
    ? '\u2318'
    : 'Ctrl'

/** Types `text` over the selection as the user would, so it can be undone. */
function insertText(ta: HTMLTextAreaElement, text: string) {
  ta.focus({ preventScroll: true })
  let ok = false
  try {
    ok = document.execCommand('insertText', false, text)
  } catch {
    ok = false
  }
  if (!ok) {
    ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, 'end')
    ta.dispatchEvent(new Event('input', { bubbles: true }))
  }
}

/** Tab indents the selected lines by two spaces, or types two; Shift+Tab outdents. */
function indent(ta: HTMLTextAreaElement, out: boolean) {
  const v = ta.value
  const s = ta.selectionStart
  const e = ta.selectionEnd
  if (!out && s === e) {
    insertText(ta, '  ')
    return
  }
  const ls = v.lastIndexOf('\n', s - 1) + 1
  let le = v.indexOf('\n', e > s && v[e - 1] === '\n' ? e - 1 : e)
  if (le < 0) le = v.length
  const block = v.slice(ls, le)
  let first = 0
  let total = 0
  const next = block
    .split('\n')
    .map((line, j) => {
      const d = out ? -(line.startsWith('  ') ? 2 : line.startsWith(' ') ? 1 : 0) : 2
      if (j === 0) first = d
      total += d
      return d < 0 ? line.slice(-d) : '  ' + line
    })
    .join('\n')
  if (next === block) return
  ta.setSelectionRange(ls, le)
  insertText(ta, next)
  ta.setSelectionRange(Math.max(ls, s + first), Math.max(ls, e + total))
}

/** Enter keeps the indentation, one level deeper after an opening bracket, and splits `{|}`. */
function newline(ta: HTMLTextAreaElement) {
  const v = ta.value
  const s = ta.selectionStart
  const e = ta.selectionEnd
  const ls = v.lastIndexOf('\n', s - 1) + 1
  const ind = /^[ \t]*/.exec(v.slice(ls, s))![0]
  const before = v.slice(ls, s).trimEnd().slice(-1)
  const open = before === '{' || before === '['
  if (open && v[e] === (before === '{' ? '}' : ']')) {
    insertText(ta, '\n' + ind + '  \n' + ind)
    const p = s + ind.length + 3
    ta.setSelectionRange(p, p)
  } else {
    insertText(ta, '\n' + ind + (open ? '  ' : ''))
  }
}

const Line = memo(function Line({
  text,
  err,
  mark,
  cur,
}: {
  text: string
  /** Column of the error character, text.length for end of line, -1 for none. */
  err: number
  mark: '' | 'bad' | 'warn'
  cur: boolean
}) {
  const parts: ReactNode[] = []
  for (const [j, token] of tokenize(text).entries()) {
    const end = token.at + token.text.length
    if (err >= token.at && err < end) {
      const k = err - token.at
      if (k)
        parts.push(
          <span key={`${j}a`} className={token.cls}>
            {token.text.slice(0, k)}
          </span>,
        )
      parts.push(
        <span key={`${j}b`} className={cx(token.cls, 'err')}>
          {token.text.slice(k, k + 1)}
        </span>,
      )
      if (k + 1 < token.text.length) {
        parts.push(
          <span key={`${j}c`} className={token.cls}>
            {token.text.slice(k + 1)}
          </span>,
        )
      }
    } else {
      parts.push(
        token.cls ? (
          <span key={j} className={token.cls}>
            {token.text}
          </span>
        ) : (
          token.text
        ),
      )
    }
  }
  if (err === text.length) parts.push(<span key="eol" className="eol" />)
  return <div className={cx('ln', mark, cur && 'cur')}>{parts}</div>
})

/** The highlighted text alone, for a read-only view such as the response body. */
export function PaintedJson({ text, className }: { text: string; className?: string }) {
  const lines = text.split('\n')
  return (
    <div
      className={cx('cx cx--ro', className)}
      style={{ ['--gw' as string]: `${Math.max(2, String(lines.length).length)}ch` }}
    >
      <div className="cx__body">
        <pre className="cx__hl">
          {lines.map((line, j) => (
            <Line key={j} text={line} err={-1} mark="" cur={false} />
          ))}
        </pre>
      </div>
    </div>
  )
}

export function CodeEditor({
  id,
  label,
  value,
  onChange,
  placeholder,
  marks = NO_MARKS,
  error,
  invalid,
  ref,
}: {
  id: string
  label: string
  value: string
  onChange: (text: string) => void
  placeholder?: string
  marks?: EditorMarks
  /** The syntax error, shown in a bar under the text that jumps to it. */
  error?: JsonSyntaxError
  invalid?: boolean
  ref?: Ref<CodeEditorHandle>
}) {
  const ta = useRef<HTMLTextAreaElement>(null)
  const pre = useRef<HTMLPreElement>(null)
  const escaped = useRef(false)
  const [cur, setCur] = useState(-1)
  const [hinting, setHinting] = useState(false)
  const hintTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const caret = useCallback(() => {
    const el = ta.current
    setCur(el && document.activeElement === el ? lineOf(el.value, el.selectionStart) : -1)
  }, [])

  useEffect(() => {
    document.addEventListener('selectionchange', caret)
    return () => {
      document.removeEventListener('selectionchange', caret)
      clearTimeout(hintTimer.current)
    }
  }, [caret])

  const select = useCallback(
    (start: number, end: number) => {
      const el = ta.current
      if (!el) return
      el.focus({ preventScroll: true })
      el.setSelectionRange(start, end)
      const row = pre.current?.children[lineOf(el.value, start)] as HTMLElement | undefined
      if (row) {
        row.scrollIntoView({ block: 'center', behavior: smoothScroll() })
        row.classList.remove('flash')
        void row.offsetWidth
        row.classList.add('flash')
      }
      caret()
    },
    [caret],
  )

  const format = useCallback(() => {
    const el = ta.current
    if (!el) return
    const parsed = tryParse(el.value)
    if (parsed.error) {
      select(parsed.error.at, parsed.error.at)
      return
    }
    const next = formatJson(parsed.ast) + '\n'
    if (next === el.value) return
    el.focus()
    el.select()
    insertText(el, next)
    el.setSelectionRange(0, 0)
  }, [select])

  useImperativeHandle(ref, () => ({ select, format, focus: () => ta.current?.focus() }), [
    select,
    format,
  ])

  /** The textarea is as tall as its text, so PageUp/PageDown move one screen of the pane. */
  const page = (dir: 1 | -1, extend: boolean) => {
    const el = ta.current
    const rows = pre.current?.children
    if (!el || !rows) return
    const v = el.value
    const scroller = el.closest('.pg-sec__body')
    const pane = scroller && getComputedStyle(scroller).overflowY !== 'visible' ? scroller : null
    const dy = dir * Math.max(20, (pane ? pane.clientHeight : window.innerHeight) - 60)
    const back = el.selectionDirection === 'backward'
    const head = back ? el.selectionStart : el.selectionEnd
    const anchor = back ? el.selectionEnd : el.selectionStart
    let li = lineOf(v, head)
    const row = (j: number) => rows[j] as HTMLElement
    if (!rows[li]) return
    const y = row(li).offsetTop + dy
    const col = head - (v.lastIndexOf('\n', head - 1) + 1)
    const off = (j: number) => Math.abs(row(j).offsetTop - y)
    if (dir > 0) while (li < rows.length - 1 && off(li + 1) <= off(li)) li++
    else while (li > 0 && off(li - 1) <= off(li)) li--
    let start = 0
    for (let k = 0; k < li; k++) start = v.indexOf('\n', start) + 1
    const end = v.indexOf('\n', start)
    const at = start + Math.min(col, (end < 0 ? v.length : end) - start)
    if (extend)
      el.setSelectionRange(
        Math.min(anchor, at),
        Math.max(anchor, at),
        at < anchor ? 'backward' : 'forward',
      )
    else el.setSelectionRange(at, at)
    ;(pane ?? window).scrollBy(0, dy)
    row(li).scrollIntoView({ block: 'nearest' })
    caret()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const el = event.currentTarget
    if (event.nativeEvent.isComposing || event.keyCode === 229) return
    if (['Shift', 'Control', 'Alt', 'Meta'].includes(event.key)) return
    if (event.key === 'Escape') {
      escaped.current = true
      return
    }
    if (event.key === 'Tab' && !event.ctrlKey && !event.altKey && !event.metaKey) {
      // Esc, then Tab: leave the editor like any other field.
      if (escaped.current) {
        escaped.current = false
        return
      }
      event.preventDefault()
      indent(el, event.shiftKey)
      return
    }
    escaped.current = false
    if (
      (event.key === 'PageDown' || event.key === 'PageUp') &&
      !event.ctrlKey &&
      !event.altKey &&
      !event.metaKey
    ) {
      event.preventDefault()
      page(event.key === 'PageDown' ? 1 : -1, event.shiftKey)
      return
    }
    if (plainEnter(event)) {
      event.preventDefault()
      newline(el)
    } else if (event.altKey && event.shiftKey && event.code === 'KeyF') {
      event.preventDefault()
      format()
    }
  }

  const lines = value.split('\n')
  let offset = 0
  const errAt = marks.errAt ?? -1
  const painted = lines.map((line, j) => {
    const start = offset
    offset += line.length + 1
    const err = errAt >= start && errAt <= start + line.length ? errAt - start : -1
    const mark = err >= 0 || marks.bad.has(j) ? 'bad' : marks.warn.has(j) ? 'warn' : ''
    return <Line key={j} text={line} err={err} mark={mark} cur={j === cur} />
  })
  const where = error ? lineCol(value, error.at) : null

  return (
    <div
      className={cx('cx', hinting && 'cx--hinting')}
      style={{ ['--gw' as string]: `${Math.max(2, String(lines.length).length)}ch` }}
    >
      <div className="cx__hint" aria-hidden="true">
        <span>Tab indents · Esc, then Tab to leave</span>
      </div>
      <div className="cx__body">
        <pre ref={pre} className="cx__hl" aria-hidden="true">
          {painted}
        </pre>
        <textarea
          ref={ta}
          id={id}
          className="cx__ta"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          onScroll={(event) => {
            if (event.currentTarget.scrollTop) event.currentTarget.scrollTop = 0
          }}
          onFocus={() => {
            caret()
            setHinting(true)
            clearTimeout(hintTimer.current)
            hintTimer.current = setTimeout(() => setHinting(false), 4000)
          }}
          onBlur={() => {
            escaped.current = false
            setHinting(false)
            caret()
          }}
          onPointerDown={() => {
            escaped.current = false
          }}
          placeholder={placeholder}
          aria-label={label}
          aria-invalid={invalid || undefined}
          aria-describedby={error ? `${id}-error` : undefined}
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          autoCorrect="off"
          wrap="soft"
        />
      </div>
      {error && where ? (
        <button
          type="button"
          id={`${id}-error`}
          className="cx__msg"
          onClick={() => select(error.at, error.at)}
        >
          <b>
            Ln {where.line}, Col {where.col}
          </b>
          <span>{error.msg}</span>
        </button>
      ) : null}
    </div>
  )
}
