// The System One playground: build a request, run it against this server's
// POST /typesafe/v1/systemone, and read each answer's distribution beside what the engine did for
// that decision.
// Ported from laya's playground (examples/server.py; Apache-2.0) and adapted to kev's System One.

import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { Tooltip } from '../components/tooltip'
import { cx, Pill } from '../components/ui'
import { count } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import { copyText } from './clipboard'
import { MOD, smoothScroll, type CodeEditorHandle } from './CodeEditor'
import type { EngineView } from './engine'
import { cardFor, modelOptions, rotatedCodec, type ModelCard } from './models'
import { Popover, usePopover } from './Popover'
import { QuestionsEditor } from './QuestionsEditor'
import { plural, quote } from './request'
import { ResponsePane } from './ResponsePane'
import { ProblemList } from './Section'
import { endpointFor } from './snippets'
import { Splitter } from './Splitter'
import { StateEditor } from './StateEditor'
import {
  SH_BOUNDS,
  SPLIT_BOUNDS,
  usePlayground,
  type PlaygroundSnapshot,
  type PlaygroundStore,
} from './store'
import { LiveRegion, ToastView } from './Toasts'
import { LOC, type Loc, type Problem } from './validate'

export type { EngineView }

/** Side by side from this window width; stacked below it. */
const WIDE = 1100
/** Neither pane narrower than this, side by side. */
const MIN_PANE = 390

const SHORTCUTS: ReadonlyArray<[string, string]> = [
  [`${MOD}+\u21b5`, 'Run the request, from anywhere'],
  ['F8 / Shift+F8', 'Jump to the next / previous problem'],
  ['Tab / Shift+Tab', 'Indent / outdent lines in a JSON editor'],
  ['Esc, then Tab / Shift+Tab', 'Leave a JSON editor, forward / back'],
  ['Shift+Alt+F', 'Format the JSON you are editing'],
  ['\u21b5', 'In a form: next field, or a new option / level'],
  ['?', 'Show these shortcuts'],
]

function splitBounds(work: HTMLElement | null): readonly [number, number] {
  const w = work?.getBoundingClientRect().width ?? 0
  const lo = Math.max(SPLIT_BOUNDS[0], MIN_PANE / w)
  const hi = Math.min(SPLIT_BOUNDS[1], 1 - (MIN_PANE + 1) / w)
  return w > 0 && lo < hi ? [lo, hi] : SPLIT_BOUNDS
}

/** The model card's fields, in the tooltip beside the picker. */
function CardTip({ card }: { card: ModelCard }) {
  const rows: Array<[string, string]> = [
    ['adapter', card.adapter],
    ['base', card.base],
    ['rank', card.rank ? String(card.rank) : ''],
    ['temperature', card.temperature ? String(card.temperature) : ''],
    ['released', card.release_date],
    ['kv cache', card.kv_cache],
    ['max context', card.max_context ? `${count(card.max_context)} tokens` : ''],
    ['max state', card.max_state_tokens ? `${count(card.max_state_tokens)} tokens` : ''],
    ['prefix reuse', card.prefix_reuse === undefined ? '' : card.prefix_reuse ? 'on' : 'off'],
  ]
  return (
    <Tooltip
      title={card.name}
      className="tip--info"
      body={
        <>
          {card.description ? <span className="pg-cardtip__desc">{card.description}</span> : null}
          <span className="pg-cardtip">
            {rows
              .filter(([, value]) => value)
              .map(([label, value]) => (
                <Fragment key={label}>
                  <span>{label}</span>
                  <span>{value}</span>
                </Fragment>
              ))}
          </span>
        </>
      }
    >
      <span aria-label={`About ${card.name}`}>?</span>
    </Tooltip>
  )
}

function ModelPicker({ s, store }: { s: PlaygroundSnapshot; store: PlaygroundStore }) {
  const { catalog, editor } = s
  const cards = catalog.state === 'ready' ? catalog.cards : []
  const options =
    catalog.state === 'ready'
      ? modelOptions(cards, editor.model, editor.protocol)
      : [
          {
            value: editor.model,
            label:
              editor.model ||
              (editor.protocol === 'openai' ? 'select a decision model' : 'server default'),
            unknown: false,
          },
        ]
  const card = cardFor(cards, editor.model)
  const warned = s.analysis.problems.some((p) => p.sec === 'model')
  return (
    <span className="pg-model">
      <input
        type="text"
        hidden={catalog.state !== 'error'}
        data-loc={catalog.state === 'error' ? LOC.model : undefined}
        className="pg-input"
        aria-label="Model name"
        value={editor.model}
        placeholder={
          editor.protocol === 'openai'
            ? 'Decision adapter name (required)'
            : 'Model name (optional)'
        }
        onChange={(event) => store.setModel(event.target.value)}
      />
      <select
        hidden={catalog.state === 'error'}
        data-loc={catalog.state !== 'error' ? LOC.model : undefined}
        className={cx('pg-select', warned && 'is-warn')}
        aria-label="Model"
        title={
          editor.protocol === 'openai'
            ? 'The actual decision adapter name; OpenAI has no SDK alias'
            : "The decision adapter that answers; left out, the server's SDK default does"
        }
        value={editor.model}
        onChange={(event) => store.setModel(event.target.value)}
      >
        {options.map((option, j) => (
          <option key={j} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {card ? <CardTip card={card} /> : null}
      {card && rotatedCodec(card.kv_cache) ? (
        <Tooltip
          title={GLOSSARY.kvCodecCaution.title}
          body={GLOSSARY.kvCodecCaution.body}
          className="pg-caution"
        >
          {card.kv_cache} kv
        </Tooltip>
      ) : null}
      {catalog.state === 'loading' ? <span className="pg-model__note">loading models…</span> : null}
      {catalog.state === 'error' ? (
        <span className="pg-model__note pg-model__note--bad" title={catalog.message}>
          models unavailable ({catalog.message})
        </span>
      ) : null}
    </span>
  )
}

export function Playground({
  store,
  engine,
  resumeLive,
}: {
  store: PlaygroundStore
  engine: EngineView
  resumeLive: () => void
}) {
  const s = usePlayground(store)
  const { editor, analysis, ui } = s
  const root = useRef<HTMLElement>(null)
  const work = useRef<HTMLDivElement>(null)
  const req = useRef<HTMLElement>(null)
  const runbar = useRef<HTMLDivElement>(null)
  const res = useRef<HTMLElement>(null)
  const runBtn = useRef<HTMLButtonElement>(null)
  const revertBtn = useRef<HTMLButtonElement>(null)
  const keysBtn = useRef<HTMLButtonElement>(null)
  const stateEd = useRef<CodeEditorHandle>(null)
  const qEd = useRef<CodeEditorHandle>(null)
  const cursor = useRef(-1)
  const keysPop = usePopover()
  const problemsPop = usePopover()
  const sharePop = usePopover()
  const [shared, setShared] = useState('')

  useEffect(() => store.boot(), [store])

  useEffect(() => {
    window.addEventListener('hashchange', store.onHashChange)
    return () => window.removeEventListener('hashchange', store.onHashChange)
  }, [store])

  // The run bar's height sizes the state section and keeps the toast clear of the bar.
  useLayoutEffect(() => {
    const bar = runbar.current
    const host = root.current
    if (!bar || !host) return
    const observer = new ResizeObserver(() =>
      host.style.setProperty('--runbar-h', `${bar.offsetHeight}px`),
    )
    observer.observe(bar)
    return () => observer.disconnect()
  }, [])

  const jump = useCallback((loc: Loc | null) => {
    if (!loc) return
    if ('text' in loc) {
      ;(loc.text === 'state' ? stateEd : qEd).current?.select(loc.at[0], loc.at[1])
      return
    }
    const el = document.querySelector<HTMLElement>(`[data-loc="${CSS.escape(loc.el)}"]`)
    if (!el) return
    el.focus({ preventScroll: true })
    el.scrollIntoView({ block: 'center', behavior: smoothScroll() })
  }, [])
  const jumpTo = useCallback((problem: Problem) => jump(problem.loc), [jump])

  const run = useCallback(async () => {
    const outcome = await store.run()
    if (outcome === 'blocked') {
      // Say so (the run bar does), and put the caret on the first error.
      const problems = store.getSnapshot().analysis.problems
      cursor.current = problems.findIndex((p) => p.severity === 'error')
      jump(problems[cursor.current]?.loc ?? null)
    } else if (
      outcome === 'done' &&
      (store.getSnapshot().ui.layout !== 'split' || window.innerWidth < WIDE)
    ) {
      // Stacked: bring the answers into view.
      res.current?.scrollIntoView({ block: 'start', behavior: smoothScroll() })
    }
  }, [store, jump])

  const nextProblem = useCallback(
    (back: boolean) => {
      const problems = store.getSnapshot().analysis.problems
      const n = problems.length
      if (!n) {
        store.showToast('No problems in the request.')
        return
      }
      const at = cursor.current
      cursor.current = at < 0 || at >= n ? (back ? n - 1 : 0) : (at + (back ? -1 : 1) + n) % n
      jump(problems[cursor.current]!.loc)
    },
    [store, jump],
  )

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        // A link keeps the browser's "open in a new tab".
        if (target?.closest('a[href]')) return
        event.preventDefault()
        void run()
        return
      }
      if (event.key === 'F8') {
        event.preventDefault()
        nextProblem(event.shiftKey)
        return
      }
      const typing = target?.closest('input, textarea, select, [contenteditable]')
      if (event.key === '?' && !typing && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault()
        keysBtn.current?.click()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [run, nextProblem])

  const share = async (button: HTMLElement) => {
    const url = store.shareLink()
    if (!url) {
      store.showToast('Fix the JSON error before copying a link.')
      return
    }
    if (await copyText(url)) {
      const kb = Math.max(1, Math.round(url.length / 1024))
      store.showToast(`Link copied. It carries the whole request (${kb} KB).`)
      return
    }
    setShared(url)
    sharePop.open(button)
  }

  const cards = s.catalog.state === 'ready' ? s.catalog.cards : null
  const noModels = cards !== null && cards.length === 0
  const layout = ui.layout

  // The request pane's height less the run bar: what the state and questions sections share.
  const editorHeight = () =>
    (req.current?.getBoundingClientRect().height ?? 0) - (runbar.current?.offsetHeight ?? 0) - 1
  const stateShare = () => {
    if (ui.sh !== null) return ui.sh
    const section = req.current?.querySelector<HTMLElement>('.pg-sec--state')
    return (section?.offsetHeight ?? 0) / (editorHeight() || 1)
  }

  return (
    <main ref={root} className="playground" data-layout={layout}>
      <h1 className="sr-only">Decision playground</h1>
      <div className="pg-bar">
        <label className="pg-bar__label" htmlFor="pg-protocol">
          protocol
        </label>
        <select
          id="pg-protocol"
          className="pg-select"
          value={editor.protocol}
          disabled={Boolean(s.pending)}
          title="Each protocol keeps its own draft; switching does not convert requests"
          onChange={(event) =>
            store.setProtocol(event.target.value === 'openai' ? 'openai' : 'typesafe')
          }
        >
          <option value="typesafe">TypeSafe System One</option>
          <option value="openai">OpenAI Decisions</option>
        </select>
        <label className="pg-bar__label" htmlFor="pg-preset">
          request
        </label>
        <select
          id="pg-preset"
          className="pg-select"
          value={s.base?.name ?? 'example'}
          onChange={(event) => {
            // Arrow keys on a closed select change it on every press, so an edited request is
            // replaced only when asked.
            const source = s.sources.find((x) => x.name === event.target.value)
            if (!source) return
            if (
              s.edited &&
              !window.confirm(`Replace your edited request with ${quote(source.label)}?`)
            )
              return
            store.loadSource(source.name, false)
          }}
        >
          {s.sources.map((source) => (
            <option key={source.name} value={source.name}>
              {source.label}
            </option>
          ))}
        </select>
        {s.edited ? <Pill tone="warning">edited</Pill> : null}
        <button
          type="button"
          className="button"
          title="Copy a link that opens this exact request"
          onClick={(event) => void share(event.currentTarget)}
        >
          copy link
        </button>
        {sharePop.anchor ? (
          <Popover anchor={sharePop.anchor} label="Share link" onClose={sharePop.close}>
            <h3>Copy this link</h3>
            <input
              className="pg-input"
              value={shared}
              readOnly
              aria-label="Share link"
              onFocus={(event) => event.currentTarget.select()}
            />
          </Popover>
        ) : null}
        <span className="pg-bar__endpoint">POST {endpointFor(editor.protocol)}</span>
        <button
          type="button"
          className="button pg-layout"
          aria-pressed={layout === 'stack'}
          title={
            layout === 'stack'
              ? 'Show request and response side by side'
              : 'Stack the response under the request'
          }
          onClick={() => store.setUi({ layout: layout === 'split' ? 'stack' : 'split' })}
        >
          stacked
        </button>
      </div>

      <div
        ref={work}
        className="pg-work"
        style={{ ['--split' as string]: `${(ui.split * 100).toFixed(2)}%` }}
      >
        <section
          ref={req}
          className={cx('pg-pane pg-req', ui.sh !== null && 'is-sized')}
          style={ui.sh !== null ? { ['--sh' as string]: ui.sh.toFixed(4) } : undefined}
          aria-label="Request"
        >
          <StateEditor
            store={store}
            editor={editor}
            result={analysis.state}
            note={s.notes.state}
            editorRef={stateEd}
            onJump={jumpTo}
          />
          <Splitter
            orientation="horizontal"
            label="Resize state and questions"
            bounds={() => SH_BOUNDS}
            value={stateShare}
            measure={(event) =>
              (event.clientY - req.current!.getBoundingClientRect().top) / (editorHeight() || 1)
            }
            onPreview={(share) => {
              req.current?.classList.add('is-sized')
              req.current?.style.setProperty('--sh', share.toFixed(4))
            }}
            onCommit={(share) => store.setUi({ sh: share })}
            onReset={() => store.setUi({ sh: null })}
          />
          <QuestionsEditor
            store={store}
            editor={editor}
            result={analysis.questions}
            note={s.notes.questions}
            editorRef={qEd}
            onJump={jumpTo}
          />
          <div ref={runbar} className="pg-runbar">
            <ModelPicker s={s} store={store} />
            {noModels ? (
              <span className="pg-runmsg pg-runmsg--bad">
                This server lists no decision model: start ninfer-serve with a decision adapter in
                --lora-dir.
              </span>
            ) : s.blocked && analysis.errors ? (
              <span className="pg-runmsg pg-runmsg--bad">
                <button
                  type="button"
                  aria-haspopup="dialog"
                  aria-expanded="false"
                  onClick={(event) => problemsPop.toggle(event.currentTarget)}
                >
                  Fix {plural(analysis.errors, 'problem')} to run
                </button>
              </span>
            ) : (
              <span className="pg-runmsg" />
            )}
            {problemsPop.anchor ? (
              <Popover anchor={problemsPop.anchor} label="Problems" onClose={problemsPop.close}>
                <ProblemList
                  problems={analysis.problems}
                  onJump={(problem) => {
                    problemsPop.close()
                    jumpTo(problem)
                  }}
                />
              </Popover>
            ) : null}
            <span className="pg-runbar__right">
              <button
                ref={keysBtn}
                type="button"
                className="pg-icon pg-runbar__keys"
                aria-label="Keyboard shortcuts"
                title="Keyboard shortcuts (?)"
                aria-haspopup="dialog"
                aria-expanded="false"
                onClick={(event) => keysPop.toggle(event.currentTarget)}
              >
                ?
              </button>
              {keysPop.anchor ? (
                <Popover anchor={keysPop.anchor} label="Keyboard shortcuts" onClose={keysPop.close}>
                  <h3>Keyboard shortcuts</h3>
                  <dl className="pg-keys">
                    {SHORTCUTS.map(([keys, what]) => (
                      <Fragment key={keys}>
                        <dt>
                          <kbd>{keys}</kbd>
                        </dt>
                        <dd>{what}</dd>
                      </Fragment>
                    ))}
                  </dl>
                </Popover>
              ) : null}
              <button
                ref={revertBtn}
                type="button"
                className="button"
                disabled={!s.edited}
                title="Go back to the request as it was loaded"
                onClick={() => {
                  // Revert disables itself, so focus moves next door to Run rather than dropping
                  // to the page.
                  if (document.activeElement === revertBtn.current)
                    runBtn.current?.focus({ preventScroll: true })
                  store.revert()
                }}
              >
                revert
              </button>
              <button
                ref={runBtn}
                type="button"
                className="button button--primary pg-run"
                disabled={Boolean(s.pending) || noModels}
                onClick={() => void run()}
              >
                {s.pending ? 'running\u2026' : 'run request'}
                <kbd>
                  {MOD}+{'\u21b5'}
                </kbd>
              </button>
            </span>
          </div>
        </section>

        <Splitter
          orientation="vertical"
          label="Resize request and response"
          bounds={() => splitBounds(work.current)}
          value={() => ui.split}
          measure={(event) => {
            const r = work.current!.getBoundingClientRect()
            return (event.clientX - r.left) / r.width
          }}
          onPreview={(share) =>
            work.current?.style.setProperty('--split', `${(share * 100).toFixed(2)}%`)
          }
          onCommit={(share) => store.setUi({ split: share })}
          onReset={() => store.setUi({ split: 0.5 })}
        />

        <ResponsePane
          s={s}
          store={store}
          engine={engine}
          onGoLive={resumeLive}
          onJump={jump}
          paneRef={res}
        />
      </div>

      <ToastView store={store} toast={s.toast} fallback={runBtn} />
      <LiveRegion announcement={s.announcement} />
    </main>
  )
}
