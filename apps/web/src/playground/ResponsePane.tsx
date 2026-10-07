// The response side: the run's status, then its answers, raw body, or the request as code.
// Ported from laya's playground (examples/server.py: status, renderAnswersView, renderError,
// renderJson, renderCode, setView; Apache-2.0).

import { useEffect, useMemo, useRef, useState, type Ref } from 'react'

import { cx, Pill, type Tone } from '../components/ui'
import { count } from '../lib/format'
import { AnswerList, PreviewList } from './AnswerList'
import { answerViews } from './answers'
import { copyText } from './clipboard'
import { MOD, PaintedJson } from './CodeEditor'
import {
  engineFacts,
  finalRecord,
  RECORD_GRACE_MS,
  type EngineView,
  type Facts,
  type RunRef,
} from './engine'
import { EngineFacts } from './EngineFacts'
import { describeFailure, type ErrorView } from './errors'
import { formatJson, tryParse, type JsonNode } from './json'
import { cardFor, type ModelCard } from './models'
import { plural, type EditorState, type RequestBody } from './request'
import { curlSnippet, endpointFor, pythonSnippet } from './snippets'
import type { CompletedRun, PlaygroundSnapshot, PlaygroundStore, View } from './store'
import { locate, type Loc } from './validate'

type Status = 'preview' | 'running' | 'stale' | 'ok' | 'error'

const STATUS_TONE: Record<Status, Tone> = {
  preview: 'neutral',
  running: 'warning',
  stale: 'neutral',
  ok: 'accent',
  error: 'danger',
}

const VIEWS: ReadonlyArray<{ value: View; label: string }> = [
  { value: 'answers', label: 'answers' },
  { value: 'json', label: 'json' },
  { value: 'code', label: 'code' },
]

const NONE: ReadonlySet<string> = new Set()

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

export const fmtMs = (ms: number) =>
  ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`

const questionCount = (node: JsonNode | null) =>
  node?.t === 'arr'
    ? node.items.length
    : node?.t === 'obj'
      ? new Set(node.entries.map((e) => e.k)).size
      : 0

function statusOf(s: PlaygroundSnapshot): { status: Status; hint: string } {
  const last = s.last
  if (s.pending) return { status: 'running', hint: 'Waiting for the server\u2026' }
  if (
    last &&
    (s.analysis.body?.canon !== last.body.canon || s.editor.protocol !== last.body.protocol)
  ) {
    return { status: 'stale', hint: 'The request changed since this run' }
  }
  if (last?.ok) {
    const answers = Object.keys((last.data as { answers: object }).answers).length
    return { status: 'ok', hint: `${plural(answers, 'answer')} in ${fmtMs(last.ms)} round trip` }
  }
  if (last) {
    const why =
      last.net !== undefined
        ? 'Network error'
        : last.odd
          ? 'Unexpected response'
          : `HTTP ${last.status}`
    return { status: 'error', hint: why }
  }
  const n = questionCount(s.preview)
  return { status: 'preview', hint: n ? `Run the request to answer ${plural(n, 'question')}` : '' }
}

function CopyButton({ text }: { text: string }) {
  const [said, setSaid] = useState<string | null>(null)
  useEffect(() => {
    if (said === null) return
    const timer = setTimeout(() => setSaid(null), 1400)
    return () => clearTimeout(timer)
  }, [said])
  return (
    <button
      type="button"
      className="button"
      onClick={async () => setSaid((await copyText(text)) ? 'copied' : 'copy failed')}
    >
      {said ?? 'copy'}
    </button>
  )
}

/** What answered and what it cost, from the response and the model catalog. */
function Strip({ run, cards }: { run: CompletedRun; cards: readonly ModelCard[] }) {
  const data = isRecord(run.data) ? run.data : {}
  const openai = run.body.protocol === 'openai'
  const answered = typeof data.model === 'string' ? data.model : ''
  const card = openai ? undefined : cardFor(cards, run.body.model)
  // The SDK alias, whether named or left out, is shown with the adapter that answered for it.
  const asked = card && card.name !== card.adapter ? card.name : run.body.model
  const model =
    asked && answered && asked !== answered
      ? `${asked} \u2192 ${answered}`
      : answered || asked || '\u2014'
  const usage = isRecord(data.usage) ? data.usage : {}
  const inputDetails = isRecord(usage.input_tokens_details) ? usage.input_tokens_details : {}
  const tokens = (value: unknown) => (typeof value === 'number' ? count(value) : '\u2014')
  const cells: Array<{ label: string; value: string; hint?: string }> = [
    { label: 'model', value: model },
    { label: 'input tokens', value: tokens(usage.input_tokens) },
    { label: 'output tokens', value: tokens(usage.output_tokens) },
    ...(openai
      ? [
          { label: 'total tokens', value: tokens(usage.total_tokens) },
          {
            label: 'cached tokens',
            value: tokens(inputDetails.cached_tokens),
            hint: 'Input state tokens reused from retained or restored state',
          },
          {
            label: 'cache write tokens',
            value: tokens(inputDetails.cache_write_tokens),
            hint: 'No separate cache-write accounting; zero does not mean the cache is disabled',
          },
        ]
      : [
          {
            label: 'latency',
            value:
              typeof data.latency_ms === 'number' ? `${data.latency_ms.toFixed(1)} ms` : '\u2014',
            hint: "latency_ms: the server's time for the decision, from the request to its answers",
          },
        ]),
    {
      label: 'round trip',
      value: fmtMs(run.ms),
      hint: 'Measured by the browser, with the network',
    },
    {
      label: 'request id',
      value: run.id,
      hint: `${openai ? 'x-request-id' : 'x-typesafe-request-id'}, the key the engine logs the decision under`,
    },
  ]
  return (
    <dl className="pg-strip">
      {cells.map((cell) => (
        <div key={cell.label} title={cell.hint}>
          <dt>{cell.label}</dt>
          <dd title={cell.value}>{cell.value}</dd>
        </div>
      ))}
    </dl>
  )
}

function ErrorBox({
  view,
  editor,
  onJump,
}: {
  view: ErrorView
  editor: EditorState | null
  onJump: (loc: Loc) => void
}) {
  return (
    <div className="pg-err" role="alert">
      <h3>{view.title}</h3>
      <ul>
        {view.lines.map((line, j) => {
          const label = line.loc.join(' \u2192 ')
          const where = editor && line.loc.length ? locate(editor, line.loc) : null
          return (
            <li key={j}>
              {line.loc.length ? (
                <>
                  {where ? (
                    <button type="button" className="pg-errloc" onClick={() => onJump(where)}>
                      {label}
                    </button>
                  ) : (
                    label
                  )}
                  :{' '}
                </>
              ) : null}
              {line.msg}
              {line.code || line.type ? (
                <span className="pg-ans__sub">
                  {' '}
                  <code>{[line.type, line.code].filter(Boolean).join(' / ')}</code>
                </span>
              ) : null}
            </li>
          )
        })}
      </ul>
      <p>{view.help}</p>
    </div>
  )
}

function JsonView({ run }: { run: CompletedRun | null }) {
  const text = useMemo(() => {
    if (!run) return ''
    const parsed = tryParse(run.text)
    // Formatted from the tree, so every number keeps the spelling the server wrote.
    return parsed.ast ? formatJson(parsed.ast) : run.text
  }, [run])
  if (!run) return <p className="pg-empty">Run the request to see the raw JSON response.</p>
  if (run.net !== undefined) return <p className="pg-empty">No response: {run.net}</p>
  const size = `${(new TextEncoder().encode(run.text).length / 1024).toFixed(1)} KB`
  return (
    <div className="pg-block pg-block--json">
      <div className="pg-block__head">
        <b>Response body</b>
        <span>
          HTTP {run.status}, {size}, {fmtMs(run.ms)}
        </span>
        <CopyButton text={text} />
      </div>
      <PaintedJson text={text} />
    </div>
  )
}

function CodeBlock({ title, sub, text }: { title: string; sub: string; text: string }) {
  return (
    <div className="pg-block">
      <div className="pg-block__head">
        <b>{title}</b>
        <span>{sub}</span>
        <CopyButton text={text} />
      </div>
      <pre>{text}</pre>
    </div>
  )
}

function CodeView({ body }: { body: RequestBody | null }) {
  if (!body)
    return <p className="pg-empty">Fix the JSON error in the request to get code for it.</p>
  const origin = window.location.origin
  return (
    <div className="pg-code">
      <CodeBlock title="curl" sub="the current request" text={curlSnippet(origin, body)} />
      <CodeBlock
        title="Python"
        sub={
          body.protocol === 'openai'
            ? 'the OpenAI SDK (pip install "openai>=3.26.0")'
            : 'the TypeSafe SDK (pip install typesafe-sdk)'
        }
        text={pythonSnippet(origin, body)}
      />
    </div>
  )
}

export function ResponsePane({
  s,
  store,
  engine,
  onGoLive,
  onJump,
  paneRef,
}: {
  s: PlaygroundSnapshot
  store: PlaygroundStore
  engine: EngineView
  onGoLive: () => void
  onJump: (loc: Loc) => void
  paneRef: Ref<HTMLElement>
}) {
  const { status, hint } = statusOf(s)
  const run = s.last
  const view = s.ui.view
  const cards = s.catalog.state === 'ready' ? s.catalog.cards : []

  // --- engine facts: the current run joined to its records -----------------------------------
  const [now, setNow] = useState(() => performance.now())
  const current: RunRef | null = s.pending
    ? { id: s.pending.id, finished: false, ok: false, net: false, doneAt: 0 }
    : run
      ? {
          id: run.id,
          finished: true,
          ok: run.status !== undefined && run.status >= 200 && run.status < 300,
          net: run.net !== undefined,
          doneAt: run.doneAt,
          pinned: run.pinned,
        }
      : null
  const facts: Facts | null = current && engineFacts(current, engine, now)

  // The final record is kept with the run, so it outlives the dashboard's bounded record window.
  useEffect(() => {
    if (!run || run.pinned) return
    const record = finalRecord(engine, run.id)
    if (record) store.pin(run.id, record)
  }, [engine, run, store])

  // Once the grace for the record has passed, say whether it came.
  const waiting = facts?.kind === 'waiting'
  useEffect(() => {
    if (!waiting || !run) return
    const left = RECORD_GRACE_MS - (performance.now() - run.doneAt)
    const timer = setTimeout(() => setNow(performance.now()), Math.max(0, left) + 50)
    return () => clearTimeout(timer)
  }, [waiting, run])

  // Each run is announced once (the store does that), and so is the first edit after it.
  const spokenStale = useRef('')
  const runId = run?.id ?? ''
  useEffect(() => {
    if (status !== 'stale' || !runId || spokenStale.current === runId) return
    spokenStale.current = runId
    store.announce('The request changed since the last run.')
  }, [status, runId, store])

  // --- answers and their folding -------------------------------------------------------------
  const views = useMemo(
    () =>
      run?.ok
        ? answerViews(
            (run.data as { answers: unknown }).answers,
            run.body.questions,
            run.body.protocol,
          )
        : [],
    [run],
  )
  const failure = useMemo(
    () =>
      run && !run.ok
        ? describeFailure(
            run,
            run.body.questions,
            window.location.origin + endpointFor(run.body.protocol),
            run.body.protocol,
          )
        : null,
    [run],
  )
  const [fold, setFold] = useState<{ run: string; keys: ReadonlySet<string> }>({
    run: '',
    keys: NONE,
  })
  const collapsed = run && fold.run === run.id ? fold.keys : NONE
  const allOpen = views.every((v) => !collapsed.has(v.key))
  const toggle = (key: string, open: boolean) => {
    if (!run) return
    const keys = new Set(fold.run === run.id ? fold.keys : NONE)
    if (open) keys.delete(key)
    else keys.add(key)
    setFold({ run: run.id, keys })
  }

  const tabs = useRef<Array<HTMLButtonElement | null>>([])
  const engineView = facts ? <EngineFacts facts={facts} onGoLive={onGoLive} /> : null

  let answers
  if (!run) {
    answers =
      questionCount(s.preview) > 0 ? (
        <>
          {s.pending ? null : (
            <p className="pg-intro">
              Nothing has run yet. Press{' '}
              <kbd>
                {MOD}+{'\u21b5'}
              </kbd>{' '}
              to answer these questions about the{' '}
              {s.editor.protocol === 'openai' ? 'input' : 'state'}.
            </p>
          )}
          <PreviewList questions={s.preview} protocol={s.editor.protocol} />
          {engineView}
        </>
      ) : (
        <>
          <p className="pg-empty">
            Add a question to the request and its answer will show up here.
          </p>
          {engineView}
        </>
      )
  } else if (failure) {
    // The questions stay listed under the error, so the pane never goes blank.
    answers = (
      <>
        <ErrorBox
          view={failure}
          editor={s.editor.protocol === run.body.protocol ? s.editor : null}
          onJump={onJump}
        />
        {engineView}
        <PreviewList questions={run.body.questions} protocol={run.body.protocol} />
      </>
    )
  } else {
    answers = (
      <>
        <Strip run={run} cards={cards} />
        <AnswerList views={views} collapsed={collapsed} onToggle={toggle} />
        {engineView}
      </>
    )
  }

  return (
    <section ref={paneRef} id="pg-res" className="pg-pane pg-res" aria-label="Response">
      <header className="pg-res__head">
        <h2>Response</h2>
        <Pill tone={STATUS_TONE[status]}>{status}</Pill>
        <span className="pg-res__hint">{hint}</span>
        <div className="pg-tabs" role="tablist" aria-label="Response views">
          {VIEWS.map((v, j) => (
            <button
              key={v.value}
              ref={(el) => {
                tabs.current[j] = el
              }}
              type="button"
              role="tab"
              id={`pg-tab-${v.value}`}
              aria-controls={`pg-view-${v.value}`}
              aria-selected={view === v.value}
              tabIndex={view === v.value ? 0 : -1}
              className="pg-tab"
              onClick={() => store.setUi({ view: v.value })}
              onKeyDown={(event) => {
                const to =
                  event.key === 'ArrowRight'
                    ? (j + 1) % VIEWS.length
                    : event.key === 'ArrowLeft'
                      ? (j + VIEWS.length - 1) % VIEWS.length
                      : event.key === 'Home'
                        ? 0
                        : event.key === 'End'
                          ? VIEWS.length - 1
                          : -1
                if (to < 0) return
                event.preventDefault()
                tabs.current[to]?.focus()
                store.setUi({ view: VIEWS[to]!.value })
              }}
            >
              {v.label}
            </button>
          ))}
        </div>
        {view === 'answers' && views.length ? (
          <button
            type="button"
            className="button pg-res__fold"
            onClick={() =>
              setFold({ run: run!.id, keys: allOpen ? new Set(views.map((v) => v.key)) : NONE })
            }
          >
            {allOpen ? 'collapse all' : 'expand all'}
          </button>
        ) : null}
      </header>
      {s.pending ? <div className="pg-progress" aria-hidden="true" /> : null}
      <div className="pg-res__body" aria-busy={s.pending ? true : undefined}>
        <div
          role="tabpanel"
          id="pg-view-answers"
          aria-labelledby="pg-tab-answers"
          hidden={view !== 'answers'}
          className={cx('pg-view', (status === 'stale' || status === 'running') && 'is-stale')}
        >
          {view === 'answers' ? answers : null}
        </div>
        <div
          role="tabpanel"
          id="pg-view-json"
          aria-labelledby="pg-tab-json"
          hidden={view !== 'json'}
          className="pg-view"
        >
          {view === 'json' ? <JsonView run={run} /> : null}
        </div>
        <div
          role="tabpanel"
          id="pg-view-code"
          aria-labelledby="pg-tab-code"
          hidden={view !== 'code'}
          className="pg-view"
        >
          {view === 'code' ? <CodeView body={s.analysis.body} /> : null}
        </div>
      </div>
    </section>
  )
}
