// The questions: one card per question for the common case, or the JSON object itself.
// Ported from laya's playground (examples/server.py: qCard, renderCrit; Apache-2.0).

import { useRef, type CSSProperties, type RefObject } from 'react'

import { cx } from '../components/ui'
import { CodeEditor, plainEnter, type CodeEditorHandle } from './CodeEditor'
import {
  copyQuestion,
  newLevel,
  newOption,
  newQuestion,
  plural,
  QUESTION_TYPES,
  uniqueKey,
  withType,
  type EditorState,
  type Level,
  type Option,
  type QuestionDraft,
  type QuestionsMode,
} from './request'
import { EditorSection, GrowArea, marksFor, severities, useFocusAfterRender } from './Section'
import type { Note, PlaygroundStore } from './store'
import { LOC, type Problem, type SectionResult, type Severity } from './validate'

const MODES: ReadonlyArray<{ value: QuestionsMode; label: string }> = [
  { value: 'form', label: 'form' },
  { value: 'json', label: 'json' },
]

const TYPE_HINT: Record<string, string> = {
  noul: 'true or false, answered with the probability of true',
  choice: 'one of named options, with the probability of each',
  score: 'a level on an ordered scale, with the probability of each',
}

type Focus = (loc: string, select?: boolean) => void

export function QuestionsEditor({
  store,
  editor,
  result,
  note,
  editorRef,
  onJump,
  style,
}: {
  store: PlaygroundStore
  editor: EditorState
  result: SectionResult
  note?: Note
  editorRef: RefObject<CodeEditorHandle | null>
  onJump: (problem: Problem) => void
  style?: CSSProperties
}) {
  const root = useRef<HTMLDivElement>(null)
  const focusAfter = useFocusAfterRender(root)
  const node = result.node
  const count =
    editor.qMode === 'form'
      ? editor.questions.length
      : node?.t === 'obj'
        ? new Set(node.entries.map((e) => e.k)).size
        : node?.t === 'arr' && editor.protocol === 'openai'
          ? node.items.length
          : null
  const sub =
    count !== null
      ? plural(count, 'question')
      : node
        ? editor.protocol === 'openai'
          ? 'not an array'
          : 'not an object'
        : 'invalid JSON'
  const sev = severities(result.problems)
  const parse = result.problems.find((p) => p.parse)?.parse

  return (
    <EditorSection
      id="pg-questions"
      title="Questions"
      sub={sub}
      modes={editor.protocol === 'openai' ? [{ value: 'json', label: 'json' }] : MODES}
      mode={editor.qMode}
      onMode={(mode) => {
        if (store.setMode('questions', mode) && mode === 'json')
          setTimeout(() => editorRef.current?.focus())
      }}
      onFormat={editor.qMode === 'json' ? () => editorRef.current?.format() : undefined}
      problems={result.problems}
      note={note}
      onShowNote={(at) => editorRef.current?.select(at, at)}
      onJump={onJump}
      className="pg-sec--questions"
      style={style}
    >
      <div ref={root} className="pg-questions">
        {editor.qMode === 'json' ? (
          <CodeEditor
            ref={editorRef}
            id="pg-questions-json"
            label="Questions as JSON"
            value={editor.qText}
            onChange={(text) => store.setQText(text)}
            placeholder={
              editor.protocol === 'openai'
                ? '[{"type":"predicate","instructions":"..."}]'
                : '{"key": {"type": "noul", "instructions": "..."}}'
            }
            marks={marksFor(result.problems, editor.qText)}
            error={parse}
            invalid={Boolean(parse)}
          />
        ) : (
          <>
            {editor.questions.map((q, at) => (
              <QuestionCard
                key={q.id}
                q={q}
                at={at}
                count={editor.questions.length}
                store={store}
                sev={sev}
                focusAfter={focusAfter}
              />
            ))}
            <button
              type="button"
              className="button pg-add"
              data-loc={LOC.addQuestion}
              onClick={() => {
                const draft = newQuestion(uniqueKey(editor.questions, 'new_question'))
                focusAfter(LOC.question(draft.id, 'key'), true)
                store.addQuestion(undefined, draft)
              }}
            >
              + add question
            </button>
          </>
        )}
      </div>
    </EditorSection>
  )
}

function QuestionCard({
  q,
  at,
  count,
  store,
  sev,
  focusAfter,
}: {
  q: QuestionDraft
  at: number
  count: number
  store: PlaygroundStore
  sev: Map<string, Severity>
  focusAfter: Focus
}) {
  const update = (fn: (q: QuestionDraft) => QuestionDraft) => store.updateQuestion(q.id, fn)
  const input = (loc: string) => ({
    'data-loc': loc,
    'aria-invalid': sev.get(loc) === 'error' || undefined,
  })
  const warn = (loc: string) => sev.get(loc) === 'warning' && 'is-warn'
  const keyLoc = LOC.question(q.id, 'key')
  const insLoc = LOC.question(q.id, 'ins')

  const move = (delta: -1 | 1) => {
    // Focus stays on the button pressed, or the key once the card cannot move further.
    const edge = at + delta === 0 || at + delta === count - 1
    focusAfter(LOC.question(q.id, edge ? 'key' : delta < 0 ? 'up' : 'down'))
    store.updateQuestions((qs) => {
      const list = qs.slice()
      const [item] = list.splice(at, 1)
      list.splice(at + delta, 0, item!)
      return list
    })
  }

  return (
    <article className="pg-card" aria-label={`Question ${q.key || at + 1}`}>
      <div className="pg-card__head">
        <input
          {...input(keyLoc)}
          className={cx('pg-input pg-card__key', warn(keyLoc))}
          value={q.key}
          placeholder="key"
          aria-label="Question key"
          spellCheck={false}
          autoComplete="off"
          onChange={(event) => update((x) => ({ ...x, key: event.target.value }))}
          onKeyDown={(event) => {
            if (!plainEnter(event)) return
            event.preventDefault()
            document.querySelector<HTMLElement>(`[data-loc="${insLoc}"]`)?.focus()
          }}
        />
        <span className="pg-seg" role="group" aria-label="Answer type">
          {QUESTION_TYPES.map((type) => (
            <button
              key={type}
              type="button"
              aria-pressed={q.type === type}
              title={TYPE_HINT[type]}
              data-loc={q.type === type ? LOC.question(q.id, 'type') : undefined}
              onClick={() => update((x) => withType(x, type))}
            >
              {type}
            </button>
          ))}
        </span>
        <span className="pg-card__tools">
          <button
            type="button"
            className="pg-icon"
            aria-label="Move up"
            title="Move up"
            data-loc={LOC.question(q.id, 'up')}
            disabled={at === 0}
            onClick={() => move(-1)}
          >
            {'\u2191'}
          </button>
          <button
            type="button"
            className="pg-icon"
            aria-label="Move down"
            title="Move down"
            data-loc={LOC.question(q.id, 'down')}
            disabled={at === count - 1}
            onClick={() => move(1)}
          >
            {'\u2193'}
          </button>
          <button
            type="button"
            className="pg-icon"
            aria-label="Duplicate question"
            title="Duplicate question"
            onClick={() => {
              const key = uniqueKey(
                store.getSnapshot().editor.questions,
                `${q.key || 'question'}_copy`,
              )
              const copy = copyQuestion(q, key)
              focusAfter(LOC.question(copy.id, 'key'), true)
              store.addQuestion(q.id, copy)
            }}
          >
            {'\u29c9'}
          </button>
          <button
            type="button"
            className="pg-icon"
            aria-label="Remove question"
            title="Remove question"
            onClick={() => {
              const rest = store.getSnapshot().editor.questions.filter((x) => x.id !== q.id)
              const next = rest[Math.min(at, rest.length - 1)]
              focusAfter(next ? LOC.question(next.id, 'key') : LOC.addQuestion)
              store.removeQuestion(q.id)
            }}
          >
            {'\u2715'}
          </button>
        </span>
      </div>
      <GrowArea
        {...input(insLoc)}
        className={cx('pg-input pg-card__ins', warn(insLoc))}
        value={q.instructions}
        placeholder="Instructions: what should be decided about the state?"
        aria-label="Instructions"
        onChange={(event) => update((x) => ({ ...x, instructions: event.target.value }))}
      />
      <Criteria q={q} update={update} input={input} warn={warn} focusAfter={focusAfter} />
      {q.extra.length ? (
        <p className="pg-card__note">
          Also sends{' '}
          {[...new Set(q.extra.map((e) => e.k))].map((k, j) => (
            <span key={k}>
              {j ? ', ' : ''}
              <code>{k}</code>
            </span>
          ))}{' '}
          (edit in JSON; System One does not read it).
        </p>
      ) : null}
    </article>
  )
}

function Criteria({
  q,
  update,
  input,
  warn,
  focusAfter,
}: {
  q: QuestionDraft
  update: (fn: (q: QuestionDraft) => QuestionDraft) => void
  input: (loc: string) => { 'data-loc': string; 'aria-invalid': true | undefined }
  warn: (loc: string) => string | false
  focusAfter: Focus
}) {
  const addLoc = LOC.question(q.id, 'add')

  if (q.type === 'noul') {
    if (!q.tf) {
      return (
        <div className="pg-crit">
          <p className="pg-crit__hint">A yes/no question, answered with the probability of true.</p>
          <button
            type="button"
            className="button pg-add"
            data-loc={addLoc}
            onClick={() => {
              const tf = [newOption('true'), newOption('false')]
              focusAfter(LOC.option(q.id, tf[0]!.id))
              update((x) => ({ ...x, tf }))
            }}
          >
            + describe true and false
          </button>
        </div>
      )
    }
    return (
      <div className="pg-crit">
        <div className="pg-crit__head">
          <span>What true and false mean</span>
          <span>optional</span>
        </div>
        {q.tf.map((o) => (
          <div className="pg-opt pg-opt--tf" key={o.id}>
            <span className="pg-opt__tf">{o.label}</span>
            <GrowArea
              {...input(LOC.option(q.id, o.id))}
              className="pg-input"
              value={o.desc}
              aria-label={`What ${o.label} means`}
              placeholder={
                o.label === 'true' ? 'yes, the statement holds' : 'no, the statement does not hold'
              }
              onChange={(event) =>
                update((x) => ({
                  ...x,
                  tf:
                    x.tf &&
                    x.tf.map((t) => (t.id === o.id ? { ...t, desc: event.target.value } : t)),
                }))
              }
            />
          </div>
        ))}
        <button
          type="button"
          className="button pg-add"
          data-loc={addLoc}
          onClick={() => {
            focusAfter(addLoc)
            update((x) => ({ ...x, tf: null }))
          }}
        >
          {'\u2715'} remove descriptions
        </button>
      </div>
    )
  }

  if (q.type === 'choice') {
    const setOptions = (fn: (options: Option[]) => Option[]) =>
      update((x) => ({ ...x, options: fn(x.options) }))
    const insertAfter = (j: number) => {
      const option = newOption()
      focusAfter(LOC.option(q.id, option.id))
      setOptions((options) => [...options.slice(0, j + 1), option, ...options.slice(j + 1)])
    }
    return (
      <div className="pg-crit">
        <div className="pg-crit__head">
          <span>Options</span>
          <span>label {'\u2192'} description</span>
        </div>
        {q.options.map((o, j) => {
          const loc = LOC.option(q.id, o.id)
          const descLoc = `${loc}:desc`
          const patch = (p: Partial<Option>) =>
            setOptions((options) => options.map((x) => (x.id === o.id ? { ...x, ...p } : x)))
          return (
            <div className="pg-opt" key={o.id}>
              <input
                {...input(loc)}
                className={cx('pg-input', warn(loc))}
                value={o.label}
                placeholder="label"
                aria-label="Option label"
                spellCheck={false}
                autoComplete="off"
                onChange={(event) => patch({ label: event.target.value })}
                onKeyDown={(event) => {
                  if (!plainEnter(event)) return
                  event.preventDefault()
                  document.querySelector<HTMLElement>(`[data-loc="${descLoc}"]`)?.focus()
                }}
              />
              <GrowArea
                data-loc={descLoc}
                className="pg-input"
                value={o.desc}
                placeholder="description (optional)"
                aria-label="Option description"
                onChange={(event) => patch({ desc: event.target.value })}
                onKeyDown={(event) => {
                  if (!plainEnter(event)) return
                  event.preventDefault()
                  insertAfter(j)
                }}
              />
              <button
                type="button"
                className="pg-icon"
                aria-label="Remove option"
                title="Remove option"
                onClick={() => {
                  const rest = q.options.filter((x) => x.id !== o.id)
                  const next = rest[Math.max(0, j - 1)]
                  focusAfter(next ? LOC.option(q.id, next.id) : addLoc)
                  setOptions(() => rest)
                }}
              >
                {'\u2715'}
              </button>
            </div>
          )
        })}
        <button
          type="button"
          className="button pg-add"
          data-loc={addLoc}
          onClick={() => insertAfter(q.options.length - 1)}
        >
          + add option
        </button>
      </div>
    )
  }

  const setLevels = (fn: (levels: Level[]) => Level[]) =>
    update((x) => ({ ...x, levels: fn(x.levels) }))
  const insertAfter = (j: number) => {
    const level = newLevel()
    focusAfter(LOC.level(q.id, level.id))
    setLevels((levels) => [...levels.slice(0, j + 1), level, ...levels.slice(j + 1)])
  }
  return (
    <div className="pg-crit">
      <div className="pg-crit__head">
        <span>Levels, lowest first</span>
        <span>index {'\u2192'} level</span>
      </div>
      {q.levels.map((l, j) => {
        const loc = LOC.level(q.id, l.id)
        const swap = (d: -1 | 1) => {
          focusAfter(loc)
          setLevels((levels) => {
            const list = levels.slice()
            ;[list[j], list[j + d]] = [list[j + d]!, list[j]!]
            return list
          })
        }
        return (
          <div className="pg-lvl" key={l.id}>
            <span className="pg-lvl__i">{j}</span>
            <GrowArea
              {...input(loc)}
              className={cx('pg-input', warn(loc))}
              value={l.text}
              aria-label={`Level ${j}`}
              placeholder={j === 0 ? 'lowest level' : 'next level'}
              onChange={(event) =>
                setLevels((levels) =>
                  levels.map((x) => (x.id === l.id ? { ...x, text: event.target.value } : x)),
                )
              }
              onKeyDown={(event) => {
                if (!plainEnter(event)) return
                event.preventDefault()
                insertAfter(j)
              }}
            />
            <span className="pg-lvl__tools">
              <button
                type="button"
                className="pg-icon"
                aria-label="Move level up"
                title="Move level up"
                disabled={j === 0}
                onClick={() => swap(-1)}
              >
                {'\u2191'}
              </button>
              <button
                type="button"
                className="pg-icon"
                aria-label="Move level down"
                title="Move level down"
                disabled={j === q.levels.length - 1}
                onClick={() => swap(1)}
              >
                {'\u2193'}
              </button>
              <button
                type="button"
                className="pg-icon"
                aria-label="Remove level"
                title="Remove level"
                onClick={() => {
                  const rest = q.levels.filter((x) => x.id !== l.id)
                  const next = rest[Math.max(0, j - 1)]
                  focusAfter(next ? LOC.level(q.id, next.id) : addLoc)
                  setLevels(() => rest)
                }}
              >
                {'\u2715'}
              </button>
            </span>
          </div>
        )
      })}
      <button
        type="button"
        className="button pg-add"
        data-loc={addLoc}
        onClick={() => insertAfter(q.levels.length - 1)}
      >
        + add level
      </button>
    </div>
  )
}
