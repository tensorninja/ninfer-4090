// The state: a list of name/value fields for the common case, or any JSON value.

import { useRef, type CSSProperties, type RefObject } from 'react'

import { cx } from '../components/ui'
import { CodeEditor, plainEnter, type CodeEditorHandle } from './CodeEditor'
import { kindOf } from './json'
import { filledFields, nextId, plural, type EditorState, type StateMode } from './request'
import { EditorSection, GrowArea, marksFor, severities, useFocusAfterRender } from './Section'
import type { Note, PlaygroundStore } from './store'
import { LOC, type Problem, type SectionResult } from './validate'

const MODES: ReadonlyArray<{ value: StateMode; label: string }> = [
  { value: 'fields', label: 'fields' },
  { value: 'json', label: 'json' },
]

export function StateEditor({
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
  const handle = editorRef
  const node = result.node
  const sub =
    editor.stateMode === 'fields'
      ? plural(filledFields(editor.fields).length, 'field')
      : !node
        ? 'invalid JSON'
        : node.t === 'str'
          ? 'text'
          : node.t === 'arr'
            ? 'list'
            : node.t === 'obj'
              ? 'object'
              : kindOf(node)
  const sev = severities(result.problems)
  const parse = result.problems.find((p) => p.parse)?.parse

  const addField = () => {
    const id = nextId()
    focusAfter(LOC.field(id, 'name'))
    store.updateFields((fields) => [...fields, { id, name: '', value: '' }])
  }

  return (
    <EditorSection
      id="pg-state"
      title={editor.protocol === 'openai' ? 'Input' : 'State'}
      sub={sub}
      modes={editor.protocol === 'openai' ? [{ value: 'json', label: 'json' }] : MODES}
      mode={editor.stateMode}
      onMode={(mode) => {
        if (store.setMode('state', mode) && mode === 'json')
          setTimeout(() => handle.current?.focus())
      }}
      onFormat={editor.stateMode === 'json' ? () => handle.current?.format() : undefined}
      problems={result.problems}
      note={note}
      onShowNote={(at) => handle.current?.select(at, at)}
      onJump={onJump}
      className="pg-sec--state"
      style={style}
    >
      <div ref={root} className="pg-state">
        {editor.stateMode === 'json' ? (
          <CodeEditor
            ref={editorRef}
            id="pg-state-json"
            label={editor.protocol === 'openai' ? 'Input as JSON' : 'State as JSON'}
            value={editor.stateText}
            onChange={(text) => store.setStateText(text)}
            placeholder={
              editor.protocol === 'openai'
                ? '"Text to classify" or [{"role":"user","content":"Text"}]'
                : '{"field": "value"} or "plain text"'
            }
            marks={marksFor(result.problems, editor.stateText)}
            error={parse}
            invalid={Boolean(parse)}
          />
        ) : (
          <>
            <div className="pg-kv-list">
              {editor.fields.map((field, j) => {
                const nameLoc = LOC.field(field.id, 'name')
                const set = (patch: { name?: string; value?: string }) =>
                  store.updateFields((fields) =>
                    fields.map((f) => (f.id === field.id ? { ...f, ...patch } : f)),
                  )
                return (
                  <div className="pg-kv" key={field.id}>
                    <input
                      data-loc={nameLoc}
                      className={cx(
                        'pg-input pg-kv__name',
                        sev.get(nameLoc) === 'warning' && 'is-warn',
                      )}
                      value={field.name}
                      placeholder="name"
                      aria-label="Field name"
                      aria-invalid={sev.get(nameLoc) === 'error' || undefined}
                      spellCheck={false}
                      autoComplete="off"
                      onChange={(event) => set({ name: event.target.value })}
                      onKeyDown={(event) => {
                        if (!plainEnter(event)) return
                        event.preventDefault()
                        root.current
                          ?.querySelector<HTMLElement>(
                            `[data-loc="${LOC.field(field.id, 'value')}"]`,
                          )
                          ?.focus()
                      }}
                    />
                    <GrowArea
                      data-loc={LOC.field(field.id, 'value')}
                      className="pg-input pg-kv__value"
                      value={field.value}
                      placeholder="value"
                      aria-label="Field value"
                      onChange={(event) => set({ value: event.target.value })}
                    />
                    <button
                      type="button"
                      className="pg-icon"
                      aria-label="Remove field"
                      title="Remove field"
                      onClick={() => {
                        const rest = editor.fields.filter((f) => f.id !== field.id)
                        const next = rest[Math.min(j, rest.length - 1)]
                        focusAfter(next ? LOC.field(next.id, 'name') : LOC.addField)
                        store.updateFields(() => rest)
                      }}
                    >
                      {'\u2715'}
                    </button>
                  </div>
                )
              })}
            </div>
            <button
              type="button"
              className="button pg-add"
              data-loc={LOC.addField}
              onClick={addField}
            >
              + add field
            </button>
          </>
        )}
      </div>
    </EditorSection>
  )
}
