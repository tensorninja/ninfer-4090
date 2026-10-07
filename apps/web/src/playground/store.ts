// The playground's state, outside React.
//
// App creates one store for the life of the page and the view reads it with
// useSyncExternalStore, as the dashboard reads EngineClient. Leaving the playground therefore
// neither loses the request being edited nor abandons a decision in flight: a run is never
// aborted, because the server would log an abandoned decision as a 499.
//
// Every edit goes through one path (laya's `changed()`, examples/server.py): the request is
// analysed again, which gives the body to send and every problem, and the draft is saved shortly
// after. Imperative DOM work (focus, selection, clipboard) stays in the components.

import { useSyncExternalStore } from 'react'

import { formatJson, lineCol, tryParse, type JsonNode, type ObjectNode } from './json'
import { fetchCatalog, type Catalog } from './models'
import { PRESETS, presetsFor, type Preset } from './presets'
import {
  editorFromSnap,
  fieldsBlocker,
  fieldsFromAst,
  fieldsNode,
  formBlocker,
  newQuestion,
  questionFromEntry,
  questionsNode,
  quote,
  snapFromNodes,
  snapFromValue,
  snapOf,
  uniqueKey,
  type EditorState,
  type Field,
  type QuestionDraft,
  type QuestionsMode,
  type RequestBody,
  type Protocol,
  type Section,
  type Snap,
  type StateMode,
} from './request'
import { readShareHash, shareUrl, type SharedRequest } from './share'
import { endpointFor } from './snippets'
import { analyze, type Analysis } from './validate'
import { requestId } from './engine'
import type { DecisionDoneRecord, DecisionErrorRecord } from '../lib/records'

export type View = 'answers' | 'json' | 'code'
export type Layout = 'split' | 'stack'

export interface UiPrefs {
  stateMode: StateMode
  qMode: QuestionsMode
  view: View
  layout: Layout
  /** The request pane's share of the width, side by side. */
  split: number
  /** The state section's share of the request pane's height; null fits its content. */
  sh: number | null
}

/** What the editors were loaded from, to tell an edited request and to revert to. */
export interface Base {
  name: string
  label: string
  snap: Snap
  canon: string
}

/** A section's note: why the form cannot show its JSON, or why a mode switch was refused. */
export interface Note {
  text: string
  kind: 'info' | 'warn'
  cause?: 'parse' | 'shape'
  /** A parse error's offset, for "Show me". */
  at?: number
}

export interface Toast {
  id: number
  text: string
  action?: { label: string; run: () => void }
}

export interface PendingRun {
  id: string
  body: RequestBody
  startedAt: number
}

export interface CompletedRun {
  /** The x-typesafe-request-id sent. */
  id: string
  body: RequestBody
  status?: number
  text: string
  data: unknown
  /** A 2xx with an answers object. */
  ok: boolean
  /** A 2xx without one. */
  odd: boolean
  /** The network error, when there was no response. */
  net?: string
  retryAfter?: string
  /** Client round trip in ms. */
  ms: number
  /** performance.now() when the response arrived. */
  doneAt: number
  /** The decision's final record once it was seen. */
  pinned?: DecisionDoneRecord | DecisionErrorRecord
}

export interface PlaygroundSnapshot {
  booted: boolean
  editor: EditorState
  analysis: Analysis
  base: Base | null
  /** The picker's entries: the presets, then a shared request once one was loaded. */
  sources: ReadonlyArray<{ name: string; label: string }>
  edited: boolean
  /**
   * The questions the Answers view previews: the current ones, or the last that parsed while their
   * JSON is being typed, so the preview does not flicker. Loading a request resets it.
   */
  preview: JsonNode | null
  pending: PendingRun | null
  last: CompletedRun | null
  /** Run was pressed while errors remained. Cleared by the next edit. */
  blocked: boolean
  catalog: Catalog
  ui: UiPrefs
  notes: Partial<Record<Section, Note>>
  toast: Toast | null
  /** The polite live region's text; `seq` makes a repeated message speak again. */
  announcement: { text: string; seq: number }
}

const DRAFT_KEY = 'ninfer.playground.draft'
const UI_KEY = 'ninfer.playground.ui'

const storage = {
  get(key: string): unknown {
    try {
      const raw = localStorage.getItem(key)
      return raw === null ? null : JSON.parse(raw)
    } catch {
      return null
    }
  },
  set(key: string, value: unknown) {
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Storage is full or disabled; the playground still works for this page.
    }
  },
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const clamp = (x: unknown, lo: number, hi: number, fallback: number) =>
  typeof x === 'number' && Number.isFinite(x) ? Math.max(lo, Math.min(hi, x)) : fallback

export const SPLIT_BOUNDS = [0.3, 0.7] as const
export const SH_BOUNDS = [0.15, 0.75] as const

function readUi(): UiPrefs {
  const ui = storage.get(UI_KEY)
  const u = isRecord(ui) ? ui : {}
  return {
    stateMode: u.stateMode === 'json' ? 'json' : 'fields',
    qMode: u.qMode === 'form' ? 'form' : 'json',
    view: u.view === 'json' || u.view === 'code' ? u.view : 'answers',
    layout: u.layout === 'stack' ? 'stack' : 'split',
    split: clamp(u.split, SPLIT_BOUNDS[0], SPLIT_BOUNDS[1], 0.5),
    sh: u.sh === null || u.sh === undefined ? null : clamp(u.sh, SH_BOUNDS[0], SH_BOUNDS[1], 0.36),
  }
}

function isSnap(value: unknown): value is Snap {
  return (
    isRecord(value) &&
    (value.protocol === 'typesafe' || value.protocol === 'openai') &&
    typeof value.stateText === 'string' &&
    typeof value.qText === 'string' &&
    (value.stateMode === 'fields' || value.stateMode === 'json') &&
    (value.qMode === 'form' || value.qMode === 'json')
  )
}

const snapModel = (snap: Snap) => (typeof snap.model === 'string' ? snap.model : '')

const canonOf = (snap: Snap) => analyze(editorFromSnap(snap), null).body?.canon ?? ''

interface Draft {
  snap: Snap
  base: Base | null
}

export class PlaygroundStore {
  private state: PlaygroundSnapshot
  private readonly listeners = new Set<() => void>()
  private readonly shared = new Map<string, { label: string; snap: Snap }>()
  private readonly drafts = new Map<Protocol, Draft>()
  private readonly catalogs = new Map<Protocol, Promise<Catalog>>()
  private saveTimer: ReturnType<typeof setTimeout> | undefined
  private toastSeq = 0
  /** The body canon an Undo toast was offered for; an edit past it withdraws the offer. */
  private undoCanon: string | null = null

  constructor() {
    const ui = readUi()
    const editor = editorFromSnap({
      protocol: 'typesafe',
      stateMode: ui.stateMode,
      qMode: ui.qMode,
      stateText: '{}\n',
      qText: '{}\n',
      model: '',
    })
    const analysis = analyze(editor, null)
    this.state = {
      booted: false,
      editor,
      analysis,
      base: null,
      sources: PRESETS.map(({ name, label }) => ({ name, label })),
      edited: false,
      preview: analysis.questions.node,
      pending: null,
      last: null,
      blocked: false,
      catalog: { state: 'loading' },
      ui,
      notes: {},
      toast: null,
      announcement: { text: '', seq: 0 },
    }
  }

  readonly subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  readonly getSnapshot = () => this.state

  private set(patch: Partial<PlaygroundSnapshot>) {
    this.state = { ...this.state, ...patch }
    for (const listener of this.listeners) listener()
  }

  // --- boot --------------------------------------------------------------------------------

  /**
   * Loads the first request, once, when the playground is first shown: a share link, else a known
   * `?preset=`, else the saved draft, else the example. `?model=` applies on top. The query is
   * then removed so a reload keeps later edits.
   */
  boot() {
    if (this.state.booted) return
    this.set({ booted: true })
    const params = new URLSearchParams(window.location.search)
    const shared = readShareHash(window.location.hash)
    if (shared !== undefined) this.clearHash()
    const saved = storage.get(DRAFT_KEY)
    if (isRecord(saved) && saved.v === 2 && isRecord(saved.drafts)) {
      for (const protocol of ['typesafe', 'openai'] as const) {
        const draft = saved.drafts[protocol]
        if (!isRecord(draft) || !isSnap(draft.snap) || draft.snap.protocol !== protocol) continue
        const base = draft.base
        this.drafts.set(protocol, {
          snap: draft.snap,
          base:
            isRecord(base) &&
            typeof base.name === 'string' &&
            typeof base.label === 'string' &&
            isSnap(base.snap) &&
            base.snap.protocol === protocol
              ? { name: base.name, label: base.label, snap: base.snap, canon: canonOf(base.snap) }
              : null,
        })
      }
    }
    const protocol =
      shared?.protocol ?? (isRecord(saved) && saved.protocol === 'openai' ? 'openai' : 'typesafe')
    const fromDraft = this.drafts.has(protocol)
    this.openProtocol(protocol)
    const preset = params.get('preset')
    if (shared) this.loadShared(shared, false)
    else if (preset && this.sourceSnap(preset)) this.loadSource(preset, false)
    if (shared === null) {
      this.showToast(
        `The link does not hold a readable request; showing ${fromDraft ? 'your last draft' : quote(this.state.base?.label ?? 'the example')} instead.`,
      )
    }
    const model = params.get('model')
    if (model !== null) {
      this.setModel(model)
      const base = this.state.base
      if (!fromDraft && base) {
        const snap = { ...base.snap, model }
        this.set({ base: { ...base, snap, canon: canonOf(snap) } })
        this.commit(this.state.editor)
      }
    }
    if (params.has('preset') || params.has('model')) {
      params.delete('preset')
      params.delete('model')
      const query = params.toString()
      window.history.replaceState(
        window.history.state,
        '',
        window.location.pathname + (query ? `?${query}` : '') + window.location.hash,
      )
    }
  }

  /** A share link pasted into the open playground. */
  readonly onHashChange = () => {
    const shared = readShareHash(window.location.hash)
    if (shared === undefined) return
    this.clearHash()
    if (this.state.pending) {
      this.showToast('Wait for the current run, then open the share link again.')
      return
    }
    if (shared) this.loadShared(shared, true)
    else this.showToast('The link does not hold a readable request.')
  }

  private clearHash() {
    window.history.replaceState(
      window.history.state,
      '',
      window.location.pathname + window.location.search,
    )
  }

  private async loadCatalog() {
    const protocol = this.state.editor.protocol
    let loading = this.catalogs.get(protocol)
    if (!loading) {
      loading = fetchCatalog(protocol).then<Catalog, Catalog>(
        (cards) => ({ state: 'ready', cards }),
        (error: unknown) => ({
          state: 'error',
          message: error instanceof Error ? error.message : String(error),
        }),
      )
      this.catalogs.set(protocol, loading)
    }
    const catalog = await loading
    if (this.state.editor.protocol !== protocol) return
    this.set({ catalog })
    if (
      protocol === 'openai' &&
      catalog.state === 'ready' &&
      !this.state.editor.model &&
      catalog.cards[0]
    ) {
      const model = catalog.cards[0].name
      const base = this.state.base
      if (base && !this.state.edited) this.setBase(base.name, base.label, { ...base.snap, model })
      this.setModel(model)
    }
    this.commit(this.state.editor)
  }

  private captureDraft() {
    this.drafts.set(this.state.editor.protocol, {
      snap: snapOf(this.state.editor),
      base: this.state.base,
    })
  }

  private openProtocol(protocol: Protocol) {
    this.set({
      editor: { ...this.state.editor, protocol, model: '' },
      sources: presetsFor(protocol).map(({ name, label }) => ({ name, label })),
      catalog: { state: 'loading' },
      base: null,
      last: null,
      preview: null,
      notes: {},
    })
    const draft = this.drafts.get(protocol)
    if (draft) {
      const base = draft.base
      if (base) {
        if (!presetsFor(protocol).some((p) => p.name === base.name))
          this.rememberShared(base.name, base.label, base.snap)
        this.setBase(base.name, base.label, base.snap)
      }
      this.restore(draft.snap)
    } else {
      this.loadSource('example', false)
    }
    void this.loadCatalog()
  }

  setProtocol(protocol: Protocol) {
    if (protocol === this.state.editor.protocol || this.state.pending) return
    this.captureDraft()
    this.dismissToast()
    this.openProtocol(protocol)
    this.saveDraft()
    this.announce(
      `Switched to ${protocol === 'openai' ? 'OpenAI Decisions' : 'TypeSafe System One'}. Each protocol keeps its own draft.`,
    )
  }

  // --- sources: presets and shared requests ------------------------------------------------

  private presetSnap(preset: Preset): Snap {
    const { stateMode, qMode } = this.state.ui
    const { protocol, model } = this.state.editor
    return snapFromValue(
      { ...preset, model: protocol === 'openai' ? model : '' },
      { stateMode, qMode },
      protocol,
    )
  }

  private sourceSnap(name: string): { label: string; snap: Snap } | undefined {
    const protocol = this.state.editor.protocol
    const preset = presetsFor(protocol).find((p) => p.name === name)
    if (preset) return { label: preset.label, snap: this.presetSnap(preset) }
    return this.shared.get(`${protocol}:${name}`)
  }

  private rememberShared(name: string, label: string, snap: Snap) {
    this.shared.set(`${snap.protocol}:${name}`, { label, snap })
    if (!this.state.sources.some((s) => s.name === name)) {
      this.set({ sources: [...this.state.sources, { name, label }] })
    }
  }

  private loadShared(shared: SharedRequest, withUndo: boolean) {
    if (shared.protocol !== this.state.editor.protocol) this.setProtocol(shared.protocol)
    const { stateMode, qMode } = this.state.ui
    const snap = snapFromNodes(
      shared.state,
      shared.questions,
      shared.model,
      { stateMode, qMode },
      shared.protocol,
    )
    this.rememberShared('shared', 'Shared request', snap)
    this.load('shared', 'Shared request', snap, withUndo)
  }

  /** Loads a picker entry. The picker asks before replacing an edited request. */
  loadSource(name: string, withUndo: boolean) {
    const source = this.sourceSnap(name)
    if (source) this.load(name, source.label, source.snap, withUndo)
  }

  private load(name: string, label: string, snap: Snap, withUndo: boolean) {
    const prev = { editor: this.state.editor, base: this.state.base, last: this.state.last }
    const edited = this.state.edited
    this.setBase(name, label, snap)
    this.restore(snap)
    this.set({ last: null })
    if (withUndo && edited && prev.base) {
      this.showToast(`Loaded ${quote(label)}.`, {
        label: 'Undo',
        run: () => {
          this.set({ base: prev.base, last: prev.last })
          this.commit(prev.editor, { notes: {} })
        },
      })
    }
  }

  private setBase(name: string, label: string, snap: Snap) {
    this.set({ base: { name, label, snap, canon: canonOf(snap) } })
  }

  /** The editors as a snapshot had them; each section opens as JSON when the form cannot show it. */
  private restore(snap: Snap) {
    this.commit(editorFromSnap({ ...snap, model: snapModel(snap) }), { notes: {}, preview: null })
  }

  /** Back to what was loaded, keeping the editors' modes where they can show it. */
  revert() {
    const base = this.state.base
    if (!base) return
    const prev = this.state.editor
    const { stateMode, qMode } = prev
    this.restore({ ...base.snap, stateMode, qMode })
    this.showToast(`Reverted to ${quote(base.label)}.`, {
      label: 'Undo',
      run: () => this.commit(prev, { notes: {} }),
    })
  }

  // --- edits ---------------------------------------------------------------------------------

  /** The one path every edit takes. */
  private commit(editor: EditorState, extra: Partial<PlaygroundSnapshot> = {}) {
    const catalog = extra.catalog ?? this.state.catalog
    const names = catalog.state === 'ready' ? catalog.cards.map((c) => c.name) : null
    const analysis = analyze(editor, names)
    const canon = analysis.body?.canon ?? ''
    const base = this.state.base
    const notes = { ...(extra.notes ?? this.state.notes) }
    if (editor.stateMode === 'json')
      syncNote(notes, 'state', editor.stateText, analysis.state.node, analysis.state.blocker)
    else delete notes.state
    if (editor.qMode === 'json')
      syncNote(
        notes,
        'questions',
        editor.qText,
        analysis.questions.node,
        analysis.questions.blocker,
      )
    else delete notes.questions
    let toast = this.state.toast
    if (toast?.action && this.undoCanon !== null && canon !== this.undoCanon) {
      toast = null
      this.undoCanon = null
    }
    this.set({
      ...extra,
      editor,
      analysis,
      notes,
      toast,
      preview:
        analysis.questions.node ??
        (extra.preview !== undefined ? extra.preview : this.state.preview),
      edited: !base || canon !== base.canon,
      blocked: false,
    })
    clearTimeout(this.saveTimer)
    this.saveTimer = setTimeout(() => this.saveDraft(), 250)
  }

  private edit(patch: Partial<EditorState>) {
    this.commit({ ...this.state.editor, ...patch })
  }

  setStateText(stateText: string) {
    this.edit({ stateText })
  }

  setQText(qText: string) {
    this.edit({ qText })
  }

  setModel(model: string) {
    this.edit({ model })
  }

  updateFields(update: (fields: Field[]) => Field[]) {
    this.edit({ fields: update(this.state.editor.fields) })
  }

  updateQuestions(update: (questions: QuestionDraft[]) => QuestionDraft[]) {
    this.edit({ questions: update(this.state.editor.questions) })
  }

  updateQuestion(id: number, update: (q: QuestionDraft) => QuestionDraft) {
    this.updateQuestions((qs) => qs.map((q) => (q.id === id ? update(q) : q)))
  }

  /** Adds a question after `after`, or at the end; returns it. */
  addQuestion(after?: number, template?: QuestionDraft): QuestionDraft {
    const qs = this.state.editor.questions
    const q = template ?? newQuestion(uniqueKey(qs, 'new_question'))
    const at = after === undefined ? qs.length : qs.findIndex((x) => x.id === after) + 1
    this.updateQuestions((list) => [...list.slice(0, at), q, ...list.slice(at)])
    return q
  }

  /** Removes a question, with an Undo toast; returns the question focus should move to. */
  removeQuestion(id: number): QuestionDraft | undefined {
    const prev = this.state.editor
    const at = prev.questions.findIndex((q) => q.id === id)
    const gone = prev.questions[at]
    if (!gone) return undefined
    this.updateQuestions((qs) => qs.filter((q) => q.id !== id))
    this.showToast(`Removed ${gone.key ? quote(gone.key) : 'the question'}.`, {
      label: 'Undo',
      run: () => this.commit(prev),
    })
    const rest = this.state.editor.questions
    return rest[Math.min(at, rest.length - 1)]
  }

  /**
   * Switches a section's editor. Leaving JSON needs JSON the form can show exactly; otherwise the
   * switch is refused with a note saying why, and every character typed is kept.
   */
  setMode(sec: Section, mode: StateMode | QuestionsMode): boolean {
    const editor = this.state.editor
    if (editor.protocol === 'openai' && mode !== 'json') return false
    const isState = sec === 'state'
    if ((isState ? editor.stateMode : editor.qMode) === mode) return true
    const notes = { ...this.state.notes }
    let next: EditorState
    if (mode === 'json') {
      next = isState
        ? { ...editor, stateMode: 'json', stateText: formatJson(fieldsNode(editor.fields)) + '\n' }
        : { ...editor, qMode: 'json', qText: formatJson(questionsNode(editor.questions)) + '\n' }
    } else {
      const text = isState ? editor.stateText : editor.qText
      const parsed = tryParse(text)
      if (parsed.error) {
        const { line, col } = lineCol(text, parsed.error.at)
        notes[sec] = {
          text: `The JSON has an error at line ${line}, column ${col}: fix it before switching.`,
          kind: 'warn',
          cause: 'parse',
          at: parsed.error.at,
        }
        this.set({ notes })
        return false
      }
      const why = isState ? fieldsBlocker(parsed.ast) : formBlocker(parsed.ast)
      if (why) {
        notes[sec] = { text: why, kind: 'warn', cause: 'shape' }
        this.set({ notes })
        return false
      }
      const ast = parsed.ast as ObjectNode
      next = isState
        ? { ...editor, stateMode: 'fields', fields: fieldsFromAst(ast) }
        : { ...editor, qMode: 'form', questions: ast.entries.map(questionFromEntry) }
    }
    delete notes[sec]
    this.setUi(isState ? { stateMode: mode as StateMode } : { qMode: mode as QuestionsMode })
    this.commit(next, { notes })
    return true
  }

  // --- running -------------------------------------------------------------------------------

  /**
   * Sends the request, unless one is in flight, errors remain ('blocked'), or the server lists no
   * System One model to answer it ('unavailable').
   */
  async run(): Promise<'busy' | 'blocked' | 'unavailable' | 'done'> {
    if (this.state.pending) return 'busy'
    const { analysis, catalog } = this.state
    if (catalog.state === 'ready' && !catalog.cards.length) return 'unavailable'
    if (analysis.errors || !analysis.body) {
      this.set({ blocked: true })
      return 'blocked'
    }
    const body = analysis.body
    const id = requestId()
    const startedAt = performance.now()
    this.set({ pending: { id, body, startedAt }, blocked: false })
    const run: CompletedRun = {
      id,
      body,
      text: '',
      data: null,
      ok: false,
      odd: false,
      ms: 0,
      doneAt: 0,
    }
    try {
      const response = await fetch(endpointFor(body.protocol), {
        method: 'POST',
        body: body.text,
        headers: {
          'content-type': 'application/json',
          accept: 'application/json',
          ...(body.protocol === 'typesafe' ? { 'x-typesafe-request-id': id } : {}),
        },
      })
      if (body.protocol === 'openai') {
        run.id = response.headers.get('x-request-id') ?? id
        this.set({ pending: { id: run.id, body, startedAt } })
      }
      run.status = response.status
      run.retryAfter = response.headers.get('retry-after') ?? undefined
      run.text = await response.text()
      try {
        run.data = JSON.parse(run.text)
      } catch {
        run.data = null
      }
      run.ok =
        response.ok &&
        isRecord(run.data) &&
        (body.protocol === 'openai' ? Array.isArray(run.data.answers) : isRecord(run.data.answers))
      run.odd = response.ok && !run.ok
    } catch (error) {
      run.net = error instanceof Error ? error.message : String(error)
    }
    run.doneAt = performance.now()
    run.ms = run.doneAt - startedAt
    this.set({ pending: null, last: run })
    // A failure in the Answers view is an alert of its own.
    if (run.ok) {
      const count = Object.keys((run.data as { answers: object }).answers).length
      this.announce(`Run finished: ${count} answer${count === 1 ? '' : 's'}.`)
    } else if (this.state.ui.view !== 'answers') {
      this.announce(
        `The run failed: ${run.net !== undefined ? 'network error' : run.odd ? 'unexpected response' : `HTTP ${run.status}`}.`,
      )
    }
    return 'done'
  }

  /** Keeps the decision's final record with the run once seen. */
  pin(id: string, record: DecisionDoneRecord | DecisionErrorRecord) {
    const last = this.state.last
    if (last && last.id === id && !last.pinned) this.set({ last: { ...last, pinned: record } })
  }

  // --- sharing and persistence ---------------------------------------------------------------

  /** A link that carries the request, or null while its JSON does not parse. */
  shareLink(): string | null {
    const body = this.state.analysis.body
    if (!body) return null
    return shareUrl(window.location.origin + window.location.pathname, body.text)
  }

  private saveDraft() {
    this.captureDraft()
    storage.set(DRAFT_KEY, {
      v: 2,
      protocol: this.state.editor.protocol,
      drafts: Object.fromEntries(this.drafts),
    })
  }

  setUi(patch: Partial<UiPrefs>) {
    const ui = { ...this.state.ui, ...patch }
    this.set({ ui })
    storage.set(UI_KEY, ui)
  }

  // --- toasts and announcements --------------------------------------------------------------

  /** A toast with an action stays until it is used, dismissed or replaced, or the request changes. */
  showToast(text: string, action?: Toast['action']) {
    this.undoCanon = action ? (this.state.analysis.body?.canon ?? '') : null
    const wrapped = action && {
      label: action.label,
      run: () => {
        this.dismissToast()
        action.run()
      },
    }
    this.set({ toast: { id: ++this.toastSeq, text, action: wrapped } })
    this.announce(text)
  }

  dismissToast() {
    this.undoCanon = null
    if (this.state.toast) this.set({ toast: null })
  }

  announce(text: string) {
    this.set({ announcement: { text, seq: this.state.announcement.seq + 1 } })
  }
}

/**
 * In JSON mode a section says quietly why the form cannot show its value. A refusal note stays,
 * kept current, while its cause does.
 */
function syncNote(
  notes: Partial<Record<Section, Note>>,
  sec: Section,
  text: string,
  node: JsonNode | null,
  blocker: string,
) {
  const note = notes[sec]
  const warn = note?.kind === 'warn'
  if (warn && note.cause === 'parse' && !node) {
    const parsed = tryParse(text)
    if (parsed.error) {
      const { line, col } = lineCol(text, parsed.error.at)
      notes[sec] = {
        ...note,
        text: `The JSON has an error at line ${line}, column ${col}: fix it before switching.`,
        at: parsed.error.at,
      }
    }
    return
  }
  if (warn && note.cause === 'shape' && node && blocker) {
    if (note.text !== blocker) notes[sec] = { text: blocker, kind: 'warn', cause: 'shape' }
    return
  }
  if (node && blocker) {
    if (!note || note.text !== blocker) notes[sec] = { text: blocker, kind: 'info' }
  } else {
    delete notes[sec]
  }
}

export function usePlayground(store: PlaygroundStore): PlaygroundSnapshot {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
}
