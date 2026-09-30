// What the engine did for one playground run.
//
// Every run sends its own `x-typesafe-request-id`. The server echoes it, and the decision's
// `decision_start` and `decision_done` or `decision_error` records on /events carry it as
// `x_request_id`, so a run is joined to its records without guessing by time or content. The
// stream is lossy by design (a reconnect drops what it missed) and a request refused while it was
// being prepared is never logged, so the join says which of those happened rather than showing
// nothing.

import type { ConnectionState } from '../lib/engine-client'
import type { DecisionDoneRecord, DecisionErrorRecord, DecisionStartRecord } from '../lib/records'

/** The slices of the engine state a run is joined against. */
export interface EngineView {
  connection: ConnectionState
  source: 'live' | 'file'
  decisions: DecisionDoneRecord[]
  activeDecisions: DecisionStartRecord[]
  decisionErrors: DecisionErrorRecord[]
}

/**
 * A uuid4 as 32 hex digits, the form the server generates. crypto.randomUUID exists only in secure
 * contexts, and the playground is usually opened over plain HTTP on a LAN address.
 */
export function requestId(): string {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)
  bytes[6] = (bytes[6]! & 0x0f) | 0x40
  bytes[8] = (bytes[8]! & 0x3f) | 0x80
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

/** How long a finished run waits for its record before saying it did not arrive. */
export const RECORD_GRACE_MS = 2000

export type Facts =
  | { kind: 'done'; record: DecisionDoneRecord }
  | { kind: 'error'; record: DecisionErrorRecord }
  /** In flight: submitted to the engine (`start`) or not yet. */
  | { kind: 'running'; start?: DecisionStartRecord }
  /** Answered; its record may still be on the way. */
  | { kind: 'waiting'; start?: DecisionStartRecord }
  /** Refused with no record: the server turned it away while preparing it. */
  | { kind: 'rejected' }
  /** Answered, but its record never arrived: the stream dropped it. */
  | { kind: 'missing'; start?: DecisionStartRecord }
  /** The dashboard is not on the live stream, so no record can arrive. */
  | { kind: 'offline'; replay: boolean }
  /** No response at all. */
  | { kind: 'unreached' }

export interface RunRef {
  id: string
  finished: boolean
  /** A 2xx status. */
  ok: boolean
  net: boolean
  /** performance.now() when the response arrived. */
  doneAt: number
  /** The final record, once seen, so it outlives the dashboard's bounded record window. */
  pinned?: DecisionDoneRecord | DecisionErrorRecord
}

const byId = <T extends { request: { x_request_id: string } }>(
  records: readonly T[],
  id: string,
) => {
  for (let j = records.length - 1; j >= 0; j--) {
    if (records[j]!.request.x_request_id === id) return records[j]
  }
  return undefined
}

/** The final record for `id` in the engine state, if it has arrived. */
export function finalRecord(
  view: EngineView,
  id: string,
): DecisionDoneRecord | DecisionErrorRecord | undefined {
  return byId(view.decisions, id) ?? byId(view.decisionErrors, id)
}

export function engineFacts(run: RunRef, view: EngineView, now: number): Facts {
  const record = run.pinned ?? finalRecord(view, run.id)
  if (record)
    return record.event === 'decision_done' ? { kind: 'done', record } : { kind: 'error', record }
  if (run.net) return { kind: 'unreached' }
  const live = view.source === 'live' && view.connection === 'live'
  const start = byId(view.activeDecisions, run.id)
  if (!run.finished)
    return live || start
      ? { kind: 'running', start }
      : { kind: 'offline', replay: view.source === 'file' }
  if (!live) return { kind: 'offline', replay: view.source === 'file' }
  if (now - run.doneAt < RECORD_GRACE_MS) return { kind: 'waiting', start }
  return run.ok || start ? { kind: 'missing', start } : { kind: 'rejected' }
}
