// Share links: `#r=` followed by the request body as base64url of its UTF-8 text, unpadded.
//
// The link carries the exact text that would be sent, so number spellings and repeated keys
// survive the trip. Decoding accepts the "=" padding Python's urlsafe_b64encode writes. The hash is
// never sent to the server, so a link works against any server that serves the playground.

import { lastEntry, tryParse, type JsonNode } from './json'
import type { Protocol } from './request'

export const MAX_SHARE_LENGTH = 32768

export function b64urlEncode(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  for (let j = 0; j < bytes.length; j += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(j, j + 0x8000))
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/** Throws when `encoded` is not base64url of UTF-8 text. */
export function b64urlDecode(encoded: string): string {
  const s = encoded.replace(/=+$/, '').replace(/-/g, '+').replace(/_/g, '/')
  const binary = atob(s + '==='.slice((s.length + 3) % 4))
  return new TextDecoder('utf-8', { fatal: true }).decode(
    Uint8Array.from(binary, (c) => c.charCodeAt(0)),
  )
}

export function shareUrl(base: string, bodyText: string): string {
  const bytes = new TextEncoder().encode(bodyText).length
  if (base.length + 3 + Math.ceil((bytes * 4) / 3) > MAX_SHARE_LENGTH)
    throw new Error(
      'This request is too large for a share link (32 KiB maximum). Export JSON instead; no images were omitted.',
    )
  return `${base}#r=${b64urlEncode(bodyText)}`
}

export interface SharedRequest {
  text: string
  protocol: Protocol
  state: JsonNode
  questions: JsonNode
  model: string
}

/**
 * The request a `#r=` hash carries: undefined when the hash is not a share link, null when it is
 * one that cannot be read.
 */
export function readShareHash(hash: string): SharedRequest | null | undefined {
  const m = /^#r=(.*)$/s.exec(hash)
  if (!m) return undefined
  if (hash.length > MAX_SHARE_LENGTH) return null
  let text: string
  try {
    text = b64urlDecode(decodeURIComponent(m[1]!))
  } catch {
    return null
  }
  return readRequest(text)
}

export function readRequest(text: string): SharedRequest | null {
  const parsed = tryParse(text)
  if (!parsed.ast || parsed.ast.t !== 'obj') return null
  const input = lastEntry(parsed.ast, 'input')?.v
  const state = lastEntry(parsed.ast, 'state')?.v
  const questions = lastEntry(parsed.ast, 'questions')?.v
  if ((!state && !input) || (state && input) || !questions) return null
  const model = lastEntry(parsed.ast, 'model')?.v
  if (model && model.t !== 'str') return null
  if (input && !model) return null
  const allowed = input
    ? ['input', 'questions', 'model', 'safety_identifier']
    : ['state', 'questions', 'model']
  if (parsed.ast.entries.some((entry) => !allowed.includes(entry.k))) return null
  return {
    text,
    protocol: input ? 'openai' : 'typesafe',
    state: (input ?? state)!,
    questions,
    model: model?.t === 'str' ? model.v : '',
  }
}
