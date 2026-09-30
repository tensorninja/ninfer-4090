// Share links: `#r=` followed by the request body as base64url of its UTF-8 text, unpadded.
//
// The link carries the exact text that would be sent, so number spellings and repeated keys
// survive the trip. Decoding accepts the "=" padding Python's urlsafe_b64encode writes. The hash is
// never sent to the server, so a link works against any server that serves the playground.

import { lastEntry, tryParse, type JsonNode } from './json'

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
  return `${base}#r=${b64urlEncode(bodyText)}`
}

export interface SharedRequest {
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
  let text: string
  try {
    text = b64urlDecode(decodeURIComponent(m[1]!))
  } catch {
    return null
  }
  const parsed = tryParse(text)
  if (!parsed.ast || parsed.ast.t !== 'obj') return null
  const state = lastEntry(parsed.ast, 'state')?.v
  const questions = lastEntry(parsed.ast, 'questions')?.v
  if (!state || !questions) return null
  const model = lastEntry(parsed.ast, 'model')?.v
  return { state, questions, model: model?.t === 'str' ? model.v : '' }
}
