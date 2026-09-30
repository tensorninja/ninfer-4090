import { expect, test } from 'bun:test'

import { compactJson } from './json'
import { b64urlDecode, b64urlEncode, readShareHash, shareUrl } from './share'

const BODY =
  '{"state":{"id":9007199254740993,"note":"caf\u00e9 \u{1F600}"},"questions":{"q":{"type":"noul"},"q":{"type":"noul","instructions":"i"}},"model":"m"}'

test('a link carries the exact body text, big integers and repeated keys included', () => {
  const url = shareUrl('http://h:8080/playground', BODY)
  expect(url).toMatch(/^http:\/\/h:8080\/playground#r=[A-Za-z0-9_-]+$/)
  const shared = readShareHash(url.slice(url.indexOf('#')))!
  expect(compactJson(shared.state)).toBe('{"id":9007199254740993,"note":"caf\u00e9 \u{1F600}"}')
  expect(compactJson(shared.questions)).toBe(
    '{"q":{"type":"noul"},"q":{"type":"noul","instructions":"i"}}',
  )
  expect(shared.model).toBe('m')
})

test('decoding accepts the padding Python writes', () => {
  const encoded = b64urlEncode('{"a":1}')
  expect(encoded).not.toContain('=')
  expect(b64urlDecode(encoded + '=')).toBe('{"a":1}')
  // base64.urlsafe_b64encode(b'{"state":"s","questions":{}}')
  expect(readShareHash('#r=eyJzdGF0ZSI6InMiLCJxdWVzdGlvbnMiOnt9fQ==')).toMatchObject({ model: '' })
})

test('a hash that is not a readable request is told apart from no link at all', () => {
  expect(readShareHash('')).toBeUndefined()
  expect(readShareHash('#section')).toBeUndefined()
  expect(readShareHash('#r=%%%')).toBeNull()
  expect(readShareHash('#r=' + b64urlEncode('{"state": 1}'))).toBeNull()
  expect(readShareHash('#r=' + b64urlEncode('not json'))).toBeNull()
  expect(readShareHash('#r=/w')).toBeNull()
})
