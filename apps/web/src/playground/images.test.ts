import { expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement, createRef } from 'react'
import {
  fileDataUrl,
  IMAGE_DETAILS,
  imagePayloadError,
  inputImages,
  insertImages,
  setImageDetail,
} from './images'
import { compactJson, parseJson } from './json'
import { StateEditor } from './StateEditor'
import { PlaygroundStore } from './store'
import { analyzeOpenAI } from './openai-validate'
import { editorFromSnap } from './request'
import type { CodeEditorHandle } from './CodeEditor'

const url = 'data:image/png;base64,AAECA/7/'

test('file insertion preserves every original byte, without image decoding or re-encoding', async () => {
  const bytes = Uint8Array.from({ length: 100000 }, (_, i) => i % 256)
  const file = new File([bytes], 'original.png', { type: 'image/png' })
  const encoded = await fileDataUrl(file)
  expect(encoded).toStartWith('data:image/png;base64,')
  expect(Uint8Array.from(atob(encoded.split(',')[1]!), (c) => c.charCodeAt(0))).toEqual(bytes)
  await expect(fileDataUrl(new File(['video'], 'v.mp4', { type: 'video/mp4' }))).rejects.toThrow(
    'image file',
  )
})

test.each([
  'https://example.test/a.png',
  'file-123',
  'file:///a.png',
  'data:video/mp4;base64,AA==',
  'data:image/;base64,AA==',
  'data:image/png;charset=utf-8;base64,AA==',
  'data:image/png,AA==',
  'data:image/png;base64,',
  'data:image/png;base64,A',
  'data:image/png;base64,AAA===',
  'data:image/png;base64,=AAA',
  'data:image/png;base64,AA A',
  'data:image/png;base64,AA-_',
])('image URL validation rejects unsupported sources and malformed payloads: %s', (source) => {
  expect(imagePayloadError(source)).not.toBeNull()
})

test('insertion targets a native message and position without reordering or rewriting existing parts', () => {
  const first = '{"role":"user","content":"first"}'
  const a = '{"type":"input_text","text":"before","text":"last\\u0020value"}'
  const b = '{"type":"input_text","text":"after"}'
  const text = `[${first}, {"role":"assistant","role":"user","content":[${a}, ${b}]}]`
  const inserted = insertImages(text, 1, 1, [url, url], 'original')
  expect(inserted).toStartWith(`[${first}, {"role":"assistant","role":"user","content":[${a}, `)
  expect(inserted).toEndWith(`${b}]}]`)
  expect(
    inputImages(parseJson(inserted)).map((image) => [image.message, image.part, image.detail]),
  ).toEqual([
    [1, 1, 'original'],
    [1, 2, 'original'],
  ])
  const parts = JSON.parse(inserted)[1].content
  expect(parts.map((part: { type: string }) => part.type)).toEqual([
    'input_text',
    'input_image',
    'input_image',
    'input_text',
  ])
})

test.each(['"hello\\u0020world"', '[{"role":"user","content":"hello\\u0020world"}]'])(
  'string input and content become ordered parts without losing the original text: %s',
  (text) => {
    const inserted = insertImages(text, 0, 1, [url], 'missing')
    expect(inserted).toContain('"text":"hello\\u0020world"')
    expect(JSON.parse(inserted)[0].content[1]).toEqual({ type: 'input_image', image_url: url })
    expect(JSON.parse(insertImages(text, 0, 0, [url], 'null'))[0].content[0].detail).toBeNull()
  },
)

test('empty input can receive images and total count includes duplicates across messages', () => {
  expect(inputImages(parseJson(insertImages('[]', 0, 0, [url], 'auto')))).toHaveLength(1)
  const full = insertImages('"s"', 0, 1, Array(128).fill(url), 'auto')
  expect(() => insertImages(full, 0, 0, [url], 'auto')).toThrow('at most 128')
  expect(() => insertImages('{invalid', 0, 0, [url], 'auto')).toThrow()
})

test('detail changes preserve the image bytes, duplicate keys and other native entries', () => {
  const text = `[{"role":"user","content":[{"type":"input_image","image_url":"old","image_url":"${url}","detail":"low","detail":"high"}]}]`
  for (const detail of IMAGE_DETAILS) {
    const changed = setImageDetail(text, 0, 0, detail)
    expect(changed).toContain(`"image_url":"old","image_url":"${url}"`)
    expect(inputImages(parseJson(changed))[0]!.detail).toBe(detail)
    if (detail !== 'missing') expect(changed).toContain('"detail":"low"')
    else expect(changed).not.toContain('"detail"')
  }
  expect(compactJson(parseJson(setImageDetail(text, 0, 0, 'high')))).toBe(text)
  for (const detail of ['null', 'missing', false]) {
    const invalid = `[{"role":"user","content":[{"type":"input_image","image_url":"${url}","detail":${JSON.stringify(detail)}}]}]`
    expect(inputImages(parseJson(invalid))[0]!.detail).toBe('invalid')
  }
})

test('image presentation renders only inline images, native detail options and no TypeSafe image controls', () => {
  const store = new PlaygroundStore({ load: async () => null, save: async () => {} })
  const editor = editorFromSnap({
    protocol: 'openai',
    stateMode: 'json',
    qMode: 'json',
    model: 'm',
    stateText: insertImages('"text"', 0, 1, [url], 'null'),
    qText: '[{"type":"predicate","instructions":"?"}]',
  })
  const render = (protocol: 'openai' | 'typesafe') =>
    renderToStaticMarkup(
      createElement(StateEditor, {
        store,
        editor: { ...editor, protocol },
        result: analyzeOpenAI(editor, null).state,
        editorRef: createRef<CodeEditorHandle>(),
        onJump: () => {},
      }),
    )
  const html = render('openai')
  expect(html).toContain(`src="${url}"`)
  expect(html).toContain('Image insertion position')
  expect(html).toContain('value="null" selected=""')
  expect(html).toContain('omit detail')
  expect(render('typesafe')).not.toContain('Image insertion position')
})
