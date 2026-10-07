import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test'

import { compactJson } from './json'
import { readShareHash, shareUrl } from './share'
import { PlaygroundStore } from './store'
import type { DraftStorage } from './drafts'
import { curlSnippet, pythonSnippet } from './snippets'

const typeSafeModels = {
  models: [
    { name: 'jev-latest', adapter: 'decision-v7' },
    { name: 'decision-v7', adapter: 'decision-v7' },
  ],
}
const openAIModels = {
  data: [
    { id: 'chat', supported_endpoints: ['/v1/chat/completions'] },
    {
      id: 'decision-v7',
      supported_endpoints: ['/v1/decisions'],
      context_window: 73728,
      max_state_tokens: 65536,
      modalities: { text: true, vision: true },
    },
  ],
}

const saved = new Map<string, string>()
let draft: unknown
const persistence: DraftStorage = {
  load: async () => structuredClone(draft),
  save: async (value) => {
    draft = structuredClone(value)
  },
}
const timers = new Map<number, () => unknown>()
let nextTimer = 0
let location: URL
const windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')
const storageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')

async function flushDrafts() {
  const callbacks = [...timers.values()]
  timers.clear()
  await Promise.all(callbacks.map((callback) => callback()))
}

function catalogReady(store: PlaygroundStore): Promise<void> {
  if (store.getSnapshot().catalog.state !== 'loading') return Promise.resolve()
  return new Promise((resolve) => {
    const unsubscribe = store.subscribe(() => {
      if (store.getSnapshot().catalog.state !== 'loading') {
        unsubscribe()
        resolve()
      }
    })
  })
}

function mockCatalogs(post?: (url: string, init: RequestInit) => Response | Promise<Response>) {
  return spyOn(globalThis, 'fetch').mockImplementation((async (input, init) => {
    const url = String(input)
    if (init?.method === 'POST' && post) return post(url, init)
    if (url === '/v1/models') return Response.json(openAIModels)
    if (url === '/typesafe/v1/models') return Response.json(typeSafeModels)
    throw new Error(`Unexpected request: ${url}`)
  }) as typeof fetch)
}

beforeEach(() => {
  saved.clear()
  draft = undefined
  timers.clear()
  location = new URL('http://localhost:8080/playground')
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      get location() {
        return location
      },
      history: {
        state: null,
        replaceState: (_state: unknown, _title: string, url: string) => {
          location = new URL(url, location)
        },
      },
    },
  })
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => saved.get(key) ?? null,
      setItem: (key: string, value: string) => saved.set(key, value),
    },
  })
  spyOn(globalThis, 'setTimeout').mockImplementation(((callback: () => void) => {
    timers.set(++nextTimer, callback)
    return nextTimer
  }) as typeof setTimeout)
  spyOn(globalThis, 'clearTimeout').mockImplementation(((id: number) =>
    timers.delete(id)) as typeof clearTimeout)
})

afterEach(() => {
  mock.restore()
  if (windowDescriptor) Object.defineProperty(globalThis, 'window', windowDescriptor)
  else Reflect.deleteProperty(globalThis, 'window')
  if (storageDescriptor) Object.defineProperty(globalThis, 'localStorage', storageDescriptor)
  else Reflect.deleteProperty(globalThis, 'localStorage')
})

test('protocols discover their own models and keep independent invalid drafts across reloads', async () => {
  const fetchMock = mockCatalogs()
  const store = new PlaygroundStore(persistence)
  await store.boot()
  await catalogReady(store)
  store.setModel('jev-latest')
  store.setQText('{ invalid TypeSafe')
  store.setProtocol('openai')
  await catalogReady(store)
  expect(store.getSnapshot().editor.model).toBe('decision-v7')
  expect(store.getSnapshot().analysis.errors).toBe(0)
  expect(store.getSnapshot().catalog).toMatchObject({
    state: 'ready',
    cards: [{ name: 'decision-v7', max_context: 73728, max_state_tokens: 65536 }],
  })
  store.setQText('[ invalid OpenAI')
  store.setProtocol('typesafe')
  await catalogReady(store)
  expect(store.getSnapshot().editor.qText).toBe('{ invalid TypeSafe')
  expect(store.getSnapshot().editor.model).toBe('jev-latest')
  store.setProtocol('openai')
  await catalogReady(store)
  expect(store.getSnapshot().editor.qText).toBe('[ invalid OpenAI')
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/typesafe/v1/models', '/v1/models'])
  await flushDrafts()
  const reloaded = new PlaygroundStore(persistence)
  await reloaded.boot()
  await catalogReady(reloaded)
  expect(reloaded.getSnapshot().editor.protocol).toBe('openai')
  expect(reloaded.getSnapshot().editor.qText).toBe('[ invalid OpenAI')
  reloaded.setProtocol('typesafe')
  await catalogReady(reloaded)
  expect(reloaded.getSnapshot().editor.qText).toBe('{ invalid TypeSafe')
})

test('submission follows the native endpoint, answer envelope, and request ID contract', async () => {
  const sent: Array<{ url: string; init: RequestInit }> = []
  mockCatalogs((url, init) => {
    sent.push({ url, init })
    return url === '/v1/decisions'
      ? Response.json(
          {
            model: 'decision-v7',
            answers: [{ type: 'predicate', name: null, probability: 0.8 }],
            usage: { input_tokens: 10, output_tokens: 0, total_tokens: 10 },
          },
          { headers: { 'x-request-id': 'req_openai' } },
        )
      : Response.json({ answers: { q: { noul: 0.8 } } })
  })
  const store = new PlaygroundStore(persistence)
  await store.boot()
  await catalogReady(store)
  expect(await store.run()).toBe('done')
  expect(sent[0]!.url).toBe('/typesafe/v1/systemone')
  expect(new Headers(sent[0]!.init.headers).get('x-typesafe-request-id')).toBe(
    store.getSnapshot().last!.id,
  )
  store.setProtocol('openai')
  await catalogReady(store)
  store.setStateText('"context"')
  store.setQText('[{"type":"predicate","instructions":"Relevant?"}]')
  expect(await store.run()).toBe('done')
  expect(sent[1]!.url).toBe('/v1/decisions')
  expect(new Headers(sent[1]!.init.headers).has('x-typesafe-request-id')).toBe(false)
  expect(JSON.parse(String(sent[1]!.init.body))).toEqual({
    input: 'context',
    questions: [{ type: 'predicate', instructions: 'Relevant?' }],
    model: 'decision-v7',
  })
  expect(store.getSnapshot().last).toMatchObject({
    id: 'req_openai',
    ok: true,
    odd: false,
    body: { protocol: 'openai' },
  })
  const shared = readShareHash(new URL(store.shareLink()!).hash)!
  expect(shared.protocol).toBe('openai')
  expect(compactJson(shared.state)).toBe('"context"')
})

test('an OpenAI share link wins over saved protocol and preserves duplicate typed choices', async () => {
  mockCatalogs()
  const store = new PlaygroundStore(persistence)
  await store.boot()
  await catalogReady(store)
  store.setQText('{ unfinished')
  await flushDrafts()
  const body =
    '{"input":"context","questions":[{"name":"q","type":"choice","instructions":"Pick","choices":[{"value":true},{"value":"true"},{"value":true}]}],"model":"decision-v7"}'
  location = new URL(shareUrl(location.href, body))
  const opened = new PlaygroundStore(persistence)
  await opened.boot()
  await catalogReady(opened)
  expect(location.hash).toBe('')
  expect(opened.getSnapshot().analysis.body!.text).toBe(body)
  opened.setProtocol('typesafe')
  await catalogReady(opened)
  expect(opened.getSnapshot().editor.qText).toBe('{ unfinished')
})

test('switching is blocked in flight and a foreign answer envelope is not rendered as success', async () => {
  let finish!: (response: Response) => void
  mockCatalogs(
    () =>
      new Promise<Response>((resolve) => {
        finish = resolve
      }),
  )
  const store = new PlaygroundStore(persistence)
  await store.boot()
  await catalogReady(store)
  store.setProtocol('openai')
  await catalogReady(store)
  const run = store.run()
  store.setProtocol('typesafe')
  expect(store.getSnapshot().editor.protocol).toBe('openai')
  expect(await store.run()).toBe('busy')
  finish(Response.json({ answers: { q: { noul: 0.8 } } }))
  await run
  expect(store.getSnapshot().last).toMatchObject({ ok: false, odd: true })
})

test('a share link rejected during a run is cleared and can be opened again afterward', async () => {
  let finish!: (response: Response) => void
  mockCatalogs(
    () =>
      new Promise<Response>((resolve) => {
        finish = resolve
      }),
  )
  const store = new PlaygroundStore(persistence)
  await store.boot()
  await catalogReady(store)
  const original = store.getSnapshot().analysis.body!.text
  const run = store.run()
  const body =
    '{"input":"shared context","questions":[{"type":"predicate","instructions":"Relevant?"}],"model":"decision-v7"}'
  const link = shareUrl(location.href, body)
  location = new URL(link)
  store.onHashChange()
  expect(location.hash).toBe('')
  expect(store.getSnapshot().analysis.body!.text).toBe(original)
  expect(store.getSnapshot().toast?.text).toContain('open the share link again')
  finish(Response.json({ answers: { q: { noul: 0.8 } } }))
  await run
  expect(store.getSnapshot().analysis.body!.text).toBe(original)
  location = new URL(link)
  store.onHashChange()
  await catalogReady(store)
  expect(location.hash).toBe('')
  expect(store.getSnapshot().analysis.body!.text).toBe(body)
})

test('a share opened during IndexedDB hydration is applied after the saved drafts load', async () => {
  mockCatalogs()
  let finish!: (value: unknown) => void
  const store = new PlaygroundStore({
    load: () =>
      new Promise((resolve) => {
        finish = resolve
      }),
    save: async () => {},
  })
  const boot = store.boot()
  const body =
    '{"input":"shared during load","questions":[{"type":"predicate","instructions":"?"}],"model":"decision-v7"}'
  location = new URL(shareUrl(location.href, body))
  store.onHashChange()
  finish(null)
  await boot
  await catalogReady(store)
  expect(store.exportRequest()).toBe(body)
  expect(location.hash).toBe('')
})

test('image requests round-trip exactly through import, preview, export, drafts and submission', async () => {
  let sent = ''
  mockCatalogs((_url, init) => {
    sent = String(init.body)
    return Response.json({ answers: [] })
  })
  const store = new PlaygroundStore(persistence)
  await store.boot()
  await catalogReady(store)
  store.setQText('{ unfinished TypeSafe')
  const url = 'data:image/png;base64,' + 'AAAA'.repeat(400000)
  const text = `{\n "model":"unused","model":"decision-v7", "questions":[{"type":"choice","instructions":"Pick","choices":[{"value":true},{"value":"true"},{"value":true}]}],\n "input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"type":"input_image","image_url":"${url}","detail":null},{"type":"input_text","text":"after"}]}],"safety_identifier":"local"\n}\n`
  store.importRequest(text)
  await catalogReady(store)
  expect(store.getSnapshot().analysis.errors).toBe(0)
  expect(store.exportRequest()).toBe(text)
  expect(curlSnippet('http://h', store.getSnapshot().analysis.body!)).toContain(text)
  expect(pythonSnippet('http://h', store.getSnapshot().analysis.body!)).toContain(url)
  expect(() => store.shareLink()).toThrow('too large')
  expect(await store.run()).toBe('done')
  expect(sent).toBe(text)
  await flushDrafts()
  expect(saved.has('ninfer.playground.draft')).toBe(false)
  const restored = new PlaygroundStore(persistence)
  await restored.boot()
  await catalogReady(restored)
  expect(restored.exportRequest()).toBe(text)
  restored.setProtocol('typesafe')
  await catalogReady(restored)
  expect(restored.getSnapshot().editor.qText).toBe('{ unfinished TypeSafe')
})

test('draft persistence failures are visible without preventing edits, with preferences still local', async () => {
  mockCatalogs()
  const failing: DraftStorage = {
    load: async () => null,
    save: async () => {
      throw new Error('quota exhausted')
    },
  }
  const store = new PlaygroundStore(failing)
  await store.boot()
  await catalogReady(store)
  expect(store.getSnapshot().persistenceError).toBeNull()
  store.setQText('{ keep invalid')
  await flushDrafts()
  expect(store.getSnapshot().editor.qText).toBe('{ keep invalid')
  expect(store.getSnapshot().persistenceError).toContain('quota exhausted')
  store.setUi({ view: 'code' })
  expect(saved.get('ninfer.playground.ui')).toContain('"view":"code"')
})

test('a failed draft read never overwrites unread persistent requests with the default example', async () => {
  mockCatalogs()
  let writes = 0
  const store = new PlaygroundStore({
    load: async () => {
      throw new Error('database unavailable')
    },
    save: async () => {
      writes++
    },
  })
  await store.boot()
  await catalogReady(store)
  store.setMode('state', 'json')
  store.setStateText('"keep this edit"')
  await flushDrafts()
  expect(writes).toBe(0)
  expect(store.getSnapshot().persistenceError).toContain('Could not load drafts')
  expect(store.exportRequest()).toContain('keep this edit')
})

test('known vision-disabled models block controls and submission; discovery failure keeps manual input usable', async () => {
  const mockFetch = mockCatalogs()
  mockFetch.mockImplementation((async (input) =>
    String(input) === '/v1/models'
      ? Response.json({
          data: [
            {
              id: 'text-model',
              supported_endpoints: ['/v1/decisions'],
              modalities: { vision: false },
            },
          ],
        })
      : Response.json(typeSafeModels)) as typeof fetch)
  const store = new PlaygroundStore(persistence)
  await store.boot()
  store.setProtocol('openai')
  await catalogReady(store)
  expect(store.imageControlsEnabled()).toBe(false)
  store.setStateText(
    '[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]',
  )
  expect(await store.run()).toBe('blocked')
  mockFetch.mockImplementation((async () => {
    throw new Error('offline')
  }) as unknown as typeof fetch)
  const manual = new PlaygroundStore(persistence)
  await manual.boot()
  manual.setProtocol('openai')
  await catalogReady(manual)
  manual.setModel('manual-model')
  manual.setStateText('"text"')
  expect(manual.imageControlsEnabled()).toBe(true)
  await manual.addImageFiles(
    [new File([new Uint8Array([0, 255])], 'a.png', { type: 'image/png' })],
    0,
    1,
    'high',
  )
  expect(manual.getSnapshot().analysis.errors).toBe(0)
  expect(manual.exportRequest()).toContain('data:image/png;base64,AP8=')
})

test('pending file reads cannot overwrite edits or switch protocols, and import is blocked in flight', async () => {
  let finish!: (response: Response) => void
  mockCatalogs(
    () =>
      new Promise<Response>((resolve) => {
        finish = resolve
      }),
  )
  const store = new PlaygroundStore(persistence)
  await store.boot()
  store.setProtocol('openai')
  await catalogReady(store)
  store.setStateText('"before"')
  const file = new File(['bytes'], 'a.png', { type: 'image/png' })
  let read!: (bytes: ArrayBuffer) => void
  spyOn(file, 'arrayBuffer').mockImplementation(
    () =>
      new Promise((resolve) => {
        read = resolve
      }),
  )
  const insertion = store.addImageFiles([file], 0, 1, 'auto')
  store.setStateText('"after"')
  read(new Uint8Array([1]).buffer)
  await expect(insertion).rejects.toThrow('request changed')
  expect(store.getSnapshot().editor.stateText).toBe('"after"')
  const request = store.exportRequest()
  const run = store.run()
  expect(() => store.importRequest(request)).toThrow('current run')
  await expect(store.addImageFiles([file], 0, 1, 'auto')).rejects.toThrow('unavailable')
  store.setProtocol('typesafe')
  expect(store.getSnapshot().editor.protocol).toBe('openai')
  finish(Response.json({ answers: [] }))
  await run
})
