import { afterEach, beforeEach, expect, mock, spyOn, test } from 'bun:test'

import { compactJson } from './json'
import { readShareHash, shareUrl } from './share'
import { PlaygroundStore } from './store'

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
    },
  ],
}

const saved = new Map<string, string>()
const timers = new Map<number, () => void>()
let nextTimer = 0
let location: URL
const windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')
const storageDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')

function flushDrafts() {
  const callbacks = [...timers.values()]
  timers.clear()
  callbacks.forEach((callback) => callback())
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
  const store = new PlaygroundStore()
  store.boot()
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
  flushDrafts()
  const reloaded = new PlaygroundStore()
  reloaded.boot()
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
  const store = new PlaygroundStore()
  store.boot()
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
  const store = new PlaygroundStore()
  store.boot()
  await catalogReady(store)
  store.setQText('{ unfinished')
  flushDrafts()
  const body =
    '{"input":"context","questions":[{"name":"q","type":"choice","instructions":"Pick","choices":[{"value":true},{"value":"true"},{"value":true}]}],"model":"decision-v7"}'
  location = new URL(shareUrl(location.href, body))
  const opened = new PlaygroundStore()
  opened.boot()
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
  const store = new PlaygroundStore()
  store.boot()
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
  const store = new PlaygroundStore()
  store.boot()
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
