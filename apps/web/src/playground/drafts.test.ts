import { afterEach, expect, test } from 'bun:test'
import { draftStorage } from './drafts'

const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB')
afterEach(() => {
  if (descriptor) Object.defineProperty(globalThis, 'indexedDB', descriptor)
  else Reflect.deleteProperty(globalThis, 'indexedDB')
})

function database() {
  let stored: unknown
  let closed = 0
  let mode = ''
  let current: IDBTransaction
  let complete: () => void = () => {}
  const request = { result: undefined as unknown }
  const db = {
    createObjectStore: (name: string) => expect(name).toBe('drafts'),
    close: () => {
      closed++
    },
    transaction: (name: string, requested: string) => {
      expect(name).toBe('drafts')
      mode = requested
      current = {
        objectStore: () => ({
          get: (key: string) => {
            expect(key).toBe('current')
            request.result = structuredClone(stored)
            return request
          },
          put: (value: unknown, key: string) => {
            expect(key).toBe('current')
            stored = structuredClone(value)
            return request
          },
        }),
      } as unknown as IDBTransaction
      complete = () => current.oncomplete?.call(current, new Event('complete'))
      return current
    },
  }
  Object.defineProperty(globalThis, 'indexedDB', {
    configurable: true,
    value: {
      open: () => {
        const opening = { result: db } as unknown as IDBOpenDBRequest
        queueMicrotask(() => opening.onsuccess?.call(opening, new Event('success')))
        return opening
      },
    },
  })
  return {
    mode: () => mode,
    closed: () => closed,
    complete: () => complete(),
    fail: () => {
      Object.defineProperty(current, 'error', {
        value: new DOMException('quota exhausted', 'QuotaExceededError'),
      })
      current.onabort?.call(current, new Event('abort'))
    },
  }
}

test('native IndexedDB stores large independent invalid drafts and waits for transaction completion', async () => {
  const db = database()
  const drafts = {
    protocol: 'openai',
    drafts: {
      typesafe: { snap: { stateText: '{ invalid' } },
      openai: {
        snap: { stateText: 'data:image/png;base64,' + 'AAAA'.repeat(1500000), qText: '[ invalid' },
      },
    },
  }
  let saved = false
  const saving = draftStorage.save(drafts).then(() => {
    saved = true
  })
  await Promise.resolve()
  await Promise.resolve()
  expect(db.mode()).toBe('readwrite')
  expect(saved).toBe(false)
  db.complete()
  await saving
  expect(db.closed()).toBe(1)
  const loading = draftStorage.load()
  await Promise.resolve()
  await Promise.resolve()
  expect(db.mode()).toBe('readonly')
  db.complete()
  expect(await loading).toEqual(drafts)
  expect(db.closed()).toBe(2)
})

test('native transaction failures reject to the UI rather than pretending a draft was saved', async () => {
  const db = database()
  const saving = draftStorage.save({ drafts: {} })
  const outcome = saving.then(
    () => null,
    (error: unknown) => error,
  )
  await Promise.resolve()
  await Promise.resolve()
  db.fail()
  expect(await outcome).toMatchObject({ name: 'QuotaExceededError', message: 'quota exhausted' })
  expect(db.closed()).toBe(1)
})
