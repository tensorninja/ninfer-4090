export interface DraftStorage {
  load(): Promise<unknown>
  save(value: unknown): Promise<void>
}

function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ninfer.playground', 1)
    let blocked = false
    request.onupgradeneeded = () => request.result.createObjectStore('drafts')
    request.onerror = () => reject(request.error)
    request.onblocked = () => {
      blocked = true
      reject(new Error('Draft database is blocked by another open tab.'))
    }
    request.onsuccess = () => {
      if (blocked) request.result.close()
      else resolve(request.result)
    }
  })
}

async function transact(mode: IDBTransactionMode, value?: unknown): Promise<unknown> {
  const db = await database()
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction('drafts', mode)
      const store = transaction.objectStore('drafts')
      const request = mode === 'readonly' ? store.get('current') : store.put(value, 'current')
      transaction.oncomplete = () => resolve(request.result)
      transaction.onabort = () =>
        reject(transaction.error ?? new Error('Draft transaction aborted.'))
      transaction.onerror = () => reject(transaction.error ?? request.error)
    })
  } finally {
    db.close()
  }
}

export const draftStorage: DraftStorage = {
  load: () => transact('readonly'),
  save: async (value) => {
    await transact('readwrite', value)
  },
}
