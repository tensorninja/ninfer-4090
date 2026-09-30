import { useEffect, useMemo, useRef, useState } from 'react'

import { EngineClient, initialEngineState, type EngineState } from './engine-client'

export interface EngineHandle {
  state: EngineState
  loadFile: (file: File) => Promise<void>
  resumeLive: () => void
}

export function useEngine(): EngineHandle {
  const [state, setState] = useState<EngineState>(initialEngineState)
  const clientRef = useRef<EngineClient | null>(null)
  const activeRef = useRef(true)

  // Created lazily with ??= so StrictMode's double invoke does not open two streams.
  clientRef.current ??= new EngineClient((next) => {
    if (activeRef.current) setState(next)
  })

  useEffect(() => {
    activeRef.current = true
    const client = clientRef.current!
    client.start()
    return () => {
      activeRef.current = false
      client.stop()
    }
  }, [])

  // Stable across renders, so a memoized view that takes them does not re-render on every poll.
  const actions = useMemo(
    () => ({
      loadFile: async (file: File) => {
        clientRef.current!.loadFile(file.name, await file.text())
      },
      resumeLive: () => clientRef.current!.resumeLive(),
    }),
    [],
  )

  return { state, ...actions }
}
