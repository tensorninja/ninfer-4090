import { useMemo, useRef } from 'react'

import { TopBar } from './components/topbar'
import { adapterPool } from './lib/derive'
import { useRoute } from './lib/route'
import { useEngine } from './lib/use-engine'
import { Playground, type EngineView } from './playground/Playground'
import { PlaygroundStore } from './playground/store'
import { Dashboard } from './views/Dashboard'

export function App() {
  const engine = useEngine()
  const route = useRoute()
  const { state } = engine

  // One playground for the life of the page: switching to the dashboard neither loses the
  // request being edited nor abandons a decision in flight, which the engine would log as a 499.
  const playground = useRef<PlaygroundStore | null>(null)
  playground.current ??= new PlaygroundStore()

  // Live inventory wins; a replayed log carries the same block on its own server_start, so a
  // registered-but-unused adapter is still named offline.
  const adapters = state.telemetry?.adapters ?? state.serverStart?.adapters
  const pool = useMemo(() => adapterPool(adapters), [adapters])
  const surface = state.telemetry?.systemone ?? state.serverStart?.systemone
  // System One earns its readings when this server can answer a decision or a log recorded one.
  const systemOne = pool.some((entry) => entry.kind === 'decision') || state.decisions.length > 0

  // The slices the playground joins its runs against. Each keeps its identity until a record of
  // its kind arrives, so the 1 Hz telemetry poll does not re-render the editor.
  const { connection, source, decisions, activeDecisions, decisionErrors } = state
  const engineView = useMemo<EngineView>(
    () => ({ connection, source, decisions, activeDecisions, decisionErrors }),
    [connection, source, decisions, activeDecisions, decisionErrors],
  )

  return (
    <div className="console" data-route={route}>
      <TopBar
        engine={engine}
        route={route}
        systemOne={systemOne || Boolean(surface?.supported)}
        binding={surface?.binding}
      />
      {route === 'playground' ? (
        <Playground store={playground.current} engine={engineView} resumeLive={engine.resumeLive} />
      ) : (
        <Dashboard
          state={state}
          adapters={adapters}
          pool={pool}
          surface={surface}
          systemOne={systemOne}
        />
      )}
    </div>
  )
}
