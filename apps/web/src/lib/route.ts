// Client-side routes.
//
// ninfer-serve answers every GET that is not an API path with the application shell when it runs
// with --web-dir, and the Vite dev server does the same, so a path route survives a reload and can
// be linked to. The URL hash stays free for the playground's share links.

import { useSyncExternalStore, type MouseEvent } from 'react'

export type Route = 'dashboard' | 'playground'

export const ROUTE_PATH: Record<Route, string> = { dashboard: '/', playground: '/playground' }

const NAVIGATE = 'ninfer:navigate'

function routeOf(pathname: string): Route {
  return pathname.replace(/\/+$/, '') === ROUTE_PATH.playground ? 'playground' : 'dashboard'
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener('popstate', onChange)
  window.addEventListener(NAVIGATE, onChange)
  return () => {
    window.removeEventListener('popstate', onChange)
    window.removeEventListener(NAVIGATE, onChange)
  }
}

export function useRoute(): Route {
  return useSyncExternalStore(subscribe, () => routeOf(window.location.pathname))
}

export function navigate(route: Route): void {
  if (routeOf(window.location.pathname) === route) return
  window.history.pushState(null, '', ROUTE_PATH[route])
  window.dispatchEvent(new Event(NAVIGATE))
}

/** Click handler for an in-app link: a modified click keeps the browser's new-tab behaviour. */
export function followRoute(route: Route) {
  return (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return
    }
    event.preventDefault()
    navigate(route)
  }
}
