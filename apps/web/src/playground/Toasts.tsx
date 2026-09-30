// The playground's toast and its one polite live region.
// Ported from laya's playground (examples/server.py: toast, hideToast, clearOfToast, announce;
// Apache-2.0).

import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react'
import { flushSync } from 'react-dom'

import type { PlaygroundStore, Toast } from './store'

const TOAST_MS = 3200
const SPEAK_DELAY_MS = 80

/** Cleared, then set a moment later, so that a repeated message is spoken again. */
export function LiveRegion({ announcement }: { announcement: { text: string; seq: number } }) {
  const [text, setText] = useState('')
  useEffect(() => {
    setText('')
    if (!announcement.text) return
    const timer = setTimeout(() => setText(announcement.text), SPEAK_DELAY_MS)
    return () => clearTimeout(timer)
  }, [announcement])
  return (
    <div className="sr-only" role="status" aria-live="polite">
      {text}
    </div>
  )
}

/**
 * A toast with an action (Undo) has no time limit: it stays until it is used or dismissed, the
 * request changes again, or another toast replaces it. A plain toast leaves after 3.2 s, later
 * while the pointer rests on it. Focus inside a toast goes back where it came from when it closes.
 */
export function ToastView({
  store,
  toast,
  fallback,
}: {
  store: PlaygroundStore
  toast: Toast | null
  /** Where focus goes when the control it came from is gone or disabled. */
  fallback: RefObject<HTMLElement | null>
}) {
  const box = useRef<HTMLDivElement>(null)
  const back = useRef<HTMLElement | null>(null)

  useLayoutEffect(() => {
    if (!toast) return
    const active = document.activeElement
    if (
      active instanceof HTMLElement &&
      active !== document.body &&
      !box.current?.contains(active)
    ) {
      back.current = active
    }
    box.current?.classList.remove('is-up')
  }, [toast])

  useEffect(() => {
    if (!toast || toast.action) return
    let timer: ReturnType<typeof setTimeout>
    const arm = () => {
      timer = setTimeout(
        () => (box.current?.matches(':hover') ? arm() : store.dismissToast()),
        TOAST_MS,
      )
    }
    arm()
    return () => clearTimeout(timer)
  }, [toast, store])

  // A toast can stay up while the user tabs on, so the focused control is scrolled clear of it;
  // when nothing can scroll that far, the toast moves to the top of the window instead.
  useEffect(() => {
    if (!toast) return
    let frame = 0
    const clear = () => {
      const el = box.current
      const target = document.activeElement
      if (
        !el ||
        !(target instanceof HTMLElement) ||
        target === document.body ||
        el.contains(target)
      )
        return
      if (!target.matches(':focus-visible')) return
      const under = () => {
        const a = target.getBoundingClientRect()
        const b = el.getBoundingClientRect()
        return a.right > b.left && a.left < b.right && a.bottom > b.top && a.top < b.bottom
          ? a.bottom - b.top + 8
          : 0
      }
      el.classList.remove('is-up')
      // An editor taller than half the window never fits clear of it.
      if (!under() || target.getBoundingClientRect().height > window.innerHeight / 2) return
      const scrolled: Array<[Element, number]> = []
      for (
        let sc = target.parentElement, d = under();
        d && sc;
        sc = sc.parentElement, d = under()
      ) {
        if (sc === document.documentElement || /auto|scroll/.test(getComputedStyle(sc).overflowY)) {
          scrolled.push([sc, sc.scrollTop])
          sc.scrollTop += d
        }
      }
      if (!under()) return
      // Scrolling did not help: leave the page where it was.
      for (const [sc, top] of scrolled) sc.scrollTop = top
      el.classList.add('is-up')
    }
    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(clear)
    }
    schedule()
    document.addEventListener('focusin', schedule)
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener('focusin', schedule)
    }
  }, [toast])

  if (!toast) return null

  /** Runs `fn`, then puts focus back where it was before the toast, if it was in the toast. */
  const act = (fn: () => void) => {
    const inside = box.current?.contains(document.activeElement) ?? false
    flushSync(fn)
    if (!inside) return
    // The first that can take focus: the control the toast came from may be gone or disabled.
    for (const el of [back.current, fallback.current]) {
      if (!el || !el.isConnected || box.current?.contains(el)) continue
      el.focus()
      if (document.activeElement === el) return
    }
  }
  const action = toast.action

  return (
    <div
      ref={box}
      className="pg-toast"
      onKeyDown={(event) => {
        if (event.key !== 'Escape') return
        event.preventDefault()
        act(() => store.dismissToast())
      }}
    >
      <span>{toast.text}</span>
      {action ? (
        <>
          <button type="button" onClick={() => act(action.run)}>
            {action.label}
          </button>
          <button
            type="button"
            className="pg-toast__x"
            aria-label="Dismiss"
            title="Dismiss"
            onClick={() => act(() => store.dismissToast())}
          >
            {'\u2715'}
          </button>
        </>
      ) : null}
    </div>
  )
}
