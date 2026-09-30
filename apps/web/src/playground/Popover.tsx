// An anchored popover for the playground's menus: problems, shortcuts, and the share link.
//
// It lives at the end of <body> (it is position: fixed), so Tab past either end closes it and
// carries on from the trigger, as if it were inline. It opens below the trigger, or above when
// there is no room, clamped to the window; it focuses its first control, and closes on Escape
// (focus back to where it was), on a click or focus outside, and on resize or scroll.
// Ported from laya's playground (examples/server.py: openPop; Apache-2.0).

import { useCallback, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

const FOCUSABLE = 'button, input, select, textarea, a[href]'

export function Popover({
  anchor,
  label,
  onClose,
  children,
}: {
  anchor: HTMLElement
  label: string
  /** `restoreFocus` is true when focus should go back where it was before opening. */
  onClose: (restoreFocus: boolean) => void
  children: ReactNode
}) {
  const box = useRef<HTMLDivElement>(null)
  const back = useRef<Element | null>(null)
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null)

  useLayoutEffect(() => {
    back.current = document.activeElement
    const el = box.current!
    const r = anchor.getBoundingClientRect()
    const w = el.offsetWidth
    const ht = el.offsetHeight
    const below = r.bottom + 6
    const top = below + ht > window.innerHeight - 8 && r.top - ht - 6 > 8 ? r.top - ht - 6 : below
    setPosition({
      top: Math.max(8, top),
      left: Math.min(Math.max(8, r.right - w), window.innerWidth - w - 8),
    })
    anchor.setAttribute('aria-expanded', 'true')
    el.querySelector<HTMLElement>(FOCUSABLE)?.focus({ preventScroll: true })
    return () => anchor.setAttribute('aria-expanded', 'false')
  }, [anchor])

  useLayoutEffect(() => {
    const close = () => onClose(false)
    const outside = (event: Event) => {
      const target = event.target as Node
      if (!box.current?.contains(target) && !anchor.contains(target)) close()
    }
    const outsideFocus = (event: FocusEvent) => {
      const target = event.target as Node
      if (!box.current?.contains(target) && target !== anchor) close()
    }
    const onScroll = (event: Event) => {
      if (!box.current?.contains(event.target as Node)) close()
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      onClose(true)
      const to = back.current
      ;(to instanceof HTMLElement && to.isConnected && to !== document.body ? to : anchor).focus()
    }
    document.addEventListener('pointerdown', outside, true)
    document.addEventListener('focusin', outsideFocus, true)
    document.addEventListener('keydown', onKey)
    window.addEventListener('resize', close)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      document.removeEventListener('pointerdown', outside, true)
      document.removeEventListener('focusin', outsideFocus, true)
      document.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', close)
      window.removeEventListener('scroll', onScroll, true)
    }
  }, [anchor, onClose])

  return createPortal(
    <div
      ref={box}
      className="pg-pop"
      role="dialog"
      aria-label={label}
      style={position ?? { left: -9999, top: -9999 }}
      onKeyDown={(event) => {
        if (event.key !== 'Tab') return
        const controls = box.current!.querySelectorAll<HTMLElement>(FOCUSABLE)
        const edge = controls[event.shiftKey ? 0 : controls.length - 1]
        if (!controls.length || document.activeElement !== edge) return
        // Past either end: close, then carry on from the trigger.
        onClose(false)
        anchor.focus()
        if (event.shiftKey) event.preventDefault()
      }}
    >
      {children}
    </div>,
    document.body,
  )
}

/** Open/close state for one popover whose trigger is the element that was clicked. */
export function usePopover() {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const toggle = useCallback(
    (element: HTMLElement) => setAnchor((current) => (current === element ? null : element)),
    [],
  )
  const close = useCallback(() => setAnchor(null), [])
  return { anchor, toggle, close, open: setAnchor }
}
