// A draggable divider between two panes, as an ARIA separator.
//
// Dragging only previews the new share (the caller moves a CSS variable, so nothing re-renders
// while the pointer moves); the share is committed when the pointer is released. Arrow keys move
// it by 2% (Shift: 10%), Home and End go to the limits, and Enter or a double click resets it.
// Ported from laya's playground (examples/server.py: splitter; Apache-2.0).

import { useLayoutEffect, useRef, type PointerEvent } from 'react'

import { cx } from '../components/ui'

const STEP = 0.02
const BIG_STEP = 0.1

export function Splitter({
  orientation,
  label,
  bounds,
  value,
  measure,
  onPreview,
  onCommit,
  onReset,
}: {
  /** A vertical divider sits between side-by-side panes; a horizontal one between stacked ones. */
  orientation: 'vertical' | 'horizontal'
  label: string
  /** The range the share may take, read whenever it is used. */
  bounds: () => readonly [number, number]
  /** The current share, read after each render and when a key moves it. */
  value: () => number
  /** The share at the pointer. */
  measure: (event: PointerEvent<HTMLDivElement>) => number
  /** Shows a share while dragging, without committing it. */
  onPreview: (share: number) => void
  onCommit: (share: number) => void
  onReset: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const drag = useRef<number | null>(null)

  const clamp = (share: number) => {
    const [lo, hi] = bounds()
    return Math.max(lo, Math.min(hi, share))
  }

  // The ARIA values follow the layout, which can change without this component rendering.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const [lo, hi] = bounds()
    el.setAttribute('aria-valuemin', String(Math.round(lo * 100)))
    el.setAttribute('aria-valuemax', String(Math.round(hi * 100)))
    el.setAttribute('aria-valuenow', String(Math.round(clamp(value()) * 100)))
  })

  const end = (event: PointerEvent<HTMLDivElement>) => {
    const share = drag.current
    if (share === null) return
    drag.current = null
    event.currentTarget.classList.remove('is-drag')
    onCommit(share)
  }

  return (
    <div
      ref={ref}
      role="separator"
      tabIndex={0}
      aria-orientation={orientation}
      aria-label={label}
      className={cx('pg-split', `pg-split--${orientation}`)}
      onPointerDown={(event) => {
        if (event.button !== 0) return
        event.preventDefault()
        event.currentTarget.setPointerCapture(event.pointerId)
        event.currentTarget.classList.add('is-drag')
        drag.current = clamp(value())
      }}
      onPointerMove={(event) => {
        if (drag.current === null) return
        drag.current = clamp(measure(event))
        onPreview(drag.current)
        event.currentTarget.setAttribute('aria-valuenow', String(Math.round(drag.current * 100)))
      }}
      onPointerUp={end}
      onPointerCancel={end}
      onKeyDown={(event) => {
        const step = event.shiftKey ? BIG_STEP : STEP
        const delta: Record<string, number> = {
          ArrowLeft: -step,
          ArrowUp: -step,
          ArrowRight: step,
          ArrowDown: step,
          Home: -1,
          End: 1,
        }
        if (event.key in delta) onCommit(clamp(value() + delta[event.key]!))
        else if (event.key === 'Enter') onReset()
        else return
        event.preventDefault()
      }}
      onDoubleClick={onReset}
    />
  )
}
