// Copying text. The Clipboard API exists only in secure contexts, and the playground is usually
// opened over plain HTTP on a LAN address, so a hidden textarea and execCommand('copy') stand in.
// Ported from laya's playground (examples/server.py: layaCopy; Apache-2.0).

export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Denied or unavailable after all: try the old way.
    }
  }
  const back = document.activeElement
  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.opacity = '0'
  document.body.append(ta)
  ta.select()
  let ok = false
  try {
    ok = document.execCommand('copy')
  } catch {
    ok = false
  }
  ta.remove()
  if (back instanceof HTMLElement) back.focus({ preventScroll: true })
  return ok
}
