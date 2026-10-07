import { lastEntry, parseJson, type JsonNode, type ObjectNode } from './json'

export const MAX_IMAGES = 128
export const MAX_IMAGE_BYTES = 256 * 1024 * 1024
export const IMAGE_DETAILS = ['auto', 'low', 'high', 'original', 'null', 'missing'] as const
export type ImageDetail = (typeof IMAGE_DETAILS)[number]

export function imagePayloadError(url: string): string | null {
  const separator = url.indexOf(';base64,')
  if (
    !url.startsWith('data:image/') ||
    separator <= 11 ||
    /[;, \t\r\n]/.exec(url.slice(5))?.index !== separator - 5
  )
    return 'Expected an inline base64 image data URI.'
  const start = separator + 8
  const encoded = url.length - start
  const padding = url.endsWith('==') ? 2 : url.endsWith('=') ? 1 : 0
  if (!encoded || encoded % 4 !== 0) return 'Invalid base64 image data.'
  if (encoded > (Math.floor(MAX_IMAGE_BYTES / 3) + 1) * 4)
    return `Image exceeds the maximum of ${MAX_IMAGE_BYTES} decoded bytes.`
  for (let i = start; i < url.length - padding; i++) {
    const code = url.charCodeAt(i)
    if (!(
      (code >= 65 && code <= 90) ||
      (code >= 97 && code <= 122) ||
      (code >= 48 && code <= 57) ||
      code === 43 ||
      code === 47
    ))
      return 'Invalid base64 image data.'
  }
  if ((encoded / 4) * 3 - padding > MAX_IMAGE_BYTES)
    return `Image exceeds the maximum of ${MAX_IMAGE_BYTES} decoded bytes.`
  return null
}

export async function fileDataUrl(file: File): Promise<string> {
  if (!file.type.startsWith('image/')) throw new Error(`${file.name}: select an image file.`)
  if (file.size > MAX_IMAGE_BYTES) throw new Error(`${file.name}: image exceeds 256 MiB.`)
  const bytes = new Uint8Array(await file.arrayBuffer())
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000)
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return `data:${file.type};base64,${btoa(binary)}`
}

export interface InputMessage {
  index: number
  content: JsonNode
}

export function inputMessages(node: JsonNode | null): InputMessage[] {
  if (node?.t === 'str') return [{ index: 0, content: node }]
  if (node?.t !== 'arr') return []
  return node.items.flatMap((message, index) => {
    if (message.t !== 'obj') return []
    const role = lastEntry(message, 'role')?.v
    const content = lastEntry(message, 'content')?.v
    return role?.t === 'str' && role.v === 'user' && (content?.t === 'str' || content?.t === 'arr')
      ? [{ index, content }]
      : []
  })
}

export interface InputImage {
  message: number
  part: number
  node: ObjectNode
  url: string | null
  detail: string
}

export function inputImages(node: JsonNode | null): InputImage[] {
  return inputMessages(node).flatMap(({ index, content }) =>
    content.t === 'arr'
      ? content.items.flatMap((part, i) => {
          if (part.t !== 'obj') return []
          const type = lastEntry(part, 'type')?.v
          if (type?.t !== 'str' || type.v !== 'input_image') return []
          const url = lastEntry(part, 'image_url')?.v
          const detail = lastEntry(part, 'detail')?.v
          return [
            {
              message: index,
              part: i,
              node: part,
              url: url?.t === 'str' ? url.v : null,
              detail: !detail
                ? 'missing'
                : detail.t === 'lit' && detail.v === null
                  ? 'null'
                  : detail.t === 'str' && ['auto', 'low', 'high', 'original'].includes(detail.v)
                    ? detail.v
                    : 'invalid',
            },
          ]
        })
      : [],
  )
}

function replace(text: string, node: JsonNode, value: string): string {
  return text.slice(0, node.s) + value + text.slice(node.e)
}

export function insertImages(
  text: string,
  message: number,
  position: number,
  urls: readonly string[],
  detail: ImageDetail,
): string {
  const root = parseJson(text)
  if (inputImages(root).length + urls.length > MAX_IMAGES)
    throw new Error(`A decision can contain at most ${MAX_IMAGES} images.`)
  const parts = urls.map((url) => {
    const error = imagePayloadError(url)
    if (error) throw new Error(error)
    return `{"type":"input_image","image_url":${JSON.stringify(url)}${detail === 'missing' ? '' : `,"detail":${detail === 'null' ? 'null' : JSON.stringify(detail)}`}}`
  })
  if (!parts.length) return text
  const content = inputMessages(root).find((item) => item.index === message)?.content
  if (!content) {
    if (root.t === 'arr' && root.items.length === 0 && message === 0)
      return replace(text, root, `[{"role":"user","content":[${parts.join(',')}]}]`)
    throw new Error('Select an existing user message with text or content parts.')
  }
  const length = content.t === 'arr' ? content.items.length : 1
  if (!Number.isInteger(position) || position < 0 || position > length)
    throw new Error('Select a valid insertion position.')
  if (content.t === 'arr') {
    const at = position === length ? content.e - 1 : content.items[position]!.s
    const insertion =
      (position === length && length > 0 ? ',' : '') +
      parts.join(',') +
      (position < length ? ',' : '')
    return text.slice(0, at) + insertion + text.slice(at)
  }
  const original = `{"type":"input_text","text":${text.slice(content.s, content.e)}}`
  const values = position === 0 ? [...parts, original] : [original, ...parts]
  const value = `[${values.join(',')}]`
  return replace(text, content, root.t === 'str' ? `[{"role":"user","content":${value}}]` : value)
}

export function setImageDetail(
  text: string,
  message: number,
  part: number,
  detail: ImageDetail,
): string {
  const image = inputImages(parseJson(text)).find(
    (image) => image.message === message && image.part === part,
  )
  if (!image) throw new Error('The image part no longer exists.')
  const previous = lastEntry(image.node, 'detail')
  const value = detail === 'null' ? 'null' : JSON.stringify(detail)
  if (detail !== 'missing' && previous) return replace(text, previous.v, value)
  if (detail !== 'missing') {
    const at = image.node.e - 1
    return text.slice(0, at) + `,"detail":${value}` + text.slice(at)
  }
  const kept = image.node.entries.filter((entry) => entry.k !== 'detail')
  return replace(
    text,
    image.node,
    `{${kept.map((entry) => text.slice(entry.ks, entry.ke) + ':' + text.slice(entry.v.s, entry.v.e)).join(',')}}`,
  )
}
