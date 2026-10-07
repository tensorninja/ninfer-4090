// The System One models this server answers, from GET /typesafe/v1/models.
//
// Each decision adapter is a model under its own name, and the SDK's default name is served as an
// alias bound to one of them. The pool is fixed when the server starts, so the catalog is read
// once per page load.

import type { Protocol } from './request'

export interface ModelCard {
  name: string
  description: string
  release_date: string
  /** The decision adapter that answers; differs from `name` only for the alias. */
  adapter: string
  base: string
  rank: number
  temperature: number
  kv_cache: string
  max_context: number
  prefix_reuse?: boolean
  max_state_tokens?: number
  vision?: boolean
}

export type Catalog =
  | { state: 'loading' }
  | { state: 'ready'; cards: ModelCard[] }
  | { state: 'error'; message: string }

export const modelsPath = (protocol: Protocol) =>
  protocol === 'openai' ? '/v1/models' : '/typesafe/v1/models'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

export function parseCatalog(data: unknown, protocol: Protocol = 'typesafe'): ModelCard[] {
  if (protocol === 'openai') {
    const models = isRecord(data) && Array.isArray(data.data) ? data.data : []
    return models.filter(isRecord).flatMap((m) =>
      typeof m.id === 'string' &&
      Array.isArray(m.supported_endpoints) &&
      m.supported_endpoints.includes('/v1/decisions')
        ? [
            {
              name: m.id,
              description: 'OpenAI Decisions. Uses a local decision adapter, not Luna.',
              release_date: '',
              adapter: m.id,
              base: '',
              rank: 0,
              temperature: 0,
              kv_cache: '',
              max_context: typeof m.context_window === 'number' ? m.context_window : 0,
              max_state_tokens:
                typeof m.max_state_tokens === 'number' ? m.max_state_tokens : undefined,
              vision: isRecord(m.modalities) ? m.modalities.vision === true : false,
            },
          ]
        : [],
    )
  }
  const models = isRecord(data) && Array.isArray(data.models) ? data.models : []
  return models.filter(isRecord).flatMap((m) =>
    typeof m.name === 'string'
      ? [
          {
            name: m.name,
            description: typeof m.description === 'string' ? m.description : '',
            release_date: typeof m.release_date === 'string' ? m.release_date : '',
            adapter: typeof m.adapter === 'string' ? m.adapter : m.name,
            base: typeof m.base === 'string' ? m.base : '',
            rank: typeof m.rank === 'number' ? m.rank : 0,
            temperature: typeof m.temperature === 'number' ? m.temperature : 0,
            kv_cache: typeof m.kv_cache === 'string' ? m.kv_cache : '',
            max_context: typeof m.max_context === 'number' ? m.max_context : 0,
            prefix_reuse: m.prefix_reuse === true,
          },
        ]
      : [],
  )
}

export async function fetchCatalog(protocol: Protocol, signal?: AbortSignal): Promise<ModelCard[]> {
  const response = await fetch(modelsPath(protocol), {
    headers: { accept: 'application/json' },
    signal,
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return parseCatalog(await response.json(), protocol)
}

/** The SDK alias's card: the one served under a name that is not its adapter's. */
export const aliasCard = (cards: readonly ModelCard[]) => cards.find((c) => c.name !== c.adapter)

/** The model a fresh request names: none when the alias exists, so the server's default answers. */
export function defaultModel(cards: readonly ModelCard[]): string {
  return aliasCard(cards) ? '' : (cards[0]?.name ?? '')
}

/** The card that answers a `model` value; '' is the alias. */
export function cardFor(cards: readonly ModelCard[], model: string): ModelCard | undefined {
  return model ? cards.find((c) => c.name === model) : aliasCard(cards)
}

export interface ModelOption {
  value: string
  label: string
  unknown: boolean
}

/** The picker's options: the alias first as "omit the model", then each adapter by name. */
export function modelOptions(
  cards: readonly ModelCard[],
  current: string,
  protocol: Protocol = 'typesafe',
): ModelOption[] {
  const alias = aliasCard(cards)
  const options: ModelOption[] = []
  if (alias)
    options.push({
      value: '',
      label: `${alias.name} \u2192 ${alias.adapter} (SDK default)`,
      unknown: false,
    })
  for (const card of cards) {
    // The alias named explicitly asks the same adapter, but it is a different request text.
    if (card !== alias || current === card.name) {
      options.push({
        value: card.name,
        label: card === alias ? `${card.name} \u2192 ${card.adapter}` : card.name,
        unknown: false,
      })
    }
  }
  if (!options.some((o) => o.value === current)) {
    options.push({
      value: current,
      label: current
        ? `${current} (unknown)`
        : protocol === 'openai'
          ? '(select a decision model)'
          : '(server default)',
      unknown: true,
    })
  }
  return options
}

/** Whether decisions on this KV codec are qualified to kev's serving tolerance; only BF16 is. */
export const rotatedCodec = (kv: string) => /^rk/i.test(kv)
