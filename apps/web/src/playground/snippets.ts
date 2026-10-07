// The current request as code: curl against this server, and Python on the selected SDK.
//
// Both are generated from the request tree, so they send what the playground sends: numbers keep
// their spelling, and a Python int has no 2^53 limit. The SDK's question classes forbid keys
// System One does not read, so those are left out of the Python with a comment, and a question
// the classes cannot express is passed as the plain dict the SDK also accepts.

import { dedupe, formatJson, lastEntry, type JsonNode, type ObjectNode } from './json'
import { bodyNode, isQuestionType, type Protocol, type RequestBody } from './request'

const ENDPOINT = '/typesafe/v1/systemone'

export const endpointFor = (protocol: Protocol): string =>
  protocol === 'openai' ? '/v1/decisions' : ENDPOINT

export function curlSnippet(origin: string, body: RequestBody): string {
  const json = formatJson(bodyNode(body.state, body.questions, body.model, body.protocol))
  return (
    `curl -s ${origin}${endpointFor(body.protocol)} \\\n` +
    `  -H 'content-type: application/json' \\\n` +
    `  -d '${json.replace(/'/g, "'\\''")}'`
  )
}

const PAD = '    '

/** A Python literal for a JSON tree; numbers keep their spelling, which Python reads alike. */
export function toPy(node: JsonNode, indent = 0): string {
  if (node.t === 'lit') return node.v === null ? 'None' : node.v ? 'True' : 'False'
  if (node.t === 'num') return node.raw
  if (node.t === 'str') return JSON.stringify(node.v)
  const pad = PAD.repeat(indent + 1)
  const end = ',\n' + PAD.repeat(indent)
  const parts =
    node.t === 'arr'
      ? node.items.map((item) => pad + toPy(item, indent + 1))
      : node.entries.map((en) => pad + JSON.stringify(en.k) + ': ' + toPy(en.v, indent + 1))
  const [open, close] = node.t === 'arr' ? ['[', ']'] : ['{', '}']
  return parts.length ? open + '\n' + parts.join(',\n') + end + close : open + close
}

const CLASS = { noul: 'Noul', choice: 'Choice', score: 'Score' } as const
const READ = new Set(['type', 'instructions', 'criteria'])

/** A question as an SDK constructor call, or null when only a plain dict can express it. */
function sdkQuestion(question: JsonNode, indent: number): { code: string; cls: string } | null {
  if (question.t !== 'obj') return null
  const type = lastEntry(question, 'type')?.v
  if (type?.t !== 'str' || !isQuestionType(type.v)) return null
  const ins = lastEntry(question, 'instructions')?.v
  let criteria = lastEntry(question, 'criteria')?.v
  if (criteria?.t === 'lit' && criteria.v === null) criteria = undefined
  const comments: string[] = []
  if (type.v === 'choice' && criteria?.t !== 'obj') return null
  if (type.v === 'score' && criteria?.t !== 'arr') return null
  if (type.v === 'noul' && criteria) {
    if (criteria.t !== 'obj') return null
    // NoulCriteria is closed to true and false, the only descriptions System One reads.
    const ignored = criteria.entries.filter((en) => en.k !== 'true' && en.k !== 'false')
    if (ignored.length) {
      comments.push(
        `criteria ${ignored.map((en) => JSON.stringify(en.k)).join(', ')} left out: noul reads only true and false`,
      )
      criteria = { ...criteria, entries: criteria.entries.filter((en) => !ignored.includes(en)) }
    }
  }
  const extra = question.entries.filter((en) => !READ.has(en.k)).map((en) => JSON.stringify(en.k))
  if (extra.length)
    comments.push(`${[...new Set(extra)].join(', ')} left out: System One does not read it`)

  const pad = PAD.repeat(indent + 1)
  const args: string[] = []
  if (ins && !(ins.t === 'lit' && ins.v === null))
    args.push(`instructions=${toPy(ins, indent + 1)}`)
  if (criteria) args.push(`criteria=${toPy(criteria, indent + 1)}`)
  const cls = CLASS[type.v]
  const notes = comments.map((c) => `${pad}# ${c}\n`).join('')
  const code =
    args.length === 1 && !notes && !args[0]!.includes('\n') && args[0]!.length < 60
      ? `${cls}(${args[0]})`
      : args.length || notes
        ? `${cls}(\n${notes}${args.map((a) => pad + a + ',\n').join('')}${PAD.repeat(indent)})`
        : `${cls}()`
  return { code, cls }
}

export function pythonSnippet(origin: string, body: RequestBody): string {
  const questions = dedupe(body.questions)
  if (body.protocol === 'openai') {
    return (
      `from openai import OpenAI\n\n` +
      `client = OpenAI(api_key="local", base_url=${JSON.stringify(origin + '/v1')})\n\n` +
      `response = client.decisions.create(\n` +
      `${PAD}model=${JSON.stringify(body.model)},\n` +
      `${PAD}input=${toPy(dedupe(body.state), 1)},\n` +
      `${PAD}questions=${toPy(questions, 1)},\n` +
      `)\n` +
      `for index, answer in enumerate(response.answers):\n` +
      `${PAD}print(index, answer)\n`
    )
  }
  const used = new Set<string>()
  let questionsCode: string
  if (questions.t === 'obj' && questions.entries.length) {
    const pad = PAD.repeat(2)
    const lines = (questions as ObjectNode).entries.map((en) => {
      const sdk = sdkQuestion(en.v, 2)
      if (sdk) used.add(sdk.cls)
      return `${pad}${JSON.stringify(en.k)}: ${sdk ? sdk.code : toPy(en.v, 2)},`
    })
    questionsCode = `{\n${lines.join('\n')}\n${PAD}}`
  } else {
    questionsCode = toPy(questions, 1)
  }
  const imports = [...[...used].sort(), 'TypeSafeClient'].join(', ')
  const model = body.model ? `${PAD}model=${JSON.stringify(body.model)},\n` : ''
  return (
    `from typesafe_sdk import ${imports}\n\n` +
    `client = TypeSafeClient(api_key="local", base_url=${JSON.stringify(origin + '/typesafe')})\n\n` +
    `state = ${toPy(dedupe(body.state))}\n\n` +
    `response = client.system_one(\n` +
    `${PAD}state,\n` +
    `${PAD}questions=${questionsCode},\n` +
    model +
    `)\n` +
    `for key, answer in response.answers.items():\n` +
    `${PAD}print(key, answer)\n`
  )
}
