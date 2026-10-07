import {
  lastEntry,
  lineCol,
  tryParse,
  type ArrayNode,
  type JsonNode,
  type ObjectNode,
} from './json'
import { buildBody, quote, type EditorState, type Section } from './request'
import type { Analysis, Problem, SectionResult } from './validate'

const STRING_LIMIT = 1048576
const INPUT_LIMIT = 10485760

function exceedsLimit(value: string, limit: number): boolean {
  if (value.length <= limit) return false
  let length = 0
  for (let i = 0; i < value.length; i++) {
    if (value.codePointAt(i)! > 0xffff) i++
    if (++length > limit) return true
  }
  return false
}

class Validator {
  readonly problems: Problem[] = []

  constructor(readonly sec: Section) {}

  error(where: string, node: JsonNode, msg: string) {
    this.problems.push({
      sec: this.sec,
      severity: 'error',
      msg,
      where,
      loc: { text: this.sec, at: [node.s, node.e] },
    })
  }

  object(node: JsonNode, path: string): node is ObjectNode {
    if (node.t === 'obj') return true
    this.error(path, node, 'Expected an object.')
    return false
  }

  fields(node: JsonNode, path: string, allowed: readonly string[]): node is ObjectNode {
    if (!this.object(node, path)) return false
    const entries = new Map(node.entries.map((entry) => [entry.k, entry]))
    for (const entry of entries.values()) {
      if (!allowed.includes(entry.k)) {
        this.problems.push({
          sec: this.sec,
          severity: 'error',
          msg: 'Unknown parameter.',
          where: `${path}.${entry.k}`,
          loc: { text: this.sec, at: [entry.ks, entry.ke] },
        })
      }
    }
    return true
  }

  required(node: ObjectNode, key: string, path: string): JsonNode | undefined {
    const entry = lastEntry(node, key)
    if (!entry) this.error(`${path}.${key}`, node, 'Missing required parameter.')
    return entry?.v
  }

  text(node: JsonNode | undefined, path: string, limit = STRING_LIMIT): string | undefined {
    if (!node) return undefined
    if (node.t !== 'str') {
      this.error(path, node, 'Expected a string.')
      return undefined
    }
    if (exceedsLimit(node.v, limit)) {
      this.error(path, node, `String exceeds the maximum of ${limit} Unicode characters.`)
    }
    return node.v
  }

  array(
    node: JsonNode | undefined,
    path: string,
    minimum: number,
    maximum: number,
  ): node is ArrayNode {
    if (!node) return false
    if (node.t !== 'arr') {
      this.error(path, node, 'Expected an array.')
      return false
    }
    if (node.items.length < minimum || node.items.length > maximum) {
      this.error(path, node, `Expected between ${minimum} and ${maximum} items.`)
      return false
    }
    return true
  }

  input(node: JsonNode) {
    if (node.t === 'str') {
      this.text(node, 'input', INPUT_LIMIT)
      return
    }
    if (!this.array(node, 'input', 0, 131072)) return
    node.items.forEach((message, i) => {
      const path = `input[${i}]`
      if (!this.fields(message, path, ['role', 'content', 'type'])) return
      const roleNode = this.required(message, 'role', path)
      const role = this.text(roleNode, `${path}.role`)
      if (role !== undefined && role !== 'user') {
        this.error(`${path}.role`, roleNode!, 'Only user messages are supported.')
      }
      const typeNode = lastEntry(message, 'type')?.v
      const type = this.text(typeNode, `${path}.type`)
      if (type !== undefined && type !== 'message') {
        this.error(`${path}.type`, typeNode!, 'Expected message.')
      }
      this.content(this.required(message, 'content', path), `${path}.content`)
    })
  }

  content(node: JsonNode | undefined, path: string) {
    if (node?.t === 'str') {
      this.text(node, path, INPUT_LIMIT)
      return
    }
    if (!this.array(node, path, 0, 16384)) return
    node.items.forEach((part, i) => {
      const partPath = `${path}[${i}]`
      if (!this.object(part, partPath)) return
      const typeNode = this.required(part, 'type', partPath)
      const type = this.text(typeNode, `${partPath}.type`)
      if (type === undefined) return
      if (type === 'input_image') {
        this.error(`${partPath}.type`, typeNode!, 'Decisions supports text input only.')
        return
      }
      this.fields(part, partPath, ['type', 'text'])
      if (type !== 'input_text') {
        this.error(`${partPath}.type`, typeNode!, 'Expected input_text.')
        return
      }
      this.text(this.required(part, 'text', partPath), `${partPath}.text`, INPUT_LIMIT)
    })
  }

  question(node: JsonNode, path: string) {
    if (!this.object(node, path)) return
    const typeNode = this.required(node, 'type', path)
    const type = this.text(typeNode, `${path}.type`)
    if (type === undefined) return
    if (type !== 'predicate' && type !== 'choice' && type !== 'score') {
      this.error(`${path}.type`, typeNode!, 'Expected predicate, choice, or score.')
      return
    }
    const optionKey = type === 'choice' ? 'choices' : 'levels'
    this.fields(
      node,
      path,
      type === 'predicate'
        ? ['type', 'name', 'instructions']
        : ['type', 'name', 'instructions', optionKey],
    )
    this.text(lastEntry(node, 'name')?.v, `${path}.name`)
    this.text(this.required(node, 'instructions', path), `${path}.instructions`)
    if (type === 'predicate') return
    const options = this.required(node, optionKey, path)
    const optionsPath = `${path}.${optionKey}`
    if (!this.array(options, optionsPath, 2, type === 'choice' ? 255 : 10)) return
    const valueKey = type === 'choice' ? 'value' : 'label'
    options.items.forEach((option, i) => {
      const optionPath = `${optionsPath}[${i}]`
      if (!this.fields(option, optionPath, [valueKey, 'description'])) return
      const value = this.required(option, valueKey, optionPath)
      if (!(type === 'choice' && value?.t === 'lit' && typeof value.v === 'boolean')) {
        this.text(value, `${optionPath}.${valueKey}`)
      }
      this.text(lastEntry(option, 'description')?.v, `${optionPath}.description`)
    })
  }

  repeatedKeys(node: JsonNode, path: string) {
    if (node.t === 'arr') {
      node.items.forEach((item, i) => this.repeatedKeys(item, `${path}[${i}]`))
    } else if (node.t === 'obj') {
      const seen = new Set<string>()
      for (const entry of node.entries) {
        const entryPath = `${path}.${entry.k}`
        if (seen.has(entry.k)) {
          this.problems.push({
            sec: this.sec,
            severity: 'warning',
            msg: `Key ${quote(entry.k)} appears twice; only the last one counts`,
            where: entryPath,
            loc: { text: this.sec, at: [entry.ks, entry.ke] },
          })
        }
        seen.add(entry.k)
        this.repeatedKeys(entry.v, entryPath)
      }
    }
  }
}

function analyzeSection(sec: Section, text: string): SectionResult {
  const parsed = tryParse(text)
  const path = sec === 'state' ? 'input' : 'questions'
  if (parsed.error) {
    const error = parsed.error
    return {
      problems: [
        {
          sec,
          severity: 'error',
          msg: error.msg,
          where: `${path}, line ${lineCol(text, error.at).line}`,
          loc: { text: sec, at: [error.at, error.at] },
          parse: error,
        },
      ],
      node: null,
      blocker: '',
    }
  }
  const node = parsed.ast
  const validator = new Validator(sec)
  if (sec === 'state') validator.input(node)
  else if (validator.array(node, path, 1, 200)) {
    node.items.forEach((question, i) => validator.question(question, `${path}[${i}]`))
  }
  validator.repeatedKeys(node, path)
  return { problems: validator.problems, node, blocker: '' }
}

export function analyzeOpenAI(editor: EditorState, models: readonly string[] | null): Analysis {
  const state = analyzeSection('state', editor.stateText)
  const questions = analyzeSection('questions', editor.qText)
  const problems = [...state.problems, ...questions.problems]
  const modelProblem = (severity: Problem['severity'], msg: string) => {
    problems.push({ sec: 'model', severity, msg, where: 'model', loc: { el: 'model' } })
  }
  if (!editor.model.trim()) {
    modelProblem('error', 'Select a decision adapter by its actual pool name; model is required.')
  } else if (editor.model === 'jev-latest') {
    modelProblem(
      'error',
      'jev-latest is a TypeSafe-only alias; use the decision adapter’s actual pool name.',
    )
  } else if (exceedsLimit(editor.model, STRING_LIMIT)) {
    modelProblem('error', `String exceeds the maximum of ${STRING_LIMIT} Unicode characters.`)
  } else if (models && !models.includes(editor.model)) {
    modelProblem(
      'warning',
      `This server has no model ${quote(editor.model)}, so it would answer 404`,
    )
  }
  const errors = problems.filter((problem) => problem.severity === 'error').length
  return {
    state,
    questions,
    problems,
    errors,
    warnings: problems.length - errors,
    body:
      state.node && questions.node
        ? buildBody(state.node, questions.node, editor.model, 'openai')
        : null,
  }
}
