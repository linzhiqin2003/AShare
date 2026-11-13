<script setup lang="ts">
import { onMounted, onUnmounted, shallowRef, watch } from 'vue'
import { EditorState } from '@codemirror/state'
import { EditorView, keymap, lineNumbers, highlightActiveLineGutter } from '@codemirror/view'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { python } from '@codemirror/lang-python'
import { autocompletion, completionKeymap } from '@codemirror/autocomplete'
import { bracketMatching, HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags } from '@lezer/highlight'
import { setDiagnostics } from '@codemirror/lint'
import { createApp, h } from 'vue'
import { FunctionSquare, Database, Library } from 'lucide-vue-next'
import type { ApiError, Completion } from '../types'

const props = defineProps<{ modelValue: string; completions: Completion[]; error?: ApiError | null; readonly?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: string]; run: []; signature: [value: Completion | null] }>()
const mount = shallowRef<HTMLDivElement | null>(null)
let editor: EditorView | undefined
const highlight = HighlightStyle.define([
  { tag: tags.comment, color: '#8A9199' }, { tag: tags.number, color: '#A14A14' },
  { tag: tags.function(tags.variableName), fontWeight: '600', color: '#14171A' },
  { tag: tags.variableName, color: '#1F4FBF' },
])
function signature() {
  if (!editor) return
  const text = editor.state.doc.sliceString(0, editor.state.selection.main.head)
  const stack: (string | null)[] = []
  const code = text.replace(/#[^\n]*/g, '')
  for (let i = 0; i < code.length; i++) {
    if (code[i] === '(') stack.push(code.slice(0, i).match(/([A-Za-z_]\w*)\s*$/)?.[1] ?? null)
    if (code[i] === ')') stack.pop()
  }
  const name = [...stack].reverse().find(v => v !== null)
  const matches = [...editor.state.doc.toString().matchAll(/([A-Za-z_]\w*)\s*\(/g)]
  const last = [...matches].reverse().find(m => m.index! <= editor!.state.selection.main.head)?.[1] ?? matches[0]?.[1]
  emit('signature', props.completions.find(c => c.label === (name ?? last) && c.kind === 'operator') ?? null)
}
function diagnostic() {
  if (!editor) return
  const error = props.error
  const line = editor.state.doc.line(Math.min(editor.state.doc.lines, Math.max(1, error?.line ?? 1)))
  const column = error?.column ?? 1
  // Python AST 列号是 UTF-8 字节；CodeMirror 位置是 UTF-16。
  let bytes = 0, offset = 0
  for (const char of line.text) { if (bytes >= column - 1) break; bytes += new TextEncoder().encode(char).length; offset += char.length }
  const from = Math.min(line.to, line.from + offset)
  editor.dispatch(setDiagnostics(editor.state, error?.line ? [{ from, to: Math.min(line.to, from + (line.text.slice(offset).match(/^\w+/)?.[0].length ?? 1)), severity: 'error', message: error.message }] : []))
}
onMounted(() => {
  editor = new EditorView({ parent: mount.value!, state: EditorState.create({ doc: props.modelValue, extensions: [
    lineNumbers(), highlightActiveLineGutter(), history(), python(), syntaxHighlighting(highlight), bracketMatching(), EditorView.lineWrapping,
    EditorState.readOnly.of(!!props.readonly), EditorView.editable.of(!props.readonly),
    keymap.of([{ key: 'Mod-Enter', run: () => { emit('run'); return true } }, ...completionKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
    autocompletion({ icons: false, defaultKeymap: false, override: [context => {
      const word = context.matchBefore(/[A-Za-z_]\w*/)
      if (!word || word.from === word.to && !context.explicit) return null
      return { from: word.from, validFor: /^\w*$/, options: props.completions.filter(c => c.label.startsWith(word.text)).sort((a,b) => ({factor:0,field:1,operator:2}[a.kind] - {factor:0,field:1,operator:2}[b.kind])).map(c => ({ label: c.label, type: c.kind, detail: c.hint, boost: {factor:99,field:40,operator:0}[c.kind] })) }
    }], addToOptions: [{ position: 10, render: completion => {
      const node = document.createElement('span'); node.className = 'completion-icon'
      const app = createApp({ render: () => h(completion.type === 'operator' ? FunctionSquare : completion.type === 'field' ? Database : Library, { size: 14, strokeWidth: 1.6 }) })
      app.mount(node)
      const icon = node.firstChild!.cloneNode(true); app.unmount(); node.append(icon)
      return node
    } }] }),
    EditorView.updateListener.of(update => {
      if (update.docChanged) emit('update:modelValue', update.state.doc.toString())
      if (update.selectionSet || update.docChanged) signature()
    }),
    EditorView.theme({ '&': { height: '100%', fontSize: '13px' }, '.cm-content': { fontFamily: 'var(--mono)', padding: '10px 0', lineHeight: '22px' }, '.cm-gutters': { background: '#fff', color: '#8A9199', border: 'none', minWidth: '36px' }, '.cm-activeLineGutter': { background: 'transparent' }, '.cm-scroller': { overflow: 'auto' }, '&.cm-focused': { outline: 'none' } }),
  ] }) })
  signature(); diagnostic()
})
watch(() => props.modelValue, value => { if (editor && value !== editor.state.doc.toString()) editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } }) })
watch(() => props.error, diagnostic)
watch(() => props.completions, signature)
onUnmounted(() => editor?.destroy())
</script>
<template><div ref="mount" class="expression-editor" /></template>
<style scoped>.expression-editor{min-height:0;flex:1;overflow:hidden}</style>
