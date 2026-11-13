<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps<{ source: string; flat?: boolean }>()
const tokens = computed(() => {
  const source = props.flat ? props.source.split('\n').filter(line => !line.trim().startsWith('#')).join('； ') : props.source
  const pattern = /#[^\n]*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b/g
  const parts: { text: string; kind: string }[] = []; let previous = 0
  for (const match of source.matchAll(pattern)) {
    const offset = match.index!; if (offset > previous) parts.push({ text: source.slice(previous, offset), kind: '' })
    const text = match[0], after = source.slice(offset + text.length)
    const assigned = /^\s*=(?!=)/.test(after)
    parts.push({ text, kind: text.startsWith('#') ? 'comment' : /^\d/.test(text) ? 'number-token' : /^\s*\(/.test(after) ? 'function-token' : assigned ? '' : 'field-token' }); previous = offset + text.length
  }
  if (previous < source.length) parts.push({ text: source.slice(previous), kind: '' })
  return parts
})
</script>
<template><code class="syntax-text"><span v-for="(token,index) in tokens" :key="index" :class="token.kind">{{ token.text }}</span></code></template>
<style scoped>.syntax-text{font-family:var(--mono);font-size:inherit}.function-token{font-weight:600;color:var(--ink)}.field-token{color:var(--accent)}.number-token{color:#A14A14}.comment{color:var(--ink3)}</style>
