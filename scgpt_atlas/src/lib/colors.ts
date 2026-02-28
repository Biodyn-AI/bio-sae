export const MODULE_COLORS = [
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
  '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
  '#8da0cb', '#e78ac3',
]

export const ONTOLOGY_COLORS: Record<string, string> = {
  GO_BP: '#3b82f6',
  KEGG: '#22c55e',
  Reactome: '#a855f7',
  STRING_edges: '#f97316',
  STRING: '#f97316',
  TRRUST_TF_enrichment: '#ef4444',
  TRRUST_edges: '#ef4444',
  TRRUST: '#ef4444',
  none: '#6b7280',
}

export const ONTOLOGY_LABELS: Record<string, string> = {
  GO_BP: 'GO Biological Process',
  KEGG: 'KEGG Pathway',
  Reactome: 'Reactome',
  STRING_edges: 'STRING PPI',
  STRING: 'STRING PPI',
  TRRUST_TF_enrichment: 'TRRUST TF',
  TRRUST_edges: 'TRRUST Edge',
  TRRUST: 'TRRUST',
  none: 'Unannotated',
}

export function moduleColor(id: number): string {
  if (id < 0) return '#374151'
  return MODULE_COLORS[id % MODULE_COLORS.length]
}

export function ontologyColor(ont: string): string {
  return ONTOLOGY_COLORS[ont] || '#6b7280'
}

export function layerColor(layer: number): string {
  const t = layer / 11
  const r = Math.round(30 + t * 180)
  const g = Math.round(100 - t * 40)
  const b = Math.round(220 - t * 140)
  return `rgb(${r},${g},${b})`
}

export function freqColor(freq: number): string {
  const t = Math.min(freq / 0.03, 1)
  const r = Math.round(20 + t * 235)
  const g = Math.round(20 + (1 - Math.abs(t - 0.5) * 2) * 180)
  const b = Math.round(80 + (1 - t) * 175)
  return `rgb(${r},${g},${b})`
}

export function richColor(n: number): string {
  if (n === 0) return '#1f2937'
  const t = Math.min(n / 50, 1)
  const r = Math.round(30 + t * 200)
  const g = Math.round(50 + (1 - t) * 150)
  const b = Math.round(120 - t * 80)
  return `rgb(${r},${g},${b})`
}

export const TISSUE_COLORS: Record<string, string> = {
  immune: '#e41a1c',
  kidney: '#377eb8',
  lung: '#4daf4a',
}

export function tissueColor(tissue: string): string {
  return TISSUE_COLORS[tissue] || '#6b7280'
}
