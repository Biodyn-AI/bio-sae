export function fmtPct(v: number): string {
  return (v * 100).toFixed(1) + '%'
}

export function fmtK(v: number): string {
  if (v >= 1000) return (v / 1000).toFixed(1) + 'K'
  return v.toString()
}

export function fmtPval(p: number): string {
  if (p < 1e-10) return '< 1e-10'
  if (p < 0.001) return p.toExponential(1)
  return p.toFixed(4)
}

export function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '\u2026' : s
}

export function featureLabel(f: { lb: string; i: number }, layer?: number): string {
  const prefix = layer !== undefined ? `L${layer}:F${f.i}` : `F${f.i}`
  if (f.lb) return `${prefix} — ${truncate(f.lb, 50)}`
  return `${prefix} (unannotated)`
}
