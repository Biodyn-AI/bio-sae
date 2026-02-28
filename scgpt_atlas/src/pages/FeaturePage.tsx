import { useMemo, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useLayerFeatures, useLayerAnnotations, useCausalPatching, usePerturbationResponse, useLayerCellTypes } from '../hooks/useData'
import { moduleColor, ontologyColor, ONTOLOGY_LABELS, tissueColor } from '../lib/colors'
import { fmtPval, fmtK } from '../lib/utils'
import type { FeatureCompact, FeatureAnnotations, AnnotationEntry, CausalFeature, PerturbationTarget, FeatureCellTypes } from '../lib/types'

const ONTOLOGY_ORDER = ['GO_BP', 'KEGG', 'Reactome', 'STRING_edges', 'TRRUST_TF_enrichment', 'TRRUST_edges']

export default function FeaturePage() {
  const { layerId, featureId } = useParams<{ layerId: string; featureId: string }>()
  const layer = Number(layerId ?? 0)
  const idx = Number(featureId ?? 0)

  const { data: features, loading: featLoading, error: featError } = useLayerFeatures(layer)
  const { data: annotations, loading: annLoading } = useLayerAnnotations(layer)
  const { data: causalData } = useCausalPatching()
  const { data: pertData } = usePerturbationResponse()
  const { data: cellTypeData } = useLayerCellTypes(layer)

  const feature: FeatureCompact | null = useMemo(() => {
    if (!features) return null
    return features.find(f => f.i === idx) ?? null
  }, [features, idx])

  const fullAnnotations: FeatureAnnotations | null = useMemo(() => {
    if (!annotations) return null
    return annotations[String(idx)] ?? null
  }, [annotations, idx])

  const causalFeature: CausalFeature | null = useMemo(() => {
    if (layer !== 11 || !causalData) return null
    return causalData.features.find(f => f.i === idx) ?? null
  }, [layer, causalData, idx])

  const featureCellTypes: FeatureCellTypes | null = useMemo(() => {
    if (!cellTypeData) return null
    return cellTypeData.features[String(idx)] ?? null
  }, [cellTypeData, idx])

  const matchingPertTargets: { target: PerturbationTarget; es: number; lb: string }[] = useMemo(() => {
    if (layer !== 11 || !pertData) return []
    const matches: { target: PerturbationTarget; es: number; lb: string }[] = []
    for (const t of pertData.targets) {
      const hit = t.top.find(entry => entry.i === idx)
      if (hit) {
        matches.push({ target: t, es: hit.es, lb: hit.lb })
      }
    }
    return matches
  }, [layer, pertData, idx])

  const loading = featLoading || annLoading

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading feature L{layer}:F{idx}...</div>
      </div>
    )
  }

  if (featError) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <div className="text-red-400">{featError}</div>
      </div>
    )
  }

  if (!feature) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <div className="text-gray-400">Feature F{idx} not found in layer {layer}.</div>
        <Link to={`/layer/${layer}`} className="text-blue-400 hover:text-blue-300 text-sm mt-2 inline-block">
          Back to Layer {layer}
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <Link
          to={`/layer/${layer}`}
          className="text-blue-400 hover:text-blue-300 text-sm inline-flex items-center gap-1 mb-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Layer {layer} Explorer
        </Link>
        <h1 className="text-2xl font-bold text-gray-100 font-mono">
          L{layer}:F{idx}
        </h1>
        {feature.lb ? (
          <p className="text-lg text-blue-300 mt-1">{feature.lb}</p>
        ) : (
          <p className="text-lg text-gray-500 italic mt-1">Unannotated feature</p>
        )}
      </div>

      {/* Metadata Grid */}
      <MetadataGrid feature={feature} layer={layer} />

      {/* Top Genes Bar Chart */}
      {fullAnnotations && fullAnnotations.genes.length > 0 && (
        <TopGenesChart genes={fullAnnotations.genes} />
      )}

      {/* Annotations by Ontology */}
      {fullAnnotations && fullAnnotations.anns.length > 0 && (
        <AnnotationsSection annotations={fullAnnotations.anns} />
      )}

      {/* Cell Type & Tissue Enrichment (all layers) */}
      {featureCellTypes && (featureCellTypes.ct.length > 0 || featureCellTypes.ti.length > 0) && (
        <CellTypeEnrichmentSection data={featureCellTypes} />
      )}

      {/* Causal Patching */}
      {causalFeature && (
        <CausalPatchingSection causal={causalFeature} />
      )}

      {/* Perturbation Response */}
      {matchingPertTargets.length > 0 && (
        <PerturbationSection matches={matchingPertTargets} featureIdx={idx} />
      )}
    </div>
  )
}

/* ------- Sub-components ------- */

function MetadataGrid({ feature }: { feature: FeatureCompact; layer?: number }) {
  const f = feature
  const cells: { label: string; value: React.ReactNode }[] = [
    {
      label: 'Activation Freq',
      value: <span className="font-mono">{f.f.toFixed(4)}</span>,
    },
    {
      label: 'Mean Activation',
      value: <span className="font-mono">{f.ma.toFixed(3)}</span>,
    },
    {
      label: 'Fire Count',
      value: <span className="font-mono">{fmtK(f.fc)}</span>,
    },
    {
      label: 'Module',
      value: f.m >= 0 ? (
        <span className="inline-flex items-center gap-1.5">
          <span
            className="w-3 h-3 rounded-full inline-block"
            style={{ backgroundColor: moduleColor(f.m) }}
          />
          <span className="font-mono">{f.m}</span>
        </span>
      ) : (
        <span className="text-gray-500 italic">Unassigned</span>
      ),
    },
    {
      label: 'Annotations',
      value: <span className="font-mono">{f.na}</span>,
    },
    {
      label: 'SVD Status',
      value: f.sv ? (
        <span className="text-amber-400">SVD-aligned</span>
      ) : (
        <span className="text-blue-400">Novel (non-SVD)</span>
      ),
    },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {cells.map(cell => (
        <div key={cell.label} className="bg-gray-900 rounded-lg border border-gray-800 p-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">{cell.label}</div>
          <div className="text-gray-200 text-sm">{cell.value}</div>
        </div>
      ))}
    </div>
  )
}

function TopGenesChart({ genes }: { genes: { n: string; a: number; fc: number }[] }) {
  const sorted = [...genes].sort((a, b) => b.a - a.a)
  const names = sorted.map(g => g.n)
  const activations = sorted.map(g => g.a)

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-1">Gene Effects</h2>
      <p className="text-xs text-gray-600 mb-3">
        Top genes by mean activation — higher values indicate stronger feature association with that gene's expression.
      </p>
      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            y: names.reverse(),
            x: activations.reverse(),
            marker: {
              color: '#3b82f6',
              opacity: 0.8,
            },
            hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
          },
        ]}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: { l: 80, r: 20, t: 10, b: 30 },
          xaxis: {
            title: { text: 'Mean Activation', font: { color: '#9ca3af', size: 11 } },
            color: '#6b7280',
            gridcolor: '#1f2937',
          },
          yaxis: {
            color: '#d1d5db',
            tickfont: { family: 'monospace', size: 11 },
          },
          height: Math.max(200, sorted.length * 22 + 60),
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

function AnnotationsSection({ annotations }: { annotations: AnnotationEntry[] }) {
  // Group by ontology
  const grouped = useMemo(() => {
    const groups: Record<string, AnnotationEntry[]> = {}
    for (const a of annotations) {
      const key = a.o
      if (!groups[key]) groups[key] = []
      groups[key].push(a)
    }
    // Sort each group by p-value
    for (const key of Object.keys(groups)) {
      groups[key].sort((a, b) => a.p - b.p)
    }
    return groups
  }, [annotations])

  // Display in canonical order, then any remaining keys
  const orderedKeys = useMemo(() => {
    const seen = new Set<string>()
    const result: string[] = []
    for (const k of ONTOLOGY_ORDER) {
      if (grouped[k]) {
        result.push(k)
        seen.add(k)
      }
    }
    for (const k of Object.keys(grouped)) {
      if (!seen.has(k)) result.push(k)
    }
    return result
  }, [grouped])

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Annotations</h2>
      {orderedKeys.map(ontKey => (
        <OntologyGroup
          key={ontKey}
          ontology={ontKey}
          entries={grouped[ontKey]}
        />
      ))}
    </div>
  )
}

function OntologyGroup({ ontology, entries }: { ontology: string; entries: AnnotationEntry[] }) {
  const [expanded, setExpanded] = useState(true)
  const color = ontologyColor(ontology)
  const label = ONTOLOGY_LABELS[ontology] || ontology

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-gray-800/50 transition-colors"
        style={{ borderLeft: `3px solid ${color}` }}
      >
        <span className="text-sm font-medium" style={{ color }}>
          {label}
          <span className="text-gray-500 ml-2 font-normal">({entries.length})</span>
        </span>
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && (
        <div className="divide-y divide-gray-800/50">
          {entries.map((entry, j) => (
            <div key={`${entry.t}-${j}`} className="px-4 py-2.5 text-sm">
              <div className="flex items-start justify-between gap-3">
                <span className="text-gray-200 flex-1">{entry.t}</span>
                <div className="flex items-center gap-3 flex-shrink-0 text-xs">
                  <span className="text-gray-500">
                    p={fmtPval(entry.p)}
                  </span>
                  <span className="text-gray-500">
                    OR={entry.or.toFixed(1)}
                  </span>
                </div>
              </div>
              {entry.g.length > 0 && (
                <div className="mt-1 text-xs text-gray-500">
                  <span className="text-gray-600">Genes: </span>
                  {entry.g.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function CausalPatchingSection({ causal }: { causal: CausalFeature }) {
  const maxAbs = Math.max(Math.abs(causal.td), Math.abs(causal.od), 0.01)

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-1">
        Logit Effects (Causal Patching)
        <span className="text-xs text-gray-500 font-normal ml-2">Layer 11</span>
      </h2>
      <p className="text-xs text-gray-600 mb-4">
        Measured change in gene prediction logits when this feature is activation-patched.
      </p>

      {/* Visual bar chart */}
      <div className="space-y-3 mb-4">
        <LogitBar
          label="Target genes"
          value={causal.td}
          maxAbs={maxAbs}
          color={causal.td < 0 ? '#ef4444' : '#22c55e'}
        />
        <LogitBar
          label="Other genes"
          value={causal.od}
          maxAbs={maxAbs}
          color={causal.od < 0 ? '#f87171' : '#86efac'}
        />
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 bg-gray-800/30 rounded-lg p-3">
        <div>
          <div className="text-xs text-gray-500 mb-1">Specificity Ratio</div>
          <div className={`text-lg font-mono ${causal.sr > 2 ? 'text-green-400' : causal.sr > 1 ? 'text-yellow-400' : 'text-gray-300'}`}>
            {causal.sr.toFixed(2)}
          </div>
          <div className="text-xs text-gray-600">
            {causal.sr > 2 ? 'Specific' : causal.sr > 1 ? 'Moderate' : 'Non-specific'}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Target Logit Diff</div>
          <div className={`text-lg font-mono ${causal.td < 0 ? 'text-red-400' : 'text-green-400'}`}>
            {causal.td > 0 ? '+' : ''}{causal.td.toFixed(3)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Other Logit Diff</div>
          <div className={`text-lg font-mono ${causal.od < 0 ? 'text-red-400' : causal.od > 0 ? 'text-green-400' : 'text-gray-400'}`}>
            {causal.od > 0 ? '+' : ''}{causal.od.toFixed(3)}
          </div>
        </div>
      </div>

      {causal.tg.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-500 mb-1.5">Target genes affected:</div>
          <div className="flex flex-wrap gap-1.5">
            {causal.tg.map(g => (
              <span
                key={g}
                className="inline-block px-2 py-0.5 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-gray-300"
              >
                {g}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function LogitBar({
  label,
  value,
  maxAbs,
  color,
}: {
  label: string
  value: number
  maxAbs: number
  color: string
}) {
  const pct = Math.abs(value) / maxAbs * 50
  const isNeg = value < 0
  return (
    <div className="flex items-center gap-3">
      <div className="text-xs text-gray-400 w-24 text-right flex-shrink-0">{label}</div>
      <div className="flex-1 h-6 relative bg-gray-800/50 rounded overflow-hidden">
        {/* Center line */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-gray-600" />
        {/* Bar */}
        <div
          className="absolute inset-y-0 rounded"
          style={{
            backgroundColor: color,
            opacity: 0.7,
            ...(isNeg
              ? { right: '50%', width: `${pct}%` }
              : { left: '50%', width: `${pct}%` }),
          }}
        />
        {/* Value label */}
        <div
          className="absolute inset-y-0 flex items-center text-xs font-mono text-gray-200"
          style={isNeg ? { right: `${50 + pct + 1}%` } : { left: `${50 + pct + 1}%` }}
        >
          {value > 0 ? '+' : ''}{value.toFixed(3)}
        </div>
      </div>
    </div>
  )
}

function PerturbationSection({
  matches,
  featureIdx,
}: {
  matches: { target: PerturbationTarget; es: number; lb: string }[]
  featureIdx: number
}) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
        Perturbation Response
        <span className="text-xs text-gray-500 font-normal ml-2">(Layer 11)</span>
      </h2>
      <p className="text-xs text-gray-500 mb-3">
        Feature F{featureIdx} appears in the top-responding features for the following perturbation targets:
      </p>
      <div className="space-y-2">
        {matches.map(({ target, es, lb }) => (
          <div
            key={target.gene}
            className="flex items-center justify-between bg-gray-800/50 rounded px-3 py-2"
          >
            <div>
              <span className="text-sm text-gray-200 font-mono">{target.gene}</span>
              {target.tf && (
                <span className="ml-2 text-xs text-amber-400/80 bg-amber-400/10 px-1.5 py-0.5 rounded">
                  TF
                </span>
              )}
              {lb && (
                <span className="ml-2 text-xs text-gray-500">{lb}</span>
              )}
            </div>
            <div className="text-right">
              <div className={`text-sm font-mono ${es > 0 ? 'text-green-400' : es < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                ES={es.toFixed(3)}
              </div>
              <div className="text-xs text-gray-600">
                {fmtK(target.nk)} KD / {fmtK(target.nr)} resp
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function CellTypeEnrichmentSection({ data }: { data: FeatureCellTypes }) {
  const [showAll, setShowAll] = useState(false)
  const visibleCt = showAll ? data.ct : data.ct.slice(0, 5)
  const maxOr = Math.max(...data.ct.map(c => c.or), 1)

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-1">
          Cell Type & Tissue Enrichment
          <span className="text-xs text-gray-500 font-normal ml-2">Tabula Sapiens (3,000 cells)</span>
        </h2>
        <p className="text-xs text-gray-600">
          Enrichment of cell types and tissues among the top 10% activating cells for this feature.
        </p>
      </div>

      {/* Tissue enrichment */}
      {data.ti.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Tissue Enrichment</div>
          <div className="grid grid-cols-3 gap-2">
            {['immune', 'kidney', 'lung'].map(tissue => {
              const match = data.ti.find(t => t.t === tissue)
              return (
                <div
                  key={tissue}
                  className="rounded-lg px-2 py-2.5 text-center overflow-hidden"
                  style={{
                    backgroundColor: match
                      ? tissueColor(tissue) + '18'
                      : '#1f293710',
                    borderLeft: `3px solid ${match ? tissueColor(tissue) : '#374151'}`,
                  }}
                >
                  <div className="text-xs capitalize truncate" style={{ color: match ? tissueColor(tissue) : '#6b7280' }}>
                    {tissue}
                  </div>
                  {match ? (
                    <>
                      <div className="text-base font-mono text-gray-200 mt-0.5 leading-tight">
                        {match.or.toFixed(1)}x
                      </div>
                      <div className="text-[10px] text-gray-500 truncate leading-tight mt-0.5">
                        p={fmtPval(match.p)}
                      </div>
                    </>
                  ) : (
                    <div className="text-sm text-gray-600 mt-1 italic">n.s.</div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Cell type enrichment */}
      {data.ct.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">
            Enriched Cell Types
            <span className="text-gray-600 ml-1">({data.ct.length})</span>
          </div>
          <div className="space-y-1.5">
            {visibleCt.map((ct, j) => (
              <div key={`${ct.c}-${j}`} className="bg-gray-800/40 rounded px-3 py-2">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="text-sm text-gray-200 truncate">{ct.c}</span>
                  <span className="text-xs text-gray-600 flex-shrink-0">n={ct.n}</span>
                </div>
                <div className="flex items-center gap-3">
                  {/* Odds ratio bar */}
                  <div className="flex-1 h-3 bg-gray-800 rounded overflow-hidden">
                    <div
                      className="h-full rounded"
                      style={{
                        width: `${Math.min(ct.or / maxOr * 100, 100)}%`,
                        backgroundColor: '#60a5fa',
                        opacity: 0.6,
                      }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-400 flex-shrink-0">
                    {ct.or.toFixed(1)}x
                  </span>
                  <span className="text-xs text-gray-500 flex-shrink-0">
                    p={fmtPval(ct.p)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          {data.ct.length > 5 && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="text-xs text-blue-400 hover:text-blue-300 mt-2"
            >
              {showAll ? 'Show fewer' : `Show all ${data.ct.length}`}
            </button>
          )}
        </div>
      )}

      {/* Top activating cells */}
      {data.tc.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Top Activating Cells</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
            {data.tc.slice(0, 5).map((cell, j) => (
              <div key={j} className="flex items-center gap-2 bg-gray-800/30 rounded px-3 py-1.5 text-sm">
                <span
                  className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: tissueColor(cell.t) }}
                />
                <span className="text-gray-300 truncate flex-1">{cell.ct}</span>
                <span className="text-xs text-gray-500 capitalize flex-shrink-0">{cell.t}</span>
                <span className="text-xs font-mono text-gray-400 flex-shrink-0">{cell.a.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
