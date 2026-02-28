import { useState, useMemo } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useLayerFeatures, useLayerPositions, useGlobalSummary } from '../hooks/useData'
import { moduleColor, ontologyColor, freqColor, richColor } from '../lib/colors'
import { fmtPct, fmtK } from '../lib/utils'
import type { FeatureCompact } from '../lib/types'

type ColorBy = 'Module' | 'Annotations' | 'Activation Freq' | 'Ontology' | 'SVD Alignment'
const COLOR_OPTIONS: ColorBy[] = ['Module', 'Annotations', 'Activation Freq', 'Ontology', 'SVD Alignment']
const LAYERS = Array.from({ length: 12 }, (_, i) => i)

function getColor(f: FeatureCompact, colorBy: ColorBy): string {
  switch (colorBy) {
    case 'Module':
      return moduleColor(f.m)
    case 'Annotations':
      return richColor(f.na)
    case 'Activation Freq':
      return freqColor(f.f)
    case 'Ontology':
      return ontologyColor(f.to)
    case 'SVD Alignment':
      return f.sv ? '#f59e0b' : '#3b82f6'
  }
}

export default function LayerExplorer() {
  const { layerId } = useParams<{ layerId: string }>()
  const navigate = useNavigate()
  const layer = Number(layerId ?? 0)

  const { data: features, loading: featLoading, error: featError } = useLayerFeatures(layer)
  const { data: positions, loading: posLoading } = useLayerPositions(layer)
  const { data: globalSummary } = useGlobalSummary()

  const [colorBy, setColorBy] = useState<ColorBy>('Module')
  const [geneSearch, setGeneSearch] = useState('')
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)

  const layerStats = useMemo(() => {
    if (!globalSummary) return null
    return globalSummary.layers.find(l => l.layer === layer) ?? null
  }, [globalSummary, layer])

  const filteredFeatures = useMemo(() => {
    if (!features) return []
    const q = geneSearch.toLowerCase().trim()
    const filtered = q
      ? features.filter(f => f.tg.some(g => g.n.toLowerCase().includes(q)))
      : features
    return [...filtered].sort((a, b) => b.na - a.na)
  }, [features, geneSearch])

  const selectedFeature = useMemo(() => {
    if (selectedIdx === null || !features) return null
    return features.find(f => f.i === selectedIdx) ?? null
  }, [features, selectedIdx])

  // Build Plotly data
  const plotData = useMemo(() => {
    if (!features || !positions) return null
    const featureSet = new Set(filteredFeatures.map(f => f.i))
    const x: number[] = []
    const y: number[] = []
    const colors: string[] = []
    const texts: string[] = []
    const ids: number[] = []

    for (const f of features) {
      if (!featureSet.has(f.i)) continue
      const pos = positions[f.i]
      if (!pos) continue
      x.push(pos[0])
      y.push(pos[1])
      colors.push(getColor(f, colorBy))
      texts.push(`F${f.i}: ${f.lb || 'unannotated'}\n${f.tg[0]?.n || ''}`)
      ids.push(f.i)
    }

    return { x, y, colors, texts, ids }
  }, [features, positions, filteredFeatures, colorBy])

  const loading = featLoading || posLoading

  return (
    <div className="flex h-[calc(100vh-3rem)]">
      {/* Left Sidebar */}
      <div className="w-[250px] flex-shrink-0 border-r border-gray-800 bg-gray-900 flex flex-col overflow-hidden">
        {/* Layer Selector */}
        <div className="p-3 border-b border-gray-800">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Layer</h3>
          <div className="flex flex-wrap gap-1">
            {LAYERS.map(l => (
              <button
                key={l}
                onClick={() => {
                  setSelectedIdx(null)
                  navigate(`/layer/${l}`)
                }}
                className={`w-8 h-7 text-xs rounded font-mono transition-colors ${
                  l === layer
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        {/* Layer Stats */}
        {layerStats && (
          <div className="p-3 border-b border-gray-800">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Layer {layer} Stats</h3>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
              <span className="text-gray-500">Alive</span>
              <span className="text-gray-200 text-right">{fmtK(layerStats.alive)}</span>
              <span className="text-gray-500">Dead</span>
              <span className="text-gray-200 text-right">{fmtK(layerStats.dead)}</span>
              <span className="text-gray-500">Annotated</span>
              <span className="text-gray-200 text-right">{fmtK(layerStats.annotated)}</span>
              <span className="text-gray-500">Annotation Rate</span>
              <span className="text-gray-200 text-right">{fmtPct(layerStats.annotation_rate)}</span>
              <span className="text-gray-500">Modules</span>
              <span className="text-gray-200 text-right">{layerStats.n_modules}</span>
              <span className="text-gray-500">SVD Aligned</span>
              <span className="text-gray-200 text-right">{fmtK(layerStats.n_svd_aligned)}</span>
            </div>
          </div>
        )}

        {/* Color-by Dropdown */}
        <div className="p-3 border-b border-gray-800">
          <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider block mb-1">
            Color by
          </label>
          <select
            value={colorBy}
            onChange={e => setColorBy(e.target.value as ColorBy)}
            className="w-full bg-gray-800 text-gray-200 text-sm rounded px-2 py-1.5 border border-gray-700 focus:border-blue-500 focus:outline-none"
          >
            {COLOR_OPTIONS.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* Gene Search Filter */}
        <div className="p-3 border-b border-gray-800">
          <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider block mb-1">
            Filter by gene
          </label>
          <input
            type="text"
            placeholder="e.g. TP53, MYC..."
            value={geneSearch}
            onChange={e => setGeneSearch(e.target.value)}
            className="w-full bg-gray-800 text-gray-200 text-sm rounded px-2 py-1.5 border border-gray-700 focus:border-blue-500 focus:outline-none placeholder-gray-600"
          />
          {geneSearch && (
            <p className="text-xs text-gray-500 mt-1">
              {filteredFeatures.length} feature{filteredFeatures.length !== 1 ? 's' : ''} match
            </p>
          )}
        </div>

        {/* Feature List */}
        <div className="flex-1 overflow-y-auto p-1">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="text-gray-500 text-sm">Loading layer {layer}...</div>
            </div>
          )}
          {featError && (
            <div className="p-3 text-red-400 text-sm">{featError}</div>
          )}
          {!loading && filteredFeatures.map(f => (
            <button
              key={f.i}
              onClick={() => setSelectedIdx(f.i)}
              className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors ${
                selectedIdx === f.i
                  ? 'bg-blue-600/20 text-blue-200'
                  : 'text-gray-300 hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center justify-between gap-1">
                <span className="font-mono text-gray-400 flex-shrink-0">F{f.i}</span>
                <span className="truncate flex-1 mx-1">
                  {f.tg[0]?.n || <span className="text-gray-600 italic">--</span>}
                </span>
                <span className={`flex-shrink-0 ${f.na > 0 ? 'text-blue-400' : 'text-gray-600'}`}>
                  {f.na}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Center: Scatter Plot */}
      <div className="flex-1 relative bg-gray-950">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="bg-gray-900/80 rounded-lg px-6 py-4 text-gray-400">
              Loading layer {layer} features...
            </div>
          </div>
        )}
        {plotData && (
          <Plot
            data={[
              {
                type: 'scattergl',
                mode: 'markers',
                x: plotData.x,
                y: plotData.y,
                marker: {
                  color: plotData.colors,
                  size: 4,
                  opacity: 0.7,
                },
                text: plotData.texts,
                hoverinfo: 'text',
                customdata: plotData.ids,
              },
            ]}
            layout={{
              xaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                title: '',
              },
              yaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                title: '',
              },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { l: 10, r: 10, t: 10, b: 10 },
              dragmode: 'pan',
              hovermode: 'closest',
              showlegend: false,
            }}
            config={{
              displayModeBar: true,
              scrollZoom: true,
              responsive: true,
            }}
            style={{ width: '100%', height: '100%' }}
            onClick={(event: unknown) => {
              const e = event as { points?: Array<{ customdata?: unknown }> }
              const point = e.points?.[0]
              if (point?.customdata !== undefined) {
                setSelectedIdx(point.customdata as number)
              }
            }}
            useResizeHandler
          />
        )}
        {/* Layer badge overlay */}
        <div className="absolute top-3 left-3 bg-gray-900/80 backdrop-blur-sm rounded px-3 py-1.5 text-sm font-mono text-gray-300 border border-gray-700">
          Layer {layer} &middot; {filteredFeatures.length} features
        </div>
      </div>

      {/* Right Detail Panel */}
      <div className="w-[300px] flex-shrink-0 border-l border-gray-800 bg-gray-900 overflow-y-auto">
        {selectedFeature ? (
          <FeatureDetailPanel feature={selectedFeature} layer={layer} />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm px-4 text-center">
            Click a point in the scatter plot or select a feature from the sidebar to see details.
          </div>
        )}
      </div>
    </div>
  )
}

function FeatureDetailPanel({ feature, layer }: { feature: FeatureCompact; layer: number }) {
  const f = feature
  const topGenes = f.tg.slice(0, 5)
  const maxActivation = topGenes.length > 0 ? Math.max(...topGenes.map(g => g.a)) : 1

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-gray-100 font-mono">
          L{layer}:F{f.i}
        </h2>
        {f.lb ? (
          <p className="text-sm text-blue-300 mt-0.5">{f.lb}</p>
        ) : (
          <p className="text-sm text-gray-500 italic mt-0.5">Unannotated</p>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <div className="text-gray-500">Activation Freq</div>
        <div className="text-gray-200 text-right font-mono">{f.f.toFixed(4)}</div>

        <div className="text-gray-500">Fire Count</div>
        <div className="text-gray-200 text-right font-mono">{fmtK(f.fc)}</div>

        <div className="text-gray-500">Module</div>
        <div className="text-right">
          {f.m >= 0 ? (
            <span className="inline-flex items-center gap-1">
              <span
                className="w-2.5 h-2.5 rounded-full inline-block"
                style={{ backgroundColor: moduleColor(f.m) }}
              />
              <span className="text-gray-200 font-mono">{f.m}</span>
            </span>
          ) : (
            <span className="text-gray-500 italic">none</span>
          )}
        </div>

        <div className="text-gray-500">Annotations</div>
        <div className="text-gray-200 text-right font-mono">{f.na}</div>

        <div className="text-gray-500">SVD Aligned</div>
        <div className="text-right">
          {f.sv ? (
            <span className="text-amber-400">Yes</span>
          ) : (
            <span className="text-blue-400">No (Novel)</span>
          )}
        </div>

        <div className="text-gray-500">Status</div>
        <div className="text-right">
          {f.d ? (
            <span className="text-red-400">Dead</span>
          ) : (
            <span className="text-green-400">Alive</span>
          )}
        </div>
      </div>

      {/* Top Genes */}
      {topGenes.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Top Genes
          </h3>
          <div className="space-y-1.5">
            {topGenes.map(g => (
              <div key={g.n} className="flex items-center gap-2">
                <span className="text-xs text-gray-300 w-14 text-right font-mono flex-shrink-0 truncate">
                  {g.n}
                </span>
                <div className="flex-1 bg-gray-800 rounded h-4 overflow-hidden">
                  <div
                    className="h-full bg-blue-500/60 rounded"
                    style={{ width: `${(g.a / maxActivation) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 font-mono w-10 text-right flex-shrink-0">
                  {g.a.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top Ontology */}
      {f.to && f.to !== 'none' && (
        <div>
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
            Top Ontology
          </h3>
          <span
            className="inline-block text-xs px-2 py-0.5 rounded"
            style={{
              backgroundColor: ontologyColor(f.to) + '22',
              color: ontologyColor(f.to),
              border: `1px solid ${ontologyColor(f.to)}44`,
            }}
          >
            {f.to}
          </span>
        </div>
      )}

      {/* Link to Full Feature Page */}
      <Link
        to={`/feature/${layer}/${f.i}`}
        className="block w-full text-center text-sm bg-blue-600/20 text-blue-300 hover:bg-blue-600/30 rounded px-3 py-2 transition-colors border border-blue-600/30"
      >
        View full feature details
      </Link>
    </div>
  )
}
