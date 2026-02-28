import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { useCrossLayerGraph, useCrossLayerTracking } from '../hooks/useData'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const PLOTLY_LAYOUT_BASE: any = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#d1d5db' },
  margin: { t: 40, r: 30, b: 50, l: 60 },
  xaxis: {
    gridcolor: '#374151',
    zerolinecolor: '#4b5563',
    tickcolor: '#6b7280',
  },
  yaxis: {
    gridcolor: '#374151',
    zerolinecolor: '#4b5563',
    tickcolor: '#6b7280',
  },
}

const PAIR_KEYS = ['L00_L04', 'L04_L08', 'L08_L11'] as const
const PAIR_LABELS: Record<string, string> = {
  L00_L04: 'L0 \u2192 L4',
  L04_L08: 'L4 \u2192 L8',
  L08_L11: 'L8 \u2192 L11',
}

export default function CrossLayerFlow() {
  const { data: graphData, loading: gLoading, error: gError } = useCrossLayerGraph()
  const { data: trackingData, loading: tLoading } = useCrossLayerTracking()
  const [selectedPair, setSelectedPair] = useState<string>(PAIR_KEYS[0])

  const loading = gLoading || tLoading
  const error = gError  // tracking data is optional — don't block on its error

  // Build persistence chart data
  const persistenceTrace = useMemo(() => {
    if (!trackingData?.adjacent_persistence) return null
    const sorted = trackingData.adjacent_persistence
      .slice()
      .sort((a, b) => (a.from_layer as number) - (b.from_layer as number))
    return {
      x: sorted.map(
        d => `L${d.from_layer}\u2192L${d.to_layer}`
      ),
      y: sorted.map(d => (d.persistence_rate as number) * 100),
      customdata: sorted.map(d => [d.n_matches, (d.mean_match_cosine as number).toFixed(3)]),
    }
  }, [trackingData])

  // Build top-20 dependencies for selected pair
  const topDeps = useMemo(() => {
    if (!graphData || !graphData[selectedPair]) return []
    const pair = graphData[selectedPair]
    const rows: { a: number; b: number; pmi: number; lb: string }[] = []
    for (const entry of pair.deps) {
      if (!entry.deps || entry.deps.length === 0) continue
      const topDep = entry.deps.reduce((best, d) => (d.pmi > best.pmi ? d : best), entry.deps[0])
      rows.push({ a: entry.a, b: topDep.b, pmi: topDep.pmi, lb: topDep.lb })
    }
    rows.sort((a, b) => b.pmi - a.pmi)
    return rows.slice(0, 20)
  }, [graphData, selectedPair])

  // Summary stats for selected pair
  const summary = graphData?.[selectedPair]?.summary

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-red-400 bg-red-900/20 border border-red-800 rounded-lg px-6 py-4">
          Failed to load cross-layer data: {error}
        </div>
      </div>
    )
  }

  if (!graphData) return null

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <h1 className="text-3xl font-bold text-white tracking-tight">
        Cross-Layer Information Flow
      </h1>

      {/* Persistence chart */}
      {persistenceTrace && (
        <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
          <h2 className="text-lg font-semibold text-gray-200 mb-4">
            Feature Persistence Across Layer Transitions
          </h2>
          <Plot
            data={[
              {
                x: persistenceTrace.x,
                y: persistenceTrace.y,
                customdata: persistenceTrace.customdata,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: '#a78bfa', size: 8 },
                line: { color: '#a78bfa', width: 2 },
                hovertemplate:
                  '%{x}<br>Persistence: %{y:.1f}%<br>Matches: %{customdata[0]}<br>Mean cosine: %{customdata[1]}<extra></extra>',
              },
            ]}
            layout={{
              ...PLOTLY_LAYOUT_BASE,
              height: 320,
              xaxis: {
                ...PLOTLY_LAYOUT_BASE.xaxis,
                title: { text: 'Layer Transition', standoff: 10 },
              },
              yaxis: {
                ...PLOTLY_LAYOUT_BASE.yaxis,
                title: { text: 'Persistence Rate (%)', standoff: 10 },
              },
            }}
            config={{ responsive: true, displayModeBar: false }}
            className="w-full"
            useResizeHandler
            style={{ width: '100%' }}
          />
        </section>
      )}

      {/* Layer pair selector */}
      <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-6">
        <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1 w-fit">
          {PAIR_KEYS.map(pk => (
            <button
              key={pk}
              onClick={() => setSelectedPair(pk)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                selectedPair === pk
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
              }`}
            >
              {PAIR_LABELS[pk]}
            </button>
          ))}
        </div>

        {/* Summary stats */}
        {summary && (
          <div className="flex flex-wrap gap-6">
            {summary.n_pmi_edges != null && (
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {Number(summary.n_pmi_edges).toLocaleString()}
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wider">
                  PMI Edges
                </div>
              </div>
            )}
            {summary.n_features_with_deps != null && (
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {String(summary.n_features_with_deps)}
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wider">
                  Connected Features
                </div>
              </div>
            )}
            {summary.mean_max_pmi != null && (
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {(summary.mean_max_pmi as number).toFixed(2)}
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wider">
                  Mean Max PMI
                </div>
              </div>
            )}
            {summary.upstream_coverage != null && (
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {((summary.upstream_coverage as number) * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wider">
                  Upstream Coverage
                </div>
              </div>
            )}
            {summary.downstream_coverage != null && (
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {((summary.downstream_coverage as number) * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wider">
                  Downstream Coverage
                </div>
              </div>
            )}
          </div>
        )}

        {/* Top dependencies table */}
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Top 20 Dependencies by PMI
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-800">
                  <th className="pb-2 pr-4">#</th>
                  <th className="pb-2 pr-4">Source Feature</th>
                  <th className="pb-2 pr-4">Target Feature</th>
                  <th className="pb-2 pr-4">PMI</th>
                  <th className="pb-2">Target Label</th>
                </tr>
              </thead>
              <tbody>
                {topDeps.map((row, idx) => (
                  <tr
                    key={idx}
                    className="border-b border-gray-800/50 hover:bg-gray-800/30"
                  >
                    <td className="py-2 pr-4 text-gray-600">{idx + 1}</td>
                    <td className="py-2 pr-4 text-gray-300 font-mono text-xs">
                      F{row.a}
                    </td>
                    <td className="py-2 pr-4 text-gray-300 font-mono text-xs">
                      F{row.b}
                    </td>
                    <td className="py-2 pr-4 text-purple-400 font-mono">
                      {row.pmi.toFixed(2)}
                    </td>
                    <td className="py-2 text-gray-400 truncate max-w-xs">
                      {row.lb || '\u2014'}
                    </td>
                  </tr>
                ))}
                {topDeps.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-gray-600">
                      No dependency data available for this pair.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  )
}
