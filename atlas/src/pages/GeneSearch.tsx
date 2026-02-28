import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useGeneIndex } from '../hooks/useData'
import { moduleColor } from '../lib/colors'

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

export default function GeneSearch() {
  const { data, loading, error } = useGeneIndex()
  const [query, setQuery] = useState('')
  const [selectedGene, setSelectedGene] = useState<string | null>(null)

  // All gene names for autocomplete
  const allGenes = useMemo(() => {
    if (!data) return []
    return Object.keys(data).sort()
  }, [data])

  // Autocomplete suggestions
  const suggestions = useMemo(() => {
    if (!query || !data) return []
    const q = query.toUpperCase()
    return allGenes.filter(g => g.toUpperCase().includes(q)).slice(0, 10)
  }, [query, allGenes, data])

  // Results for selected gene
  const results = useMemo(() => {
    if (!selectedGene || !data || !data[selectedGene]) return null
    const entries = data[selectedGene].slice().sort((a, b) => a.l - b.l || a.r - b.r)
    const layers = new Set(entries.map(e => e.l))
    return { entries, nLayers: layers.size }
  }, [selectedGene, data])

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
          Failed to load gene index: {error}
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* Search input */}
      <div className="flex flex-col items-center space-y-4">
        <h1 className="text-3xl font-bold text-white tracking-tight">
          Gene Search
        </h1>
        <div className="w-full max-w-xl relative">
          <input
            type="text"
            value={query}
            onChange={e => {
              setQuery(e.target.value)
              setSelectedGene(null)
            }}
            placeholder="Search gene name (e.g. CDK1, GATA1, TP53)..."
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-5 py-3 text-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />

          {/* Autocomplete dropdown */}
          {suggestions.length > 0 && !selectedGene && (
            <ul className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl overflow-hidden">
              {suggestions.map(gene => (
                <li key={gene}>
                  <button
                    onClick={() => {
                      setSelectedGene(gene)
                      setQuery(gene)
                    }}
                    className="w-full text-left px-5 py-2.5 text-gray-200 hover:bg-gray-700 transition-colors"
                  >
                    {gene}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Results */}
      {results && selectedGene && (
        <div className="space-y-6">
          {/* Summary */}
          <p className="text-gray-400 text-center">
            <span className="text-white font-semibold">{selectedGene}</span> found in{' '}
            <span className="text-blue-400 font-semibold">{results.entries.length}</span>{' '}
            feature{results.entries.length !== 1 ? 's' : ''} across{' '}
            <span className="text-blue-400 font-semibold">{results.nLayers}</span>{' '}
            layer{results.nLayers !== 1 ? 's' : ''}
          </p>

          {/* Scatter: layer vs rank */}
          <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
            <Plot
              data={[
                {
                  x: results.entries.map(e => e.l),
                  y: results.entries.map(e => e.r),
                  text: results.entries.map(e => e.lb || `F${e.i}`),
                  customdata: results.entries.map(e => [e.i, e.m]),
                  type: 'scatter',
                  mode: 'markers',
                  marker: {
                    color: results.entries.map(e => moduleColor(e.m)),
                    size: 12,
                    line: { color: '#1f2937', width: 1 },
                  },
                  hovertemplate:
                    'Layer %{x}, Rank %{y}<br>F%{customdata[0]}<br>%{text}<br>Module %{customdata[1]}<extra></extra>',
                },
              ]}
              layout={{
                ...PLOTLY_LAYOUT_BASE,
                height: 340,
                title: {
                  text: `${selectedGene} across layers`,
                  font: { color: '#e5e7eb', size: 14 },
                },
                xaxis: {
                  ...PLOTLY_LAYOUT_BASE.xaxis,
                  title: { text: 'Layer', standoff: 10 },
                  dtick: 1,
                  range: [-0.5, 17.5],
                },
                yaxis: {
                  ...PLOTLY_LAYOUT_BASE.yaxis,
                  title: { text: 'Rank (0 = top gene)', standoff: 10 },
                  autorange: 'reversed',
                  dtick: 1,
                },
              }}
              config={{ responsive: true, displayModeBar: false }}
              className="w-full"
              useResizeHandler
              style={{ width: '100%' }}
            />
          </section>

          {/* Results table */}
          <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b border-gray-800">
                    <th className="pb-2 pr-4">Layer</th>
                    <th className="pb-2 pr-4">Feature</th>
                    <th className="pb-2 pr-4">Rank</th>
                    <th className="pb-2 pr-4">Label</th>
                    <th className="pb-2">Module</th>
                  </tr>
                </thead>
                <tbody>
                  {results.entries.map((e, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-gray-800/50 hover:bg-gray-800/30"
                    >
                      <td className="py-2 pr-4 text-gray-300">L{e.l}</td>
                      <td className="py-2 pr-4">
                        <Link
                          to={`/feature/${e.l}/${e.i}`}
                          className="text-blue-400 hover:text-blue-300 font-mono text-xs"
                        >
                          F{e.i}
                        </Link>
                      </td>
                      <td className="py-2 pr-4 text-gray-400">{e.r}</td>
                      <td className="py-2 pr-4 text-gray-400 truncate max-w-xs">
                        {e.lb || '\u2014'}
                      </td>
                      <td className="py-2">
                        {e.m >= 0 ? (
                          <span
                            className="inline-block w-3 h-3 rounded-sm mr-1.5"
                            style={{ backgroundColor: moduleColor(e.m) }}
                          />
                        ) : null}
                        <span className="text-gray-400 text-xs">
                          {e.m >= 0 ? e.m : '\u2014'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
