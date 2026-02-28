import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useOntologyIndex } from '../hooks/useData'
import { ontologyColor, ONTOLOGY_LABELS } from '../lib/colors'
import { fmtPval } from '../lib/utils'
import { InfoIcon } from '../components/Tooltip'
import { useIsMobile } from '../hooks/useIsMobile'

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

export default function OntologySearch() {
  const { data, loading, error } = useOntologyIndex()
  const [query, setQuery] = useState('')
  const [selectedTerm, setSelectedTerm] = useState<string | null>(null)
  const isMobile = useIsMobile()

  // All term names for autocomplete
  const allTerms = useMemo(() => {
    if (!data) return []
    return Object.keys(data).sort()
  }, [data])

  // Autocomplete suggestions
  const suggestions = useMemo(() => {
    if (!query || !data) return []
    const q = query.toLowerCase()
    return allTerms.filter(t => t.toLowerCase().includes(q)).slice(0, 10)
  }, [query, allTerms, data])

  // Results for selected term
  const results = useMemo(() => {
    if (!selectedTerm || !data || !data[selectedTerm]) return null
    const entries = data[selectedTerm].slice().sort((a, b) => a.p - b.p)
    const layers = new Set(entries.map(e => e.l))
    return { entries, nLayers: layers.size }
  }, [selectedTerm, data])

  // Bar chart data: count per layer, colored by ontology source
  const barData = useMemo(() => {
    if (!results) return []
    // Group by ontology source
    const byOntology: Record<string, Map<number, number>> = {}
    for (const e of results.entries) {
      const ont = e.o || 'none'
      if (!byOntology[ont]) byOntology[ont] = new Map()
      const counts = byOntology[ont]
      counts.set(e.l, (counts.get(e.l) || 0) + 1)
    }

    const layers = Array.from({ length: 18 }, (_, i) => i)
    return Object.entries(byOntology).map(([ont, counts]) => ({
      x: layers,
      y: layers.map(l => counts.get(l) || 0),
      type: 'bar' as const,
      name: ONTOLOGY_LABELS[ont] || ont,
      marker: { color: ontologyColor(ont) },
      hovertemplate: `%{x}: %{y} feature(s)<br>${ONTOLOGY_LABELS[ont] || ont}<extra></extra>`,
    }))
  }, [results])

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
          Failed to load ontology index: {error}
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* Search input */}
      <div className="flex flex-col items-center space-y-4">
        <h1 className="text-3xl font-bold text-white tracking-tight flex items-center justify-center gap-2">
          Pathway &amp; Ontology Search
          <InfoIcon tip="Search for GO terms, KEGG pathways, Reactome processes, or other ontology terms to find related SAE features." position="bottom" />
        </h1>
        <div className="w-full max-w-xl relative">
          <input
            type="text"
            value={query}
            onChange={e => {
              setQuery(e.target.value)
              setSelectedTerm(null)
            }}
            placeholder="Search pathway or GO term (e.g. Cell Cycle, Apoptosis, mTOR)..."
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-5 py-3 text-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />

          {/* Autocomplete dropdown */}
          {suggestions.length > 0 && !selectedTerm && (
            <ul className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl overflow-hidden max-h-80 overflow-y-auto">
              {suggestions.map(term => (
                <li key={term}>
                  <button
                    onClick={() => {
                      setSelectedTerm(term)
                      setQuery(term)
                    }}
                    className="w-full text-left px-5 py-2.5 text-gray-200 hover:bg-gray-700 transition-colors truncate"
                  >
                    {term}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Results */}
      {results && selectedTerm && (
        <div className="space-y-6">
          {/* Summary */}
          <p className="text-gray-400 text-center">
            <span className="text-white font-semibold">{selectedTerm}</span> found in{' '}
            <span className="text-green-400 font-semibold">{results.entries.length}</span>{' '}
            feature{results.entries.length !== 1 ? 's' : ''} across{' '}
            <span className="text-green-400 font-semibold">{results.nLayers}</span>{' '}
            layer{results.nLayers !== 1 ? 's' : ''}
          </p>

          {/* Bar chart: features per layer */}
          <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
            <Plot
              data={barData}
              layout={{
                ...PLOTLY_LAYOUT_BASE,
                height: isMobile ? 260 : 340,
                barmode: 'stack',
                title: {
                  text: `${selectedTerm} - features per layer`,
                  font: { color: '#e5e7eb', size: 14 },
                },
                xaxis: {
                  ...PLOTLY_LAYOUT_BASE.xaxis,
                  title: { text: 'Layer', standoff: 10 },
                  dtick: 1,
                },
                yaxis: {
                  ...PLOTLY_LAYOUT_BASE.yaxis,
                  title: { text: 'Feature Count', standoff: 10 },
                },
                legend: {
                  font: { color: '#9ca3af' },
                  bgcolor: 'transparent',
                },
                showlegend: barData.length > 1,
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
                    <th className="pb-2 pr-4">p-value</th>
                    <th className="pb-2">Ontology</th>
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
                      <td className="py-2 pr-4 text-gray-400 font-mono text-xs">
                        {fmtPval(e.p)}
                      </td>
                      <td className="py-2">
                        <span className="flex items-center gap-1.5">
                          <span
                            className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                            style={{ backgroundColor: ontologyColor(e.o) }}
                          />
                          <span className="text-gray-400 text-xs">
                            {ONTOLOGY_LABELS[e.o] || e.o}
                          </span>
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
