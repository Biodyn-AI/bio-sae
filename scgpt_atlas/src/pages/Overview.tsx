import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useGlobalSummary } from '../hooks/useData'
import { ONTOLOGY_COLORS, ONTOLOGY_LABELS, layerColor } from '../lib/colors'
import { fmtPct, fmtK } from '../lib/utils'

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

const HEATMAP_ONTOLOGIES = ['GO_BP', 'KEGG', 'Reactome', 'STRING', 'TRRUST'] as const

const KEY_FINDINGS = [
  {
    title: 'Layer Explorer',
    description: 'Dive into per-layer feature catalogs with activation patterns, gene loadings, and ontology annotations.',
    to: '/layer/7',
    accent: 'blue',
  },
  {
    title: 'Biological Modules',
    description: 'Co-activating feature clusters that map to coherent biological programs across cell types.',
    to: '/modules',
    accent: 'green',
  },
  {
    title: 'Cross-Layer Information Flow',
    description: 'Track how biological representations transform and persist across the 12-layer residual stream.',
    to: '/flow',
    accent: 'purple',
  },
  {
    title: 'SVD vs SAE Comparison',
    description: 'SAE features capture novel biological directions invisible to PCA.',
    to: '/comparisons',
    accent: 'orange',
  },
] as const

const ACCENT_RING: Record<string, string> = {
  blue: 'ring-blue-500/30 hover:ring-blue-400/50',
  green: 'ring-green-500/30 hover:ring-green-400/50',
  purple: 'ring-purple-500/30 hover:ring-purple-400/50',
  orange: 'ring-orange-500/30 hover:ring-orange-400/50',
}

const ACCENT_TEXT: Record<string, string> = {
  blue: 'text-blue-400',
  green: 'text-green-400',
  purple: 'text-purple-400',
  orange: 'text-orange-400',
}

export default function Overview() {
  const { data, loading, error } = useGlobalSummary()

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
          Failed to load summary: {error}
        </div>
      </div>
    )
  }

  if (!data) return null

  const layers = data.layers.slice().sort((a, b) => a.layer - b.layer)
  const layerNumbers = layers.map(l => l.layer)

  // Variance explained range
  const veValues = layers.map(l => l.variance_explained).filter((v): v is number => v != null)
  const veMin = veValues.length > 0 ? Math.min(...veValues) : 0
  const veMax = veValues.length > 0 ? Math.max(...veValues) : 0

  // Heatmap data: rows = layers, columns = ontology types
  const heatmapRaw = layers.map(l =>
    HEATMAP_ONTOLOGIES.map(ont => {
      if (l.ontology_counts[ont] != null) return l.ontology_counts[ont]
      const match = Object.keys(l.ontology_counts).find(k => k.startsWith(ont))
      return match ? l.ontology_counts[match] : 0
    })
  )

  // Normalize each column independently to 0-1 so all ontologies show variation
  const colMins = HEATMAP_ONTOLOGIES.map((_, ci) => Math.min(...heatmapRaw.map(r => r[ci])))
  const colMaxs = HEATMAP_ONTOLOGIES.map((_, ci) => Math.max(...heatmapRaw.map(r => r[ci])))
  const heatmapZ = heatmapRaw.map(row =>
    row.map((val, ci) => {
      const range = colMaxs[ci] - colMins[ci]
      return range > 0 ? (val - colMins[ci]) / range : 0
    })
  )

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-10">
      {/* Hero */}
      <section className="text-center space-y-3 py-6">
        <h1 className="text-4xl font-bold text-white tracking-tight">
          scGPT Feature Atlas
        </h1>
        <p className="text-lg text-gray-400 max-w-3xl mx-auto leading-relaxed">
          Biological features across 12 layers of scGPT whole-human (33M cells),
          decomposed by Sparse Autoencoders
        </p>
      </section>

      {/* Stat cards */}
      <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard label="Total Features" value={fmtK(data.total_features)} />
        <StatCard label="Alive" value={fmtK(data.total_alive)} />
        <StatCard
          label="Annotated"
          value={fmtK(data.total_annotated)}
          sub={fmtPct(data.total_annotated / data.total_alive)}
        />
        <StatCard label="Modules" value={data.total_modules.toString()} />
        <StatCard
          label="Features / Layer"
          value={data.n_features_per_layer.toLocaleString()}
          sub={`${data.n_layers} layers`}
        />
        <StatCard
          label="Var. Explained"
          value={`${fmtPct(veMin)}-${fmtPct(veMax)}`}
        />
      </section>

      {/* Annotation rate line chart */}
      <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-200 mb-4">
          Annotation Rate by Layer
        </h2>
        <Plot
          data={[
            {
              x: layerNumbers,
              y: layers.map(l => l.annotation_rate * 100),
              type: 'scatter',
              mode: 'lines+markers',
              marker: {
                color: layerNumbers.map(l => layerColor(l)),
                size: 8,
              },
              line: { color: '#60a5fa', width: 2 },
              hovertemplate: 'Layer %{x}<br>Annotation rate: %{y:.1f}%<extra></extra>',
            },
          ]}
          layout={{
            ...PLOTLY_LAYOUT_BASE,
            height: 340,
            xaxis: {
              ...PLOTLY_LAYOUT_BASE.xaxis,
              title: { text: 'Layer', standoff: 10 },
              dtick: 1,
            },
            yaxis: {
              ...PLOTLY_LAYOUT_BASE.yaxis,
              title: { text: 'Annotation Rate (%)', standoff: 10 },
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          useResizeHandler
          style={{ width: '100%' }}
        />
      </section>

      {/* Layer x ontology heatmap */}
      <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-200 mb-4">
          Ontology Coverage by Layer
        </h2>
        <Plot
          data={[
            {
              z: heatmapZ,
              x: HEATMAP_ONTOLOGIES.map(o => ONTOLOGY_LABELS[o] || o),
              y: layerNumbers.map(l => `L${l}`),
              type: 'heatmap',
              colorscale: [
                [0, '#111827'],
                [0.25, '#1e3a5f'],
                [0.5, '#2563eb'],
                [0.75, '#60a5fa'],
                [1, '#dbeafe'],
              ],
              customdata: heatmapRaw,
              hovertemplate:
                '%{y} &mdash; %{x}<br>Count: %{customdata}<extra></extra>',
              showscale: false,
            },
          ]}
          layout={{
            ...PLOTLY_LAYOUT_BASE,
            height: 500,
            margin: { t: 40, r: 30, b: 100, l: 60 },
            xaxis: {
              ...PLOTLY_LAYOUT_BASE.xaxis,
              tickangle: -35,
              tickfont: { size: 11, color: '#d1d5db' },
            },
            yaxis: {
              ...PLOTLY_LAYOUT_BASE.yaxis,
              autorange: 'reversed' as const,
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          useResizeHandler
          style={{ width: '100%' }}
        />
        <div className="mt-3 flex flex-wrap gap-4 text-xs text-gray-500">
          {HEATMAP_ONTOLOGIES.map(ont => (
            <span key={ont} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ backgroundColor: ONTOLOGY_COLORS[ont] || '#6b7280' }}
              />
              {ONTOLOGY_LABELS[ont] || ont}
            </span>
          ))}
        </div>
        <p className="mt-2 text-xs text-gray-600 italic">
          Colors are normalized per ontology column to show within-ontology variation. Hover for raw annotation counts.
        </p>
      </section>

      {/* Key findings */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-200">Key Findings</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {KEY_FINDINGS.map(card => (
            <Link
              key={card.to}
              to={card.to}
              className={`block bg-gray-900/50 rounded-xl border border-gray-800 p-5 ring-1 ${ACCENT_RING[card.accent]} transition-all hover:bg-gray-900/80`}
            >
              <h3 className={`font-semibold mb-2 ${ACCENT_TEXT[card.accent]}`}>
                {card.title}
              </h3>
              <p className="text-sm text-gray-400 leading-relaxed">
                {card.description}
              </p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800 px-4 py-5 text-center">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs text-gray-500 mt-1 uppercase tracking-wider">{label}</div>
      {sub && <div className="text-sm text-blue-400 mt-0.5">{sub}</div>}
    </div>
  )
}
