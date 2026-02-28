import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useGlobalSummary } from '../hooks/useData'
import { ONTOLOGY_COLORS, ONTOLOGY_LABELS, layerColor } from '../lib/colors'
import { fmtPct, fmtK } from '../lib/utils'
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

const HEATMAP_ONTOLOGIES = ['GO_BP', 'KEGG', 'Reactome', 'STRING', 'TRRUST'] as const

const KEY_FINDINGS = [
  {
    title: 'Layer Explorer',
    description: 'Dive into per-layer feature catalogs with activation patterns, gene loadings, and ontology annotations.',
    to: '/layer/11',
    accent: 'blue',
    tip: 'Browse all features at each transformer layer, visualized as a UMAP scatter plot.',
  },
  {
    title: '141 Biological Modules',
    description: 'Co-activating feature clusters that map to coherent biological programs across cell types.',
    to: '/modules',
    accent: 'green',
    tip: 'Co-activating feature clusters that represent coherent biological programs (e.g., cell cycle, immune signaling).',
  },
  {
    title: 'Cross-Layer Information Flow',
    description: 'Track how biological representations transform and persist across the 18-layer residual stream.',
    to: '/flow',
    accent: 'purple',
    tip: 'How features transform across layers \u2014 which early-layer features have analogs in later layers.',
  },
  {
    title: 'SVD vs SAE Comparison',
    description: '99.8% of SAE features are novel directions invisible to PCA, capturing fine-grained biology.',
    to: '/comparisons',
    accent: 'orange',
    tip: 'Comparison with PCA/SVD baseline. Most SAE features capture novel directions invisible to linear decomposition.',
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
  const isMobile = useIsMobile()

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

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-10">
      {/* Hero */}
      <section className="text-center space-y-3 py-6">
        <h1 className="text-4xl font-bold text-white tracking-tight">
          Geneformer Feature Atlas
        </h1>
        <p className="text-lg text-gray-400 max-w-3xl mx-auto leading-relaxed">
          82,525 biological features across 18 layers of Geneformer V2-316M,
          decomposed by Sparse Autoencoders
        </p>
      </section>

      {/* Stat cards */}
      <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard label="Total Features" value={fmtK(data.total_features)} tip="Total number of SAE dictionary elements (4,608 per layer x 18 layers). Each feature is a learned direction in the model's activation space." />
        <StatCard label="Alive" value={fmtK(data.total_alive)} tip="Features that activate at least once on test data. Dead features never fire and are excluded from analysis." />
        <StatCard
          label="Annotated"
          value={fmtK(data.total_annotated)}
          sub={fmtPct(data.total_annotated / data.total_alive)}
          tip="Features with at least one statistically significant ontology enrichment (Fisher's exact test, BH-corrected p < 0.05)."
        />
        <StatCard label="Modules" value={data.total_modules.toString()} tip="Groups of features that co-activate on the same cells, identified by Leiden clustering of the co-activation graph." />
        <StatCard
          label="Features / Layer"
          value={data.n_features_per_layer.toLocaleString()}
          sub={`${data.n_layers} layers`}
          tip="Number of dictionary elements per SAE. The 4x expansion ratio (1,152 to 4,608) enables fine-grained decomposition."
        />
        <StatCard
          label="Var. Explained"
          value={`${fmtPct(veMin)}-${fmtPct(veMax)}`}
          tip="Fraction of the original activation variance captured by the SAE reconstruction. Higher is better."
        />
      </section>

      {/* Annotation rate line chart */}
      <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-200 mb-4 flex items-center">
          Annotation Rate by Layer
          <InfoIcon tip="Percentage of alive features with significant ontology enrichment at each layer. The U-shape suggests early and late layers capture the most interpretable biology." />
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
            height: isMobile ? 260 : 340,
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

      {/* Layer x ontology grouped bar chart */}
      <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-gray-200 mb-4 flex items-center">
          Ontology Coverage by Layer
          <InfoIcon tip="Number of features enriched in each ontology database per layer. Hover bars for exact counts." />
        </h2>
        <Plot
          data={HEATMAP_ONTOLOGIES.map(ont => ({
            x: layerNumbers.map(l => `L${l}`),
            y: layers.map(l => {
              if (l.ontology_counts[ont] != null) return l.ontology_counts[ont]
              const match = Object.keys(l.ontology_counts).find(k => k.startsWith(ont))
              return match ? l.ontology_counts[match] : 0
            }),
            type: 'bar' as const,
            name: ONTOLOGY_LABELS[ont] || ont,
            marker: { color: ONTOLOGY_COLORS[ont] || '#6b7280' },
            hovertemplate: `${ONTOLOGY_LABELS[ont] || ont}<br>%{x}: %{y:,}<extra></extra>`,
          }))}
          layout={{
            ...PLOTLY_LAYOUT_BASE,
            height: isMobile ? 350 : 500,
            margin: { t: 40, r: 30, b: 50, l: 70 },
            barmode: 'group',
            xaxis: {
              ...PLOTLY_LAYOUT_BASE.xaxis,
              title: { text: 'Layer', standoff: 10 },
              tickfont: { size: 11, color: '#d1d5db' },
            },
            yaxis: {
              ...PLOTLY_LAYOUT_BASE.yaxis,
              title: { text: 'Enrichment Count', standoff: 10 },
            },
            legend: {
              font: { color: '#d1d5db', size: 11 },
              bgcolor: 'transparent',
              orientation: 'h' as const,
              x: 0.5,
              xanchor: 'center' as const,
              y: -0.15,
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          useResizeHandler
          style={{ width: '100%' }}
        />
        <p className="mt-2 text-xs text-gray-600 italic">
          Raw annotation counts per ontology database. Hover for exact values.
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
              <h3 className={`font-semibold mb-2 flex items-center ${ACCENT_TEXT[card.accent]}`}>
                {card.title}
                <InfoIcon tip={card.tip} position="bottom" />
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

function StatCard({ label, value, sub, tip }: { label: string; value: string; sub?: string; tip?: string }) {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800 px-4 py-5 text-center">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs text-gray-500 mt-1 uppercase tracking-wider flex items-center justify-center">
        {label}
        {tip && <InfoIcon tip={tip} position="bottom" />}
      </div>
      {sub && <div className="text-sm text-blue-400 mt-0.5">{sub}</div>}
    </div>
  )
}
