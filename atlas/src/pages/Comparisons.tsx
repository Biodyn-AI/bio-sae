import Plot from 'react-plotly.js'
import { useSvdComparison, useCrossLayerTracking, useCausalPatching } from '../hooks/useData'
import { fmtPct } from '../lib/utils'
import { InfoIcon } from '../components/Tooltip'
import { useIsMobile } from '../hooks/useIsMobile'
import type { CausalFeature } from '../lib/types'

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

function Spinner() {
  return (
    <div className="flex items-center justify-center h-48">
      <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

function ErrorBox({ msg }: { msg: string }) {
  return (
    <div className="text-red-400 bg-red-900/20 border border-red-800 rounded-lg px-4 py-3 text-sm">
      {msg}
    </div>
  )
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xl font-semibold text-gray-200 mb-4">{children}</h2>
  )
}

// ---------------------------------------------------------------------------
// Section 1: SVD vs SAE
// ---------------------------------------------------------------------------

function SvdVsSaeSection() {
  const { data, loading, error } = useSvdComparison()
  const isMobile = useIsMobile()

  if (loading) return <Spinner />
  if (error) return <ErrorBox msg={`SVD comparison: ${error}`} />
  if (!data) return null

  const layerKeys = Object.keys(data.per_layer)
    .sort((a, b) => parseInt(a) - parseInt(b))
  const layerLabels = layerKeys.map(k => `L${k}`)

  const svdVar = layerKeys.map(k => data.per_layer[k].svd_variance ?? 0)
  const saeVar = layerKeys.map(k => data.per_layer[k].sae_variance ?? 0)

  // Aggregate stats for display
  const agg = data.aggregate as Record<string, number | string>

  return (
    <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-6">
      <SectionHeading>
        <span className="flex items-center">SVD vs SAE Variance Explained<InfoIcon tip="Variance explained by PCA/SVD (gray) vs SAE (blue) at each layer. SAEs capture more variance because they use overcomplete, non-orthogonal bases." /></span>
      </SectionHeading>

      <Plot
        data={[
          {
            x: layerLabels,
            y: svdVar.map(v => v * 100),
            name: 'SVD (PCA)',
            type: 'bar',
            marker: { color: '#6b7280' },
            hovertemplate: '%{x}<br>SVD: %{y:.1f}%<extra></extra>',
          },
          {
            x: layerLabels,
            y: saeVar.map(v => v * 100),
            name: 'SAE',
            type: 'bar',
            marker: { color: '#3b82f6' },
            hovertemplate: '%{x}<br>SAE: %{y:.1f}%<extra></extra>',
          },
        ]}
        layout={{
          ...PLOTLY_LAYOUT_BASE,
          height: isMobile ? 280 : 380,
          barmode: 'group',
          legend: {
            font: { color: '#d1d5db' },
            bgcolor: 'transparent',
            x: 0.01,
            y: 0.99,
          },
          xaxis: {
            ...PLOTLY_LAYOUT_BASE.xaxis,
            title: { text: 'Layer', standoff: 10 },
          },
          yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            title: { text: 'Variance Explained (%)', standoff: 10 },
          },
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
        useResizeHandler
        style={{ width: '100%' }}
      />

      {/* Aggregate stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(agg).map(([key, val]) => (
          <div
            key={key}
            className="bg-gray-800/50 rounded-lg px-4 py-3 text-center"
          >
            <div className="text-lg font-bold text-white">
              {typeof val === 'number'
                ? val < 1 && val > 0
                  ? fmtPct(val)
                  : val.toLocaleString()
                : String(val)}
            </div>
            <div className="text-xs text-gray-500 mt-1 uppercase tracking-wider">
              {key.replace(/_/g, ' ')}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 2: Feature Persistence
// ---------------------------------------------------------------------------

function FeaturePersistenceSection() {
  const { data, loading, error } = useCrossLayerTracking()
  const isMobile = useIsMobile()

  if (loading) return <Spinner />
  if (error) return <ErrorBox msg={`Cross-layer tracking: ${error}`} />
  if (!data) return null

  const ap = data.adjacent_persistence as Array<{
    from_layer: number
    to_layer: number
    n_matches: number
    persistence_rate: number
    mean_match_cosine: number
  }>

  if (!ap || ap.length === 0) {
    return (
      <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
        <SectionHeading><span className="flex items-center">Feature Persistence Across Layers<InfoIcon tip="Fraction of features at layer N that have a close match at layer N+1. Low persistence = the layer is transforming representations." /></span></SectionHeading>
        <p className="text-gray-500">No adjacent persistence data available.</p>
      </section>
    )
  }

  const sorted = [...ap].sort((a, b) => a.from_layer - b.from_layer)
  const transitionLabels = sorted.map(
    d => `L${d.from_layer}\u2192L${d.to_layer}`
  )
  const persistenceRates = sorted.map(d => d.persistence_rate * 100)
  const nMatches = sorted.map(d => d.n_matches)

  return (
    <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-4">
      <SectionHeading><span className="flex items-center">Feature Persistence Across Layers<InfoIcon tip="Fraction of features at layer N that have a close match at layer N+1. Low persistence = the layer is transforming representations." /></span></SectionHeading>

      <Plot
        data={[
          {
            x: transitionLabels,
            y: persistenceRates,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#a855f7', size: 7 },
            line: { color: '#a855f7', width: 2 },
            hovertemplate:
              '%{x}<br>Persistence: %{y:.1f}%<br>Matches: %{customdata}<extra></extra>',
            customdata: nMatches,
          },
        ]}
        layout={{
          ...PLOTLY_LAYOUT_BASE,
          height: isMobile ? 260 : 340,
          xaxis: {
            ...PLOTLY_LAYOUT_BASE.xaxis,
            title: { text: 'Layer Transition', standoff: 10 },
            tickangle: -45,
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

      <p className="text-sm text-gray-500 leading-relaxed">
        Persistence rate measures the fraction of features in layer N that have a
        close cosine-similarity match in layer N+1. Low persistence indicates
        representational transformation between layers.
      </p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Section 3: Causal Specificity
// ---------------------------------------------------------------------------

function CausalSpecificitySection() {
  const { data, loading, error } = useCausalPatching()
  const isMobile = useIsMobile()

  if (loading) return <Spinner />
  if (error) return <ErrorBox msg={`Causal patching: ${error}`} />
  if (!data) return null

  const features = (data.features as CausalFeature[]) || []

  // Sort by specificity ratio descending, take top 50
  const sorted = [...features]
    .sort((a, b) => b.sr - a.sr)
    .slice(0, 50)

  const labels = sorted.map(
    f => f.lb ? `F${f.i}: ${f.lb.length > 30 ? f.lb.slice(0, 29) + '\u2026' : f.lb}` : `F${f.i}`
  )
  const srValues = sorted.map(f => f.sr)

  // Color by specificity tier
  const barColors = sorted.map(f => {
    if (f.sr > 10) return '#ef4444'  // red: highly specific
    if (f.sr > 2) return '#f97316'   // orange: moderately specific
    return '#6b7280'                  // gray: low specificity
  })

  return (
    <section className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-4">
      <SectionHeading><span className="flex items-center">Causal Specificity of SAE Features<InfoIcon tip="Top 50 features ranked by specificity ratio from activation patching experiments." /></span></SectionHeading>

      <div className="flex flex-wrap gap-4 text-xs text-gray-500 mb-2">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#ef4444' }} />
          {'> 10x specificity'}
          <InfoIcon tip="Features with >10x specificity — they affect their target genes much more than off-target genes." position="bottom" />
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#f97316' }} />
          {'> 2x specificity'}
          <InfoIcon tip="Moderately specific features (2-10x specificity ratio)." position="bottom" />
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#6b7280' }} />
          {'< 2x specificity'}
          <InfoIcon tip="Features with <2x specificity — they affect many genes similarly (polysemantic)." position="bottom" />
        </span>
      </div>

      <Plot
        data={[
          {
            x: srValues,
            y: labels,
            type: 'bar',
            orientation: 'h',
            marker: { color: barColors },
            hovertemplate:
              '%{y}<br>Specificity ratio: %{x:.1f}x<extra></extra>',
          },
        ]}
        layout={{
          ...PLOTLY_LAYOUT_BASE,
          height: Math.max(400, sorted.length * 22),
          margin: { t: 20, r: 30, b: 50, l: isMobile ? 150 : 250 },
          xaxis: {
            ...PLOTLY_LAYOUT_BASE.xaxis,
            title: { text: 'Specificity Ratio (target / off-target)', standoff: 10 },
            type: 'log',
          },
          yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            autorange: 'reversed' as const,
            tickfont: { size: 10, color: '#9ca3af' },
          },
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
        useResizeHandler
        style={{ width: '100%' }}
      />

      <p className="text-sm text-gray-500 leading-relaxed">
        Specificity ratio measures how selectively a feature&apos;s causal patch
        affects its target gene logit versus off-target genes. Higher values
        indicate more precise, monosemantic causal roles.
      </p>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Comparisons() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-10">
      <section className="space-y-2">
        <h1 className="text-3xl font-bold text-white">Comparisons</h1>
        <p className="text-gray-400">
          SVD baseline comparisons, cross-layer feature persistence, and causal
          specificity analysis for SAE features.
        </p>
      </section>

      <SvdVsSaeSection />
      <FeaturePersistenceSection />
      <CausalSpecificitySection />
    </div>
  )
}
