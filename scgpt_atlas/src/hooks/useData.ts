import { useState, useEffect, useRef } from 'react'
import type {
  GlobalSummary, FeatureCompact, FeatureAnnotations,
  Module, CausalFeature, PerturbationTarget,
  GeneIndexEntry, OntologyIndexEntry, CrossLayerPair,
  CellTypeData,
} from '../lib/types'

const BASE = import.meta.env.BASE_URL + 'data/'
const cache = new Map<string, unknown>()

async function fetchJson<T>(path: string): Promise<T> {
  const key = path
  if (cache.has(key)) return cache.get(key) as T
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`)
  const data = await res.json()
  cache.set(key, data)
  return data as T
}

function useAsync<T>(loader: () => Promise<T>, deps: unknown[]): { data: T | null; loading: boolean; error: string | null } {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    setLoading(true)
    setError(null)
    loader()
      .then(d => { if (mounted.current) { setData(d); setLoading(false) } })
      .catch(e => { if (mounted.current) { setError(e.message); setLoading(false) } })
    return () => { mounted.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { data, loading, error }
}

export function useGlobalSummary() {
  return useAsync(() => fetchJson<GlobalSummary>('global_summary.json'), [])
}

export function useLayerFeatures(layer: number) {
  return useAsync(
    () => fetchJson<FeatureCompact[]>(`layer_${String(layer).padStart(2, '0')}_features.json`),
    [layer]
  )
}

export function useLayerPositions(layer: number) {
  return useAsync(
    () => fetchJson<number[][]>(`layer_${String(layer).padStart(2, '0')}_positions.json`),
    [layer]
  )
}

export function useLayerAnnotations(layer: number) {
  return useAsync(
    () => fetchJson<Record<string, FeatureAnnotations>>(`layer_${String(layer).padStart(2, '0')}_annotations.json`),
    [layer]
  )
}

export function useModules() {
  return useAsync(() => fetchJson<Module[]>('modules.json'), [])
}

export function useCrossLayerGraph() {
  return useAsync(() => fetchJson<Record<string, CrossLayerPair>>('cross_layer_graph.json'), [])
}

export function useCausalPatching() {
  return useAsync(() => fetchJson<{ summary: Record<string, unknown>; features: CausalFeature[] }>('causal_patching.json'), [])
}

export function usePerturbationResponse() {
  return useAsync(() => fetchJson<{ summary: Record<string, unknown>; targets: PerturbationTarget[] }>('perturbation_response.json'), [])
}

export function useSvdComparison() {
  return useAsync(() => fetchJson<{ aggregate: Record<string, unknown>; per_layer: Record<string, Record<string, number>> }>('svd_comparison.json'), [])
}

export function useCrossLayerTracking() {
  return useAsync(async () => {
    try {
      return await fetchJson<{ adjacent_persistence: Array<Record<string, unknown>>; long_range_persistence?: Array<Record<string, unknown>> }>('cross_layer_tracking.json')
    } catch {
      return null  // optional — persistence chart will be skipped
    }
  }, [])
}

export function useNovelClusters() {
  return useAsync(() => fetchJson<Record<string, { summary: Record<string, unknown>; clusters: Array<Record<string, unknown>> }>>('novel_clusters.json'), [])
}

export function useGeneIndex() {
  return useAsync(() => fetchJson<Record<string, GeneIndexEntry[]>>('gene_index.json'), [])
}

export function useOntologyIndex() {
  return useAsync(() => fetchJson<Record<string, OntologyIndexEntry[]>>('ontology_index.json'), [])
}

export function useLayerCellTypes(layer: number) {
  return useAsync(
    async () => {
      if (layer < 0 || layer > 11) return null
      try {
        return await fetchJson<CellTypeData>(
          `layer_${String(layer).padStart(2, '0')}_celltypes.json`
        )
      } catch {
        return null
      }
    },
    [layer]
  )
}
