export interface LayerSummary {
  layer: number
  alive: number
  dead: number
  annotated: number
  annotation_rate: number
  n_modules: number
  n_svd_aligned: number
  n_novel: number
  ontology_counts: Record<string, number>
  mean_feature_cosine: number
  variance_explained?: number
}

export interface GlobalSummary {
  total_features: number
  total_alive: number
  total_annotated: number
  total_modules: number
  total_novel: number
  n_layers: number
  n_features_per_layer: number
  layers: LayerSummary[]
}

export interface FeatureCompact {
  i: number    // index
  d: boolean   // dead
  f: number    // activation_freq
  ma: number   // mean_activation
  fc: number   // fire_count
  na: number   // n_annotations
  m: number    // module_id (-1 = unassigned)
  sv: boolean  // svd_aligned
  lb: string   // best annotation label
  to: string   // top ontology
  tg: { n: string; a: number }[] // top genes
}

export interface AnnotationEntry {
  o: string   // ontology
  t: string   // term
  p: number   // p_adjusted
  or: number  // odds_ratio
  n: number   // n_overlap
  g: string[] // overlap_genes
}

export interface FeatureAnnotations {
  genes: { n: string; a: number; fc: number }[]
  anns: AnnotationEntry[]
}

export interface Module {
  layer: number
  id: number
  n: number
  features: number[]
  top_anns: { t: string; c: number }[]
}

export interface CrossLayerDep {
  a: number
  deps: { b: number; pmi: number; lb: string }[]
}

export interface CrossLayerPair {
  summary: Record<string, unknown>
  deps: CrossLayerDep[]
}

export interface CausalFeature {
  i: number
  lb: string
  na: number
  af: number
  tg: string[]
  td: number  // target_logit_diff
  od: number  // other_logit_diff
  sr: number  // specificity_ratio
}

export interface PerturbationTarget {
  gene: string
  tf: boolean
  nk: number
  nr: number
  ns: number
  top: { i: number; es: number; lb: string }[]
}

export interface GeneIndexEntry {
  l: number  // layer
  i: number  // feature index
  r: number  // rank in top genes
  lb: string // label
  m: number  // module
}

export interface OntologyIndexEntry {
  l: number
  i: number
  p: number
  o: string
}

export interface CellTypeEnrichment {
  c: string   // cell type name
  p: number   // p_adjusted
  or: number  // odds_ratio
  n: number   // n_top (count in top set)
}

export interface TissueEnrichment {
  t: string   // tissue name
  p: number   // p_adjusted
  or: number  // odds_ratio
}

export interface TopCell {
  ct: string  // cell type
  t: string   // tissue
  a: number   // activation (mean-when-active)
}

export interface FeatureCellTypes {
  ct: CellTypeEnrichment[]
  ti: TissueEnrichment[]
  tc: TopCell[]
}

export interface CellTypeMeta {
  n: number   // count
  t: string   // tissues (comma-separated)
}

export interface CellTypeData {
  summary: {
    n_cells: number
    tissues: string[]
    n_cell_types: number
    n_features_with_enrichment: number
  }
  cell_type_meta: Record<string, CellTypeMeta>
  features: Record<string, FeatureCellTypes>
}
