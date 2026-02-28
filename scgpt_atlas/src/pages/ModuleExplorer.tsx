import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useModules } from '../hooks/useData'
import { moduleColor } from '../lib/colors'

type SortKey = 'size' | 'layer'

export default function ModuleExplorer() {
  const { data, loading, error } = useModules()
  const [layerFilter, setLayerFilter] = useState<number | 'all'>('all')
  const [sortBy, setSortBy] = useState<SortKey>('layer')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const filtered = useMemo(() => {
    if (!data) return []
    let modules = layerFilter === 'all'
      ? data
      : data.filter(m => m.layer === layerFilter)
    if (sortBy === 'size') {
      modules = modules.slice().sort((a, b) => b.n - a.n)
    } else {
      modules = modules.slice().sort((a, b) => a.layer - b.layer || a.id - b.id)
    }
    return modules
  }, [data, layerFilter, sortBy])

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
          Failed to load modules: {error}
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <h1 className="text-3xl font-bold text-white tracking-tight">
        {data.length} Biological Modules across 12 Layers
      </h1>

      {/* Filter row */}
      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-gray-400">
          Layer
          <select
            value={layerFilter === 'all' ? 'all' : layerFilter}
            onChange={e => {
              const v = e.target.value
              setLayerFilter(v === 'all' ? 'all' : Number(v))
            }}
            className="bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-gray-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">All</option>
            {Array.from({ length: 12 }, (_, i) => (
              <option key={i} value={i}>
                Layer {i}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 text-sm text-gray-400">
          Sort by
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value as SortKey)}
            className="bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-gray-200 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="layer">Layer</option>
            <option value="size">Size</option>
          </select>
        </label>

        <span className="text-sm text-gray-500">
          Showing {filtered.length} module{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Module grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map(mod => {
          const key = `L${mod.layer}_M${mod.id}`
          const isExpanded = expandedId === key
          return (
            <div
              key={key}
              className="bg-gray-900/50 rounded-xl border border-gray-800 overflow-hidden cursor-pointer hover:bg-gray-900/80 transition-colors"
              onClick={() => setExpandedId(isExpanded ? null : key)}
            >
              {/* Card header with colored left border */}
              <div
                className="px-4 py-3 border-l-4"
                style={{ borderLeftColor: moduleColor(mod.id) }}
              >
                <div className="flex items-baseline justify-between">
                  <h3 className="font-semibold text-gray-100">
                    L{mod.layer} Module {mod.id}
                  </h3>
                  <span className="text-xs text-gray-500">
                    {mod.n} features
                  </span>
                </div>
              </div>

              {/* Top annotations */}
              <div className="px-4 pb-3 space-y-1">
                {mod.top_anns.length > 0 ? (
                  <ul className="space-y-0.5">
                    {mod.top_anns.map((ann, j) => (
                      <li
                        key={j}
                        className="flex items-baseline justify-between text-sm"
                      >
                        <span className="text-gray-300 truncate mr-2">
                          {ann.t}
                        </span>
                        <span className="text-gray-500 text-xs shrink-0">
                          {ann.c}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-gray-600 italic">No annotations</p>
                )}
              </div>

              {/* Expanded feature list */}
              {isExpanded && (
                <div className="border-t border-gray-800 px-4 py-3">
                  <p className="text-xs text-gray-500 mb-2">
                    Feature indices ({mod.features.length})
                  </p>
                  <div className="max-h-40 overflow-y-auto flex flex-wrap gap-1.5">
                    {mod.features.map(fi => (
                      <Link
                        key={fi}
                        to={`/feature/${mod.layer}/${fi}`}
                        onClick={e => e.stopPropagation()}
                        className="inline-block text-xs bg-gray-800 hover:bg-gray-700 text-blue-400 hover:text-blue-300 rounded px-2 py-0.5 transition-colors"
                      >
                        F{fi}
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
