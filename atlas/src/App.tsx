import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import Overview from './pages/Overview'
import LayerExplorer from './pages/LayerExplorer'
import FeaturePage from './pages/FeaturePage'
import ModuleExplorer from './pages/ModuleExplorer'
import CrossLayerFlow from './pages/CrossLayerFlow'
import GeneSearch from './pages/GeneSearch'
import OntologySearch from './pages/OntologySearch'
import Comparisons from './pages/Comparisons'
import About from './pages/About'

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/layer/0', label: 'Layer Explorer' },
  { to: '/modules', label: 'Modules' },
  { to: '/flow', label: 'Cross-Layer Flow' },
  { to: '/genes', label: 'Gene Search' },
  { to: '/pathways', label: 'Pathway Search' },
  { to: '/comparisons', label: 'Comparisons' },
  { to: '/about', label: 'About' },
]

function isLayerActive(_: { isActive: boolean }): string {
  return ''
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()

  // Close mobile menu on route change
  useEffect(() => { setMenuOpen(false) }, [location.pathname])

  return (
    <div className="min-h-screen flex flex-col">
      <nav className="bg-gray-900 border-b border-gray-800 px-4 py-2 flex items-center gap-6 flex-shrink-0 relative">
        <NavLink to="/" className="text-lg font-bold text-blue-400 hover:text-blue-300 whitespace-nowrap">
          Geneformer Atlas
        </NavLink>
        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-1 overflow-x-auto">
          {navItems.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm whitespace-nowrap transition-colors ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-300'
                    : item.to.startsWith('/layer')
                      ? isLayerActive({ isActive }) || 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
        {/* Mobile hamburger */}
        <button
          className="md:hidden ml-auto p-1.5 rounded text-gray-400 hover:text-gray-200 hover:bg-gray-800"
          onClick={() => setMenuOpen(o => !o)}
          aria-label="Toggle navigation menu"
        >
          {menuOpen ? (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
        {/* Mobile dropdown */}
        {menuOpen && (
          <div className="absolute top-full left-0 right-0 bg-gray-900 border-b border-gray-800 shadow-xl z-50 md:hidden">
            <div className="flex flex-col p-2 gap-1">
              {navItems.map(item => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    `px-4 py-2.5 rounded text-sm transition-colors ${
                      isActive
                        ? 'bg-blue-600/20 text-blue-300'
                        : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
        )}
      </nav>
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/layer/:layerId" element={<LayerExplorer />} />
          <Route path="/feature/:layerId/:featureId" element={<FeaturePage />} />
          <Route path="/modules" element={<ModuleExplorer />} />
          <Route path="/flow" element={<CrossLayerFlow />} />
          <Route path="/genes" element={<GeneSearch />} />
          <Route path="/pathways" element={<OntologySearch />} />
          <Route path="/comparisons" element={<Comparisons />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  )
}
