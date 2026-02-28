/**
 * Tooltip + InfoIcon — hover tooltip with fixed positioning.
 * Uses position:fixed so tooltips escape parent overflow/clipping (e.g. narrow sidebars).
 * Auto-flips vertically and clamps horizontally to stay within the viewport.
 */
import { useState, useRef, useCallback } from 'react'

type Position = 'top' | 'bottom' | 'left' | 'right'

interface TooltipProps {
  tip: string
  position?: Position
  children: React.ReactNode
}

const TW = 288 // tooltip width: w-72 = 18rem = 288px
const GAP = 8  // spacing between trigger and tooltip

export function Tooltip({ tip, position = 'top', children }: TooltipProps) {
  const [coords, setCoords] = useState<React.CSSProperties>({ top: -9999, left: -9999 })
  const ref = useRef<HTMLSpanElement>(null)

  const handleMouseEnter = useCallback(() => {
    if (!ref.current) return
    const r = ref.current.getBoundingClientRect()

    // Determine effective vertical position (flip if near edge)
    let vPos = position
    if (position === 'top' && r.top < 100) vPos = 'bottom'
    else if (position === 'bottom' && window.innerHeight - r.bottom < 100) vPos = 'top'

    let top: number, left: number, transform: string

    switch (vPos) {
      case 'top':
        top = r.top - GAP
        left = r.left + r.width / 2
        transform = 'translate(-50%, -100%)'
        break
      case 'bottom':
        top = r.bottom + GAP
        left = r.left + r.width / 2
        transform = 'translateX(-50%)'
        break
      case 'left':
        top = r.top + r.height / 2
        left = r.left - GAP
        transform = 'translate(-100%, -50%)'
        break
      default: // right
        top = r.top + r.height / 2
        left = r.right + GAP
        transform = 'translateY(-50%)'
        break
    }

    // Clamp horizontally for top/bottom so tooltip stays in viewport
    if (vPos === 'top' || vPos === 'bottom') {
      const tooltipLeft = left - TW / 2
      if (tooltipLeft < GAP) {
        left = GAP
        transform = vPos === 'top' ? 'translateY(-100%)' : ''
      } else if (tooltipLeft + TW > window.innerWidth - GAP) {
        left = window.innerWidth - TW - GAP
        transform = vPos === 'top' ? 'translateY(-100%)' : ''
      }
    }

    setCoords({ top, left, transform })
  }, [position])

  return (
    <span className="relative inline-flex group" ref={ref} onMouseEnter={handleMouseEnter}>
      {children}
      <span
        className="fixed z-[100] pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 delay-150 w-72 px-3 py-2 text-xs leading-relaxed text-gray-200 bg-gray-800 border border-gray-700 rounded-lg shadow-xl whitespace-normal"
        style={coords}
      >
        {tip}
      </span>
    </span>
  )
}

/**
 * Small (?) icon that shows a tooltip on hover. Place next to labels.
 */
export function InfoIcon({ tip, position = 'top' }: { tip: string; position?: Position }) {
  return (
    <Tooltip tip={tip} position={position}>
      <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-gray-700 text-gray-400 text-[9px] font-bold cursor-help ml-1 flex-shrink-0 hover:bg-gray-600 hover:text-gray-300 transition-colors">
        ?
      </span>
    </Tooltip>
  )
}
