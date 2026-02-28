/**
 * Tooltip + InfoIcon — hover tooltip with smart viewport-aware positioning.
 * Auto-flips vertically and adjusts horizontal alignment to stay on-screen.
 */
import { useState, useRef, useCallback } from 'react'

type Position = 'top' | 'bottom' | 'left' | 'right'
type HAlign = 'center' | 'left' | 'right'

interface TooltipProps {
  tip: string
  position?: Position
  children: React.ReactNode
}

const TOOLTIP_WIDTH = 288 // w-72 = 18rem = 288px

/**
 * Wraps any element with a hover tooltip.
 * Auto-flips top↔bottom and shifts horizontally to stay within viewport.
 */
export function Tooltip({ tip, position = 'top', children }: TooltipProps) {
  const [pos, setPos] = useState<Position>(position)
  const [hAlign, setHAlign] = useState<HAlign>('center')
  const ref = useRef<HTMLSpanElement>(null)

  const handleMouseEnter = useCallback(() => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()

    // Vertical: flip top↔bottom if too close to edge
    if (position === 'top' && rect.top < 100) {
      setPos('bottom')
    } else if (position === 'bottom' && window.innerHeight - rect.bottom < 100) {
      setPos('top')
    } else {
      setPos(position)
    }

    // Horizontal: adjust alignment for top/bottom tooltips
    if (position === 'top' || position === 'bottom') {
      const centerX = rect.left + rect.width / 2
      if (centerX - TOOLTIP_WIDTH / 2 < 8) {
        setHAlign('left')
      } else if (centerX + TOOLTIP_WIDTH / 2 > window.innerWidth - 8) {
        setHAlign('right')
      } else {
        setHAlign('center')
      }
    }
  }, [position])

  // Build position classes
  let posClass: string
  if (pos === 'top' || pos === 'bottom') {
    const vertical = pos === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'
    const horizontal =
      hAlign === 'center' ? 'left-1/2 -translate-x-1/2' :
      hAlign === 'left' ? 'left-0' : 'right-0'
    posClass = `${vertical} ${horizontal}`
  } else if (pos === 'left') {
    posClass = 'right-full top-1/2 -translate-y-1/2 mr-2'
  } else {
    posClass = 'left-full top-1/2 -translate-y-1/2 ml-2'
  }

  return (
    <span className="relative inline-flex group" ref={ref} onMouseEnter={handleMouseEnter}>
      {children}
      <span
        className={`absolute ${posClass} z-50 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 delay-150 w-72 px-3 py-2 text-xs leading-relaxed text-gray-200 bg-gray-800 border border-gray-700 rounded-lg shadow-xl whitespace-normal`}
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
