/**
 * Tooltip + InfoIcon — hover tooltip with smart viewport-aware positioning.
 */
import { useState, useRef, useCallback } from 'react'

type Position = 'top' | 'bottom' | 'left' | 'right'

interface TooltipProps {
  tip: string
  position?: Position
  children: React.ReactNode
}

const POS_CLASSES: Record<Position, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
}

/**
 * Wraps any element with a hover tooltip.
 * Auto-flips top↔bottom when too close to the viewport edge.
 */
export function Tooltip({ tip, position = 'top', children }: TooltipProps) {
  const [pos, setPos] = useState<Position>(position)
  const ref = useRef<HTMLSpanElement>(null)

  const handleMouseEnter = useCallback(() => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    if (position === 'top' && rect.top < 100) {
      setPos('bottom')
    } else if (position === 'bottom' && window.innerHeight - rect.bottom < 100) {
      setPos('top')
    } else {
      setPos(position)
    }
  }, [position])

  return (
    <span className="relative inline-flex group" ref={ref} onMouseEnter={handleMouseEnter}>
      {children}
      <span
        className={`absolute ${POS_CLASSES[pos]} z-50 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 delay-150 w-72 px-3 py-2 text-xs leading-relaxed text-gray-200 bg-gray-800 border border-gray-700 rounded-lg shadow-xl whitespace-normal`}
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
