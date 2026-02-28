declare module 'react-plotly.js' {
  import { Component } from 'react'

  interface PlotParams {
    data: Array<Record<string, unknown>>
    layout?: Record<string, unknown>
    config?: Record<string, unknown>
    style?: React.CSSProperties
    className?: string
    useResizeHandler?: boolean
    onHover?: (event: unknown) => void
    onClick?: (event: unknown) => void
    onSelected?: (event: unknown) => void
    revision?: number
  }

  class Plot extends Component<PlotParams> {}
  export default Plot
}
